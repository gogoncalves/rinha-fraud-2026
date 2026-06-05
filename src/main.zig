const std = @import("std");
const posix = std.posix;
const linux = std.os.linux;
const Index = @import("index.zig").Index;
const http = @import("http.zig");
const norm = @import("normalize.zig");

const MAX_CONNS: u32 = 1024;
const BUF_SIZE: usize = 8192;

const Conn = struct {
    fd: posix.fd_t = -1,
    in_len: u32 = 0,
    out: []const u8 = "",
    out_pos: u32 = 0,
    in_buf: [BUF_SIZE]u8 align(64) = undefined,
};

const KIND_LISTEN: u64 = 0;
const KIND_CTRL_ACCEPT: u64 = 1;
const KIND_CTRL_CONN: u64 = 2;
const KIND_CLIENT: u64 = 3;

inline fn epollKey(kind: u64, val: u32) u64 {
    return (kind << 32) | val;
}
inline fn epollKind(k: u64) u64 { return k >> 32; }
inline fn epollVal(k: u64) u32 { return @intCast(k & 0xFFFF_FFFF); }

// Per-worker state. Each worker has its own epoll fd, conn pool, control socket
// and listen socket. The mmap'd index is read-only and shared across workers
// via the parent process (no COW, no copy).
const Worker = struct {
    idx: *const Index,
    conns: [MAX_CONNS]Conn = undefined,
    free_idx: [MAX_CONNS]u16 = undefined,
    free_count: u32 = 0,
    ctrl_conn_fd: posix.fd_t = -1,
    ctrl_listen_fd: posix.fd_t = -1,
    legacy_listen_fd: posix.fd_t = -1,
    epfd: posix.fd_t = -1,
    ctrl_path: ?[*:0]const u8,
    sock_path: ?[*:0]const u8,

    fn initPool(self: *Worker) void {
        for (0..MAX_CONNS) |i| {
            self.conns[i] = .{};
            self.free_idx[MAX_CONNS - 1 - i] = @intCast(i);
        }
        self.free_count = MAX_CONNS;
    }

    fn run(self: *Worker) !void {
        // Per-thread scheduling tweaks. Both best-effort: shrink timer slack
        // from the default 50µs to 1ns (cuts wake-up jitter on p99) and promote
        // to SCHED_FIFO so an inbound packet preempts SCHED_OTHER context
        // (softirq, client work). Return values ignored — running without
        // CAP_SYS_NICE / RLIMIT_RTPRIO degrades gracefully to SCHED_OTHER.
        tunePerThreadSched();

        self.epfd = try posix.epoll_create1(linux.EPOLL.CLOEXEC);
        defer posix.close(self.epfd);

        setEpollBusyPoll(self.epfd);

        if (self.ctrl_path) |cp| {
            self.ctrl_listen_fd = try openCtrlListener(cp);
            var ev: linux.epoll_event = .{ .events = linux.EPOLL.IN, .data = .{ .u64 = epollKey(KIND_CTRL_ACCEPT, 0) } };
            try posix.epoll_ctl(self.epfd, linux.EPOLL.CTL_ADD, self.ctrl_listen_fd, &ev);
        } else {
            self.legacy_listen_fd = try openListener(self.sock_path);
            var ev: linux.epoll_event = .{ .events = linux.EPOLL.IN, .data = .{ .u64 = epollKey(KIND_LISTEN, 0) } };
            try posix.epoll_ctl(self.epfd, linux.EPOLL.CTL_ADD, self.legacy_listen_fd, &ev);
        }

        var events: [128]linux.epoll_event = undefined;
        const spin_ns: u64 = envU64("EPOLL_SPIN_US", 0) *% 1000;
        const idle_us: u64 = envU64("EPOLL_IDLE_US", 0);
        while (true) {
            // Phase 1: non-blocking probe — costs ~1 syscall when traffic is hot.
            var n: usize = epollWaitNoBlock(self.epfd, &events);

            // Phase 2: busy-spin tight loop probing epoll_wait(0). Burns CPU but
            // keeps the worker hot and avoids context-switch latency between bursts.
            if (n == 0 and spin_ns != 0) {
                const start = std.time.Instant.now() catch unreachable;
                while (true) {
                    std.atomic.spinLoopHint();
                    n = epollWaitNoBlock(self.epfd, &events);
                    if (n != 0) break;
                    const now = std.time.Instant.now() catch unreachable;
                    if (now.since(start) >= spin_ns) break;
                }
            }

            // Phase 3: blocking wait — epoll_pwait2 (ns precision) with fallback
            // to epoll_wait on EINTR or kernels < 5.11 (ENOSYS).
            if (n == 0) n = epollWaitBlocking(self.epfd, &events, idle_us);
            var i: usize = 0;
            while (i < n) : (i += 1) {
                const e = events[i];
                const k = epollKind(e.data.u64);
                switch (k) {
                    KIND_LISTEN => self.acceptAll(),
                    KIND_CTRL_ACCEPT => self.acceptCtrl(),
                    KIND_CTRL_CONN => self.onCtrlRecv(),
                    else => {
                        const ci: u32 = epollVal(e.data.u64);
                        if ((e.events & (linux.EPOLL.HUP | linux.EPOLL.ERR | linux.EPOLL.RDHUP)) != 0) {
                            self.closeConn(ci);
                            continue;
                        }
                        if ((e.events & linux.EPOLL.OUT) != 0) self.drainSend(ci);
                        if ((e.events & linux.EPOLL.IN) != 0) self.onRecv(ci);
                    },
                }
            }
        }
    }

    fn acceptCtrl(self: *Worker) void {
        if (self.ctrl_conn_fd >= 0) {
            const c = posix.accept(self.ctrl_listen_fd, null, null, posix.SOCK.CLOEXEC) catch return;
            posix.close(c);
            return;
        }
        const c = posix.accept(self.ctrl_listen_fd, null, null, posix.SOCK.NONBLOCK | posix.SOCK.CLOEXEC) catch return;
        self.ctrl_conn_fd = c;
        var ev: linux.epoll_event = .{ .events = linux.EPOLL.IN | linux.EPOLL.RDHUP, .data = .{ .u64 = epollKey(KIND_CTRL_CONN, 0) } };
        posix.epoll_ctl(self.epfd, linux.EPOLL.CTL_ADD, c, &ev) catch {
            posix.close(c);
            self.ctrl_conn_fd = -1;
        };
    }

    fn onCtrlRecv(self: *Worker) void {
        while (true) {
            var len_hdr: u16 = 0;
            var prefix_buf: [BUF_SIZE]u8 = undefined;
            var iov_buf: [2]Iovec_t = .{
                .{ .base = @ptrCast(&len_hdr), .len = @sizeOf(u16) },
                .{ .base = &prefix_buf, .len = prefix_buf.len },
            };
            const CTRL_LEN = cmsgSpace(@sizeOf(c_int));
            var ctrl_buf: [CTRL_LEN]u8 align(@alignOf(Cmsghdr_t)) = undefined;
            var msg = Msghdr_t{
                .name = null,
                .namelen = 0,
                .iov = &iov_buf[0],
                .iovlen = 2,
                .control = @ptrCast(&ctrl_buf),
                .controllen = CTRL_LEN,
                .flags = 0,
            };
            const r = recvmsg(self.ctrl_conn_fd, &msg, MSG_CMSG_CLOEXEC_C | MSG_DONTWAIT_C);
            if (r <= 0) {
                const e = std.posix.errno(r);
                if (e == .AGAIN) return;
                _ = posix.epoll_ctl(self.epfd, linux.EPOLL.CTL_DEL, self.ctrl_conn_fd, null) catch {};
                posix.close(self.ctrl_conn_fd);
                self.ctrl_conn_fd = -1;
                return;
            }
            const cmsg: *const Cmsghdr_t = @ptrCast(@alignCast(&ctrl_buf));
            if (cmsg.level != SOL_SOCKET_C or cmsg.type != SCM_RIGHTS_C) continue;
            const data_off = cmsgAlign(@sizeOf(Cmsghdr_t));
            const fd_ptr: *const c_int = @ptrCast(@alignCast(&ctrl_buf[data_off]));
            const cfd: posix.fd_t = @intCast(fd_ptr.*);

            const total_recv: usize = @intCast(r);
            var prefix_n: usize = 0;
            if (total_recv >= @sizeOf(u16)) {
                prefix_n = @min(@as(usize, len_hdr), total_recv - @sizeOf(u16));
            }
            self.addClientFd(cfd, prefix_buf[0..prefix_n]);
        }
    }

    fn addClientFd(self: *Worker, cfd: posix.fd_t, prefix: []const u8) void {
        if (self.free_count == 0) { posix.close(cfd); return; }
        self.free_count -= 1;
        const ci = self.free_idx[self.free_count];
        self.conns[ci] = .{ .fd = cfd };
        if (prefix.len > 0) {
            const n = @min(prefix.len, self.conns[ci].in_buf.len);
            @memcpy(self.conns[ci].in_buf[0..n], prefix[0..n]);
            self.conns[ci].in_len = @intCast(n);
        }
        var ev: linux.epoll_event = .{ .events = linux.EPOLL.IN | linux.EPOLL.RDHUP | linux.EPOLL.ET, .data = .{ .u64 = epollKey(KIND_CLIENT, ci) } };
        posix.epoll_ctl(self.epfd, linux.EPOLL.CTL_ADD, cfd, &ev) catch {
            posix.close(cfd);
            self.conns[ci].fd = -1;
            self.conns[ci].in_len = 0;
            self.free_idx[self.free_count] = ci;
            self.free_count += 1;
            return;
        };
        if (prefix.len > 0) {
            self.processBuffered(ci);
        }
    }

    fn processBuffered(self: *Worker, ci: u32) void {
        const c = &self.conns[ci];
        while (c.in_len > 0) {
            const req = http.parse(c.in_buf[0..c.in_len]) catch |err| switch (err) {
                error.Incomplete => return,
                error.Bad => { self.closeConn(ci); return; },
            };
            const resp = http.respond(self.idx, req);
            startSend(c, resp);
            if (drainSendNow(c)) {
                shiftBuf(c, req.end);
                continue;
            }
            shiftBuf(c, req.end);
            var ev: linux.epoll_event = .{ .events = linux.EPOLL.IN | linux.EPOLL.OUT | linux.EPOLL.RDHUP | linux.EPOLL.ET, .data = .{ .u64 = epollKey(KIND_CLIENT, ci) } };
            _ = posix.epoll_ctl(self.epfd, linux.EPOLL.CTL_MOD, c.fd, &ev) catch {};
            return;
        }
    }

    fn acceptAll(self: *Worker) void {
        while (true) {
            const cfd = posix.accept(self.legacy_listen_fd, null, null, posix.SOCK.NONBLOCK | posix.SOCK.CLOEXEC) catch return;
            self.addClientFd(cfd, &[_]u8{});
        }
    }

    fn onRecv(self: *Worker, ci: u32) void {
        const c = &self.conns[ci];
        while (true) {
            if (c.in_len >= c.in_buf.len) { self.closeConn(ci); return; }
            const got = posix.recv(c.fd, c.in_buf[c.in_len..], 0) catch |err| switch (err) {
                error.WouldBlock => return,
                else => { self.closeConn(ci); return; },
            };
            if (got == 0) { self.closeConn(ci); return; }
            c.in_len += @intCast(got);
            while (c.in_len > 0) {
                const req = http.parse(c.in_buf[0..c.in_len]) catch |err| switch (err) {
                    error.Incomplete => break,
                    error.Bad => { self.closeConn(ci); return; },
                };
                const resp = http.respond(self.idx, req);
                startSend(c, resp);
                if (drainSendNow(c)) {
                    shiftBuf(c, req.end);
                    continue;
                }
                shiftBuf(c, req.end);
                var ev: linux.epoll_event = .{ .events = linux.EPOLL.IN | linux.EPOLL.OUT | linux.EPOLL.RDHUP | linux.EPOLL.ET, .data = .{ .u64 = epollKey(KIND_CLIENT, ci) } };
                _ = posix.epoll_ctl(self.epfd, linux.EPOLL.CTL_MOD, c.fd, &ev) catch {};
                return;
            }
        }
    }

    fn drainSend(self: *Worker, ci: u32) void {
        const c = &self.conns[ci];
        if (drainSendNow(c)) {
            var ev: linux.epoll_event = .{ .events = linux.EPOLL.IN | linux.EPOLL.RDHUP | linux.EPOLL.ET, .data = .{ .u64 = epollKey(KIND_CLIENT, ci) } };
            _ = posix.epoll_ctl(self.epfd, linux.EPOLL.CTL_MOD, c.fd, &ev) catch {};
        }
    }

    fn closeConn(self: *Worker, ci: u32) void {
        const c = &self.conns[ci];
        if (c.fd >= 0) {
            _ = posix.epoll_ctl(self.epfd, linux.EPOLL.CTL_DEL, c.fd, null) catch {};
            posix.close(c.fd);
            c.fd = -1;
        }
        c.in_len = 0;
        c.out = "";
        c.out_pos = 0;
        self.free_idx[self.free_count] = @intCast(ci);
        self.free_count += 1;
    }
};

fn workerEntry(w: *Worker) void {
    w.initPool();
    w.run() catch |err| {
        std.debug.print("[worker] run error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}

pub fn main() !void {
    const index_path = envCstr("INDEX_PATH") orelse "/data/index.bin";
    const ctrl_path_env = envCstr("CTRL_SOCK_PATH");
    const sock_path_env = envCstr("SOCK_PATH");
    const sock_prefix_env = envCstr("API_SOCKET_PREFIX");

    // API_WORKERS=N spawns N threads, each binding its own listening socket.
    // With API_SOCKET_PREFIX=/sockets/api1 we get /sockets/api1-w0.sock,
    // /sockets/api1-w1.sock, ... — the lucasmontano convention. API_WORKERS=1
    // (default) preserves the original single-loop behavior bit-for-bit.
    const workers: u32 = blk: {
        if (std.c.getenv("API_WORKERS")) |p| {
            const v = std.mem.span(p);
            const n = std.fmt.parseInt(u32, v, 10) catch 1;
            break :blk if (n == 0) 1 else n;
        }
        break :blk 1;
    };

    var idx = try Index.open(index_path);
    defer idx.close();

    runWarmup(&idx);

    if (mlockallEnabled()) {
        const MCL_CURRENT: c_int = 1;
        const MCL_FUTURE: c_int = 2;
        _ = mlockall(MCL_CURRENT | MCL_FUTURE);
    }

    var gpa = std.heap.page_allocator;

    var workers_state = try gpa.alloc(Worker, workers);
    defer gpa.free(workers_state);

    // Path buffers must outlive the workers (each Worker keeps a pointer into
    // its slot for the lifetime of the listener).
    var path_bufs = try gpa.alloc([256]u8, workers);
    defer gpa.free(path_bufs);

    for (0..workers) |i| {
        workers_state[i] = .{
            .idx = &idx,
            .ctrl_path = null,
            .sock_path = null,
        };

        if (sock_prefix_env) |pfx| {
            // /sockets/api1 → /sockets/api1-w0.sock — lucasmontano convention.
            // The lb-c speaks SOCK_SEQPACKET + SCM_RIGHTS, so the worker binds
            // a CTRL listener (not a stream listener). This keeps the fd-passing
            // pipeline identical to the single-worker case.
            const pfx_slice = std.mem.span(pfx);
            const written = std.fmt.bufPrintZ(&path_bufs[i], "{s}-w{d}.sock", .{ pfx_slice, i }) catch return error.PathTooLong;
            workers_state[i].ctrl_path = written.ptr;
        } else if (ctrl_path_env) |cp| {
            if (workers == 1) {
                workers_state[i].ctrl_path = cp;
            } else {
                // CTRL_SOCK_PATH=/sock/api1.sock with workers>1 → suffix -wN.
                const cp_slice = std.mem.span(cp);
                var base_len = cp_slice.len;
                if (base_len >= 5 and std.mem.endsWith(u8, cp_slice, ".sock")) {
                    base_len -= 5;
                }
                const written = std.fmt.bufPrintZ(&path_bufs[i], "{s}-w{d}.sock", .{ cp_slice[0..base_len], i }) catch return error.PathTooLong;
                workers_state[i].ctrl_path = written.ptr;
            }
        } else {
            workers_state[i].sock_path = sock_path_env;
        }
    }

    if (workers == 1) {
        workerEntry(&workers_state[0]);
        return;
    }

    // Spawn N-1 additional threads; the main thread runs worker 0.
    var threads = try gpa.alloc(std.Thread, workers - 1);
    defer gpa.free(threads);

    for (1..workers) |i| {
        threads[i - 1] = try std.Thread.spawn(.{}, workerEntry, .{&workers_state[i]});
    }

    workerEntry(&workers_state[0]);

    for (threads) |t| t.join();
}

fn runWarmup(idx: *const Index) void {
    // Default 800ms (proven on v37). Contest can bump via WARMUP_MS=5000 in
    // compose to prime branch predictor, L1i, page cache and the k6 client
    // connection pool ramp-up window. Reference: top-bmtec API_WARMUP_QUERIES
    // = 2048 + top-fksegundo-tree RINHA_SELF_WARMUP_DURATION_MS=30000.
    var ms: u64 = 800;
    if (std.c.getenv("WARMUP_MS")) |p| {
        const v = std.mem.span(p);
        ms = std.fmt.parseInt(u64, v, 10) catch ms;
    }
    if (ms == 0) return;
    const target_ns: u64 = ms * std.time.ns_per_ms;
    const t0 = std.time.nanoTimestamp();
    var rng: u64 = 0xDEADBEEFCAFEBABE;
    var count: u64 = 0;
    var sink: u32 = 0;
    while (true) {
        const now = std.time.nanoTimestamp();
        const elapsed: u64 = @intCast(now - t0);
        if (elapsed >= target_ns) break;
        var v: [norm.DIMS]f32 = undefined;
        inline for (0..norm.DIMS) |i| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            const r: f32 = @floatFromInt(@as(u32, @intCast((rng >> 32) & 0x7FFFFFFF)));
            v[i] = (r / @as(f32, @floatFromInt(std.math.maxInt(i32)))) * 2.0 - 1.0;
        }
        sink +%= idx.score(v);
        count += 1;
    }
    std.mem.doNotOptimizeAway(sink);
    const now = std.time.nanoTimestamp();
    const ms_actual: u64 = @intCast(@divTrunc(now - t0, std.time.ns_per_ms));
    std.debug.print("[warmup] {d} queries in {d}ms\n", .{ count, ms_actual });
}

inline fn mlockallEnabled() bool {
    if (std.c.getenv("MLOCK")) |p| {
        const v = std.mem.span(p);
        return v.len > 0 and v[0] != '0';
    }
    return true;
}

const EpollParams = extern struct {
    busy_poll_usecs: u32,
    busy_poll_budget: u16,
    prefer_busy_poll: u8,
    _pad: u8,
};
const EPIOCSPARAMS: c_int = @bitCast(@as(u32, 0x4008_7001));
extern fn ioctl(fd: c_int, req: c_int, ...) c_int;

fn envU64(name: [*:0]const u8, default: u64) u64 {
    if (std.c.getenv(name)) |p| {
        const v = std.mem.span(p);
        return std.fmt.parseInt(u64, v, 10) catch default;
    }
    return default;
}

// Tracks pwait2 ENOSYS once per worker so we don't keep paying the failed syscall
// on kernels < 5.11.
var pwait2_unsupported: bool = false;

inline fn epollWaitNoBlock(epfd: posix.fd_t, events: []linux.epoll_event) usize {
    while (true) {
        const r = linux.syscall4(
            .epoll_wait,
            @as(usize, @bitCast(@as(isize, epfd))),
            @intFromPtr(events.ptr),
            events.len,
            0, // timeout = 0 → non-blocking
        );
        const s: isize = @bitCast(r);
        if (s >= 0) return @intCast(s);
        // EINTR retry; any other error → return 0 (caller will fall through to blocking wait).
        if (s != -@as(isize, @intFromEnum(linux.E.INTR))) return 0;
    }
}

fn epollWaitBlocking(epfd: posix.fd_t, events: []linux.epoll_event, idle_us: u64) usize {
    // Build a finite timespec when EPOLL_IDLE_US>0, else block forever (NULL ts).
    var ts: linux.timespec = undefined;
    const ts_ptr: usize = if (idle_us == 0) 0 else blk: {
        ts = .{
            .sec = @intCast(idle_us / 1_000_000),
            .nsec = @intCast((idle_us % 1_000_000) * 1000),
        };
        break :blk @intFromPtr(&ts);
    };

    while (true) {
        if (!pwait2_unsupported) {
            const r = linux.syscall6(
                .epoll_pwait2,
                @as(usize, @bitCast(@as(isize, epfd))),
                @intFromPtr(events.ptr),
                events.len,
                ts_ptr,
                0, // NULL sigmask
                8, // sizeof(sigset_t)
            );
            const s: isize = @bitCast(r);
            if (s >= 0) return @intCast(s);
            if (s == -@as(isize, @intFromEnum(linux.E.INTR))) continue;
            if (s == -@as(isize, @intFromEnum(linux.E.NOSYS))) {
                pwait2_unsupported = true;
                // fall through to epoll_wait
            } else {
                return 0;
            }
        }
        // Fallback for kernel < 5.11: epoll_wait with ms timeout (rounded up).
        const timeout_ms: i32 = if (idle_us == 0) -1 else @intCast(@max(@as(u64, 1), (idle_us + 999) / 1000));
        const got = posix.epoll_wait(epfd, events, timeout_ms);
        return got;
    }
}

// Best-effort per-thread scheduler tweaks. Mirrors the piassa-asm path:
//   prctl(PR_SET_TIMERSLACK, 1)  → default 50µs slack → 1ns
//   sched_setscheduler(0, FIFO, prio=$API_RT_PRIO)  [opt-in]
// Both can EPERM without CAP_SYS_NICE; both ignore failure so the worker still
// boots in container environments that refuse realtime promotion.
// SCHED_FIFO is opt-in via API_RT_PRIO=N because in CFS-throttled cgroups
// (CPU < 1.0) realtime promotion causes 50-93ms cgroup-bandwidth stalls when
// the quota expires. In contest cpuset (0.475 CPU per worker on dedicated
// cores) the kernel runs the FIFO thread within the quota window with no
// preemption — exactly the latency reduction we want.
fn tunePerThreadSched() void {
    const PR_SET_TIMERSLACK: i32 = 29;
    _ = linux.prctl(PR_SET_TIMERSLACK, 1, 0, 0, 0);

    var rt_prio: i32 = 0;
    if (std.c.getenv("API_RT_PRIO")) |p| {
        const v = std.mem.span(p);
        rt_prio = std.fmt.parseInt(i32, v, 10) catch 0;
    }
    if (rt_prio > 0) {
        const param = linux.sched_param{ .priority = rt_prio };
        _ = linux.sched_setscheduler(0, .{ .mode = .FIFO }, &param);
    }
}

fn setEpollBusyPoll(epfd: posix.fd_t) void {
    var us: u32 = 100;
    var budget: u16 = 8;
    var prefer: u8 = 1;
    if (std.c.getenv("EPOLL_BUSY_POLL_US")) |p| {
        const v = std.mem.span(p);
        us = std.fmt.parseInt(u32, v, 10) catch us;
    }
    if (std.c.getenv("EPOLL_BUSY_POLL_BUDGET")) |p| {
        const v = std.mem.span(p);
        budget = std.fmt.parseInt(u16, v, 10) catch budget;
    }
    if (std.c.getenv("EPOLL_PREFER_BUSY_POLL")) |p| {
        const v = std.mem.span(p);
        if (v.len > 0 and v[0] == '0') prefer = 0;
    }
    if (us == 0) return;
    var params: EpollParams = .{ .busy_poll_usecs = us, .busy_poll_budget = budget, .prefer_busy_poll = prefer, ._pad = 0 };
    _ = ioctl(epfd, EPIOCSPARAMS, &params);
}

const SOCK_SEQPACKET: u32 = 5;

fn openCtrlListener(sp_z: [*:0]const u8) !posix.fd_t {
    const sp = std.mem.span(sp_z);
    _ = posix.unlinkZ(sp_z) catch {};
    const fd = try posix.socket(posix.AF.UNIX, SOCK_SEQPACKET | posix.SOCK.NONBLOCK | posix.SOCK.CLOEXEC, 0);
    var addr: posix.sockaddr.un = .{ .family = posix.AF.UNIX, .path = undefined };
    @memset(&addr.path, 0);
    const n = @min(addr.path.len - 1, sp.len);
    @memcpy(addr.path[0..n], sp[0..n]);
    const addrlen: posix.socklen_t = @intCast(@offsetOf(posix.sockaddr.un, "path") + n + 1);
    try posix.bind(fd, @ptrCast(&addr), addrlen);
    try posix.listen(fd, 8);
    _ = std.c.chmod(sp_z, 0o666);
    return fd;
}

const Iovec_t = extern struct { base: ?*anyopaque, len: usize };
const Msghdr_t = extern struct {
    name: ?*anyopaque,
    namelen: u32,
    iov: ?*Iovec_t,
    iovlen: usize,
    control: ?*anyopaque,
    controllen: usize,
    flags: c_int,
};
const Cmsghdr_t = extern struct { len: usize, level: c_int, type: c_int };

const SCM_RIGHTS_C: c_int = 1;
const SOL_SOCKET_C: c_int = 1;
const MSG_CMSG_CLOEXEC_C: c_int = 0x40000000;
const MSG_DONTWAIT_C: c_int = 0x40;

extern fn recvmsg(sockfd: c_int, msg: *Msghdr_t, flags: c_int) isize;
extern fn mlockall(flags: c_int) c_int;

inline fn cmsgAlign(len: usize) usize {
    const a: usize = @sizeOf(usize);
    return (len + a - 1) & ~(a - 1);
}
inline fn cmsgSpace(len: usize) usize {
    return cmsgAlign(len) + cmsgAlign(@sizeOf(Cmsghdr_t));
}

inline fn envCstr(name: [*:0]const u8) ?[*:0]const u8 {
    if (std.c.getenv(name)) |p| return p;
    return null;
}

fn openListener(sock_path_z: ?[*:0]const u8) !posix.fd_t {
    if (sock_path_z) |sp| {
        const sp_slice = std.mem.span(sp);
        _ = posix.unlinkZ(sp) catch {};
        const fd = try posix.socket(posix.AF.UNIX, posix.SOCK.STREAM | posix.SOCK.NONBLOCK | posix.SOCK.CLOEXEC, 0);
        var addr: posix.sockaddr.un = .{ .family = posix.AF.UNIX, .path = undefined };
        @memset(&addr.path, 0);
        const copy_len = @min(addr.path.len - 1, sp_slice.len);
        @memcpy(addr.path[0..copy_len], sp_slice[0..copy_len]);
        const addrlen: posix.socklen_t = @intCast(@offsetOf(posix.sockaddr.un, "path") + copy_len + 1);
        try posix.bind(fd, @ptrCast(&addr), addrlen);
        try posix.listen(fd, 512);
        _ = std.c.chmod(sp, 0o666);
        return fd;
    }
    const port_cstr = envCstr("PORT");
    const port: u16 = blk: {
        if (port_cstr) |p| {
            const s = std.mem.span(p);
            break :blk std.fmt.parseInt(u16, s, 10) catch 9999;
        }
        break :blk 9999;
    };
    const fd = try posix.socket(posix.AF.INET, posix.SOCK.STREAM | posix.SOCK.NONBLOCK | posix.SOCK.CLOEXEC, 0);
    try posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, &std.mem.toBytes(@as(c_int, 1)));
    const addr: posix.sockaddr.in = .{
        .family = posix.AF.INET,
        .port = std.mem.nativeToBig(u16, port),
        .addr = 0,
        .zero = .{0} ** 8,
    };
    try posix.bind(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in));
    try posix.listen(fd, 512);
    return fd;
}

inline fn shiftBuf(c: *Conn, consumed: usize) void {
    const used: usize = consumed;
    if (used >= c.in_len) {
        c.in_len = 0;
        return;
    }
    const rem: usize = c.in_len - used;
    std.mem.copyForwards(u8, c.in_buf[0..rem], c.in_buf[used .. used + rem]);
    c.in_len = @intCast(rem);
}

inline fn startSend(c: *Conn, resp: []const u8) void {
    c.out = resp;
    c.out_pos = 0;
}

fn drainSendNow(c: *Conn) bool {
    while (c.out_pos < c.out.len) {
        const sent = posix.send(c.fd, c.out[c.out_pos..], linux.MSG.NOSIGNAL) catch return false;
        c.out_pos += @intCast(sent);
    }
    return true;
}
