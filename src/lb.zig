const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;

const MAX_BACKENDS: usize = 8;
const DEFAULT_BACKLOG: u32 = 4096;
const DEFAULT_ACCEPT_BATCH: u32 = 128;

const Iovec = extern struct {
    base: ?*anyopaque,
    len: usize,
};

const Msghdr = extern struct {
    name: ?*anyopaque,
    namelen: u32,
    iov: ?*Iovec,
    iovlen: usize,
    control: ?*anyopaque,
    controllen: usize,
    flags: c_int,
};

const Cmsghdr = extern struct {
    len: usize,
    level: c_int,
    type: c_int,
};

const SCM_RIGHTS: c_int = 1;
const SOL_SOCKET: c_int = 1;
const MSG_NOSIGNAL: c_int = 0x4000;
const MSG_DONTWAIT: c_int = 0x40;
const SOCK_SEQPACKET: u32 = 5;
const TCP_NODELAY: c_int = 1;
const TCP_DEFER_ACCEPT: c_int = 9;
const TCP_QUICKACK: c_int = 12;

const MAX_PREFIX: usize = 4096;

inline fn cmsgAlign(len: usize) usize {
    const a: usize = @sizeOf(usize);
    return (len + a - 1) & ~(a - 1);
}
inline fn cmsgSpace(len: usize) usize {
    return cmsgAlign(len) + cmsgAlign(@sizeOf(Cmsghdr));
}
inline fn cmsgLen(len: usize) usize {
    return cmsgAlign(@sizeOf(Cmsghdr)) + len;
}

extern fn sendmsg(sockfd: c_int, msg: *const Msghdr, flags: c_int) isize;
extern fn chmod(path: [*:0]const u8, mode: c_uint) c_int;

pub fn main() !void {
    const port_cstr = std.c.getenv("LB_PORT");
    const port: u16 = if (port_cstr) |p| (std.fmt.parseInt(u16, std.mem.span(p), 10) catch 9999) else 9999;
    const backlog_cstr = std.c.getenv("LB_BACKLOG");
    const backlog: u32 = if (backlog_cstr) |p| (std.fmt.parseInt(u32, std.mem.span(p), 10) catch DEFAULT_BACKLOG) else DEFAULT_BACKLOG;
    const batch_cstr = std.c.getenv("LB_ACCEPT_BATCH");
    const accept_batch: u32 = if (batch_cstr) |p| (std.fmt.parseInt(u32, std.mem.span(p), 10) catch DEFAULT_ACCEPT_BATCH) else DEFAULT_ACCEPT_BATCH;

    const sockets_cstr = std.c.getenv("API_SOCKETS") orelse {
        std.debug.print("[lb] set API_SOCKETS=/sock/api1.sock,/sock/api2.sock\n", .{});
        return error.MissingEnv;
    };
    const sockets = std.mem.span(sockets_cstr);

    var backend_fds: [MAX_BACKENDS]c_int = undefined;
    var nb: usize = 0;
    {
        var it = std.mem.splitScalar(u8, sockets, ',');
        while (it.next()) |raw_p| {
            const p = std.mem.trim(u8, raw_p, " ");
            if (p.len == 0) continue;
            if (nb >= MAX_BACKENDS) break;
            try waitForPath(p);
            var conn_tries: u32 = 0;
            backend_fds[nb] = while (conn_tries < 600) : (conn_tries += 1) {
                if (connectUds(p)) |fd| break fd else |_| {}
                std.time.sleep(100 * std.time.ns_per_ms);
            } else return error.BackendConnectTimeout;
            std.debug.print("[lb] connected to {s} fd={d}\n", .{ p, backend_fds[nb] });
            nb += 1;
        }
    }
    if (nb == 0) return error.NoBackends;

    const lfd = try posix.socket(posix.AF.INET, posix.SOCK.STREAM | posix.SOCK.NONBLOCK | posix.SOCK.CLOEXEC, 0);
    defer posix.close(lfd);
    try posix.setsockopt(lfd, posix.SOL.SOCKET, posix.SO.REUSEADDR, &std.mem.toBytes(@as(c_int, 1)));
    try posix.setsockopt(lfd, posix.SOL.SOCKET, posix.SO.REUSEPORT, &std.mem.toBytes(@as(c_int, 1)));
    _ = std.c.setsockopt(lfd, std.c.IPPROTO.TCP, @intCast(TCP_DEFER_ACCEPT), &std.mem.toBytes(@as(c_int, 1)), 4);

    const addr: posix.sockaddr.in = .{
        .family = posix.AF.INET,
        .port = std.mem.nativeToBig(u16, port),
        .addr = 0,
        .zero = .{0} ** 8,
    };
    try posix.bind(lfd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in));
    try posix.listen(lfd, @intCast(@min(backlog, std.math.maxInt(u31))));
    std.debug.print("[lb] listening :{d} batch={d} backends={d}\n", .{ port, accept_batch, nb });

    var rr: usize = 0;
    while (true) {
        var accepted: u32 = 0;
        var got_one: bool = false;
        while (accepted < accept_batch) : (accepted += 1) {
            const cfd = posix.accept(lfd, null, null, posix.SOCK.NONBLOCK | posix.SOCK.CLOEXEC) catch break;
            got_one = true;
            _ = posix.setsockopt(cfd, std.c.IPPROTO.TCP, @intCast(TCP_NODELAY), &std.mem.toBytes(@as(c_int, 1))) catch {};
            _ = std.c.setsockopt(cfd, std.c.IPPROTO.TCP, @intCast(TCP_QUICKACK), &std.mem.toBytes(@as(c_int, 1)), 4);

            var prefix_buf: [MAX_PREFIX]u8 = undefined;
            const prefix_len = readReadyPrefix(cfd, &prefix_buf);

            const start = rr;
            var attempt: usize = 0;
            while (attempt < nb) : (attempt += 1) {
                const idx = (start + attempt) % nb;
                if (sendFdWithBytes(backend_fds[idx], cfd, prefix_buf[0..prefix_len])) {
                    rr = (idx + 1) % nb;
                    break;
                } else |_| {}
            }
            posix.close(cfd);
        }
        if (!got_one) {
            var pfd = [_]posix.pollfd{.{ .fd = lfd, .events = posix.POLL.IN, .revents = 0 }};
            _ = posix.poll(&pfd, 60_000) catch {};
        }
    }
}

fn waitForPath(path: []const u8) !void {
    var tries: u32 = 0;
    while (tries < 600) : (tries += 1) {
        var z_buf: [256]u8 = undefined;
        if (path.len >= z_buf.len) return error.PathTooLong;
        @memcpy(z_buf[0..path.len], path);
        z_buf[path.len] = 0;
        if (std.fs.cwd().accessZ(@ptrCast(&z_buf), .{})) {
            return;
        } else |_| {}
        std.time.sleep(100 * std.time.ns_per_ms);
    }
    return error.Timeout;
}

fn connectUds(path: []const u8) !c_int {
    const fd_pos: posix.fd_t = try posix.socket(posix.AF.UNIX, SOCK_SEQPACKET | posix.SOCK.CLOEXEC, 0);
    errdefer posix.close(fd_pos);
    const sndbuf: c_int = 256 * 1024;
    try posix.setsockopt(fd_pos, posix.SOL.SOCKET, posix.SO.SNDBUF, &std.mem.toBytes(sndbuf));
    var addr: posix.sockaddr.un = .{ .family = posix.AF.UNIX, .path = undefined };
    @memset(&addr.path, 0);
    const n = @min(addr.path.len - 1, path.len);
    @memcpy(addr.path[0..n], path[0..n]);
    const addrlen: posix.socklen_t = @intCast(@offsetOf(posix.sockaddr.un, "path") + n + 1);
    try posix.connect(fd_pos, @ptrCast(&addr), addrlen);
    return @intCast(fd_pos);
}

fn readReadyPrefix(fd: posix.fd_t, buf: []u8) usize {
    // Drain whatever is currently in the kernel buffer. We consume the bytes
    // (no MSG_PEEK), so we MUST forward them to the worker — anything we leave
    // behind would still arrive via epoll on the duplicated fd in the worker.
    var len: usize = 0;
    while (len < buf.len) {
        const got = posix.recv(fd, buf[len..], MSG_DONTWAIT) catch |err| switch (err) {
            error.WouldBlock => return len,
            else => return len,
        };
        if (got == 0) return len;
        len += got;
    }
    return len;
}

fn sendFdWithBytes(uds_fd: c_int, client_fd: posix.fd_t, prefix: []const u8) !void {
    var len_hdr: u16 = @intCast(prefix.len);
    var dummy: u8 = 1;
    // First iov: 2-byte length prefix (always present, even when 0).
    // Second iov: prefix bytes (may be empty, then we pad with a dummy byte for SEQPACKET reception parity).
    var iov_buf: [2]Iovec = .{
        .{ .base = @ptrCast(&len_hdr), .len = @sizeOf(u16) },
        .{ .base = if (prefix.len > 0) @constCast(@ptrCast(prefix.ptr)) else @ptrCast(&dummy), .len = if (prefix.len > 0) prefix.len else 1 },
    };
    const iovlen: usize = if (prefix.len > 0) 2 else 1;

    const CTRL_LEN = cmsgSpace(@sizeOf(c_int));
    var ctrl_buf: [CTRL_LEN]u8 align(@alignOf(Cmsghdr)) = undefined;

    var msg = Msghdr{
        .name = null,
        .namelen = 0,
        .iov = &iov_buf[0],
        .iovlen = iovlen,
        .control = @ptrCast(&ctrl_buf),
        .controllen = CTRL_LEN,
        .flags = 0,
    };

    const cmsg: *Cmsghdr = @ptrCast(@alignCast(&ctrl_buf));
    cmsg.len = cmsgLen(@sizeOf(c_int));
    cmsg.level = SOL_SOCKET;
    cmsg.type = SCM_RIGHTS;
    const data_off = cmsgAlign(@sizeOf(Cmsghdr));
    const data_ptr: *c_int = @ptrCast(@alignCast(&ctrl_buf[data_off]));
    data_ptr.* = @intCast(client_fd);
    msg.controllen = cmsg.len;

    while (true) {
        const r = sendmsg(uds_fd, &msg, MSG_NOSIGNAL);
        if (r > 0) return;
        const e = std.posix.errno(r);
        if (e == .INTR) continue;
        return error.SendFailed;
    }
}

fn sendFd(uds_fd: c_int, client_fd: posix.fd_t) !void {
    var dummy: u8 = 1;
    var iov = Iovec{ .base = @ptrCast(&dummy), .len = 1 };
    const CTRL_LEN = cmsgSpace(@sizeOf(c_int));
    var ctrl_buf: [CTRL_LEN]u8 align(@alignOf(Cmsghdr)) = undefined;

    var msg = Msghdr{
        .name = null,
        .namelen = 0,
        .iov = &iov,
        .iovlen = 1,
        .control = @ptrCast(&ctrl_buf),
        .controllen = CTRL_LEN,
        .flags = 0,
    };

    const cmsg: *Cmsghdr = @ptrCast(@alignCast(&ctrl_buf));
    cmsg.len = cmsgLen(@sizeOf(c_int));
    cmsg.level = SOL_SOCKET;
    cmsg.type = SCM_RIGHTS;
    const data_off = cmsgAlign(@sizeOf(Cmsghdr));
    const data_ptr: *c_int = @ptrCast(@alignCast(&ctrl_buf[data_off]));
    data_ptr.* = @intCast(client_fd);
    msg.controllen = cmsg.len;

    while (true) {
        const r = sendmsg(uds_fd, &msg, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (r > 0) return;
        const e = std.posix.errno(r);
        if (e == .INTR) continue;
        return error.SendFailed;
    }
}
