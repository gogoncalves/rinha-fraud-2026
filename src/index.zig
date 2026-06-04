const std = @import("std");
const posix = std.posix;
const norm = @import("normalize.zig");
const kdtree = @import("kdtree.zig");

pub const MAGIC: u32 = 0x52494E48;
pub const VERSION: u32 = 4;
pub const KD_VERSION: u32 = kdtree.VERSION;
pub const QUANT_SCALE: f32 = 10000.0;
pub const QUANT_MAX: f32 = 10000.0;

pub const LANES: usize = 8;
pub const PAIRS: usize = (norm.DIMS + 1) / 2;
pub const BLOCK_BYTES: usize = norm.PADDED_DIMS * LANES * 2;

pub const NPROBE: usize = 2;
pub const REPAIR_EXTRA: usize = 30;
pub const MAX_PROBES: usize = NPROBE + REPAIR_EXTRA;
pub const MAX_K: usize = 4096;
pub const SEEN_WORDS: usize = (MAX_K + 63) / 64;
pub const TOP_K: usize = 5;
pub const REPAIR_MIN: u8 = 1;
pub const REPAIR_MAX: u8 = 4;
pub const EARLY_DIST_FRAC: i32 = 120;
pub const EARLY_DIST: i64 = blk: {
    const v: i64 = @as(i64, @intFromFloat(QUANT_SCALE)) * EARLY_DIST_FRAC / 1000;
    break :blk v * v * norm.DIMS;
};

pub const Header = extern struct {
    magic: u32,
    version: u32,
    k: u32,
    n: u32,
    n_blocks: u32,
    scale: f32,
    _r: [40]u8,
};

pub const Backend = enum { ivf, kd };

pub const Index = struct {
    backend: Backend,
    map_ptr: [*]align(4096) const u8,
    map_len: usize,
    // IVF fields (valid when backend == .ivf):
    centroids_base: [*]const i16 = undefined,
    n_centroid_blocks: usize = 0,
    bbox_min: []const [norm.PADDED_DIMS]i16 = &[_][norm.PADDED_DIMS]i16{},
    bbox_max: []const [norm.PADDED_DIMS]i16 = &[_][norm.PADDED_DIMS]i16{},
    block_offsets: []const u32 = &[_]u32{},
    counts: []const u32 = &[_]u32{},
    vectors_base: [*]const i16 = undefined,
    labels: []const u8 = &[_]u8{},
    k: usize = 0,
    n: usize = 0,
    n_blocks: usize = 0,
    // KD fields (valid when backend == .kd):
    kd: kdtree.Tree = undefined,

    pub fn open(path_z: [*:0]const u8) !Index {
        const fd = try posix.openatZ(posix.AT.FDCWD, path_z, .{ .ACCMODE = .RDONLY }, 0);
        defer posix.close(fd);
        const stat = try posix.fstat(fd);
        const size: usize = @intCast(stat.size);

        // Map the index file. We rely on MADV_HUGEPAGE + transparent huge pages.
        // MAP_HUGETLB cannot be combined with file-backed PRIVATE mmap on most kernels,
        // so we keep the standard PRIVATE mapping and rely on THP via madvise below.
        const ptr = try posix.mmap(null, size, posix.PROT.READ, .{ .TYPE = .PRIVATE, .POPULATE = true }, fd, 0);

        const hdr: *const Header = @ptrCast(@alignCast(ptr.ptr));
        if (hdr.magic != MAGIC) return error.BadMagic;
        if (hdr.version == KD_VERSION) return openKd(ptr.ptr, size, hdr);
        if (hdr.version != VERSION) return error.BadVersion;
        const k: usize = hdr.k;
        const n: usize = hdr.n;
        const nb: usize = hdr.n_blocks;

        var cur: usize = @sizeOf(Header);
        const n_centroid_blocks: usize = (k + LANES - 1) / LANES;
        const centroids_base: [*]const i16 = @ptrCast(@alignCast(ptr.ptr + cur));
        cur += n_centroid_blocks * BLOCK_BYTES;

        const bmin_ptr: [*]const [norm.PADDED_DIMS]i16 = @ptrCast(@alignCast(ptr.ptr + cur));
        const bbox_min = bmin_ptr[0..k];
        cur += k * norm.PADDED_DIMS * 2;

        const bmax_ptr: [*]const [norm.PADDED_DIMS]i16 = @ptrCast(@alignCast(ptr.ptr + cur));
        const bbox_max = bmax_ptr[0..k];
        cur += k * norm.PADDED_DIMS * 2;

        const off_ptr: [*]const u32 = @ptrCast(@alignCast(ptr.ptr + cur));
        const block_offsets = off_ptr[0 .. k + 1];
        cur += (k + 1) * 4;

        const cnt_ptr: [*]const u32 = @ptrCast(@alignCast(ptr.ptr + cur));
        const counts = cnt_ptr[0..k];
        cur += k * 4;

        const vectors_base: [*]const i16 = @ptrCast(@alignCast(ptr.ptr + cur));
        cur += nb * BLOCK_BYTES;

        const labels_ptr: [*]const u8 = @ptrCast(ptr.ptr + cur);
        const labels = labels_ptr[0 .. nb * LANES];

        advise(ptr.ptr, size);

        return .{
            .backend = .ivf,
            .map_ptr = @alignCast(ptr.ptr),
            .map_len = size,
            .centroids_base = centroids_base,
            .n_centroid_blocks = n_centroid_blocks,
            .bbox_min = bbox_min,
            .bbox_max = bbox_max,
            .block_offsets = block_offsets,
            .counts = counts,
            .vectors_base = vectors_base,
            .labels = labels,
            .k = k,
            .n = n,
            .n_blocks = nb,
        };
    }

    pub fn close(self: *Index) void {
        posix.munmap(self.map_ptr[0..self.map_len]);
    }

    pub fn score(self: *const Index, q: [norm.DIMS]f32) u32 {
        const qq = quantize(q);
        return self.score_qq(&qq);
    }

    pub fn score_qq(self: *const Index, qq: *const [norm.PADDED_DIMS]i16) u32 {
        if (self.backend == .kd) return self.kd.score(qq);
        return self.score_qq_ivf(qq);
    }

    fn score_qq_ivf(self: *const Index, qq: *const [norm.PADDED_DIMS]i16) u32 {
        const empty_probe: Probe = .{ .cluster = std.math.maxInt(u32), .dist = std.math.maxInt(i64) };
        var probes: [MAX_PROBES]Probe = .{empty_probe} ** MAX_PROBES;
        var probe_count: usize = 0;

        var cb: usize = 0;
        while (cb < self.n_centroid_blocks) : (cb += 1) {
            const dists = blk_dist(self.centroids_base, cb, qq);
            const ci_base: u32 = @intCast(cb * LANES);
            const lane_max: u32 = @min(@as(u32, LANES), @as(u32, @intCast(self.k - ci_base)));
            var lane: u32 = 0;
            while (lane < lane_max) : (lane += 1) {
                const ci = ci_base + lane;
                if (self.counts[ci] == 0) continue;
                insert_probe(&probes, &probe_count, ci, dists[lane]);
            }
        }
        heap_to_sorted(&probes, probe_count);

        var best_d: [TOP_K]i64 = .{std.math.maxInt(i64)} ** TOP_K;
        var best_l: [TOP_K]u8 = .{0} ** TOP_K;

        const n_initial = @min(NPROBE, probe_count);
        var pi: usize = 0;
        while (pi < n_initial) : (pi += 1) {
            self.scan_cluster(qq, probes[pi].cluster, &best_d, &best_l);
        }

        var frauds: u32 = 0;
        inline for (0..TOP_K) |j| frauds += best_l[j];

        const unanimous = frauds == 0 or frauds == TOP_K;
        const tight = best_d[TOP_K - 1] <= EARLY_DIST;
        if (unanimous and tight) {
            @branchHint(.likely);
            return frauds;
        }
        self.repair_fast(qq, probes[0..probe_count], &best_d, &best_l);
        frauds = 0;
        inline for (0..TOP_K) |j| frauds += best_l[j];

        const still_borderline = frauds >= REPAIR_MIN and frauds <= REPAIR_MAX;
        if (still_borderline) {
            @branchHint(.unlikely);
            self.repair_full(qq, probes[0..probe_count], &best_d, &best_l);
            frauds = 0;
            inline for (0..TOP_K) |j| frauds += best_l[j];
        }
        return frauds;
    }

    fn repair_full(
        self: *const Index,
        q: *const [norm.PADDED_DIMS]i16,
        probes: []const Probe,
        best_d: *[TOP_K]i64,
        best_l: *[TOP_K]u8,
    ) void {
        var seen_mask: [SEEN_WORDS]u64 = [_]u64{0} ** SEEN_WORDS;
        for (probes) |p| {
            if (p.cluster == std.math.maxInt(u32)) break;
            const w: usize = @as(usize, p.cluster) / 64;
            const b: u6 = @intCast(@as(usize, p.cluster) % 64);
            if (w < SEEN_WORDS) seen_mask[w] |= @as(u64, 1) << b;
        }
        var ci: u32 = 0;
        while (ci < self.k) : (ci += 1) {
            if (self.counts[ci] == 0) continue;
            const w: usize = @as(usize, ci) / 64;
            const b: u6 = @intCast(@as(usize, ci) % 64);
            if ((seen_mask[w] >> b) & 1 != 0) continue;
            const lb = bbox_lower_bound(q, &self.bbox_min[ci], &self.bbox_max[ci]);
            if (lb >= best_d[TOP_K - 1]) continue;
            self.scan_cluster(q, ci, best_d, best_l);
        }
    }

    fn repair_fast(
        self: *const Index,
        q: *const [norm.PADDED_DIMS]i16,
        probes: []const Probe,
        best_d: *[TOP_K]i64,
        best_l: *[TOP_K]u8,
    ) void {
        var pi: usize = NPROBE;
        while (pi < probes.len) : (pi += 1) {
            const p = probes[pi];
            if (p.cluster == std.math.maxInt(u32)) break;
            const lb = bbox_lower_bound(q, &self.bbox_min[p.cluster], &self.bbox_max[p.cluster]);
            if (lb >= best_d[TOP_K - 1]) continue;
            self.scan_cluster(q, p.cluster, best_d, best_l);
            var frauds: u8 = 0;
            inline for (0..TOP_K) |j| frauds += best_l[j];
            if (frauds < REPAIR_MIN or frauds > REPAIR_MAX) {
                if (best_d[TOP_K - 1] <= EARLY_DIST) break;
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Parametric IVF score for the offline bench. Mirrors score_qq_ivf, but
    // every "knob" (NPROBE, NPROBE_BORDER, REPAIR_MIN/MAX) is passed in. Lives
    // alongside the production path so we don't duplicate distance kernels.
    // Only used by tools/bench.zig; production server keeps the comptime-const
    // hot path untouched.
    // ─────────────────────────────────────────────────────────────────────────
    pub fn score_qq_ivf_param(
        self: *const Index,
        qq: *const [norm.PADDED_DIMS]i16,
        nprobe: usize,
        repair_min: u8,
        repair_max: u8,
    ) u32 {
        const empty_probe: Probe = .{ .cluster = std.math.maxInt(u32), .dist = std.math.maxInt(i64) };
        var probes: [MAX_PROBES]Probe = .{empty_probe} ** MAX_PROBES;
        var probe_count: usize = 0;

        var cb: usize = 0;
        while (cb < self.n_centroid_blocks) : (cb += 1) {
            const dists = blk_dist(self.centroids_base, cb, qq);
            const ci_base: u32 = @intCast(cb * LANES);
            const lane_max: u32 = @min(@as(u32, LANES), @as(u32, @intCast(self.k - ci_base)));
            var lane: u32 = 0;
            while (lane < lane_max) : (lane += 1) {
                const ci = ci_base + lane;
                if (self.counts[ci] == 0) continue;
                insert_probe(&probes, &probe_count, ci, dists[lane]);
            }
        }
        heap_to_sorted(&probes, probe_count);

        var best_d: [TOP_K]i64 = .{std.math.maxInt(i64)} ** TOP_K;
        var best_l: [TOP_K]u8 = .{0} ** TOP_K;

        const n_initial = @min(nprobe, probe_count);
        var pi: usize = 0;
        while (pi < n_initial) : (pi += 1) {
            self.scan_cluster(qq, probes[pi].cluster, &best_d, &best_l);
        }

        var frauds: u32 = 0;
        inline for (0..TOP_K) |j| frauds += best_l[j];

        const unanimous = frauds == 0 or frauds == TOP_K;
        const tight = best_d[TOP_K - 1] <= EARLY_DIST;
        if (unanimous and tight) return frauds;

        self.repair_fast_param(qq, probes[0..probe_count], &best_d, &best_l, nprobe, repair_min, repair_max);
        frauds = 0;
        inline for (0..TOP_K) |j| frauds += best_l[j];

        const still_borderline = frauds >= repair_min and frauds <= repair_max;
        if (still_borderline) {
            self.repair_full(qq, probes[0..probe_count], &best_d, &best_l);
            frauds = 0;
            inline for (0..TOP_K) |j| frauds += best_l[j];
        }
        return frauds;
    }

    fn repair_fast_param(
        self: *const Index,
        q: *const [norm.PADDED_DIMS]i16,
        probes: []const Probe,
        best_d: *[TOP_K]i64,
        best_l: *[TOP_K]u8,
        nprobe: usize,
        repair_min: u8,
        repair_max: u8,
    ) void {
        var pi: usize = nprobe;
        while (pi < probes.len) : (pi += 1) {
            const p = probes[pi];
            if (p.cluster == std.math.maxInt(u32)) break;
            const lb = bbox_lower_bound(q, &self.bbox_min[p.cluster], &self.bbox_max[p.cluster]);
            if (lb >= best_d[TOP_K - 1]) continue;
            self.scan_cluster(q, p.cluster, best_d, best_l);
            var frauds: u8 = 0;
            inline for (0..TOP_K) |j| frauds += best_l[j];
            if (frauds < repair_min or frauds > repair_max) {
                if (best_d[TOP_K - 1] <= EARLY_DIST) break;
            }
        }
    }

    fn scan_cluster(
        self: *const Index,
        q: *const [norm.PADDED_DIMS]i16,
        cluster: u32,
        best_d: *[TOP_K]i64,
        best_l: *[TOP_K]u8,
    ) void {
        const start_block: usize = self.block_offsets[cluster];
        const end_block: usize = self.block_offsets[cluster + 1];
        const total = self.counts[cluster];
        if (total == 0) return;

        const PREFETCH_AHEAD: usize = 4;
        var blk: usize = start_block;
        var processed: u32 = 0;
        while (blk < end_block) : (blk += 1) {
            if (blk + PREFETCH_AHEAD < end_block) {
                const off = (blk + PREFETCH_AHEAD) * (norm.PADDED_DIMS * LANES);
                @prefetch(self.vectors_base + off, .{ .rw = .read, .locality = 1, .cache = .data });
            }
            const threshold = best_d[TOP_K - 1];
            const maybe_dists = blk_dist_prune(self.vectors_base, blk, q, threshold);
            const lane_n: u32 = @min(@as(u32, LANES), total - processed);
            processed += lane_n;
            if (maybe_dists == null) {
                @branchHint(.likely);
                continue;
            }
            const dists = maybe_dists.?;
            const lab_base = blk * LANES;
            var lane: u32 = 0;
            while (lane < lane_n) : (lane += 1) {
                const d = dists[lane];
                if (d < best_d[TOP_K - 1]) {
                    const label = self.labels[lab_base + lane];
                    insert_best(d, label, best_d, best_l);
                }
            }
        }
    }

    fn openKd(
        ptr: [*]u8,
        size: usize,
        hdr: *const Header,
    ) !Index {
        // KD layout: header(64) | partitions | nodes | vectors | labels
        // n            → n_points
        // k            → part_count
        // n_blocks     → block_count
        // _r[0..4]     → node_count (u32 LE)
        const part_count: usize = hdr.k;
        const node_count: usize = std.mem.readInt(u32, hdr._r[0..4], .little);
        const block_count: usize = hdr.n_blocks;

        const part_off: usize = @sizeOf(Header);
        const nodes_off: usize = part_off + part_count * kdtree.PART_BYTES;
        const vectors_off: usize = nodes_off + node_count * kdtree.NODE_BYTES;
        const labels_off: usize = vectors_off + block_count * kdtree.BLOCK_BYTES;
        const end_off: usize = labels_off + block_count * kdtree.LANES;
        if (end_off != size) return error.KdSizeMismatch;

        const part_bytes = ptr[part_off .. part_off + part_count * kdtree.PART_BYTES];
        const node_bytes = ptr[nodes_off .. nodes_off + node_count * kdtree.NODE_BYTES];
        const vec_bytes = ptr[vectors_off .. vectors_off + block_count * kdtree.BLOCK_BYTES];
        const lab_bytes = ptr[labels_off .. labels_off + block_count * kdtree.LANES];

        advise(ptr, size);

        return .{
            .backend = .kd,
            .map_ptr = @alignCast(ptr),
            .map_len = size,
            .kd = kdtree.Tree.fromBytes(part_bytes, node_bytes, vec_bytes, lab_bytes),
        };
    }
};

pub const Probe = struct { cluster: u32, dist: i64 };

inline fn heap_sift_up(heap: *[MAX_PROBES]Probe, start: usize) void {
    var i = start;
    while (i > 0) {
        const parent = (i - 1) / 2;
        if (heap[i].dist > heap[parent].dist) {
            const tmp = heap[i];
            heap[i] = heap[parent];
            heap[parent] = tmp;
            i = parent;
        } else break;
    }
}

inline fn heap_sift_down(heap: *[MAX_PROBES]Probe, start: usize, size: usize) void {
    var i = start;
    while (true) {
        const left = 2 * i + 1;
        const right = 2 * i + 2;
        var largest = i;
        if (left < size and heap[left].dist > heap[largest].dist) largest = left;
        if (right < size and heap[right].dist > heap[largest].dist) largest = right;
        if (largest == i) break;
        const tmp = heap[i];
        heap[i] = heap[largest];
        heap[largest] = tmp;
        i = largest;
    }
}

inline fn insert_probe(probes: *[MAX_PROBES]Probe, count: *usize, cluster: u32, dist: i64) void {
    if (count.* < MAX_PROBES) {
        probes[count.*] = .{ .cluster = cluster, .dist = dist };
        heap_sift_up(probes, count.*);
        count.* += 1;
        return;
    }
    if (dist >= probes[0].dist) return;
    probes[0] = .{ .cluster = cluster, .dist = dist };
    heap_sift_down(probes, 0, MAX_PROBES);
}

inline fn heap_to_sorted(probes: *[MAX_PROBES]Probe, count: usize) void {
    var n = count;
    while (n > 1) {
        n -= 1;
        const tmp = probes[0];
        probes[0] = probes[n];
        probes[n] = tmp;
        heap_sift_down(probes, 0, n);
    }
}

inline fn insert_best(dist: i64, label: u8, best_d: *[TOP_K]i64, best_l: *[TOP_K]u8) void {
    if (dist >= best_d[TOP_K - 1]) return;
    var pos: usize = TOP_K - 1;
    while (pos > 0 and dist < best_d[pos - 1]) : (pos -= 1) {
        best_d[pos] = best_d[pos - 1];
        best_l[pos] = best_l[pos - 1];
    }
    best_d[pos] = dist;
    best_l[pos] = label;
}

pub inline fn quantize(v: [norm.DIMS]f32) [norm.PADDED_DIMS]i16 {
    var out: [norm.PADDED_DIMS]i16 = .{0} ** norm.PADDED_DIMS;
    inline for (0..norm.DIMS) |i| {
        var x = v[i] * QUANT_SCALE;
        if (x > QUANT_MAX) x = QUANT_MAX else if (x < -QUANT_MAX) x = -QUANT_MAX;
        out[i] = @intFromFloat(@round(x));
    }
    return out;
}

inline fn dist2_qv(a: *const [norm.PADDED_DIMS]i16, b: *const [norm.PADDED_DIMS]i16) i64 {
    const va: @Vector(16, i16) = a.*;
    const vb: @Vector(16, i16) = b.*;
    const a32: @Vector(16, i32) = va;
    const b32: @Vector(16, i32) = vb;
    const d = a32 - b32;
    const sq = d * d;
    return @reduce(.Add, sq);
}

inline fn bbox_lower_bound(q: *const [norm.PADDED_DIMS]i16, mn: *const [norm.PADDED_DIMS]i16, mx: *const [norm.PADDED_DIMS]i16) i64 {
    const qv: @Vector(16, i16) = q.*;
    const mnv: @Vector(16, i16) = mn.*;
    const mxv: @Vector(16, i16) = mx.*;
    const zero: @Vector(16, i16) = @splat(0);
    const below = @max(mnv - qv, zero);
    const above = @max(qv - mxv, zero);
    const gap = @max(below, above);
    const g32: @Vector(16, i32) = gap;
    const sq = g32 * g32;
    return @reduce(.Add, sq);
}

const LO_MASK: @Vector(LANES, i32) = .{ 0, 2, 4, 6, 8, 10, 12, 14 };
const HI_MASK: @Vector(LANES, i32) = .{ 1, 3, 5, 7, 9, 11, 13, 15 };

inline fn pair_sum_vec(q_pair_int: i32, vectors: [*]const i16, base_off: usize) @Vector(LANES, i64) {
    const q_v: @Vector(LANES, i32) = @splat(q_pair_int);
    const q_pair: @Vector(16, i16) = @bitCast(q_v);
    const block_ptr: *const [16]i16 = @ptrCast(@alignCast(vectors + base_off));
    const block_v: @Vector(16, i16) = block_ptr.*;
    const d: @Vector(16, i16) = q_pair - block_v;
    const d32: @Vector(16, i32) = d;
    const sq: @Vector(16, i32) = d32 * d32;
    const lo: @Vector(LANES, i32) = @shuffle(i32, sq, undefined, LO_MASK);
    const hi: @Vector(LANES, i32) = @shuffle(i32, sq, undefined, HI_MASK);
    const pair_sum_i32: @Vector(LANES, i32) = lo + hi;
    return pair_sum_i32;
}

inline fn blk_dist(vectors: [*]const i16, block_idx: usize, q: *const [norm.PADDED_DIMS]i16) [LANES]i64 {
    const block_off_i16 = block_idx * (norm.PADDED_DIMS * LANES);
    var acc0: @Vector(LANES, i64) = @splat(0);
    var acc1: @Vector(LANES, i64) = @splat(0);
    inline for (0..PAIRS) |p| {
        const pair_lo: i32 = @as(i32, q[p * 2]) & 0xFFFF;
        const pair_hi: i32 = @as(i32, q[p * 2 + 1]) << 16;
        const q_pair_int: i32 = pair_lo | pair_hi;
        const ps = pair_sum_vec(q_pair_int, vectors, block_off_i16 + p * LANES * 2);
        if ((p & 1) == 0) acc0 += ps else acc1 += ps;
    }
    const acc = acc0 + acc1;
    var out: [LANES]i64 = undefined;
    inline for (0..LANES) |i| out[i] = acc[i];
    return out;
}

inline fn blk_dist_prune(vectors: [*]const i16, block_idx: usize, q: *const [norm.PADDED_DIMS]i16, threshold: i64) ?[LANES]i64 {
    const block_off_i16 = block_idx * (norm.PADDED_DIMS * LANES);
    var acc0: @Vector(LANES, i64) = @splat(0);
    var acc1: @Vector(LANES, i64) = @splat(0);
    inline for (0..PAIRS) |p| {
        const pair_lo: i32 = @as(i32, q[p * 2]) & 0xFFFF;
        const pair_hi: i32 = @as(i32, q[p * 2 + 1]) << 16;
        const q_pair_int: i32 = pair_lo | pair_hi;
        const ps = pair_sum_vec(q_pair_int, vectors, block_off_i16 + p * LANES * 2);
        if ((p & 1) == 0) acc0 += ps else acc1 += ps;
        if (p == 2 or p == 4) {
            const acc_partial = acc0 + acc1;
            const t: @Vector(LANES, i64) = @splat(threshold);
            const exceeds = acc_partial >= t;
            if (@reduce(.And, exceeds)) return null;
        }
    }
    const acc = acc0 + acc1;
    var out: [LANES]i64 = undefined;
    inline for (0..LANES) |i| out[i] = acc[i];
    return out;
}

extern fn madvise(addr: ?*anyopaque, length: usize, advice: c_int) c_int;

fn advise(ptr: [*]const u8, len: usize) void {
    const MADV_RANDOM: c_int = 1;
    const MADV_WILLNEED: c_int = 3;
    const MADV_HUGEPAGE: c_int = 14;
    const p: *anyopaque = @ptrCast(@constCast(ptr));
    _ = madvise(p, len, MADV_HUGEPAGE);
    _ = madvise(p, len, MADV_WILLNEED);
    _ = madvise(p, len, MADV_RANDOM);
    var i: usize = 0;
    var acc: u8 = 0;
    while (i < len) : (i += 4096) acc ^= ptr[i];
    std.mem.doNotOptimizeAway(acc);
}
