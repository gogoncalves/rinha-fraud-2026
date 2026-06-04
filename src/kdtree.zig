// 256 partitioned KD-trees over the quantised reference set.
//
// Replicates the lucasmontano top-1 layout (Rust, src/index.rs):
//   * partition_key(query) selects 1 of 256 buckets
//   * each bucket has its own KD-tree (pre-built offline)
//   * search probes the primary tree first; if best5 isn't tight enough,
//     it walks the other partitions in lower-bound order until pruning kicks in
//
// File layout (suffix to the existing index.bin, gated by version=7):
//   [Header 64B]                        (see index.zig::Header)
//   [Partitions: part_count × PART_SIZE=72]
//     key:u32 root:i32 min[16]:i16 max[16]:i16
//   [Nodes: node_count × NODE_SIZE=80]
//     left:i32 right:i32 start:i32 len:i32 min[16]:i16 max[16]:i16
//     - leaf: left<0, start is *block index* (lane-aligned), len in points
//     - inner: start is the absolute first-block of its leftmost descendant
//   [Vectors: block_count × BLOCK_BYTES] — pair-SOA (same as IVF blocks)
//   [Labels:  block_count × LANES bytes]
const std = @import("std");
const posix = std.posix;
const norm = @import("normalize.zig");

pub const VERSION: u32 = 7;
pub const LANES: usize = 8;
pub const PAIRS: usize = norm.PADDED_DIMS / 2; // 8
pub const BLOCK_BYTES: usize = norm.PADDED_DIMS * LANES * 2;
pub const TOP_K: usize = 5;
pub const PART_BYTES: usize = 4 + 4 + 32 + 32; // 72
pub const NODE_BYTES: usize = 4 + 4 + 4 + 4 + 32 + 32; // 80
pub const NUM_PARTITIONS: usize = 256;
pub const STACK_DEPTH: usize = 128;

pub const Partition = extern struct {
    key: u32,
    root: i32,
    min: [norm.PADDED_DIMS]i16,
    max: [norm.PADDED_DIMS]i16,
};

pub const Node = extern struct {
    left: i32,
    right: i32,
    start: i32,
    len: i32,
    min: [norm.PADDED_DIMS]i16,
    max: [norm.PADDED_DIMS]i16,
};

comptime {
    std.debug.assert(@sizeOf(Partition) == PART_BYTES);
    std.debug.assert(@sizeOf(Node) == NODE_BYTES);
}

pub const Tree = struct {
    partitions: [*]const Partition,
    nodes: [*]const Node,
    vectors_base: [*]const i16, // pair-SOA, LANES per block, PADDED_DIMS dims
    labels: [*]const u8,
    part_count: u32,
    node_count: u32,
    block_count: u32,
    // Inverted lookup: bucket key (0..256) → partition slot, -1 if empty.
    part_by_key: [NUM_PARTITIONS]i32,

    pub fn fromBytes(
        partitions: []const u8,
        nodes: []const u8,
        vectors: []const u8,
        labels: []const u8,
    ) Tree {
        const p_count: u32 = @intCast(partitions.len / PART_BYTES);
        const n_count: u32 = @intCast(nodes.len / NODE_BYTES);
        const b_count: u32 = @intCast(vectors.len / BLOCK_BYTES);
        const p_ptr: [*]const Partition = @ptrCast(@alignCast(partitions.ptr));
        const n_ptr: [*]const Node = @ptrCast(@alignCast(nodes.ptr));
        const v_ptr: [*]const i16 = @ptrCast(@alignCast(vectors.ptr));
        var by_key: [NUM_PARTITIONS]i32 = .{-1} ** NUM_PARTITIONS;
        var i: u32 = 0;
        while (i < p_count) : (i += 1) {
            const k = p_ptr[i].key;
            if (k < NUM_PARTITIONS) by_key[k] = @intCast(i);
        }
        return .{
            .partitions = p_ptr,
            .nodes = n_ptr,
            .vectors_base = v_ptr,
            .labels = labels.ptr,
            .part_count = p_count,
            .node_count = n_count,
            .block_count = b_count,
            .part_by_key = by_key,
        };
    }

    pub fn score(self: *const Tree, q: *const [norm.PADDED_DIMS]i16) u32 {
        const part_key = @import("partition.zig").partitionKey(q);
        var best_d: [TOP_K]i64 = .{std.math.maxInt(i64)} ** TOP_K;
        var best_l: [TOP_K]u8 = .{0} ** TOP_K;

        // Primary partition: walk its KD-tree first to seed best5.
        const primary = self.part_by_key[part_key & 0xFF];
        if (primary >= 0) {
            const root = self.partitions[@intCast(primary)].root;
            self.searchNode(root, 0, q, &best_d, &best_l);
        }

        // Other partitions: prune by their bbox lower bound, walk in order.
        // (256 max — fine to keep on stack.)
        var probes: [NUM_PARTITIONS]Probe = undefined;
        var n_probes: usize = 0;
        var i: u32 = 0;
        while (i < self.part_count) : (i += 1) {
            if (@as(i32, @intCast(i)) == primary) continue;
            const p = &self.partitions[i];
            const lb = bboxLowerBound(q, &p.min, &p.max);
            if (lb >= best_d[TOP_K - 1]) continue;
            probes[n_probes] = .{ .part = i, .lb = lb };
            n_probes += 1;
        }
        sortProbes(probes[0..n_probes]);

        var pi: usize = 0;
        while (pi < n_probes) : (pi += 1) {
            const probe = probes[pi];
            if (probe.lb >= best_d[TOP_K - 1]) break;
            const root = self.partitions[probe.part].root;
            self.searchNode(root, probe.lb, q, &best_d, &best_l);
        }

        var frauds: u32 = 0;
        inline for (0..TOP_K) |j| frauds += best_l[j];
        return frauds;
    }

    fn searchNode(
        self: *const Tree,
        root: i32,
        root_bound: i64,
        q: *const [norm.PADDED_DIMS]i16,
        best_d: *[TOP_K]i64,
        best_l: *[TOP_K]u8,
    ) void {
        if (root < 0 or @as(u32, @intCast(root)) >= self.node_count) return;
        var stack_node: [STACK_DEPTH]i32 = undefined;
        var stack_bound: [STACK_DEPTH]i64 = undefined;
        var sp: usize = 0;
        var current: i32 = root;
        var current_bound: i64 = root_bound;
        while (true) {
            if (current_bound < best_d[TOP_K - 1]) {
                const node = &self.nodes[@intCast(current)];
                if (node.left < 0) {
                    self.scanLeaf(node.start, @intCast(node.len), q, best_d, best_l);
                } else {
                    const lnode = &self.nodes[@intCast(node.left)];
                    const rnode = &self.nodes[@intCast(node.right)];
                    const lb = bboxLowerBound(q, &lnode.min, &lnode.max);
                    const rb = bboxLowerBound(q, &rnode.min, &rnode.max);
                    var near = node.left;
                    var near_b = lb;
                    var far = node.right;
                    var far_b = rb;
                    if (rb < lb) {
                        near = node.right;
                        near_b = rb;
                        far = node.left;
                        far_b = lb;
                    }
                    if (far_b < best_d[TOP_K - 1] and sp < STACK_DEPTH) {
                        stack_node[sp] = far;
                        stack_bound[sp] = far_b;
                        sp += 1;
                    }
                    current = near;
                    current_bound = near_b;
                    continue;
                }
            }
            if (sp == 0) return;
            sp -= 1;
            current = stack_node[sp];
            current_bound = stack_bound[sp];
        }
    }

    fn scanLeaf(
        self: *const Tree,
        start_block: i32,
        total_len: u32,
        q: *const [norm.PADDED_DIMS]i16,
        best_d: *[TOP_K]i64,
        best_l: *[TOP_K]u8,
    ) void {
        if (total_len == 0) return;
        const sb: usize = @intCast(start_block);
        const full_blocks: usize = total_len / LANES;
        const tail: usize = total_len % LANES;
        var b: usize = 0;
        while (b < full_blocks) : (b += 1) {
            const block_idx = sb + b;
            const dists = blkDist(self.vectors_base, block_idx, q);
            const lab_base = block_idx * LANES;
            inline for (0..LANES) |lane| {
                const d = dists[lane];
                if (d < best_d[TOP_K - 1]) {
                    const label = self.labels[lab_base + lane];
                    insertBest(d, label, best_d, best_l);
                }
            }
        }
        if (tail != 0) {
            const block_idx = sb + full_blocks;
            const dists = blkDist(self.vectors_base, block_idx, q);
            const lab_base = block_idx * LANES;
            var lane: usize = 0;
            while (lane < tail) : (lane += 1) {
                const d = dists[lane];
                if (d < best_d[TOP_K - 1]) {
                    const label = self.labels[lab_base + lane];
                    insertBest(d, label, best_d, best_l);
                }
            }
        }
    }
};

const Probe = struct { part: u32, lb: i64 };

inline fn sortProbes(probes: []Probe) void {
    // Insertion sort — n is small (≤256, usually much smaller after pruning).
    var i: usize = 1;
    while (i < probes.len) : (i += 1) {
        const cur = probes[i];
        var j: usize = i;
        while (j > 0 and probes[j - 1].lb > cur.lb) : (j -= 1) {
            probes[j] = probes[j - 1];
        }
        probes[j] = cur;
    }
}

inline fn insertBest(dist: i64, label: u8, best_d: *[TOP_K]i64, best_l: *[TOP_K]u8) void {
    if (dist >= best_d[TOP_K - 1]) return;
    var pos: usize = TOP_K - 1;
    while (pos > 0 and dist < best_d[pos - 1]) : (pos -= 1) {
        best_d[pos] = best_d[pos - 1];
        best_l[pos] = best_l[pos - 1];
    }
    best_d[pos] = dist;
    best_l[pos] = label;
}

inline fn bboxLowerBound(
    q: *const [norm.PADDED_DIMS]i16,
    mn: *const [norm.PADDED_DIMS]i16,
    mx: *const [norm.PADDED_DIMS]i16,
) i64 {
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

// --- Pair-SOA distance kernels (lifted from index.zig, kept self-contained
//     so kdtree.zig works without IVF). ---

const LO_MASK: @Vector(LANES, i32) = .{ 0, 2, 4, 6, 8, 10, 12, 14 };
const HI_MASK: @Vector(LANES, i32) = .{ 1, 3, 5, 7, 9, 11, 13, 15 };

inline fn pairSumVec(q_pair_int: i32, vectors: [*]const i16, base_off: usize) @Vector(LANES, i32) {
    const q_v: @Vector(LANES, i32) = @splat(q_pair_int);
    const q_pair: @Vector(16, i16) = @bitCast(q_v);
    const block_ptr: *const [16]i16 = @ptrCast(@alignCast(vectors + base_off));
    const block_v: @Vector(16, i16) = block_ptr.*;
    const d: @Vector(16, i16) = q_pair - block_v;
    const d32: @Vector(16, i32) = d;
    const sq: @Vector(16, i32) = d32 * d32;
    const lo: @Vector(LANES, i32) = @shuffle(i32, sq, undefined, LO_MASK);
    const hi: @Vector(LANES, i32) = @shuffle(i32, sq, undefined, HI_MASK);
    return lo + hi;
}

inline fn blkDist(vectors: [*]const i16, block_idx: usize, q: *const [norm.PADDED_DIMS]i16) [LANES]i64 {
    const block_off_i16 = block_idx * (norm.PADDED_DIMS * LANES);
    var acc0: @Vector(LANES, i32) = @splat(0);
    var acc1: @Vector(LANES, i32) = @splat(0);
    inline for (0..PAIRS) |p| {
        const pair_lo: i32 = @as(i32, q[p * 2]) & 0xFFFF;
        const pair_hi: i32 = @as(i32, q[p * 2 + 1]) << 16;
        const q_pair_int: i32 = pair_lo | pair_hi;
        const ps = pairSumVec(q_pair_int, vectors, block_off_i16 + p * LANES * 2);
        if ((p & 1) == 0) acc0 += ps else acc1 += ps;
    }
    const acc = acc0 + acc1;
    var out: [LANES]i64 = undefined;
    inline for (0..LANES) |i| out[i] = @intCast(acc[i]);
    return out;
}
