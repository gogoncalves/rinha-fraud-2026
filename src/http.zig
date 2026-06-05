const std = @import("std");
const json = @import("json.zig");
const norm = @import("normalize.zig");
const idx_mod = @import("index.zig");
const Index = idx_mod.Index;
const tree = @import("tree.zig");
const partition = @import("partition.zig");

pub const Parsed = struct {
    method: enum { get_ready, post_score, other },
    body: []const u8,
    end: usize,
};

pub const ParseErr = error{ Incomplete, Bad };

pub fn parse(buf: []const u8) ParseErr!Parsed {
    if (buf.len < 16) return ParseErr.Incomplete;

    if (std.mem.startsWith(u8, buf, "POST /fraud-score")) {
        const hdr_end = std.mem.indexOf(u8, buf, "\r\n\r\n") orelse return ParseErr.Incomplete;
        const cl = findContentLength(buf[0 .. hdr_end + 2]) orelse return ParseErr.Bad;
        const body_start = hdr_end + 4;
        const total = body_start + cl;
        if (buf.len < total) return ParseErr.Incomplete;
        return .{ .method = .post_score, .body = buf[body_start..total], .end = total };
    }
    if (std.mem.startsWith(u8, buf, "GET /ready")) {
        const sep = std.mem.indexOf(u8, buf, "\r\n\r\n") orelse return ParseErr.Incomplete;
        return .{ .method = .get_ready, .body = "", .end = sep + 4 };
    }
    const sep = std.mem.indexOf(u8, buf, "\r\n\r\n") orelse return ParseErr.Incomplete;
    return .{ .method = .other, .body = "", .end = sep + 4 };
}

fn findContentLength(hdrs: []const u8) ?usize {
    var i: usize = 0;
    while (i < hdrs.len) {
        const nl = std.mem.indexOfScalarPos(u8, hdrs, i, '\n') orelse return null;
        const line_end = if (nl > i and hdrs[nl - 1] == '\r') nl - 1 else nl;
        const line = hdrs[i..line_end];
        i = nl + 1;
        if (line.len < 15) continue;
        if (asciiEqlIgnoreCase(line[0..15], "content-length:")) {
            var v: usize = 0;
            var j: usize = 15;
            while (j < line.len and (line[j] == ' ' or line[j] == '\t')) : (j += 1) {}
            while (j < line.len and line[j] >= '0' and line[j] <= '9') : (j += 1) v = v * 10 + (line[j] - '0');
            return v;
        }
    }
    return null;
}

inline fn asciiEqlIgnoreCase(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| {
        const xl = if (x >= 'A' and x <= 'Z') x + 32 else x;
        const yl = if (y >= 'A' and y <= 'Z') y + 32 else y;
        if (xl != yl) return false;
    }
    return true;
}

pub const READY = "HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\ncontent-length: 2\r\nconnection: keep-alive\r\n\r\nok";
pub const NOT_FOUND = "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: keep-alive\r\n\r\n";

const APPROVED_HDR = "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 35\r\nconnection: keep-alive\r\n\r\n";
const DENIED_HDR = "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 36\r\nconnection: keep-alive\r\n\r\n";

pub const SCORES = [_][]const u8{
    APPROVED_HDR ++ "{\"approved\":true,\"fraud_score\":0.0}",
    APPROVED_HDR ++ "{\"approved\":true,\"fraud_score\":0.2}",
    APPROVED_HDR ++ "{\"approved\":true,\"fraud_score\":0.4}",
    DENIED_HDR ++ "{\"approved\":false,\"fraud_score\":0.6}",
    DENIED_HDR ++ "{\"approved\":false,\"fraud_score\":0.8}",
    DENIED_HDR ++ "{\"approved\":false,\"fraud_score\":1.0}",
};

pub fn respond(idx: *const Index, req: Parsed) []const u8 {
    switch (req.method) {
        .get_ready => return READY,
        .post_score => {
            const payload = json.parse(req.body) catch return SCORES[0];
            const v = norm.vectorize(&payload);
            const qq = idx_mod.quantize(v);
            // Tree fast-path (Mode A): if the leaf is "safe", trust its label count.
            const leaf = tree.predict(&qq);
            if (tree.LEAF_SAFE_A[leaf]) {
                @branchHint(.likely);
                const c = tree.LEAF_COUNT_OF[leaf];
                return SCORES[@min(c, 5)];
            }
            // Mode B: route-restricted partition bypass. Fires only when both
            // the leaf and the partition_key are in a tightly-audited whitelist
            // where every observed sample agreed on the cached verdict. Final
            // safety net before the KNN scan.
            if (tree.LEAF_SAFE_B[leaf]) {
                @branchHint(.unlikely);
                const part = partition.partitionKey(&qq);
                const mask = tree.LEAF_B_PART_MASK[leaf];
                const word = part >> 6;
                const bit: u6 = @intCast(part & 63);
                if (((mask[word] >> bit) & 1) != 0) {
                    const c = tree.LEAF_B_VERDICT[leaf];
                    return SCORES[@min(c, 5)];
                }
            }
            const frauds = idx.score_qq(&qq);
            return SCORES[@min(frauds, 5)];
        },
        .other => return NOT_FOUND,
    }
}
