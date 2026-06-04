const std = @import("std");

pub const Payload = struct {
    amount: f32 = 0,
    installments: u32 = 0,
    requested_at: []const u8 = "",
    avg_amount: f32 = 0,
    tx_count_24h: u32 = 0,
    known_merchants: [16][]const u8 = .{""} ** 16,
    known_n: u32 = 0,
    merchant_id: []const u8 = "",
    mcc: []const u8 = "",
    merchant_avg_amount: f32 = 0,
    is_online: bool = false,
    card_present: bool = false,
    km_from_home: f32 = 0,
    has_last: bool = false,
    last_timestamp: []const u8 = "",
    last_km_from_current: f32 = 0,
};

pub const Err = error{Malformed};

pub fn parse(buf: []const u8) Err!Payload {
    var p: Payload = .{};
    var i: usize = 0;
    try expect(buf, &i, '{');
    while (true) {
        skipWs(buf, &i);
        if (peek(buf, i) == '}') { i += 1; break; }
        const k = try readKey(buf, &i);
        try expect(buf, &i, ':');
        skipWs(buf, &i);
        if (eql(k, "transaction")) try readTx(buf, &i, &p)
        else if (eql(k, "customer")) try readCust(buf, &i, &p)
        else if (eql(k, "merchant")) try readMer(buf, &i, &p)
        else if (eql(k, "terminal")) try readTerm(buf, &i, &p)
        else if (eql(k, "last_transaction")) try readLast(buf, &i, &p)
        else skipVal(buf, &i);
        skipWs(buf, &i);
        if (peek(buf, i) == ',') i += 1;
    }
    return p;
}

fn readTx(buf: []const u8, i: *usize, p: *Payload) Err!void {
    try expect(buf, i, '{');
    while (true) {
        skipWs(buf, i);
        if (peek(buf, i.*) == '}') { i.* += 1; return; }
        const k = try readKey(buf, i);
        try expect(buf, i, ':');
        skipWs(buf, i);
        if (eql(k, "amount")) p.amount = try readF32(buf, i)
        else if (eql(k, "installments")) p.installments = try readU32(buf, i)
        else if (eql(k, "requested_at")) p.requested_at = try readStr(buf, i)
        else skipVal(buf, i);
        skipWs(buf, i);
        if (peek(buf, i.*) == ',') i.* += 1;
    }
}

fn readCust(buf: []const u8, i: *usize, p: *Payload) Err!void {
    try expect(buf, i, '{');
    while (true) {
        skipWs(buf, i);
        if (peek(buf, i.*) == '}') { i.* += 1; return; }
        const k = try readKey(buf, i);
        try expect(buf, i, ':');
        skipWs(buf, i);
        if (eql(k, "avg_amount")) p.avg_amount = try readF32(buf, i)
        else if (eql(k, "tx_count_24h")) p.tx_count_24h = try readU32(buf, i)
        else if (eql(k, "known_merchants")) try readKnown(buf, i, p)
        else skipVal(buf, i);
        skipWs(buf, i);
        if (peek(buf, i.*) == ',') i.* += 1;
    }
}

fn readMer(buf: []const u8, i: *usize, p: *Payload) Err!void {
    try expect(buf, i, '{');
    while (true) {
        skipWs(buf, i);
        if (peek(buf, i.*) == '}') { i.* += 1; return; }
        const k = try readKey(buf, i);
        try expect(buf, i, ':');
        skipWs(buf, i);
        if (eql(k, "id")) p.merchant_id = try readStr(buf, i)
        else if (eql(k, "mcc")) p.mcc = try readStr(buf, i)
        else if (eql(k, "avg_amount")) p.merchant_avg_amount = try readF32(buf, i)
        else skipVal(buf, i);
        skipWs(buf, i);
        if (peek(buf, i.*) == ',') i.* += 1;
    }
}

fn readTerm(buf: []const u8, i: *usize, p: *Payload) Err!void {
    try expect(buf, i, '{');
    while (true) {
        skipWs(buf, i);
        if (peek(buf, i.*) == '}') { i.* += 1; return; }
        const k = try readKey(buf, i);
        try expect(buf, i, ':');
        skipWs(buf, i);
        if (eql(k, "is_online")) p.is_online = try readBool(buf, i)
        else if (eql(k, "card_present")) p.card_present = try readBool(buf, i)
        else if (eql(k, "km_from_home")) p.km_from_home = try readF32(buf, i)
        else skipVal(buf, i);
        skipWs(buf, i);
        if (peek(buf, i.*) == ',') i.* += 1;
    }
}

fn readLast(buf: []const u8, i: *usize, p: *Payload) Err!void {
    if (i.* + 4 <= buf.len and std.mem.eql(u8, buf[i.* .. i.* + 4], "null")) {
        i.* += 4;
        p.has_last = false;
        return;
    }
    try expect(buf, i, '{');
    p.has_last = true;
    while (true) {
        skipWs(buf, i);
        if (peek(buf, i.*) == '}') { i.* += 1; return; }
        const k = try readKey(buf, i);
        try expect(buf, i, ':');
        skipWs(buf, i);
        if (eql(k, "timestamp")) p.last_timestamp = try readStr(buf, i)
        else if (eql(k, "km_from_current")) p.last_km_from_current = try readF32(buf, i)
        else skipVal(buf, i);
        skipWs(buf, i);
        if (peek(buf, i.*) == ',') i.* += 1;
    }
}

fn readKnown(buf: []const u8, i: *usize, p: *Payload) Err!void {
    try expect(buf, i, '[');
    p.known_n = 0;
    while (true) {
        skipWs(buf, i);
        if (peek(buf, i.*) == ']') { i.* += 1; return; }
        const s = try readStr(buf, i);
        if (p.known_n < p.known_merchants.len) {
            p.known_merchants[p.known_n] = s;
            p.known_n += 1;
        }
        skipWs(buf, i);
        if (peek(buf, i.*) == ',') i.* += 1;
    }
}

inline fn peek(buf: []const u8, i: usize) i32 {
    return if (i < buf.len) @intCast(buf[i]) else -1;
}

inline fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

inline fn skipWs(buf: []const u8, i: *usize) void {
    while (i.* < buf.len) {
        const c = buf[i.*];
        if (c != ' ' and c != '\t' and c != '\r' and c != '\n') break;
        i.* += 1;
    }
}

inline fn expect(buf: []const u8, i: *usize, c: u8) Err!void {
    skipWs(buf, i);
    if (i.* >= buf.len or buf[i.*] != c) return Err.Malformed;
    i.* += 1;
}

fn readKey(buf: []const u8, i: *usize) Err![]const u8 {
    skipWs(buf, i);
    if (i.* >= buf.len or buf[i.*] != '"') return Err.Malformed;
    i.* += 1;
    const start = i.*;
    while (i.* < buf.len and buf[i.*] != '"') : (i.* += 1) {}
    if (i.* >= buf.len) return Err.Malformed;
    const k = buf[start..i.*];
    i.* += 1;
    return k;
}

fn readStr(buf: []const u8, i: *usize) Err![]const u8 {
    skipWs(buf, i);
    if (i.* >= buf.len or buf[i.*] != '"') return Err.Malformed;
    i.* += 1;
    const start = i.*;
    while (i.* < buf.len and buf[i.*] != '"') : (i.* += 1) {}
    if (i.* >= buf.len) return Err.Malformed;
    const s = buf[start..i.*];
    i.* += 1;
    return s;
}

fn readBool(buf: []const u8, i: *usize) Err!bool {
    skipWs(buf, i);
    if (i.* + 4 <= buf.len and std.mem.eql(u8, buf[i.* .. i.* + 4], "true")) {
        i.* += 4;
        return true;
    }
    if (i.* + 5 <= buf.len and std.mem.eql(u8, buf[i.* .. i.* + 5], "false")) {
        i.* += 5;
        return false;
    }
    return Err.Malformed;
}

fn readU32(buf: []const u8, i: *usize) Err!u32 {
    skipWs(buf, i);
    var v: u32 = 0;
    var saw = false;
    while (i.* < buf.len) : (i.* += 1) {
        const c = buf[i.*];
        if (c < '0' or c > '9') break;
        v = v * 10 + (c - '0');
        saw = true;
    }
    if (!saw) return Err.Malformed;
    return v;
}

fn readF32(buf: []const u8, i: *usize) Err!f32 {
    skipWs(buf, i);
    const start = i.*;
    if (i.* < buf.len and (buf[i.*] == '-' or buf[i.*] == '+')) i.* += 1;
    while (i.* < buf.len) : (i.* += 1) {
        const c = buf[i.*];
        if (!((c >= '0' and c <= '9') or c == '.' or c == 'e' or c == 'E' or c == '+' or c == '-')) break;
    }
    if (i.* == start) return Err.Malformed;
    return std.fmt.parseFloat(f32, buf[start..i.*]) catch return Err.Malformed;
}

fn skipVal(buf: []const u8, i: *usize) void {
    skipWs(buf, i);
    if (i.* >= buf.len) return;
    const c = buf[i.*];
    if (c == '"') {
        i.* += 1;
        while (i.* < buf.len and buf[i.*] != '"') : (i.* += 1) {}
        if (i.* < buf.len) i.* += 1;
        return;
    }
    if (c == '{' or c == '[') {
        const close: u8 = if (c == '{') '}' else ']';
        var depth: u32 = 1;
        i.* += 1;
        while (i.* < buf.len and depth > 0) {
            if (buf[i.*] == c) depth += 1
            else if (buf[i.*] == close) depth -= 1;
            i.* += 1;
        }
        return;
    }
    while (i.* < buf.len) : (i.* += 1) {
        const cc = buf[i.*];
        if (cc == ',' or cc == '}' or cc == ']') return;
    }
}
