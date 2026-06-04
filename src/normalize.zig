const time = @import("time.zig");
const json = @import("json.zig");

pub const DIMS: usize = 14;
pub const PADDED_DIMS: usize = 16;

pub const MAX_AMOUNT: f32 = 10_000.0;
pub const MAX_INSTALLMENTS: f32 = 12.0;
pub const AMOUNT_VS_AVG_RATIO: f32 = 10.0;
pub const MAX_MINUTES: f32 = 1440.0;
pub const MAX_KM: f32 = 1000.0;
pub const MAX_TX_COUNT_24H: f32 = 20.0;
pub const MAX_MERCHANT_AVG_AMOUNT: f32 = 10_000.0;

pub fn mccRisk(mcc: []const u8) f32 {
    if (mcc.len != 4) return 0.5;
    const x = pack(mcc);
    return switch (x) {
        packLit("5411") => 0.15,
        packLit("5812") => 0.30,
        packLit("5912") => 0.20,
        packLit("5944") => 0.45,
        packLit("7801") => 0.80,
        packLit("7802") => 0.75,
        packLit("7995") => 0.85,
        packLit("4511") => 0.35,
        packLit("5311") => 0.25,
        packLit("5999") => 0.50,
        else => 0.5,
    };
}

inline fn pack(s: []const u8) u32 {
    return (@as(u32, s[0]) << 24) | (@as(u32, s[1]) << 16) | (@as(u32, s[2]) << 8) | s[3];
}
inline fn packLit(comptime s: *const [4:0]u8) u32 {
    return (@as(u32, s[0]) << 24) | (@as(u32, s[1]) << 16) | (@as(u32, s[2]) << 8) | s[3];
}

inline fn clamp01(x: f32) f32 {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

pub fn vectorize(p: *const json.Payload) [DIMS]f32 {
    const ts = time.parse(p.requested_at);
    const cur = time.epochSeconds(ts);
    const dow = time.dayOfWeek(ts.year, ts.month, ts.day);

    var known = false;
    var i: usize = 0;
    while (i < p.known_n) : (i += 1) {
        const m = p.known_merchants[i];
        if (m.len == p.merchant_id.len) {
            var same = true;
            var j: usize = 0;
            while (j < m.len) : (j += 1) if (m[j] != p.merchant_id[j]) { same = false; break; };
            if (same) { known = true; break; }
        }
    }

    var d5: f32 = -1.0;
    var d6: f32 = -1.0;
    if (p.has_last) {
        const lts = time.parse(p.last_timestamp);
        const last = time.epochSeconds(lts);
        const mins_raw: f32 = @as(f32, @floatFromInt(cur - last)) / 60.0;
        const mins: f32 = if (mins_raw < 0.0) 0.0 else mins_raw;
        d5 = clamp01(mins / MAX_MINUTES);
        d6 = clamp01(p.last_km_from_current / MAX_KM);
    }

    return .{
        clamp01(p.amount / MAX_AMOUNT),
        clamp01(@as(f32, @floatFromInt(p.installments)) / MAX_INSTALLMENTS),
        clamp01((p.amount / p.avg_amount) / AMOUNT_VS_AVG_RATIO),
        @as(f32, @floatFromInt(ts.hour)) / 23.0,
        @as(f32, @floatFromInt(dow)) / 6.0,
        d5,
        d6,
        clamp01(p.km_from_home / MAX_KM),
        clamp01(@as(f32, @floatFromInt(p.tx_count_24h)) / MAX_TX_COUNT_24H),
        if (p.is_online) 1.0 else 0.0,
        if (p.card_present) 1.0 else 0.0,
        if (known) 0.0 else 1.0,
        mccRisk(p.mcc),
        clamp01(p.merchant_avg_amount / MAX_MERCHANT_AVG_AMOUNT),
    };
}
