pub const Stamp = struct {
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
};

pub fn parse(s: []const u8) Stamp {
    return .{
        .year = di(s[0]) * 1000 + di(s[1]) * 100 + di(s[2]) * 10 + di(s[3]),
        .month = d(s[5]) * 10 + d(s[6]),
        .day = d(s[8]) * 10 + d(s[9]),
        .hour = d(s[11]) * 10 + d(s[12]),
        .minute = d(s[14]) * 10 + d(s[15]),
        .second = d(s[17]) * 10 + d(s[18]),
    };
}

inline fn d(c: u8) u32 { return c - '0'; }
inline fn di(c: u8) i32 { return @as(i32, c) - '0'; }

pub fn dayOfWeek(year: i32, month: u32, day: u32) u32 {
    const t = [_]i32{ 0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4 };
    var y = year;
    if (month < 3) y -= 1;
    const raw = @mod(y + @divFloor(y, 4) - @divFloor(y, 100) + @divFloor(y, 400) + t[month - 1] + @as(i32, @intCast(day)), 7);
    const s = @as(u32, @intCast(@mod(raw + 7, 7)));
    return (s + 6) % 7;
}

pub fn daysSinceEpoch(year: i32, month: u32, day: u32) i64 {
    var y: i64 = year;
    if (month <= 2) y -= 1;
    const era = @divFloor(if (y >= 0) y else y - 399, 400);
    const yoe = y - era * 400;
    const m: i64 = @intCast(month);
    const m_shift: i64 = if (m > 2) m - 3 else m + 9;
    const doy = @divTrunc(153 * m_shift + 2, 5) + @as(i64, @intCast(day)) - 1;
    const doe = yoe * 365 + @divTrunc(yoe, 4) - @divTrunc(yoe, 100) + doy;
    return era * 146097 + doe - 719468;
}

pub fn epochSeconds(s: Stamp) i64 {
    return daysSinceEpoch(s.year, s.month, s.day) * 86400 +
        @as(i64, s.hour) * 3600 +
        @as(i64, s.minute) * 60 +
        @as(i64, s.second);
}
