// 256-bucket partition key.
//
// Ported from lucasmontano top-1 (Rust): src/index.rs::partition_key.
// Splits the query space via sign/threshold checks on five dims, packing
// the result into 8 bits → 256 buckets. Each bucket gets its own KD sub-tree
// so we narrow the search radius dramatically before touching geometry.
//
// Layout (bit → dim/test):
//   bit 0  : v[5]  >= 0                  (last-tx minutes non-negative, i.e. has_last_tx)
//   bit 1  : v[9]  >  0                  (is_online)
//   bit 2  : v[10] >  0                  (card_present)
//   bit 3  : v[11] >  0                  (merchant_unknown)
//   bits 4-5 : v[12] (mcc_risk) bucket   (0..2047 / ..4095 / ..6143 / >6143)
//   bit 6  : v[2]  > 4096                (amount-vs-avg ratio high)
//   bit 7  : v[8]  > 2048                (tx_count_24h high)
const norm = @import("normalize.zig");

pub const NUM_BUCKETS: usize = 256;

pub inline fn partitionKey(v: *const [norm.PADDED_DIMS]i16) u32 {
    var key: u32 = 0;
    if (v[5] >= 0) key |= 1 << 0;
    if (v[9] > 0) key |= 1 << 1;
    if (v[10] > 0) key |= 1 << 2;
    if (v[11] > 0) key |= 1 << 3;
    const mr = v[12];
    if (mr <= 2047) {
        // bits 4-5 stay 0
    } else if (mr <= 4095) {
        key |= 1 << 4;
    } else if (mr <= 6143) {
        key |= 2 << 4;
    } else {
        key |= 3 << 4;
    }
    if (v[2] > 4096) key |= 1 << 6;
    if (v[8] > 2048) key |= 1 << 7;
    return key;
}
