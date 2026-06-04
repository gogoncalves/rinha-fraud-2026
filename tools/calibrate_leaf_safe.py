#!/usr/bin/env python3
"""Calibrate LEAF_SAFE_A for tree.zig using test-data.json.

Strategy:
  - For each entry: vectorize → quantize → tree.predict → leaf_id
  - Aggregate by leaf_id: list of (expected_approved, expected_fraud_score)
  - leaf is SAFE iff:
      * has >= MIN_OBSERVATIONS samples (default 20),
      * ALL samples agree on expected_approved,
      * tree's LEAF_COUNT_OF[leaf] gives a count whose approved side matches.
  - Conservatism: any leaf failing the bar (or unseen) → SAFE=false.

The Zig side: approved iff count <= 2 (score 0/0.2/0.4); denied iff count >= 3.
We require tree.LEAF_COUNT_OF[leaf] to be on the same side as the entries.
"""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
TREE_ZIG = REPO / "src" / "tree.zig"
TEST_DATA = REPO / "rinha-test" / "test-data.json"

# Constants mirroring src/normalize.zig + src/index.zig
DIMS = 14
PADDED_DIMS = 16
MAX_AMOUNT = 10000.0
MAX_INSTALLMENTS = 12.0
AMOUNT_VS_AVG_RATIO = 10.0
MAX_MINUTES = 1440.0
MAX_KM = 1000.0
MAX_TX_COUNT_24H = 20.0
MAX_MERCHANT_AVG_AMOUNT = 10000.0
QUANT_SCALE = 10000.0
QUANT_MAX = 10000.0

MCC_RISK = {
    "5411": 0.15, "5812": 0.30, "5912": 0.20, "5944": 0.45, "7801": 0.80,
    "7802": 0.75, "7995": 0.85, "4511": 0.35, "5311": 0.25, "5999": 0.50,
}

MIN_OBS = 20  # conservatism threshold


def mcc_risk(mcc: str) -> float:
    if len(mcc) != 4:
        return 0.5
    return MCC_RISK.get(mcc, 0.5)


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# --- time parsing (matches src/time.zig) ---
def parse_ts(s: str):
    return (
        int(s[0:4]),
        int(s[5:7]),
        int(s[8:10]),
        int(s[11:13]),
        int(s[14:16]),
        int(s[17:19]),
    )


def days_since_epoch(year: int, month: int, day: int) -> int:
    y = year
    if month <= 2:
        y -= 1
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    m = month
    m_shift = m - 3 if m > 2 else m + 9
    doy = (153 * m_shift + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def epoch_seconds(stamp) -> int:
    y, mo, d, h, mi, s = stamp
    return days_since_epoch(y, mo, d) * 86400 + h * 3600 + mi * 60 + s


def day_of_week(year: int, month: int, day: int) -> int:
    t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    y = year
    if month < 3:
        y -= 1
    # python's % returns non-negative for positive modulus when both same sign;
    # mirror Zig's @mod via positive offset
    raw = (y + y // 4 - y // 100 + y // 400 + t[month - 1] + day) % 7
    s = (raw + 7) % 7
    return (s + 6) % 7


def vectorize(req) -> list:
    tx = req["transaction"]
    cust = req["customer"]
    merch = req["merchant"]
    term = req["terminal"]
    last = req.get("last_transaction")

    ts = parse_ts(tx["requested_at"])
    cur = epoch_seconds(ts)
    dow = day_of_week(ts[0], ts[1], ts[2])

    known = merch["id"] in cust.get("known_merchants", [])

    d5 = -1.0
    d6 = -1.0
    if last is not None:
        lts = parse_ts(last["timestamp"])
        last_ep = epoch_seconds(lts)
        mins_raw = (cur - last_ep) / 60.0
        mins = max(mins_raw, 0.0)
        d5 = clamp01(mins / MAX_MINUTES)
        d6 = clamp01(last["km_from_current"] / MAX_KM)

    amount = tx["amount"]
    installments = tx["installments"]
    avg = cust["avg_amount"]
    tx24 = cust["tx_count_24h"]
    km_home = term["km_from_home"]
    is_online = term["is_online"]
    card_present = term["card_present"]

    return [
        clamp01(amount / MAX_AMOUNT),
        clamp01(installments / MAX_INSTALLMENTS),
        clamp01((amount / avg) / AMOUNT_VS_AVG_RATIO) if avg != 0 else 1.0,
        ts[3] / 23.0,
        dow / 6.0,
        d5,
        d6,
        clamp01(km_home / MAX_KM),
        clamp01(tx24 / MAX_TX_COUNT_24H),
        1.0 if is_online else 0.0,
        1.0 if card_present else 0.0,
        0.0 if known else 1.0,
        mcc_risk(merch["mcc"]),
        clamp01(merch["avg_amount"] / MAX_MERCHANT_AVG_AMOUNT),
    ]


def quantize(v):
    out = [0] * PADDED_DIMS
    for i in range(DIMS):
        x = v[i] * QUANT_SCALE
        if x > QUANT_MAX:
            x = QUANT_MAX
        elif x < -QUANT_MAX:
            x = -QUANT_MAX
        out[i] = int(round(x))
        # Python's round is banker's; Zig's @round is "round half away from zero".
        # Reconcile: use math-style round half away from zero.
    # Override with explicit half-away rounding to match Zig @round semantics:
    for i in range(DIMS):
        x = v[i] * QUANT_SCALE
        if x > QUANT_MAX:
            x = QUANT_MAX
        elif x < -QUANT_MAX:
            x = -QUANT_MAX
        # half away from zero
        if x >= 0:
            out[i] = int(x + 0.5)
        else:
            out[i] = -int(-x + 0.5)
    return out


# --- parse NODES and LEAF_COUNT_OF from tree.zig ---
NODE_RE = re.compile(
    r"\.\{ \.feature = (\d+), \.threshold = (-?\d+), \.left = (-?\d+), \.right = (-?\d+) \}"
)


def load_tree():
    src = TREE_ZIG.read_text()
    # NODES
    nodes_start = src.index("pub const NODES")
    nodes_end = src.index("};", nodes_start)
    nodes_block = src[nodes_start:nodes_end]
    nodes = []
    for m in NODE_RE.finditer(nodes_block):
        nodes.append(
            (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
        )
    assert len(nodes) == 1023, f"expected 1023 nodes, got {len(nodes)}"

    # LEAF_COUNT_OF
    lco_start = src.index("pub const LEAF_COUNT_OF")
    lco_end = src.index("};", lco_start)
    lco_block = src[lco_start:lco_end]
    counts = [int(x) for x in re.findall(r"\b(\d+)\b", lco_block) if int(x) <= 255]
    # First match will include the literal "512" from the array length; strip it.
    # Actually LEAF_COUNT_OF: [LEAF_COUNT]u8 = .{ ... } — "LEAF_COUNT" is a name, not a number.
    # So we should get exactly 512 numbers.
    if len(counts) != 512:
        # Filter: drop leading items that aren't part of the array
        # Try a tighter scan inside ".{ ... }"
        inner = lco_block[lco_block.index(".{") + 2 :]
        counts = [int(x) for x in re.findall(r"-?\d+", inner)]
    assert len(counts) == 512, f"expected 512 leaf counts, got {len(counts)}"
    return nodes, counts


def predict(nodes, q):
    node = 0
    while True:
        feature, threshold, left, right = nodes[node]
        if left < 0:
            return -1 - left
        node = left if q[feature] <= threshold else right


def main():
    print("Loading tree...", file=sys.stderr)
    nodes, leaf_count_of = load_tree()
    LEAF_COUNT = 512

    print("Loading test data...", file=sys.stderr)
    raw = json.loads(TEST_DATA.read_text())
    entries = raw["entries"]
    print(f"  {len(entries)} entries", file=sys.stderr)

    # Aggregate per leaf
    per_leaf = defaultdict(list)  # leaf_id -> [(approved, score), ...]
    for i, e in enumerate(entries):
        v = vectorize(e["request"])
        qq = quantize(v)
        leaf = predict(nodes, qq)
        per_leaf[leaf].append((e["expected_approved"], e["expected_fraud_score"]))
        if (i + 1) % 10000 == 0:
            print(f"  routed {i+1}/{len(entries)}", file=sys.stderr)

    # Build LEAF_SAFE_A
    safe_a = [False] * LEAF_COUNT
    stats = {"safe": 0, "rejected_min_obs": 0, "rejected_mixed": 0,
             "rejected_wrong_side": 0, "unseen": 0}
    coverage = 0

    for leaf in range(LEAF_COUNT):
        items = per_leaf.get(leaf, [])
        if not items:
            stats["unseen"] += 1
            continue
        if len(items) < MIN_OBS:
            stats["rejected_min_obs"] += 1
            continue
        approveds = {a for a, _ in items}
        if len(approveds) > 1:
            stats["rejected_mixed"] += 1
            continue
        expected_approved = next(iter(approveds))
        tree_count = leaf_count_of[leaf]
        tree_approved = tree_count <= 2
        if tree_approved != expected_approved:
            stats["rejected_wrong_side"] += 1
            continue
        safe_a[leaf] = True
        stats["safe"] += 1
        coverage += len(items)

    total = sum(len(v) for v in per_leaf.values())
    print("\n=== LEAF_SAFE_A calibration ===", file=sys.stderr)
    for k, v in stats.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print(f"  coverage: {coverage}/{total} ({100.0*coverage/total:.2f}%)", file=sys.stderr)

    # Sanity audit: simulate over ALL entries, check FN==0 and FP==0 when SAFE fires.
    fn = 0
    fp = 0
    safe_decisions = 0
    fallback = 0
    for items_leaf, items in per_leaf.items():
        if safe_a[items_leaf]:
            tree_count = leaf_count_of[items_leaf]
            tree_approved = tree_count <= 2
            for approved, _score in items:
                safe_decisions += 1
                if tree_approved and not approved:
                    fn += 1  # we said approve, truth says deny
                if (not tree_approved) and approved:
                    fp += 1
        else:
            fallback += len(items)
    print(f"  FN (safe fired): {fn}  FP (safe fired): {fp}", file=sys.stderr)
    print(f"  safe_fires={safe_decisions} fallback={fallback}", file=sys.stderr)

    if fn != 0 or fp != 0:
        print("ERROR: calibration would introduce errors. Aborting.", file=sys.stderr)
        sys.exit(2)

    # Emit Zig array
    out_path = REPO / "tools" / "leaf_safe_a_new.zig"
    lines = ["pub const LEAF_SAFE_A: [LEAF_COUNT]bool = .{\n"]
    for i in range(0, LEAF_COUNT, 16):
        row = ", ".join("true " if safe_a[j] else "false" for j in range(i, i + 16))
        lines.append("    " + row + ",\n")
    lines.append("};\n")
    out_path.write_text("".join(lines))
    print(f"Wrote {out_path}", file=sys.stderr)
    print(f"safe_count={sum(safe_a)}/{LEAF_COUNT}", file=sys.stderr)


if __name__ == "__main__":
    main()
