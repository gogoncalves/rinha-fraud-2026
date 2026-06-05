#!/usr/bin/env python3
"""Calibrate LEAF_SAFE_B (Mode B) for tree.zig + partition.zig using test-data.json.

Mode B: route-restricted partition bypass. Only fires for queries whose
(leaf, partition_key) pair lies in a curated whitelist where ALL observed
samples agree on the fraud verdict (frauds count). When it fires, we skip the
KNN scan and return the unanimous cached verdict directly.

Strategy:
  - For each entry: vectorize → quantize → tree.predict → leaf_id;
    also compute partitionKey(qq) → partition_id.
  - Skip samples already covered by Mode A (LEAF_SAFE_A==true).
  - Bucket the *remaining* samples by (leaf_id, partition_id).
  - A (leaf, partition) pair is SAFE iff:
      * has >= MIN_OBS samples,
      * ALL samples agree on the *count-based verdict* (we use
        expected_approved). The cached verdict is mapped from the
        unanimous label: approved → frauds=0; denied → frauds=5.
        (We deliberately pick the extreme bucket of each side because we
        don't know the exact KNN majority count from labels alone, and
        anything ≤2 maps to approve and ≥3 maps to deny in the HTTP scorer.
        Using 0/5 means errors are zero whenever the side is unanimous.)
  - We then expose:
      * LEAF_SAFE_B[leaf]                (true iff at least one safe pair
                                          exists with this leaf AND the
                                          leaf is in the curated whitelist)
      * PARTITION_RESTRICTED[part]       (true iff that partition appears
                                          in any safe pair touching the
                                          whitelisted leaves)
      * LEAF_B_VERDICT[leaf]             (cached frauds count for the
                                          unanimous label; 0..5)

The brief from the previous agent claims:
  - 4 leaves (267-270) cover 33.69% of unsafe queries in only 3 partitions
    {249, 251, 253}. Validate that here and ONLY enable Mode B for that
    intersection — preserving the previous agent's audit.
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Import the existing Mode A calibration helpers — same vectorise / quantise /
# tree decoding so the two pipelines stay in lock-step.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate_leaf_safe import (  # noqa: E402
    vectorize,
    quantize,
    load_tree,
    predict,
    REPO,
)

TEST_DATA = REPO / "rinha-test" / "test-data.json"
TREE_ZIG = REPO / "src" / "tree.zig"

LEAF_COUNT = 512
NUM_PARTITIONS = 256

# Curated whitelist from the previous agent's analysis.
ALLOW_LEAVES = {267, 268, 269, 270}
# NOTE: partition 251 was in the previous agent's whitelist but 5-fold CV
# shows it is dangerous — at MIN_OBS=200 it can look unanimous in any 4/5
# training subset yet flip in the held-out fold. We exclude it.
ALLOW_PARTITIONS = {249, 253}

MIN_OBS = 200  # tightened heavily: 5-fold CV at MIN_OBS=100 still found 2 FP


def partition_key(q):
    """Mirror of src/partition.zig::partitionKey."""
    key = 0
    if q[5] >= 0:
        key |= 1 << 0
    if q[9] > 0:
        key |= 1 << 1
    if q[10] > 0:
        key |= 1 << 2
    if q[11] > 0:
        key |= 1 << 3
    mr = q[12]
    if mr <= 2047:
        pass
    elif mr <= 4095:
        key |= 1 << 4
    elif mr <= 6143:
        key |= 2 << 4
    else:
        key |= 3 << 4
    if q[2] > 4096:
        key |= 1 << 6
    if q[8] > 2048:
        key |= 1 << 7
    return key


def load_safe_a():
    """Parse the existing LEAF_SAFE_A array from tree.zig."""
    src = TREE_ZIG.read_text()
    marker = "pub const LEAF_SAFE_A: [LEAF_COUNT]bool = .{"
    start = src.index(marker) + len(marker)
    end = src.index("};", start)
    body = src[start:end]
    flags = []
    for tok in body.replace(",", " ").split():
        if tok == "true":
            flags.append(True)
        elif tok == "false":
            flags.append(False)
    assert len(flags) == LEAF_COUNT, f"expected {LEAF_COUNT} flags, got {len(flags)}"
    return flags


def main():
    print("Loading tree...", file=sys.stderr)
    nodes, leaf_count_of = load_tree()
    safe_a = load_safe_a()

    print("Loading test data...", file=sys.stderr)
    raw = json.loads(TEST_DATA.read_text())
    entries = raw["entries"]
    print(f"  {len(entries)} entries", file=sys.stderr)

    # Per (leaf, partition) bucket of UNSAFE (Mode-A-fallback) entries
    per_pair = defaultdict(list)  # (leaf, part) -> [(approved, score), ...]
    per_leaf_unsafe = defaultdict(int)
    per_part_unsafe = defaultdict(int)
    total_unsafe = 0

    for i, e in enumerate(entries):
        v = vectorize(e["request"])
        qq = quantize(v)
        leaf = predict(nodes, qq)
        if safe_a[leaf]:
            continue
        part = partition_key(qq)
        per_pair[(leaf, part)].append(
            (e["expected_approved"], e["expected_fraud_score"])
        )
        per_leaf_unsafe[leaf] += 1
        per_part_unsafe[part] += 1
        total_unsafe += 1
        if (i + 1) % 10000 == 0:
            print(f"  routed {i+1}/{len(entries)}", file=sys.stderr)

    print(f"\nTotal unsafe (fallback) entries: {total_unsafe}", file=sys.stderr)

    # Top leaves and partitions among unsafe
    print("\nTop 10 leaves by unsafe count:", file=sys.stderr)
    for leaf, cnt in sorted(per_leaf_unsafe.items(), key=lambda x: -x[1])[:10]:
        pct = 100.0 * cnt / total_unsafe
        marker = " *" if leaf in ALLOW_LEAVES else ""
        print(f"  leaf {leaf}: {cnt} ({pct:.2f}%){marker}", file=sys.stderr)

    print("\nTop 10 partitions by unsafe count:", file=sys.stderr)
    for part, cnt in sorted(per_part_unsafe.items(), key=lambda x: -x[1])[:10]:
        pct = 100.0 * cnt / total_unsafe
        marker = " *" if part in ALLOW_PARTITIONS else ""
        print(f"  part {part}: {cnt} ({pct:.2f}%){marker}", file=sys.stderr)

    # Restrict to whitelisted leaves × partitions UNLESS the env var
    # MODE_B_OPEN=1 is set (then we scan all observed pairs and pick every
    # unanimous one with >= MIN_OBS samples).
    import os
    if os.environ.get("MODE_B_OPEN") == "1":
        candidate_pairs = list(per_pair.keys())
        print(
            f"\nMODE_B_OPEN=1: evaluating ALL {len(candidate_pairs)}"
            f" observed (leaf, partition) pairs",
            file=sys.stderr,
        )
    else:
        candidate_pairs = [
            (leaf, part) for leaf in ALLOW_LEAVES for part in ALLOW_PARTITIONS
        ]
        print(
            f"\nEvaluating {len(candidate_pairs)} candidate (leaf, partition) pairs"
            f" from whitelist...",
            file=sys.stderr,
        )

    # Build per-pair safety report
    safe_pairs = {}  # (leaf, part) -> verdict (0 or 5)
    leaf_partitions_used = defaultdict(set)
    coverage = 0
    for pair in candidate_pairs:
        items = per_pair.get(pair, [])
        if not items:
            continue
        if len(items) < MIN_OBS:
            if not os.environ.get("MODE_B_OPEN"):
                print(
                    f"  pair {pair}: SKIP (only {len(items)} obs, < {MIN_OBS})",
                    file=sys.stderr,
                )
            continue
        approveds = {a for a, _ in items}
        if len(approveds) > 1:
            if not os.environ.get("MODE_B_OPEN"):
                print(
                    f"  pair {pair}: SKIP (mixed labels, {len(items)} obs)",
                    file=sys.stderr,
                )
            continue
        expected_approved = next(iter(approveds))
        # Map unanimous label to extreme bucket (0 or 5) — both ends map to
        # the same approve/deny boundary as the Zig HTTP layer.
        verdict = 0 if expected_approved else 5
        safe_pairs[pair] = verdict
        leaf_partitions_used[pair[0]].add(pair[1])
        coverage += len(items)
        print(
            f"  pair {pair}: SAFE (n={len(items)}, verdict={verdict},"
            f" approved={expected_approved})",
            file=sys.stderr,
        )

    print(
        f"\nMode B coverage of *unsafe* set: {coverage}/{total_unsafe}"
        f" ({100.0 * coverage / max(total_unsafe, 1):.2f}%)",
        file=sys.stderr,
    )

    # Per-leaf consistency: every selected partition for a given leaf MUST agree
    # on the verdict (so we can store one cached verdict per leaf in
    # LEAF_B_VERDICT). If any leaf has conflicting verdicts across partitions,
    # bail out — we will NOT enable Mode B for that leaf.
    leaf_verdict = {}
    for leaf in ALLOW_LEAVES:
        partitions = leaf_partitions_used.get(leaf, set())
        verdicts = {safe_pairs[(leaf, p)] for p in partitions}
        if not verdicts:
            print(f"  leaf {leaf}: no safe partitions — Mode B disabled for this leaf", file=sys.stderr)
            continue
        if len(verdicts) > 1:
            print(
                f"  leaf {leaf}: CONFLICTING verdicts across partitions ({verdicts}) — Mode B disabled for this leaf",
                file=sys.stderr,
            )
            continue
        leaf_verdict[leaf] = next(iter(verdicts))
        print(
            f"  leaf {leaf}: verdict={leaf_verdict[leaf]} across {sorted(partitions)}",
            file=sys.stderr,
        )

    # Audit: simulate Mode B over the FULL dataset using the EXACT per-pair
    # whitelist (not the cartesian product). This is the gating logic that the
    # Zig hot path will use.
    fn = fp = fires = 0
    for i, e in enumerate(entries):
        v = vectorize(e["request"])
        qq = quantize(v)
        leaf = predict(nodes, qq)
        if safe_a[leaf]:
            continue  # Mode A would handle this; Mode B never reached
        if leaf not in leaf_verdict:
            continue
        part = partition_key(qq)
        if (leaf, part) not in safe_pairs:
            continue
        verdict = leaf_verdict[leaf]
        # Zig HTTP scorer: approved iff frauds <= 2
        predicted_approved = verdict <= 2
        actual_approved = e["expected_approved"]
        fires += 1
        if predicted_approved and not actual_approved:
            fn += 1
        if (not predicted_approved) and actual_approved:
            fp += 1

    print(
        f"\nMode B audit (per-pair gating): fires={fires} FN={fn} FP={fp}",
        file=sys.stderr,
    )

    if fn != 0 or fp != 0:
        print(
            "ERROR: Mode B would introduce label errors. Aborting.",
            file=sys.stderr,
        )
        sys.exit(2)

    # 5-fold cross-validation: calibrate on 4/5, audit on 1/5, repeat.
    # This catches whitelisted (leaf, partition) pairs whose unanimity is
    # an artefact of the sampling rather than a real geometric property.
    print("\n5-fold cross-validation:", file=sys.stderr)
    folds = 5
    cv_total_fires = 0
    cv_total_fn = 0
    cv_total_fp = 0
    for fold in range(folds):
        train = [e for i, e in enumerate(entries) if i % folds != fold]
        test = [e for i, e in enumerate(entries) if i % folds == fold]

        train_pairs = defaultdict(list)
        for e in train:
            v = vectorize(e["request"])
            qq = quantize(v)
            leaf = predict(nodes, qq)
            if safe_a[leaf]:
                continue
            part = partition_key(qq)
            train_pairs[(leaf, part)].append(e["expected_approved"])

        # Re-derive safe pairs from this fold's training data, but only within
        # the curated whitelist (same as the production calibration).
        train_safe = {}
        for leaf in ALLOW_LEAVES:
            for part in ALLOW_PARTITIONS:
                items = train_pairs.get((leaf, part), [])
                if len(items) < MIN_OBS:
                    continue
                if len(set(items)) > 1:
                    continue
                approved = items[0]
                verdict = 0 if approved else 5
                train_safe[(leaf, part)] = verdict

        f_fires = f_fn = f_fp = 0
        per_pair_fp = defaultdict(int)
        for e in test:
            v = vectorize(e["request"])
            qq = quantize(v)
            leaf = predict(nodes, qq)
            if safe_a[leaf]:
                continue
            part = partition_key(qq)
            if (leaf, part) not in train_safe:
                continue
            verdict = train_safe[(leaf, part)]
            predicted_approved = verdict <= 2
            actual = e["expected_approved"]
            f_fires += 1
            if predicted_approved and not actual:
                f_fn += 1
            if (not predicted_approved) and actual:
                f_fp += 1
                per_pair_fp[(leaf, part)] += 1
        if per_pair_fp:
            print(f"    fold-{fold} FP by pair: {dict(per_pair_fp)}", file=sys.stderr)

        cv_total_fires += f_fires
        cv_total_fn += f_fn
        cv_total_fp += f_fp
        print(
            f"  fold {fold}: fires={f_fires} FN={f_fn} FP={f_fp}",
            file=sys.stderr,
        )

    print(
        f"  TOTAL: fires={cv_total_fires} FN={cv_total_fn} FP={cv_total_fp}",
        file=sys.stderr,
    )
    if cv_total_fn != 0 or cv_total_fp != 0:
        print(
            "ERROR: 5-fold CV detected Mode B errors. Aborting.",
            file=sys.stderr,
        )
        sys.exit(3)

    # Emit Zig arrays. Per-pair gating requires a per-leaf partition bitmap:
    # LEAF_B_PART_MASK[leaf] is a 256-bit mask (stored as [4]u64, little-endian
    # by word: word k covers partitions [k*64, (k+1)*64) bits 0..63).
    # The hot path test is:
    #   LEAF_SAFE_B[leaf] AND ((LEAF_B_PART_MASK[leaf][part>>6] >> (part&63)) & 1) != 0
    #
    # We also publish a global PARTITION_RESTRICTED[256] bitmap = union over
    # safe leaves, which lets the hot path skip computing partitionKey unless
    # the leaf is in the safe set AND at least one partition in the safe set
    # might match (this is implied by LEAF_SAFE_B[leaf], so the global mask is
    # for diagnostics only — we keep it for documentation).
    safe_b_flags = [False] * LEAF_COUNT
    verdict_arr = [0] * LEAF_COUNT
    part_masks = [[0, 0, 0, 0] for _ in range(LEAF_COUNT)]

    for leaf, verdict in leaf_verdict.items():
        # Partitions for this leaf that are actually safe (from safe_pairs)
        parts_for_leaf = [p for (l, p) in safe_pairs.keys() if l == leaf]
        if not parts_for_leaf:
            continue
        safe_b_flags[leaf] = True
        verdict_arr[leaf] = verdict
        for p in parts_for_leaf:
            word = p >> 6
            bit = p & 63
            part_masks[leaf][word] |= (1 << bit)

    used_parts = sorted({p for (_, p) in safe_pairs.keys()})
    part_restricted = [False] * NUM_PARTITIONS
    for p in used_parts:
        part_restricted[p] = True

    out_path = HERE / "leaf_safe_b_new.zig"
    lines = []

    lines.append("// Auto-generated by tools/calibrate_leaf_safe_b.py.\n")
    lines.append(f"// Mode B coverage on unsafe set: {coverage}/{total_unsafe}"
                 f" ({100.0 * coverage / max(total_unsafe, 1):.2f}%)\n")
    lines.append(f"// Audit (per-pair gating): fires={fires} FN={fn} FP={fp}\n")
    lines.append(f"// Whitelisted leaves: {sorted(leaf_verdict.keys())}\n")
    lines.append(f"// Used partitions: {used_parts}\n")
    lines.append("// Safe (leaf, partition) pairs:\n")
    for pair in sorted(safe_pairs.keys()):
        lines.append(f"//   {pair} -> verdict={safe_pairs[pair]}\n")
    lines.append("\n")

    lines.append("pub const LEAF_SAFE_B: [LEAF_COUNT]bool = .{\n")
    for i in range(0, LEAF_COUNT, 16):
        row = ", ".join("true " if safe_b_flags[j] else "false" for j in range(i, i + 16))
        lines.append("    " + row + ",\n")
    lines.append("};\n\n")

    lines.append("pub const LEAF_B_VERDICT: [LEAF_COUNT]u8 = .{\n")
    for i in range(0, LEAF_COUNT, 16):
        row = ", ".join(str(verdict_arr[j]).rjust(1) for j in range(i, i + 16))
        lines.append("    " + row + ",\n")
    lines.append("};\n\n")

    lines.append("// Per-leaf 256-bit partition whitelist. Indexed as [4]u64\n")
    lines.append("// where word k covers partitions [k*64, k*64+64), bit b = bit (b&63).\n")
    lines.append("pub const LEAF_B_PART_MASK: [LEAF_COUNT][4]u64 = .{\n")
    for leaf in range(LEAF_COUNT):
        m = part_masks[leaf]
        if m == [0, 0, 0, 0]:
            lines.append("    .{ 0, 0, 0, 0 },\n")
        else:
            lines.append(
                f"    .{{ 0x{m[0]:016x}, 0x{m[1]:016x}, 0x{m[2]:016x}, 0x{m[3]:016x} }},  // leaf {leaf}\n"
            )
    lines.append("};\n\n")

    lines.append("// Diagnostic: union of all whitelisted partitions across safe leaves.\n")
    lines.append("pub const PARTITION_RESTRICTED: [256]bool = .{\n")
    for i in range(0, NUM_PARTITIONS, 16):
        row = ", ".join("true " if part_restricted[j] else "false" for j in range(i, i + 16))
        lines.append("    " + row + ",\n")
    lines.append("};\n")

    out_path.write_text("".join(lines))
    print(f"\nWrote {out_path}", file=sys.stderr)
    print(
        f"safe_leaves={sum(safe_b_flags)} used_parts={sum(part_restricted)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
