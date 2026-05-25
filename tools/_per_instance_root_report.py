"""Per-instance root-solved breakdown across VSB / FSB / DEFAULT / GNN baselines.

Replaces average-based summaries with explicit per-instance "did SCIP need to
branch?" reporting, plus head-to-head time/node comparison versus DEFAULT.

Usage:
    python tools/_per_instance_root_report.py results/<tag>_four_way.csv
"""
import csv
import sys

paths = sys.argv[1:]
if not paths:
    print("usage: _per_instance_root_report.py <csv> [csv2 ...]")
    sys.exit(1)

ROOT_THRESHOLD = 1  # SCIP node count <= this counts as "root-solved"

for path in paths:
    rows = list(csv.DictReader(open(path)))
    n = len(rows)
    print(f"\n=== Per-instance root analysis ({path}) ===")
    header = (
        f"{'inst':>4} | "
        f"{'V_n':>5} {'F_n':>5} {'D_n':>5} {'G_n':>5} | "
        f"{'D_root':>6} | "
        f"{'V_t':>7} {'F_t':>7} {'D_t':>7} {'G_t':>7} | "
        f"{'GvsD_t':>7} {'GvsD_n':>7}"
    )
    print(header)
    print("-" * len(header))

    root_count = 0
    gnn_wins_d_time = 0
    gnn_wins_d_nodes = 0
    gnn_wins_v_time = 0
    non_trivial = []
    sum_d_t = sum_g_t = sum_v_t = sum_f_t = 0.0
    sum_d_n = sum_g_n = sum_v_n = sum_f_n = 0
    for r in rows:
        vn = int(float(r["vsb_nodes"])); fn = int(float(r["fsb_nodes"]))
        dn = int(float(r["default_nodes"])); gn = int(float(r["gnn_nodes"]))
        vt = float(r["vsb_time"]); ft = float(r["fsb_time"])
        dt = float(r["default_time"]); gt = float(r["gnn_time"])
        sum_v_t += vt; sum_f_t += ft; sum_d_t += dt; sum_g_t += gt
        sum_v_n += vn; sum_f_n += fn; sum_d_n += dn; sum_g_n += gn
        d_root = dn <= ROOT_THRESHOLD
        if d_root: root_count += 1
        if gt < dt: gnn_wins_d_time += 1
        if gn < dn: gnn_wins_d_nodes += 1
        if gt < vt: gnn_wins_v_time += 1
        if dn > 5: non_trivial.append((int(r["instance_id"]), vn, fn, dn, gn, vt, ft, dt, gt))
        gvd_t_pct = 100*(1-gt/dt) if dt > 0 else 0
        gvd_n_pct = 100*(1-gn/dn) if dn > 0 else 0
        flag = "ROOT" if d_root else "branch"
        print(
            f"{r['instance_id']:>4} | "
            f"{vn:>5} {fn:>5} {dn:>5} {gn:>5} | "
            f"{flag:>6} | "
            f"{vt:>7.2f} {ft:>7.2f} {dt:>7.2f} {gt:>7.2f} | "
            f"{gvd_t_pct:>+7.1f} {gvd_n_pct:>+7.1f}"
        )

    print(f"\n--- Summary (n={n}) ---")
    print(f"  SCIP DEFAULT root-solved (nodes<={ROOT_THRESHOLD}): {root_count}/{n} = {100*root_count/n:.0f}%")
    print(f"  GNN beats DEFAULT on TIME:  {gnn_wins_d_time}/{n} = {100*gnn_wins_d_time/n:.0f}%")
    print(f"  GNN beats DEFAULT on NODES: {gnn_wins_d_nodes}/{n} = {100*gnn_wins_d_nodes/n:.0f}%")
    print(f"  GNN beats VSB on TIME:      {gnn_wins_v_time}/{n} = {100*gnn_wins_v_time/n:.0f}%")
    print(f"\n  Aggregate TIME (sum-of-time): VSB={sum_v_t:.1f}s  FSB={sum_f_t:.1f}s  DEFAULT={sum_d_t:.1f}s  GNN={sum_g_t:.1f}s")
    print(f"  Aggregate NODES (sum):        VSB={sum_v_n}  FSB={sum_f_n}  DEFAULT={sum_d_n}  GNN={sum_g_n}")
    print(f"\n  Non-trivial instances (default_nodes>5): {len(non_trivial)}/{n}")
    for inst, vn, fn, dn, gn, vt, ft, dt, gt in non_trivial:
        print(f"    inst {inst:>3}:  V={vn:>5}({vt:.1f}s)  F={fn:>5}({ft:.1f}s)  D={dn:>5}({dt:.1f}s)  G={gn:>5}({gt:.1f}s)")
