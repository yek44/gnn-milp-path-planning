"""Quick per-instance breakdown of a four_way CSV including ms/node estimates."""
import csv
import sys

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))

# Estimate reset overhead from a row with small node count where GNN finished quickly
# (reset = ~2-3s typically; use gnn_wall - gnn_tensor - gnn_forward - gnn_select - gnn_step as residual)
def gnn_reset_est(r):
    return float(r['gnn_wall_time']) - (
        float(r['gnn_tensor_wall']) + float(r['gnn_forward_wall'])
        + float(r['gnn_select_wall']) + float(r['gnn_env_step_wall'])
    )

reset_estimates = [gnn_reset_est(r) for r in rows]
reset_typical = sum(reset_estimates) / len(reset_estimates)
print(f"GNN reset overhead (typical across 20 inst): {reset_typical:.2f}s")
print()

print(f"{'inst':>4} {'vN':>5} {'vT':>7} {'fN':>5} {'fT':>7} {'dN':>5} {'dT':>6} {'gN':>5} {'gT':>6} "
      f"{'V_ms/n':>7} {'F_ms/n':>7} {'D_ms/n':>7} {'G_ms/n':>7} {'spdV':>5} {'spdD':>5}")
print("-" * 105)

wins_vsb = 0
wins_fsb = 0
wins_def = 0
for r in rows:
    vt = float(r['vsb_time']); gt = float(r['gnn_time'])
    ft = float(r['fsb_time']); dt = float(r['default_time'])
    vn = int(float(r['vsb_nodes'])); gn = int(float(r['gnn_nodes']))
    fn = int(float(r['fsb_nodes'])); dn = int(float(r['default_nodes']))
    gw = float(r['gnn_wall_time'])
    gnn_reset = gnn_reset_est(r)
    gnn_solve = max(0.001, gw - gnn_reset)
    gnn_ms = 1000 * gnn_solve / max(gn, 1)
    # For SCIP, assume same reset overhead applies
    vsb_solve = max(0.001, vt - gnn_reset)
    fsb_solve = max(0.001, ft - gnn_reset)
    def_solve = max(0.001, dt - gnn_reset)
    vsb_ms = 1000 * vsb_solve / max(vn, 1)
    fsb_ms = 1000 * fsb_solve / max(fn, 1)
    def_ms = 1000 * def_solve / max(dn, 1)
    spd_v = vt / gt if gt > 0 else float('nan')
    spd_d = dt / gt if gt > 0 else float('nan')
    if gt < vt: wins_vsb += 1
    if gt < ft: wins_fsb += 1
    if gt < dt: wins_def += 1
    print(f"{r['instance_id']:>4} {vn:>5} {vt:>7.2f} {fn:>5} {ft:>7.2f} {dn:>5} {dt:>6.2f} {gn:>5} {gt:>6.2f} "
          f"{vsb_ms:>7.0f} {fsb_ms:>7.0f} {def_ms:>7.0f} {gnn_ms:>7.0f} {spd_v:>5.2f} {spd_d:>5.2f}")

n = len(rows)
print()
print(f"GNN time win rate: vs VSB {wins_vsb}/{n}={100*wins_vsb/n:.0f}%  vs FSB {wins_fsb}/{n}={100*wins_fsb/n:.0f}%  vs DEFAULT {wins_def}/{n}={100*wins_def/n:.0f}%")

# Aggregate ms/node (weighted by nodes, excluding outliers)
total_v_nodes = sum(int(float(r['vsb_nodes'])) for r in rows)
total_v_time = sum(max(0.001, float(r['vsb_time']) - gnn_reset_est(r)) for r in rows)
total_g_nodes = sum(int(float(r['gnn_nodes'])) for r in rows)
total_g_time = sum(max(0.001, float(r['gnn_wall_time']) - gnn_reset_est(r)) for r in rows)
total_d_nodes = sum(int(float(r['default_nodes'])) for r in rows)
total_d_time = sum(max(0.001, float(r['default_time']) - gnn_reset_est(r)) for r in rows)
print()
print(f"Aggregate (sum-of-time / sum-of-nodes, reset-subtracted):")
print(f"  VSB:     {1000*total_v_time/max(total_v_nodes,1):.1f} ms/node  (total {total_v_nodes} nodes, {total_v_time:.1f}s solve)")
print(f"  DEFAULT: {1000*total_d_time/max(total_d_nodes,1):.1f} ms/node  (total {total_d_nodes} nodes, {total_d_time:.1f}s solve)")
print(f"  GNN:     {1000*total_g_time/max(total_g_nodes,1):.1f} ms/node  (total {total_g_nodes} nodes, {total_g_time:.1f}s solve)")
