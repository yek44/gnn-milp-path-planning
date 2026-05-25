"""
01_generate_rh_instances.py
Generate MILP instances from Receding Horizon path planning scenarios.

Each scenario defines vehicles, obstacles, start/goal positions. The receding
horizon loop is executed and at every RH step the resulting MILP is exported as
a CPLEX LP file (.lp) via PySCIPOpt's ``model.writeProblem()``. After export
the model is solved so the simulation can advance to the next state.

The generator supports single- and multi-vehicle scenarios and randomizes
starts, goals, and obstacle layouts while keeping each scenario feasible inside
the configured workspace.

Directory layout follows learn2branch conventions:
    data/instances/rh_milp/<split>/instance_<i>.lp

Usage:
    python 01_generate_rh_instances.py --map_mode random --seed 0
"""

import os
import sys
import json
import argparse
import glob
import re
import time as _time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from true_receding_horizon import (
    double_integrator_matrices,
    filter_active_obstacles,
    assemble_milp_data,
    build_milp_model,
    build_base_scenario,
    RH_POS_TOLERANCE,
    RH_VEL_TOLERANCE,
    RH_MAX_ITERATIONS,
)

SEMANTIC_FEATURE_NAMES = [
    'state_px',
    'state_py',
    'state_vx',
    'state_vy',
    'input_ux',
    'input_uy',
    'fuel_slack',
    'terminal_slack',
    'obstacle_binary',
    'obstacle_selector',
    'separation_binary',
    'separation_selector',
    'vehicle_id',
    'time_index',
    'obstacle_id',
    'side_id',
    'pair_id',
    'axis_binary',
    'sign_binary',
    'side_left',
    'side_right',
    'side_bottom',
    'side_top',
    'is_binary',
    'is_obstacle',
    'is_separation',
    'solver_index',
]


# ---------------------------------------------------------------------------
# Scenario generation helpers
# ---------------------------------------------------------------------------

def _rectangles_overlap(r1, r2, gap=0.0):
    """Check if two rectangles [xmin, xmax, ymin, ymax] overlap (with gap)."""
    return not (r1[1] + gap < r2[0] or r2[1] + gap < r1[0] or
                r1[3] + gap < r2[2] or r2[3] + gap < r1[2])


def _point_inside_rect(px, py, rect, margin=0.0):
    """True if (px, py) is inside rect [xmin, xmax, ymin, ymax] + margin."""
    return (rect[0] - margin <= px <= rect[1] + margin and
            rect[2] - margin <= py <= rect[3] + margin)


def _pairwise_min_distance(points):
    """Return minimum pairwise Euclidean distance for an (N, 2) array."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return float('inf')
    min_dist = float('inf')
    for i in range(pts.shape[0]):
        for j in range(i + 1, pts.shape[0]):
            dist = np.linalg.norm(pts[i] - pts[j])
            min_dist = min(min_dist, float(dist))
    return min_dist


def _sample_start_goal_uniform(rng, n_vehicles, obstacles, workspace,
                               min_pair_dist=0.45, obstacle_margin=0.35,
                               border_margin=0.5, max_attempts=1000):
    """Sample start/goal states uniformly inside the workspace, avoiding obstacles."""
    obs_arr = np.asarray(obstacles, dtype=float).reshape((-1, 4))
    x_lo, x_hi = workspace['x']
    y_lo, y_hi = workspace['y']

    x_lo = float(x_lo) + border_margin
    x_hi = float(x_hi) - border_margin
    y_lo = float(y_lo) + border_margin
    y_hi = float(y_hi) - border_margin

    for _ in range(max_attempts):
        starts_xy = np.zeros((n_vehicles, 2), dtype=float)
        goals_xy = np.zeros((n_vehicles, 2), dtype=float)
        blocked = False

        for v in range(n_vehicles):
            sx = rng.uniform(x_lo, x_hi)
            sy = rng.uniform(y_lo, y_hi)
            gx = rng.uniform(x_lo, x_hi)
            gy = rng.uniform(y_lo, y_hi)

            for rect in obs_arr:
                if _point_inside_rect(sx, sy, rect, margin=obstacle_margin) or \
                   _point_inside_rect(gx, gy, rect, margin=obstacle_margin):
                    blocked = True
                    break
            if blocked:
                break

            starts_xy[v, 0] = sx
            starts_xy[v, 1] = sy
            goals_xy[v, 0] = gx
            goals_xy[v, 1] = gy

        if blocked:
            continue

        all_pts = np.vstack([starts_xy, goals_xy])
        if _pairwise_min_distance(all_pts) < min_pair_dist:
            continue

        starts = np.zeros((n_vehicles, 4), dtype=float)
        goals = np.zeros((n_vehicles, 4), dtype=float)
        starts[:, 0:2] = starts_xy
        goals[:, 0:2] = goals_xy
        return starts, goals

    # Fallback: ignore pairwise distance but still avoid obstacles.
    starts = np.zeros((n_vehicles, 4), dtype=float)
    goals = np.zeros((n_vehicles, 4), dtype=float)
    v = 0
    while v < n_vehicles:
        sx = rng.uniform(x_lo, x_hi)
        sy = rng.uniform(y_lo, y_hi)
        gx = rng.uniform(x_lo, x_hi)
        gy = rng.uniform(y_lo, y_hi)
        blocked = False
        for rect in obs_arr:
            if _point_inside_rect(sx, sy, rect, margin=obstacle_margin) or \
               _point_inside_rect(gx, gy, rect, margin=obstacle_margin):
                blocked = True
                break
        if blocked:
            continue
        starts[v, 0:2] = (sx, sy)
        goals[v, 0:2] = (gx, gy)
        v += 1

    return starts, goals


def _sample_city_3agent_crossing(rng):
    """Structured 3-agent city scenario with a shared central junction."""
    # The canonical city grid has several horizontal streets and vertical
    # streets between aligned rectangular blocks. These routes intentionally
    # intersect around the central blocks.
    starts_xy = np.array([
        [-2.35, 4.45],   # west -> east through an upper-middle street
        [14.65, 0.55],   # east -> west through a lower-middle street
        [7.80, -7.05],   # south -> north through a central vertical street
    ], dtype=float)
    goals_xy = np.array([
        [14.65, 4.45],
        [-2.35, 0.55],
        [7.80, 10.15],
    ], dtype=float)

    # Small scenario-level jitter gives train/valid/test variation while keeping
    # all agents inside the same streets and preserving the bottleneck.
    jitter = rng.uniform(-0.08, 0.08, size=starts_xy.shape)
    starts_xy = starts_xy + jitter
    goals_xy = goals_xy - jitter

    starts = np.zeros((3, 4), dtype=float)
    goals = np.zeros((3, 4), dtype=float)
    starts[:, 0:2] = starts_xy
    goals[:, 0:2] = goals_xy
    return starts, goals


def generate_random_obstacles(rng, n_obstacles, workspace, forbidden_points,
                              min_size=0.8, max_size=2.5, point_margin=0.5,
                              max_attempts=200):
    """Generate *n_obstacles* non-overlapping axis-aligned rectangles.

    Parameters
    ----------
    rng : numpy.random.RandomState
    n_obstacles : int
    workspace : dict  ``{'x': (lo, hi), 'y': (lo, hi)}``
    forbidden_points : list of (x, y)
        Start / goal positions that obstacles must not cover.
    min_size, max_size : float
        Range for each rectangle side length.
    point_margin : float
        Extra clearance around forbidden points.
    max_attempts : int
        Placement retries per obstacle.

    Returns
    -------
    numpy.ndarray  shape (M, 4) with columns [xmin, xmax, ymin, ymax].
        M <= n_obstacles (some may be dropped if placement fails).
    """
    x_lo, x_hi = workspace['x']
    y_lo, y_hi = workspace['y']
    placed = []

    for _ in range(n_obstacles):
        for _attempt in range(max_attempts):
            w = rng.uniform(min_size, max_size)
            h = rng.uniform(min_size, max_size)
            cx = rng.uniform(x_lo + w / 2, x_hi - w / 2)
            cy = rng.uniform(y_lo + h / 2, y_hi - h / 2)
            rect = np.array([cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2])

            # Must not cover any forbidden point
            blocked = False
            for (fpx, fpy) in forbidden_points:
                if _point_inside_rect(fpx, fpy, rect, margin=point_margin):
                    blocked = True
                    break
            if blocked:
                continue

            # Must not overlap previously placed obstacles (small gap)
            overlap = False
            for prev in placed:
                if _rectangles_overlap(rect, prev, gap=0.15):
                    overlap = True
                    break
            if overlap:
                continue

            placed.append(rect)
            break  # success

    if len(placed) == 0:
        return np.zeros((0, 4))
    return np.array(placed)


def generate_random_scenario(
        rng, n_vehicles, workspace_size=15.0,
        horizon_time=3.0, dt=0.1, map_mode='random'):
    """Create a scenario dict compatible with ``assemble_milp_data``.

    This implementation uses the fixed canonical RH obstacle map and bounds
    and only randomises the start/goal states, while ensuring they:
    - lie inside the workspace bounds, and
    - do not intersect any obstacle (with a safety margin).
    """
    if map_mode != 'random':
        raise ValueError(f'Unsupported map_mode: {map_mode}')

    horizon_steps = int(round(horizon_time / dt))

    # ------------------------------------------------------------------
    # Fixed map from the canonical base scenario
    # ------------------------------------------------------------------
    template = build_base_scenario()
    template_obs = np.asarray(template.get('obstacles', []), dtype=float).reshape((-1, 4))

    template_pos_bounds = np.asarray(template.get('posBounds', []), dtype=float)
    if template_pos_bounds.shape == (2, 2):
        x_lo, x_hi = map(float, template_pos_bounds[0])
        y_lo, y_hi = map(float, template_pos_bounds[1])
    elif template_obs.size == 0:
        # Fallback to a square workspace if, for some reason, no obstacles exist.
        x_lo, x_hi = 0.0, float(workspace_size)
        y_lo, y_hi = 0.0, float(workspace_size)
    else:
        x_lo = float(np.min(template_obs[:, 0])) - 1.0
        x_hi = float(np.max(template_obs[:, 1])) + 1.0
        y_lo = float(np.min(template_obs[:, 2])) - 1.0
        y_hi = float(np.max(template_obs[:, 3])) + 1.0
    workspace = {'x': (x_lo, x_hi), 'y': (y_lo, y_hi)}
    pos_bounds = np.array([[x_lo, x_hi], [y_lo, y_hi]], dtype=float)

    obstacles = template_obs.copy()

    # ------------------------------------------------------------------
    # Random start / goal inside workspace, avoiding obstacles
    # ------------------------------------------------------------------
    if int(n_vehicles) == 3:
        starts, goals = _sample_city_3agent_crossing(rng)
    else:
        starts, goals = _sample_start_goal_uniform(
            rng=rng,
            n_vehicles=n_vehicles,
            obstacles=obstacles,
            workspace=workspace,
            min_pair_dist=0.45,
            obstacle_margin=0.20,
        )

    # ------------------------------------------------------------------
    # Assemble scenario dict
    # ------------------------------------------------------------------
    enforce_sep = n_vehicles > 1
    scenario = {
        'dt': float(dt),
        'horizonSteps': horizon_steps,
        'start': starts,
        'goal': goals,
        'obstacles': obstacles,
        'posBounds': pos_bounds,
        'velBounds': np.asarray(template.get('velBounds'), dtype=float),
        'inputBounds': np.asarray(template.get('inputBounds'), dtype=float),
        'obstacleBuffer': float(template.get('obstacleBuffer', 0.05)),
        'obstaclePruneMargin': float(template.get('obstaclePruneMargin', 2.0)),
        'obstacleBigM': float(template.get('obstacleBigM', 800.0)),
        'safeSeparation': float(template.get('safeSeparation', 0.08)) if enforce_sep else 0.0,
        'enforceSeparation': bool(template.get('enforceSeparation', True)) and enforce_sep,
        'pairPruneMargin': float(template.get('pairPruneMargin', 1.0)),
        'separationBigM': float(template.get('separationBigM', 25.0)),
        'useDynamicObstacleBigM': bool(template.get('useDynamicObstacleBigM', False)),
        'useDynamicSeparationBigM': bool(template.get('useDynamicSeparationBigM', False)),
        'bigMScale': float(template.get('bigMScale', 1.0)),
        'binaryTieBreak': float(template.get('binaryTieBreak', 0.0)),
        'nVehicles': int(n_vehicles),
        'terminalWeights': dict(template.get('terminalWeights', {'position': 10.0, 'velocity': 5.0})),
    }
    return scenario


# ---------------------------------------------------------------------------
# Split worker – used both sequentially and in parallel
# ---------------------------------------------------------------------------


def _generate_split(
    split_name,
    n_scenarios,
    target_instances,
    nv_min,
    nv_max,
    use_dynamic_obstacle_bigm,
    use_dynamic_separation_bigm,
    bigm_scale,
    binary_tiebreak,
    clean_split_dirs,
    base_dir,
    rng_seed,
    args,
):
    rng = np.random.RandomState(int(rng_seed))

    split_dir = os.path.join(base_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    existing_before = 0
    global_idx = 1
    if clean_split_dirs:
        old_instance_files = (
            glob.glob(os.path.join(split_dir, 'instance_*.lp')) +
            glob.glob(os.path.join(split_dir, 'instance_*.json'))
        )
        for path in old_instance_files:
            os.remove(path)
    else:
        existing_lp_files = glob.glob(os.path.join(split_dir, 'instance_*.lp'))
        existing_ids = []
        for path in existing_lp_files:
            match = re.search(r'instance_(\d+)\.lp$', os.path.basename(path))
            if match is not None:
                existing_ids.append(int(match.group(1)))
        existing_before = len(existing_ids)
        if existing_ids:
            global_idx = max(existing_ids) + 1

    total_instances = 0
    new_start_idx = global_idx
    scenario_log = []      # per-scenario metadata
    split_instance_log_lines = [
        f"Generation log for split={split_name}",
        f"timestamp={datetime.now().isoformat(timespec='seconds')}",
        f"seed={args.seed} map_mode={args.map_mode} dt={args.dt} horizon_time={args.horizon_time}",
        (
            f"formulation_profile={args.formulation_profile} "
            f"useDynamicObstacleBigM={int(use_dynamic_obstacle_bigm)} "
            f"useDynamicSeparationBigM={int(use_dynamic_separation_bigm)} "
            f"bigMScale={bigm_scale} binaryTieBreak={binary_tiebreak} "
            f"obstacleEncoding={args.obstacle_encoding} "
            f"reachabilityPruning={int(not args.no_reachability_obstacle_pruning)} "
            f"corridorObstacleFilter={int(not args.no_corridor_obstacle_filter)} "
            f"obstaclePruneMargin={args.obstacle_prune_margin}"
        ),
        "",
    ]
    t0 = _time.time()
    print(f'\n=== {split_name}: {n_scenarios} scenarios in {split_dir} ===')
    if not clean_split_dirs:
        print(f'  append mode: existing instances={existing_before}, next_instance_idx={global_idx}')

    for s in range(n_scenarios):
        if target_instances > 0 and total_instances >= target_instances:
            break
        scenario_max_rh_steps = int(args.max_rh_steps)
        if target_instances > 0:
            remaining_instances = target_instances - total_instances
            scenario_max_rh_steps = min(scenario_max_rh_steps, max(0, remaining_instances))

        # Sample n_vehicles within the configured range. Obstacles come from the
        # fixed canonical map.
        nv = int(rng.randint(nv_min, nv_max + 1))

        scenario = generate_random_scenario(
            rng, n_vehicles=nv,
            workspace_size=args.workspace_size,
            horizon_time=args.horizon_time, dt=args.dt,
            map_mode=args.map_mode,
        )
        scenario['useDynamicObstacleBigM'] = use_dynamic_obstacle_bigm
        scenario['useDynamicSeparationBigM'] = use_dynamic_separation_bigm
        scenario['bigMScale'] = bigm_scale
        scenario['binaryTieBreak'] = binary_tiebreak
        scenario['binaryTieBreakLexScale'] = args.binary_tiebreak_lex_scale
        scenario['obstacleEncoding'] = args.obstacle_encoding
        scenario['separationEncoding'] = (
            'side_select' if args.obstacle_encoding == 'side_pruned' else args.obstacle_encoding
        )
        scenario['useReachabilityObstaclePruning'] = not args.no_reachability_obstacle_pruning
        if args.obstacle_prune_margin is not None:
            scenario['obstaclePruneMargin'] = float(args.obstacle_prune_margin)
        scenario['disableCorridorObstacleFilter'] = bool(args.no_corridor_obstacle_filter)
        scenario['obstacleSidePruneKeep'] = args.obstacle_side_prune_keep
        scenario['obstacleSideEpsScale'] = args.obstacle_side_eps_scale
        print(f'  scenario {s + 1}/{n_scenarios} '
              f'(veh={nv}, obs={len(scenario["obstacles"])}) ... ',
              end='', flush=True)

        n_written = generate_rh_instances(
            scenario, Thor=args.horizon_time, output_dir=split_dir,
            start_idx=global_idx, time_limit=args.time_limit, verbose=False,
            instance_log_lines=split_instance_log_lines,
            scenario_index=s + 1,
            split_name=split_name,
            trajectory_json_path=os.path.join(split_dir, f'scenario_{s + 1}_trajectory.json'),
            max_rh_steps=scenario_max_rh_steps,
        )

        print(f'{n_written} instances')

        scenario_log.append({
            'scenario_index': s + 1,
            'n_vehicles': nv,
            'n_obstacles': int(len(scenario['obstacles'])),
            'instance_start': global_idx,
            'instance_end': global_idx + n_written - 1,
            'n_instances': n_written,
        })

        global_idx += n_written
        total_instances += n_written

    elapsed = _time.time() - t0
    print(f'  >> {total_instances} total instances, {elapsed:.1f} s')
    if total_instances > 0:
        new_end_idx = new_start_idx + total_instances - 1
    else:
        new_end_idx = None

    # Save metadata (includes per-scenario breakdown)
    meta = {
        'split': split_name,
        'n_scenarios': n_scenarios,
        'existing_instances_before': existing_before,
        'n_instances': total_instances,
        'n_instances_total_after': existing_before + total_instances,
        'target_instances': target_instances,
        'n_vehicles_range': [nv_min, nv_max],
        'fixed_n_vehicles': args.fixed_n_vehicles,
        'horizon_time': args.horizon_time,
        'dt': args.dt,
        'map_mode': args.map_mode,
        'formulation_profile': args.formulation_profile,
        'useDynamicObstacleBigM': use_dynamic_obstacle_bigm,
        'useDynamicSeparationBigM': use_dynamic_separation_bigm,
        'bigMScale': bigm_scale,
        'binaryTieBreak': binary_tiebreak,
        'binaryTieBreakLexScale': args.binary_tiebreak_lex_scale,
        'obstacleEncoding': args.obstacle_encoding,
        'obstacleSidePruneKeep': args.obstacle_side_prune_keep,
        'obstacleSideEpsScale': args.obstacle_side_eps_scale,
        'reachabilityObstaclePruning': not args.no_reachability_obstacle_pruning,
        'obstaclePruneMargin': args.obstacle_prune_margin,
        'corridorObstacleFilter': not args.no_corridor_obstacle_filter,
        'workspace_size': args.workspace_size,
        'clean_split_dirs': clean_split_dirs,
        'seed': args.seed,
        'scenarios': scenario_log,
    }
    with open(os.path.join(split_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    latest_range = {
        'split': split_name,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'append_mode': (not clean_split_dirs),
        'existing_instances_before': existing_before,
        'new_instances': total_instances,
        'new_instance_start': new_start_idx if total_instances > 0 else None,
        'new_instance_end': new_end_idx,
        'n_instances_total_after': existing_before + total_instances,
    }
    latest_range_path = os.path.join(split_dir, 'latest_append_range.json')
    with open(latest_range_path, 'w', encoding='utf-8') as f:
        json.dump(latest_range, f, indent=2)
    txt_log_path = os.path.join(split_dir, 'generation_log.txt')
    with open(txt_log_path, 'a', encoding='utf-8') as f:
        if os.path.exists(txt_log_path) and os.path.getsize(txt_log_path) > 0:
            f.write("\n")
        f.write("\n".join(split_instance_log_lines).rstrip() + "\n")
    print(f'  log saved: {txt_log_path}')
    print(f'  latest range saved: {latest_range_path}')


# ---------------------------------------------------------------------------
# Instance export via receding-horizon loop
# ---------------------------------------------------------------------------

def _normalise_index(value, max_value):
    """Map a non-negative index to [0, 1], using 0 for singleton domains."""
    if value is None:
        return 0.0
    max_value = int(max_value)
    if max_value <= 0:
        return 0.0
    return float(value) / float(max_value)


def _build_semantic_variable_features(milp_data):
    """Build variable metadata aligned with PySCIPOpt variable names x0..xN."""
    idx = milp_data['idx']
    nvar = int(idx['nvar'])
    n_vehicles = int(idx.get('nVehicles', milp_data.get('nVehicles', 1)))
    horizon = int(idx.get('T', 0))
    n_obstacles = int(idx.get('nObstacles', 0))
    vehicle_pairs = list(milp_data.get('vehicle_pairs', []))
    n_pairs = len(vehicle_pairs)
    feature_pos = {name: i for i, name in enumerate(SEMANTIC_FEATURE_NAMES)}
    features = np.zeros((nvar, len(SEMANTIC_FEATURE_NAMES)), dtype=np.float32)
    if nvar > 1:
        features[:, feature_pos['solver_index']] = np.linspace(0.0, 1.0, nvar, dtype=np.float32)
    variable_semantics = [{'kind': 'unknown'} for _ in range(nvar)]
    side_names = ['left', 'right', 'bottom', 'top']
    bit_names = ['axis', 'sign']

    def set_feature(var_idx, name, value=1.0):
        features[int(var_idx), feature_pos[name]] = float(value)

    def set_common(var_idx, kind, veh=None, time_idx=None, is_binary=False):
        var_idx = int(var_idx)
        set_feature(var_idx, kind)
        set_feature(var_idx, 'vehicle_id', _normalise_index(veh, n_vehicles - 1))
        set_feature(var_idx, 'time_index', _normalise_index(time_idx, horizon))
        if is_binary:
            set_feature(var_idx, 'is_binary')
        variable_semantics[var_idx] = {
            'kind': kind,
            'vehicle': None if veh is None else int(veh),
            'time': None if time_idx is None else int(time_idx),
        }

    for veh in range(n_vehicles):
        for k, var_idx in enumerate(idx['px'][veh]):
            set_common(var_idx, 'state_px', veh=veh, time_idx=k)
        for k, var_idx in enumerate(idx['py'][veh]):
            set_common(var_idx, 'state_py', veh=veh, time_idx=k)
        for k, var_idx in enumerate(idx['vx'][veh]):
            set_common(var_idx, 'state_vx', veh=veh, time_idx=k)
        for k, var_idx in enumerate(idx['vy'][veh]):
            set_common(var_idx, 'state_vy', veh=veh, time_idx=k)
        for k, var_idx in enumerate(idx['ux'][veh]):
            set_common(var_idx, 'input_ux', veh=veh, time_idx=k)
        for k, var_idx in enumerate(idx['uy'][veh]):
            set_common(var_idx, 'input_uy', veh=veh, time_idx=k)
        for k, var_idx in enumerate(idx['sx'][veh]):
            set_common(var_idx, 'fuel_slack', veh=veh, time_idx=k)
        for k, var_idx in enumerate(idx['sy'][veh]):
            set_common(var_idx, 'fuel_slack', veh=veh, time_idx=k)

        terminal_slacks = [
            idx['tpx'][veh],
            idx['tpy'][veh],
            idx['tvx'][veh],
            idx['tvy'][veh],
        ]
        for var_idx in terminal_slacks:
            set_common(var_idx, 'terminal_slack', veh=veh, time_idx=horizon)

    a_cube = idx.get('aCube')
    if a_cube is not None and np.asarray(a_cube).size > 0:
        a_names = bit_names if a_cube.shape[1] == 2 else side_names
        for veh in range(a_cube.shape[0]):
            for side in range(a_cube.shape[1]):
                side_name = a_names[side] if side < len(a_names) else str(side)
                for obs in range(a_cube.shape[2]):
                    for k in range(a_cube.shape[3]):
                        var_idx = int(a_cube[veh, side, obs, k])
                        if var_idx < 0:
                            continue
                        set_common(var_idx, 'obstacle_binary', veh=veh, time_idx=k, is_binary=True)
                        set_feature(var_idx, 'obstacle_id', _normalise_index(obs, n_obstacles - 1))
                        set_feature(var_idx, 'side_id', _normalise_index(side, len(a_names) - 1))
                        if side_name == 'axis':
                            set_feature(var_idx, 'axis_binary')
                        elif side_name == 'sign':
                            set_feature(var_idx, 'sign_binary')
                        elif side < len(side_names):
                            set_feature(var_idx, f'side_{side_name}')
                        set_feature(var_idx, 'is_obstacle')
                        variable_semantics[var_idx].update({
                            'obstacle': int(obs),
                            'side': side_name,
                        })

    a_selector_cube = idx.get('aSelectorCube')
    if a_selector_cube is not None and np.asarray(a_selector_cube).size > 0:
        for veh in range(a_selector_cube.shape[0]):
            for side in range(a_selector_cube.shape[1]):
                side_name = side_names[side] if side < len(side_names) else str(side)
                for obs in range(a_selector_cube.shape[2]):
                    for k in range(a_selector_cube.shape[3]):
                        var_idx = int(a_selector_cube[veh, side, obs, k])
                        if var_idx < 0:
                            continue
                        set_common(var_idx, 'obstacle_selector', veh=veh, time_idx=k, is_binary=False)
                        set_feature(var_idx, 'obstacle_id', _normalise_index(obs, n_obstacles - 1))
                        set_feature(var_idx, 'side_id', _normalise_index(side, len(side_names) - 1))
                        if side < len(side_names):
                            set_feature(var_idx, f'side_{side_name}')
                        set_feature(var_idx, 'is_obstacle')
                        variable_semantics[var_idx].update({
                            'obstacle': int(obs),
                            'side': side_name,
                            'selector': True,
                        })

    pair_cube = idx.get('pairCube')
    if pair_cube is not None and np.asarray(pair_cube).size > 0:
        p_names = bit_names if pair_cube.shape[1] == 2 else side_names
        for pair in range(pair_cube.shape[0]):
            for side in range(pair_cube.shape[1]):
                side_name = p_names[side] if side < len(p_names) else str(side)
                for k in range(pair_cube.shape[2]):
                    var_idx = int(pair_cube[pair, side, k])
                    if var_idx < 0:
                        continue
                    set_common(var_idx, 'separation_binary', time_idx=k, is_binary=True)
                    set_feature(var_idx, 'pair_id', _normalise_index(pair, n_pairs - 1))
                    set_feature(var_idx, 'side_id', _normalise_index(side, len(p_names) - 1))
                    if side_name == 'axis':
                        set_feature(var_idx, 'axis_binary')
                    elif side_name == 'sign':
                        set_feature(var_idx, 'sign_binary')
                    elif side < len(side_names):
                        set_feature(var_idx, f'side_{side_name}')
                    set_feature(var_idx, 'is_separation')
                    variable_semantics[var_idx].update({
                        'pair': int(pair),
                        'vehicle_pair': (
                            list(map(int, vehicle_pairs[pair]))
                            if pair < len(vehicle_pairs) else None
                        ),
                        'side': side_name,
                    })

    pair_selector_cube = idx.get('pairSelectorCube')
    if pair_selector_cube is not None and np.asarray(pair_selector_cube).size > 0:
        for pair in range(pair_selector_cube.shape[0]):
            for side in range(pair_selector_cube.shape[1]):
                side_name = side_names[side] if side < len(side_names) else str(side)
                for k in range(pair_selector_cube.shape[2]):
                    var_idx = int(pair_selector_cube[pair, side, k])
                    if var_idx < 0:
                        continue
                    set_common(var_idx, 'separation_selector', time_idx=k, is_binary=False)
                    set_feature(var_idx, 'pair_id', _normalise_index(pair, n_pairs - 1))
                    set_feature(var_idx, 'side_id', _normalise_index(side, len(side_names) - 1))
                    if side < len(side_names):
                        set_feature(var_idx, f'side_{side_name}')
                    set_feature(var_idx, 'is_separation')
                    variable_semantics[var_idx].update({
                        'pair': int(pair),
                        'vehicle_pair': (
                            list(map(int, vehicle_pairs[pair]))
                            if pair < len(vehicle_pairs) else None
                        ),
                        'side': side_name,
                        'selector': True,
                    })

    return features, variable_semantics


def _write_instance_sidecar_json(json_path, lp_path, milp_data, scenario_index, split_name, iteration):
    """Write semantic sidecar metadata without changing the LP formulation."""
    semantic_features, variable_semantics = _build_semantic_variable_features(milp_data)
    idx = milp_data['idx']
    payload = {
        'instance_path': str(lp_path),
        'split': split_name,
        'scenario_index': None if scenario_index is None else int(scenario_index),
        'rh_iteration': int(iteration),
        'model_metadata': {
            'variable_names': [f'x{i}' for i in range(int(idx['nvar']))],
            'semantic_feature_names': list(SEMANTIC_FEATURE_NAMES),
            'semantic_variable_features': semantic_features.tolist(),
            'variable_semantics': variable_semantics,
            'n_variables': int(idx['nvar']),
            'n_vehicles': int(idx.get('nVehicles', milp_data.get('nVehicles', 0))),
            'horizon': int(idx.get('T', 0)),
            'n_obstacles': int(idx.get('nObstacles', 0)),
            'n_pairs': int(len(milp_data.get('vehicle_pairs', []))),
        },
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def _extract_solution(model, vars_dict, data):
    """Extract state/input arrays from a solved PySCIPOpt model.

    Returns a dict identical to the one produced by ``solve_milp`` but without
    calling model.freeProb() (caller is responsible for that).
    """
    idx = data['idx']
    nvar = idx['nvar']
    n_vehicles = idx['nVehicles']
    horizon = idx['T']

    solution = {}
    status = model.getStatus()
    solution['status'] = status

    if status == 'timelimit':
        solution['status'] = 'timeout'
        solution['exitflag'] = -1
        solution['cost'] = None
        solution['energy'] = np.nan
        solution['state'] = []
        solution['input'] = []
        return solution

    if status == 'optimal':
        solution['exitflag'] = 1
        solution['cost'] = model.getObjVal()
        z = np.zeros(nvar)
        for i in range(nvar):
            z[i] = model.getVal(vars_dict[i])

        energy = 0.0
        for veh in range(n_vehicles):
            energy += np.sum(z[idx['sx'][veh]]) + np.sum(z[idx['sy'][veh]])
        solution['energy'] = energy
        solution['state'] = np.zeros((n_vehicles, 4, horizon + 1))
        solution['input'] = np.zeros((n_vehicles, 2, horizon))

        for veh in range(n_vehicles):
            for k in range(horizon + 1):
                solution['state'][veh, 0, k] = z[idx['px'][veh][k]]
                solution['state'][veh, 1, k] = z[idx['py'][veh][k]]
                solution['state'][veh, 2, k] = z[idx['vx'][veh][k]]
                solution['state'][veh, 3, k] = z[idx['vy'][veh][k]]
            for k in range(horizon):
                solution['input'][veh, 0, k] = z[idx['ux'][veh][k]]
                solution['input'][veh, 1, k] = z[idx['uy'][veh][k]]
    else:
        solution['exitflag'] = 0
        solution['cost'] = None
        solution['energy'] = np.nan
        solution['state'] = []
        solution['input'] = []

    return solution


def generate_rh_instances(scenario, Thor, output_dir, start_idx,
                          time_limit=60.0, verbose=True,
                          instance_log_lines=None, scenario_index=None,
                          split_name=None, trajectory_json_path=None,
                          max_rh_steps=None):
    """Run the receding-horizon loop and export every step's MILP as ``.lp``.

    Parameters
    ----------
    scenario : dict
        Full scenario (from ``generate_random_scenario``).
    Thor : float
        Horizon time in seconds.
    output_dir : str
        Directory to write ``.lp`` files into.
    start_idx : int
        Global instance counter – first file will be ``instance_{start_idx}.lp``.
    time_limit : float
        Per-step SCIP solver time limit in seconds.
    verbose : bool
    instance_log_lines : list[str] or None
        If provided, per-instance solve diagnostics are appended as text lines.
    scenario_index : int or None
        1-based scenario index used only for logging.
    split_name : str or None
        Split label used only for logging.
    trajectory_json_path : str or None
        If provided, write trajectory (states, obstacles, posBounds, goal) to
        this JSON file for instant visualization without re-solving.

    Returns
    -------
    n_exported : int
        Number of ``.lp`` files successfully written.
    """
    dt = scenario['dt']
    horizon_steps = int(round(Thor / dt))
    A, B = double_integrator_matrices(dt)

    start = np.atleast_2d(np.asarray(scenario['start'], dtype=float))
    goal = np.atleast_2d(np.asarray(scenario['goal'], dtype=float))
    n_vehicles = start.shape[0]
    current_state = start.copy()

    pos_tolerance = RH_POS_TOLERANCE
    vel_tolerance = RH_VEL_TOLERANCE
    max_iterations = RH_MAX_ITERATIONS
    if max_rh_steps is not None and max_rh_steps > 0:
        try:
            max_iterations = min(max_iterations, int(max_rh_steps))
        except Exception:
            pass

    n_exported = 0
    state_history = [start.copy()]  # for trajectory JSON

    for iteration in range(max_iterations):
        # ----- convergence check -----
        pos_error = np.linalg.norm(current_state[:, 0:2] - goal[:, 0:2], axis=1)
        vel_error = np.linalg.norm(current_state[:, 2:4] - goal[:, 2:4], axis=1)
        if np.all(pos_error < pos_tolerance) and np.all(vel_error < vel_tolerance):
            if verbose:
                print(f'    goal reached at iteration {iteration}')
            break

        # ----- build local scenario -----
        local = scenario.copy()
        local['start'] = current_state.copy()
        local['goal'] = goal.copy()
        local['horizonSteps'] = horizon_steps
        local['nVehicles'] = n_vehicles
        if bool(scenario.get('disableCorridorObstacleFilter', True)):
            local['obstacles'] = np.asarray(scenario.get('obstacles', []), dtype=float).reshape((-1, 4))
        else:
            prune_margin = scenario.get('obstaclePruneMargin', 2.0)
            local['obstacles'] = filter_active_obstacles(
                current_state, goal, scenario.get('obstacles', []), prune_margin)

        # ----- assemble & build model -----
        milp_data = assemble_milp_data(local)
        model, vars_dict = build_milp_model(milp_data, time_limit=time_limit)

        # ----- export .lp BEFORE solving -----
        instance_idx = start_idx + n_exported
        lp_path = os.path.join(output_dir, f'instance_{instance_idx}.lp')
        json_path = os.path.splitext(lp_path)[0] + '.json'
        model.writeProblem(lp_path)
        _write_instance_sidecar_json(
            json_path=json_path,
            lp_path=lp_path,
            milp_data=milp_data,
            scenario_index=scenario_index,
            split_name=split_name,
            iteration=iteration,
        )

        # ----- solve -----
        model.optimize()
        scip_status = str(model.getStatus())
        try:
            scip_time = float(model.getSolvingTime())
        except Exception:
            scip_time = float('nan')
        try:
            scip_nodes = int(model.getNNodes())
        except Exception:
            scip_nodes = -1
        n_vars = int(model.getNVars())
        n_cons = int(model.getNConss())
        n_bin = int(model.getNBinVars())
        solution = _extract_solution(model, vars_dict, milp_data)
        model.freeProb()

        if instance_log_lines is not None:
            split_tag = split_name if split_name is not None else "unknown_split"
            scen_tag = scenario_index if scenario_index is not None else -1
            line = (
                f"instance {instance_idx} SCIP nb nodes {scip_nodes} | "
                f"SCIP time {scip_time:.3f} s | split={split_tag} "
                f"scenario={scen_tag} iter={iteration} status={scip_status} "
                f"n_vars={n_vars} n_cons={n_cons} n_bin={n_bin}"
            )
            if solution.get('exitflag', 0) <= 0:
                line += " exported_lp_removed=1"
            instance_log_lines.append(line)

        if solution.get('exitflag', 0) <= 0:
            if verbose:
                print(f'    infeasible / timeout at iteration {iteration} – '
                      f'removing {lp_path}')
            # Remove the exported file for the infeasible instance
            if os.path.exists(lp_path):
                os.remove(lp_path)
            if os.path.exists(json_path):
                os.remove(json_path)
            break

        n_exported += 1

        # ----- apply first control -----
        applied_input = solution.get('input', None)
        if (isinstance(applied_input, np.ndarray) and applied_input.ndim == 3
                and applied_input.shape[2] > 0):
            applied_input = applied_input[:, :, 0]
        else:
            applied_input = np.zeros((n_vehicles, 2))

        next_state = np.zeros_like(current_state)
        for veh in range(n_vehicles):
            next_state[veh, :] = A @ current_state[veh, :] + B @ applied_input[veh, :]
        current_state = next_state
        state_history.append(current_state.copy())

    # ----- write trajectory JSON for instant visualization -----
    if trajectory_json_path is not None and len(state_history) > 0:
        obstacles = np.asarray(scenario.get('obstacles', []), dtype=float).reshape(-1, 4)
        pos_bounds = np.asarray(scenario.get('posBounds', [[0, 20], [0, 20]]), dtype=float)
        traj = {
            'scenario_index': int(scenario_index) if scenario_index is not None else -1,
            'n_vehicles': int(n_vehicles),
            'n_steps': len(state_history) - 1,
            'states': [np.asarray(s).tolist() for s in state_history],
            'obstacles': obstacles.tolist(),
            'posBounds': pos_bounds.tolist(),
            'goal': np.asarray(goal).tolist(),
        }
        with open(trajectory_json_path, 'w', encoding='utf-8') as f:
            json.dump(traj, f, indent=2)

    return n_exported


# ---------------------------------------------------------------------------
# Main – follows learn2branch/01_generate_instances.py structure
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate MILP instances from Receding Horizon scenarios.')
    template_map = build_base_scenario()
    template_n_vehicles = int(template_map.get('nVehicles', np.asarray(template_map['start']).shape[0]))
    default_workers = os.cpu_count()
    if default_workers is None or default_workers < 1:
        default_workers = 4

    parser.add_argument('--n_vehicles_min', type=int, default=template_n_vehicles,
                        help='Min number of vehicles per scenario.')
    parser.add_argument('--n_vehicles_max', type=int, default=template_n_vehicles,
                        help='Max number of vehicles per scenario.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random generator seed (default 0).')
    parser.add_argument('--horizon_time', type=float, default=3.0,
                        help='Planning horizon in seconds (default 3.0).')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Discretisation time step (default 0.1).')
    parser.add_argument('--workspace_size', type=float, default=15.0,
                        help='Square workspace side length (kept for backwards compatibility).')
    parser.add_argument(
        '--base_dir',
        type=str,
        default=os.path.join('data', 'instances', 'rh_milp'),
        help='Base output directory for train_mixed/valid_mixed/test_mixed splits.',
    )
    parser.add_argument(
        '--map_mode',
        choices=['random'],
        default='random',
        help='Randomized RH maps with starts/goals sampled inside the configured workspace.',
    )
    parser.add_argument('--time_limit', type=float, default=360.0,
                        help='SCIP per-step time limit in seconds (default 360).')
    parser.add_argument(
        '--formulation_profile',
        choices=['baseline', 'safe_tight'],
        default='baseline',
        help='baseline: fixed Big-M/no tie-break; safe_tight: dynamic Big-M with tiny binary tie-break.',
    )
    parser.add_argument(
        '--bigm_scale',
        type=float,
        default=1.02,
        help='Safety multiplier for dynamic Big-M (>=1.0). Used when formulation_profile=safe_tight.',
    )
    parser.add_argument(
        '--binary_tiebreak',
        type=float,
        default=1e-6,
        help='Tiny objective coefficient added to disjunction binaries in safe_tight profile.',
    )
    parser.add_argument(
        '--binary_tiebreak_lex_scale',
        type=float,
        default=1e-3,
        help='Tiny variable-dependent multiplier for binary_tiebreak coefficients.',
    )
    parser.add_argument(
        '--obstacle_encoding',
        choices=['side_select', 'side_pruned', 'axis_sign', 'axis_sign_extended', 'relax_flags'],
        default='relax_flags',
        help=(
            'Obstacle/separation disjunction encoding. relax_flags is the paper-faithful Big-M encoding; '
            'side_select uses exactly-one side binaries; '
            'side_pruned keeps a small geometric side subset; '
            'axis_sign uses two binaries for axis and sign; axis_sign_extended keeps side-select '
            'continuous selectors but branches only on axis/sign binaries.'
        ),
    )
    parser.add_argument(
        '--obstacle_side_prune_keep',
        type=int,
        default=2,
        help='For --obstacle_encoding side_pruned, number of sides retained per active obstacle-time group.',
    )
    parser.add_argument(
        '--obstacle_side_eps_scale',
        type=float,
        default=0.0,
        help=(
            'Scale for deterministic side/time/obstacle epsilon perturbation in obstacle constraints. '
            '0 disables it; positive values slightly inflate sides to break exact score ties.'
        ),
    )
    reachability_group = parser.add_mutually_exclusive_group()
    reachability_group.add_argument(
        '--reachability_obstacle_pruning',
        dest='no_reachability_obstacle_pruning',
        action='store_false',
        help=(
            'Enable per-time obstacle pruning based on a loose reachable-position box. '
            'Off by default for Schouwenaars-faithful instance generation.'
        ),
    )
    reachability_group.add_argument(
        '--no_reachability_obstacle_pruning',
        dest='no_reachability_obstacle_pruning',
        action='store_true',
        help='Disable per-time reachability obstacle pruning (default; paper-faithful).',
    )
    parser.set_defaults(no_reachability_obstacle_pruning=True)
    parser.add_argument(
        '--obstacle_prune_margin',
        type=float,
        default=None,
        help=(
            'Override corridor obstacle filter margin when --corridor_obstacle_filter is enabled. '
            'Larger values keep more map obstacles. If omitted, uses the scenario default.'
        ),
    )
    corridor_group = parser.add_mutually_exclusive_group()
    corridor_group.add_argument(
        '--corridor_obstacle_filter',
        dest='no_corridor_obstacle_filter',
        action='store_false',
        help=(
            'Enable current-goal corridor obstacle filtering. '
            'Off by default for Schouwenaars-faithful instance generation.'
        ),
    )
    corridor_group.add_argument(
        '--no_corridor_obstacle_filter',
        dest='no_corridor_obstacle_filter',
        action='store_true',
        help='Keep all scenario obstacles in every RH MILP (default; paper-faithful).',
    )
    parser.set_defaults(no_corridor_obstacle_filter=True)
    parser.add_argument(
        '--max_rh_steps',
        type=int,
        default=300,
        help='Maximum receding-horizon iterations per scenario (cap on number of LP instances per scenario).',
    )
    parser.add_argument('--train_target_instances', type=int, default=0,
                        help='Target number of train instances (0 disables cap).')
    parser.add_argument('--valid_target_instances', type=int, default=0,
                        help='Target number of valid instances (0 disables cap).')
    parser.add_argument('--test_target_instances', type=int, default=0,
                        help='Target number of test instances (0 disables cap).')
    parser.add_argument('--train_scenarios', type=int, default=30,
                        help='Number of train scenarios to generate (default 30).')
    parser.add_argument('--valid_scenarios', type=int, default=10,
                        help='Number of valid scenarios to generate (default 10).')
    parser.add_argument('--test_scenarios', type=int, default=10,
                        help='Number of test scenarios to generate (default 10).')
    parser.add_argument(
        '--fixed_n_vehicles',
        type=int,
        default=None,
        help='If set, force the same number of vehicles in all scenarios.',
    )
    parser.add_argument(
        '--no_clean_split_dirs',
        action='store_true',
        help='Keep existing instance_*.lp files in split directories.',
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        default=default_workers,
        help=(
            'Number of parallel worker processes for split generation (>=1). '
            f'Default: CPU count = {default_workers}.'
        ),
    )
    args = parser.parse_args()

    if args.bigm_scale < 1.0:
        raise ValueError('--bigm_scale must be >= 1.0')
    if args.binary_tiebreak < 0.0:
        raise ValueError('--binary_tiebreak must be >= 0.0')
    if args.binary_tiebreak_lex_scale < 0.0:
        raise ValueError('--binary_tiebreak_lex_scale must be >= 0.0')
    if args.obstacle_side_prune_keep < 1 or args.obstacle_side_prune_keep > 4:
        raise ValueError('--obstacle_side_prune_keep must be in [1, 4].')
    if args.obstacle_side_eps_scale < 0.0:
        raise ValueError('--obstacle_side_eps_scale must be >= 0.0')
    if args.obstacle_prune_margin is not None and args.obstacle_prune_margin < 0.0:
        raise ValueError('--obstacle_prune_margin must be >= 0.0')
    if args.n_workers < 1:
        raise ValueError('--n_workers must be >= 1')
    if args.train_scenarios < 1 or args.valid_scenarios < 1 or args.test_scenarios < 1:
        raise ValueError('--*_scenarios must be >= 1')

    if args.formulation_profile == 'safe_tight':
        use_dynamic_obstacle_bigm = True
        use_dynamic_separation_bigm = True
        bigm_scale = float(args.bigm_scale)
        binary_tiebreak = float(args.binary_tiebreak)
    else:
        use_dynamic_obstacle_bigm = False
        use_dynamic_separation_bigm = False
        bigm_scale = 1.0
        binary_tiebreak = 0.0

    nv_min = args.n_vehicles_min
    nv_max = args.n_vehicles_max
    if args.fixed_n_vehicles is not None:
        if args.fixed_n_vehicles < 1:
            raise ValueError('--fixed_n_vehicles must be >= 1.')
        nv_min = nv_max = int(args.fixed_n_vehicles)
    if nv_min > nv_max:
        raise ValueError('n_vehicles_min must be <= n_vehicles_max.')

    # Fixed-map mode: by default use the canonical number of vehicles.
    # However, if the user explicitly requests a fixed number of vehicles via
    # --fixed_n_vehicles, respect that so we can easily switch to e.g. single-
    # vehicle experiments for faster instance generation.
    if args.fixed_n_vehicles is None:
        if nv_min != template_n_vehicles or nv_max != template_n_vehicles:
            print(
                'Fixed-map mode enforces '
                f'{template_n_vehicles} vehicles; overriding provided vehicle range.'
            )
        nv_min = nv_max = template_n_vehicles
        args.fixed_n_vehicles = template_n_vehicles

    clean_split_dirs = not args.no_clean_split_dirs

    # ------------------------------------------------------------------
    # Split definitions  (scenarios, NOT individual instances)
    # Each scenario yields ~20-60 instances depending on distance / obstacles.
    # Map (obstacles) and vehicle count are fixed; only start/goal states change.
    # ------------------------------------------------------------------
    splits = [
        ('train_mixed', int(args.train_scenarios)),
        ('valid_mixed', int(args.valid_scenarios)),
        ('test_mixed', int(args.test_scenarios)),
    ]
    split_targets = {
        'train_mixed': max(0, int(args.train_target_instances)),
        'valid_mixed': max(0, int(args.valid_target_instances)),
        'test_mixed': max(0, int(args.test_target_instances)),
    }

    base_dir = args.base_dir

    # ------------------------------------------------------------------
    # Generate main splits (train / valid / test) – mixed params
    # Optionally in parallel across splits.
    # ------------------------------------------------------------------
    if args.n_workers == 1:
        for split_idx, (split_name, n_scenarios) in enumerate(splits):
            split_seed = args.seed + split_idx
            target_instances = split_targets.get(split_name, 0)
            _generate_split(
                split_name=split_name,
                n_scenarios=n_scenarios,
                target_instances=target_instances,
                nv_min=nv_min,
                nv_max=nv_max,
                use_dynamic_obstacle_bigm=use_dynamic_obstacle_bigm,
                use_dynamic_separation_bigm=use_dynamic_separation_bigm,
                bigm_scale=bigm_scale,
                binary_tiebreak=binary_tiebreak,
                clean_split_dirs=clean_split_dirs,
                base_dir=base_dir,
                rng_seed=split_seed,
                args=args,
            )
    else:
        jobs = []
        for split_idx, (split_name, n_scenarios) in enumerate(splits):
            split_seed = args.seed + split_idx
            target_instances = split_targets.get(split_name, 0)
            jobs.append(
                (
                    split_name,
                    n_scenarios,
                    target_instances,
                    nv_min,
                    nv_max,
                    use_dynamic_obstacle_bigm,
                    use_dynamic_separation_bigm,
                    bigm_scale,
                    binary_tiebreak,
                    clean_split_dirs,
                    base_dir,
                    split_seed,
                    args,
                )
            )

        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            future_to_split = {
                executor.submit(_generate_split, *job): job[0] for job in jobs
            }
            for future in as_completed(future_to_split):
                split_name = future_to_split[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f'Error while generating split {split_name}: {exc}', file=sys.stderr)

    print('\nDone.')


if __name__ == '__main__':
    main()
