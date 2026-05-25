import os
import json
import argparse
import importlib.util

import numpy as np
import matplotlib.pyplot as plt

from true_receding_horizon import (
    double_integrator_matrices,
    filter_active_obstacles,
    assemble_milp_data,
    build_milp_model,
    RH_POS_TOLERANCE,
    RH_VEL_TOLERANCE,
    RH_MAX_ITERATIONS,
)


def _load_generation_module():
    """Dynamically load helpers from 01_generate_rh_instances.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    gen_path = os.path.join(here, "01_generate_rh_instances.py")
    if not os.path.exists(gen_path):
        raise FileNotFoundError(f"Could not find generator script at {gen_path}")

    spec = importlib.util.spec_from_file_location("rh_instance_gen", gen_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_GEN_MODULE = _load_generation_module()
generate_random_scenario = _GEN_MODULE.generate_random_scenario
_extract_solution = _GEN_MODULE._extract_solution


def recreate_scenario_from_metadata(meta, scenario_index):
    """Recreate the *scenario_index*-th scenario using metadata.json.

    This mirrors the sampling logic in 01_generate_rh_instances.main so that
    the regenerated scenario matches the one used to export the LP instances.
    """
    n_scenarios = int(meta["n_scenarios"])
    if scenario_index < 1 or scenario_index > n_scenarios:
        raise ValueError(
            f"scenario_index must be in [1, {n_scenarios}], got {scenario_index}."
        )

    rng = np.random.RandomState(int(meta["seed"]))
    nv_min, nv_max = [int(x) for x in meta["n_vehicles_range"]]

    map_mode = meta["map_mode"]
    workspace_size = float(meta["workspace_size"])
    horizon_time = float(meta["horizon_time"])
    dt = float(meta["dt"])

    scenario = None
    sampled_nv = None
    sampled_no = None

    for s in range(1, n_scenarios + 1):
        nv = int(rng.randint(nv_min, nv_max + 1))

        sc = generate_random_scenario(
            rng,
            n_vehicles=nv,
            workspace_size=workspace_size,
            horizon_time=horizon_time,
            dt=dt,
            map_mode=map_mode,
        )

        if s == scenario_index:
            scenario = sc
            sampled_nv = nv
            sampled_no = int(len(sc.get("obstacles", [])))
            break

    assert scenario is not None, "Failed to recreate scenario."
    return scenario, sampled_nv, sampled_no


def simulate_rh_trajectory(
    scenario,
    Thor,
    time_limit=60.0,
    max_steps=None,
    verbose=True,
):
    """Run the receding-horizon loop and record vehicle states over time.

    Returns
    -------
    states : list of np.ndarray
        List of arrays with shape (n_vehicles, 4). The first entry is the
        initial state, subsequent entries are the state after each RH step.
    """
    dt = float(scenario["dt"])
    horizon_steps = int(round(Thor / dt))
    A, B = double_integrator_matrices(dt)

    start = np.atleast_2d(np.asarray(scenario["start"], dtype=float))
    goal = np.atleast_2d(np.asarray(scenario["goal"], dtype=float))
    n_vehicles = start.shape[0]
    current_state = start.copy()

    pos_tolerance = RH_POS_TOLERANCE
    vel_tolerance = RH_VEL_TOLERANCE
    max_iterations = RH_MAX_ITERATIONS
    if max_steps is not None:
        max_iterations = min(max_iterations, int(max_steps))

    state_history = [current_state.copy()]

    for iteration in range(max_iterations):
        # Convergence check
        pos_error = np.linalg.norm(current_state[:, 0:2] - goal[:, 0:2], axis=1)
        vel_error = np.linalg.norm(current_state[:, 2:4] - goal[:, 2:4], axis=1)
        if np.all(pos_error < pos_tolerance) and np.all(vel_error < vel_tolerance):
            if verbose:
                print(f"  goal reached at iteration {iteration}")
            break

        # Build local scenario
        local = scenario.copy()
        local["start"] = current_state.copy()
        local["goal"] = goal.copy()
        local["horizonSteps"] = horizon_steps
        local["nVehicles"] = n_vehicles
        prune_margin = scenario.get("obstaclePruneMargin", 2.0)
        local["obstacles"] = filter_active_obstacles(
            current_state, goal, scenario.get("obstacles", []), prune_margin
        )

        # Assemble & build model
        milp_data = assemble_milp_data(local)
        model, vars_dict = build_milp_model(milp_data, time_limit=time_limit)

        # Solve
        model.optimize()
        solution = _extract_solution(model, vars_dict, milp_data)
        model.freeProb()

        if solution.get("exitflag", 0) <= 0:
            if verbose:
                print(
                    f"  infeasible / timeout at iteration {iteration} – "
                    f"stopping simulation"
                )
            break

        applied_input = solution.get("input", None)
        if (
            isinstance(applied_input, np.ndarray)
            and applied_input.ndim == 3
            and applied_input.shape[2] > 0
        ):
            applied_input = applied_input[:, :, 0]
        else:
            applied_input = np.zeros((n_vehicles, 2))

        next_state = np.zeros_like(current_state)
        for veh in range(n_vehicles):
            next_state[veh, :] = A @ current_state[veh, :] + B @ applied_input[veh, :]
        current_state = next_state
        state_history.append(current_state.copy())

    return state_history


def plot_trajectory(scenario, state_history, title=None, save_path=None, show=True):
    """Plot vehicle paths together with obstacles."""
    states = np.stack(state_history, axis=0)  # (T+1, n_vehicles, 4)
    n_steps, n_vehicles, _ = states.shape

    obstacles = np.asarray(scenario.get("obstacles", []), dtype=float).reshape(-1, 4)
    goal_raw = scenario.get("goal", None)
    if goal_raw is not None:
        goal = np.asarray(goal_raw, dtype=float).reshape(-1, 4)
    else:
        goal = np.zeros((0, 4))

    fig, ax = plt.subplots(figsize=(6, 6))

    # Obstacles
    for rect in obstacles:
        xmin, xmax, ymin, ymax = rect
        width = xmax - xmin
        height = ymax - ymin
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                width,
                height,
                facecolor="tab:gray",
                edgecolor="k",
                linewidth=0.5,
                alpha=0.4,
            )
        )

    # Paths
    cmap = plt.cm.get_cmap("tab10", n_vehicles)
    for v in range(n_vehicles):
        xs = states[:, v, 0]
        ys = states[:, v, 1]
        ax.plot(xs, ys, "-o", color=cmap(v), label=f"vehicle {v + 1}", markersize=3)

        start_label = "start (initial position)" if v == 0 else None
        final_label = "final position" if v == 0 else None
        ax.scatter(
            xs[0],
            ys[0],
            color=cmap(v),
            edgecolor="k",
            zorder=5,
            label=start_label,
        )
        ax.scatter(
            xs[-1],
            ys[-1],
            marker="*",
            color=cmap(v),
            s=80,
            zorder=5,
            label=final_label,
        )

        if goal.shape[0] > v:
            gx = float(goal[v, 0])
            gy = float(goal[v, 1])
            goal_label = "goal" if v == 0 else None
            ax.scatter(
                gx,
                gy,
                marker="X",
                color=cmap(v),
                s=80,
                zorder=6,
                label=goal_label,
            )

    # Bounds: tight around trajectories and obstacles (zoomed view)
    xs_all = states[:, :, 0].ravel()
    ys_all = states[:, :, 1].ravel()
    x_min = float(np.min(xs_all))
    x_max = float(np.max(xs_all))
    y_min = float(np.min(ys_all))
    y_max = float(np.max(ys_all))

    if obstacles.size > 0:
        x_min = min(x_min, float(np.min(obstacles[:, 0])))
        x_max = max(x_max, float(np.max(obstacles[:, 1])))
        y_min = min(y_min, float(np.min(obstacles[:, 2])))
        y_max = max(y_max, float(np.max(obstacles[:, 3])))

    if goal.size > 0:
        x_min = min(x_min, float(np.min(goal[:, 0])))
        x_max = max(x_max, float(np.max(goal[:, 0])))
        y_min = min(y_min, float(np.min(goal[:, 1])))
        y_max = max(y_max, float(np.max(goal[:, 1])))

    # Handle degenerate ranges
    dx = x_max - x_min
    dy = y_max - y_min
    span = max(dx, dy, 1.0)
    pad = 0.1 * span
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)
    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Custom legend ordering: start, final, goal, then vehicles (1-based index).
    handles, labels = ax.get_legend_handles_labels()

    def _legend_sort_key(item):
        _, lbl = item
        if lbl.startswith("start"):
            return (0, 0)
        if lbl == "final position":
            return (1, 0)
        if lbl == "goal":
            return (2, 0)
        if lbl.startswith("vehicle "):
            try:
                num = int(lbl.split()[1])
            except Exception:
                num = 999
            return (3, num)
        return (99, 0)

    if handles:
        paired = list(zip(handles, labels))
        paired_sorted = sorted(paired, key=_legend_sort_key)
        handles_sorted, labels_sorted = zip(*paired_sorted)
        ax.legend(handles_sorted, labels_sorted, loc="best", fontsize=8)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualise RH trajectories for a given scenario index using "
            "existing metadata and the same random generator settings."
        )
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train_mixed",
        help="Data split to use (e.g. train_mixed, valid_mixed, test_mixed).",
    )
    parser.add_argument(
        "--scenario_index",
        type=int,
        required=True,
        help="1-based index of the scenario to visualize within the split.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.path.join("data", "instances", "rh_milp"),
        help="Root directory containing split subfolders and metadata.json.",
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        default=360.0,
        help="Per-step SCIP time limit in seconds.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of RH steps to simulate "
            "(e.g. 20 to show only the first 20 instances). "
            "If not set, uses n_instances from metadata when available."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the plot instead of (or in addition to) showing it.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="If set, do not open an interactive plot window.",
    )
    args = parser.parse_args()

    scenario_idx = int(args.scenario_index)

    # Default output path: data/visualization/rh_milp/<split>/scenario_<idx>.png
    vis_root = os.path.join("data", "visualization", "rh_milp")
    split_vis_dir = os.path.join(vis_root, args.split)
    os.makedirs(split_vis_dir, exist_ok=True)
    if args.output is None:
        output_path = os.path.join(split_vis_dir, f"scenario_{scenario_idx}.png")
    else:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ----- Prefer trajectory JSON from generator (no solving) -----
    traj_path = os.path.join(
        args.dataset_root, args.split, f"scenario_{scenario_idx}_trajectory.json"
    )
    if os.path.exists(traj_path):
        print(f"Loading trajectory from {traj_path} (no solving).")
        with open(traj_path, "r", encoding="utf-8") as f:
            traj = json.load(f)
        states_raw = traj["states"]
        state_history = [np.asarray(s) for s in states_raw]
        scenario_for_plot = {
            "obstacles": np.asarray(traj["obstacles"], dtype=float),
            "posBounds": np.asarray(traj["posBounds"], dtype=float),
            "goal": np.asarray(traj.get("goal", []), dtype=float),
        }
        if len(state_history) <= 1:
            print("Trajectory JSON has at most one state; nothing to plot.")
            return
        if args.max_steps is not None and args.max_steps < len(state_history):
            state_history = state_history[: args.max_steps + 1]
        title = f"{args.split} – scenario {scenario_idx} (from JSON)"
        plot_trajectory(
            scenario_for_plot,
            state_history,
            title=title,
            save_path=output_path,
            show=not args.no_show,
        )
        return

    # ----- Fallback: need metadata to recreate scenario and simulate (solving) -----
    meta_path = os.path.join(args.dataset_root, args.split, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Could not find metadata.json at {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Find per-scenario metadata (instance range etc.)
    scenario_meta = None
    for sc in meta.get("scenarios", []):
        if int(sc.get("scenario_index", -1)) == scenario_idx:
            scenario_meta = sc
            break
    if scenario_meta is None:
        raise ValueError(f"No entry for scenario_index={scenario_idx} in metadata.")

    if args.max_steps is not None:
        max_steps = int(args.max_steps)
        print(f"Using user-provided max_steps={max_steps}.")
    else:
        n_instances = int(scenario_meta.get("n_instances", 0))
        if n_instances == 0:
            print(
                f"Scenario {scenario_idx} has n_instances=0 according to metadata; "
                f"re-simulating without a hard step cap."
            )
            max_steps = None
        else:
            max_steps = n_instances

    print(
        f"Recreating scenario {scenario_idx} in split '{args.split}' "
        f"(vehicles={scenario_meta['n_vehicles']}, "
        f"obstacles={scenario_meta['n_obstacles']})."
    )
    scenario, sampled_nv, sampled_no = recreate_scenario_from_metadata(
        meta, scenario_idx
    )

    states = simulate_rh_trajectory(
        scenario,
        Thor=float(meta["horizon_time"]),
        time_limit=float(args.time_limit),
        max_steps=max_steps,
        verbose=True,
    )

    if len(states) <= 1:
        print("No successful RH steps; nothing to plot.")
        return

    title = f"{args.split} – scenario {scenario_idx}"
    plot_trajectory(
        scenario,
        states,
        title=title,
        save_path=output_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
