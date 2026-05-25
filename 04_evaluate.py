"""
04_evaluate.py
Evaluate RH MILP branching performance: SCIP vanilla full strong branching vs trained GNN.

References:
  - SCIP/Ecole learned-branching evaluation workflow
"""

import argparse
import csv
import datetime
import importlib.util
import json
import os
import pathlib
import re
import sys
import time

import ecole
import numpy as np
import torch

try:
    from pyscipopt import SCIP_PARAMSETTING
except Exception:  # pragma: no cover - PySCIPOpt is environment-dependent
    SCIP_PARAMSETTING = None

try:
    from pyscipopt import Model as PySCIPModel
except Exception:  # pragma: no cover - optional outside the Ecole environment
    PySCIPModel = None


UNSUPPORTED_HEURISTIC_PARAMS = {
    "heuristics/dks/freq",
    "heuristics/indicatordiving/freq",
    "heuristics/scheduler/freq",
}
BASE_VAR_NFEATS = 19
SEMANTIC_FEATURE_NAMES = [
    "state_px",
    "state_py",
    "state_vx",
    "state_vy",
    "input_ux",
    "input_uy",
    "fuel_slack",
    "terminal_slack",
    "obstacle_binary",
    "obstacle_selector",
    "separation_binary",
    "separation_selector",
    "vehicle_id",
    "time_index",
    "obstacle_id",
    "side_id",
    "pair_id",
    "axis_binary",
    "sign_binary",
    "side_left",
    "side_right",
    "side_bottom",
    "side_top",
    "is_binary",
    "is_obstacle",
    "is_separation",
    "solver_index",
]
SEMANTIC_VAR_NFEATS = len(SEMANTIC_FEATURE_NAMES)
CANDIDATE_SCHEMES = ("raw", "obstacle_group")


def _load_heuristics_off_overrides():
    path = pathlib.Path("tools/scip_heuristics_off_params.json")
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        overrides = payload.get("overrides", {})
        if not isinstance(overrides, dict):
            return {}
        return {k: v for k, v in overrides.items() if k not in UNSUPPORTED_HEURISTIC_PARAMS}
    except Exception:
        return {}


def _disable_heuristics_via_pyscipopt(model_or_env):
    """Apply the strict SCIP-off settings from the reference Gasse-style runs."""
    if SCIP_PARAMSETTING is None:
        return False
    try:
        model = getattr(model_or_env, "model", model_or_env)
        if not hasattr(model, "as_pyscipopt"):
            return False
        scip_model = model.as_pyscipopt()
        applied = False
        for setter in (
            lambda: scip_model.setPresolve(SCIP_PARAMSETTING.OFF),
            lambda: scip_model.setSeparating(SCIP_PARAMSETTING.OFF),
            lambda: scip_model.setHeuristics(SCIP_PARAMSETTING.OFF),
            lambda: scip_model.setParam("propagating/maxrounds", 0),
            lambda: scip_model.setParam("propagating/maxroundsroot", 0),
        ):
            try:
                setter()
                applied = True
            except Exception:
                pass
        return applied
    except Exception:
        return False


def build_branching_only_scip_params(
    time_limit,
    *,
    fullstrong=False,
    heuristics_off=True,
    fullstrong_scoreall=True,
    fullstrong_donotbranch=True,
):
    params = {
        "separating/maxrounds": 0,
        "separating/maxroundsroot": 0,
        "presolving/maxrestarts": 0,
        "presolving/maxrounds": 0,
        "propagating/maxrounds": 0,
        "propagating/maxroundsroot": 0,
        "limits/time": float(time_limit),
        "timing/clocktype": 2,
    }
    if fullstrong:
        params.update(
            {
                "branching/vanillafullstrong/priority": 1_000_000,
                "branching/vanillafullstrong/collectscores": True,
            }
        )
        if fullstrong_scoreall:
            params["branching/vanillafullstrong/scoreall"] = True
        if fullstrong_donotbranch:
            params["branching/vanillafullstrong/donotbranch"] = True
    if heuristics_off:
        params.update(_load_heuristics_off_overrides())
    return params


def build_example_scip_params(
    time_limit,
    *,
    fullstrong=False,
    fullstrong_scoreall=True,
    fullstrong_donotbranch=True,
):
    """Light SCIP profile: limited cuts/presolve restarts with the default brancher."""
    params = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": float(time_limit),
    }
    if fullstrong:
        params.update(
            {
                "branching/vanillafullstrong/priority": 1_000_000,
                "branching/vanillafullstrong/collectscores": True,
            }
        )
        if fullstrong_scoreall:
            params["branching/vanillafullstrong/scoreall"] = True
        if fullstrong_donotbranch:
            params["branching/vanillafullstrong/donotbranch"] = True
    return params


# ---------------------------------------------------------------------------
# Gasse-style SCIP parameter profiles for the four-way
# VSB / FSB / DEFAULT / GNN baseline.
# ---------------------------------------------------------------------------


def build_gasse_vsb_params(time_limit):
    """Vanilla Full Strong Branching baseline (uses vanillafullstrong AS the brancher)."""
    return {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": float(time_limit),
        "branching/vanillafullstrong/priority": 1_000_000,
        "branching/vanillafullstrong/collectscores": False,
        "branching/vanillafullstrong/donotbranch": False,
        "branching/vanillafullstrong/scoreall": False,
    }


def build_gasse_fsb_params(time_limit):
    """Full Strong Branching baseline (fullstrong + other branchers disabled)."""
    return {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": float(time_limit),
        "branching/fullstrong/priority": 1_000_000,
        "branching/relpscost/priority": -1,
        "branching/pscost/priority": -1,
        "branching/leastinf/priority": -1,
        "branching/mostinf/priority": -1,
        "branching/inference/priority": -1,
        "branching/random/priority": -1,
        "branching/fullstrong/maxdepth": -1,
        "branching/fullstrong/maxbounddist": 1.0,
    }


def build_gasse_default_params(time_limit):
    """Hybrid SCIP default brancher baseline (no override of branching/*)."""
    return {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": float(time_limit),
    }


def build_gasse_gnn_params(time_limit):
    """SCIP profile used while the GNN is the active brancher (no SB overrides)."""
    return {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": float(time_limit),
    }


class HeuristicsOffObservation:
    """Delegate to an Ecole observation while disabling SCIP heuristics before reset."""

    def __init__(self, observation, *, heuristics_off=True, pyscipopt_heuristics_off=False):
        self.observation = observation
        self.heuristics_off = bool(heuristics_off)
        self.pyscipopt_heuristics_off = bool(pyscipopt_heuristics_off)

    def before_reset(self, model):
        if self.heuristics_off and self.pyscipopt_heuristics_off:
            _disable_heuristics_via_pyscipopt(model)
        if hasattr(self.observation, "before_reset"):
            self.observation.before_reset(model)

    def extract(self, model, done):
        return self.observation.extract(model, done)


def extract_status_from_info(info, default="unknown"):
    if isinstance(info, dict):
        status = info.get("status", default)
    else:
        status = default
    if isinstance(status, bytes):
        return status.decode("utf-8", errors="replace")
    return str(status)


def instance_sort_key(path: pathlib.Path):
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        return (float("inf"), path.name)
    return (int(match.group(1)), path.name)


def select_instance_subset(instance_paths, subset_mode, subset_size, subset_seed):
    if subset_mode == "all":
        return list(instance_paths)
    if subset_size <= 0 or subset_size >= len(instance_paths):
        return list(instance_paths)
    if subset_mode != "random_fixed":
        raise ValueError(f"Unsupported subset_mode: {subset_mode}")
    rng = np.random.RandomState(subset_seed)
    indices = rng.choice(len(instance_paths), size=subset_size, replace=False)
    indices = np.sort(indices)
    return [instance_paths[i] for i in indices]


def _sanitize_array(values, clip=1e3):
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(values, -clip, clip)


def _semantic_col(name, semantic_dim=SEMANTIC_VAR_NFEATS):
    try:
        idx = SEMANTIC_FEATURE_NAMES.index(name)
    except ValueError:
        return None
    return idx if idx < int(semantic_dim) else None


def _obstacle_group_key(var_idx, semantic_features, semantic_dim=SEMANTIC_VAR_NFEATS):
    if semantic_features is None:
        return None
    if var_idx < 0 or var_idx >= semantic_features.shape[0]:
        return None

    obstacle_binary_col = _semantic_col("obstacle_binary", semantic_dim)
    obstacle_selector_col = _semantic_col("obstacle_selector", semantic_dim)
    vehicle_col = _semantic_col("vehicle_id", semantic_dim)
    time_col = _semantic_col("time_index", semantic_dim)
    obstacle_col = _semantic_col("obstacle_id", semantic_dim)
    if vehicle_col is None or time_col is None or obstacle_col is None:
        return None

    row = semantic_features[int(var_idx)]
    is_obstacle_branch = False
    if obstacle_binary_col is not None:
        is_obstacle_branch = is_obstacle_branch or bool(row[obstacle_binary_col] > 0.5)
    if obstacle_selector_col is not None:
        is_obstacle_branch = is_obstacle_branch or bool(row[obstacle_selector_col] > 0.5)
    if not is_obstacle_branch:
        return None

    return (
        "obstacle_group",
        round(float(row[vehicle_col]), 6),
        round(float(row[time_col]), 6),
        round(float(row[obstacle_col]), 6),
    )


def collapse_policy_action_set(
    action_set,
    semantic_features,
    *,
    candidate_scheme="raw",
    semantic_dim=SEMANTIC_VAR_NFEATS,
):
    action_set = np.asarray(action_set, dtype=np.int64)
    if action_set.size == 0 or candidate_scheme == "raw" or semantic_features is None:
        return action_set
    if candidate_scheme != "obstacle_group":
        raise ValueError(f"Unsupported candidate_scheme: {candidate_scheme}")

    groups = {}
    group_order = []
    for var_idx in action_set:
        var_idx = int(var_idx)
        group_key = _obstacle_group_key(var_idx, semantic_features, semantic_dim)
        if group_key is None:
            group_key = ("raw", var_idx)
        if group_key not in groups:
            groups[group_key] = []
            group_order.append(group_key)
        groups[group_key].append(var_idx)

    representatives = []
    for group_key in group_order:
        members = np.asarray(groups[group_key], dtype=np.int64)
        if group_key[0] == "obstacle_group":
            representatives.append(int(np.min(members)))
        else:
            representatives.append(int(members[0]))
    return np.asarray(representatives, dtype=np.int64)


def _instance_metadata_path(instance_path):
    path = pathlib.Path(instance_path)
    if path.suffix == ".lp":
        return path.with_suffix(".json")
    return path


def _solver_variable_names(instance_path):
    if PySCIPModel is None:
        return None
    path = pathlib.Path(instance_path)
    if path.suffix != ".lp" or not path.exists():
        return None
    try:
        model = PySCIPModel()
        model.hideOutput()
        model.readProblem(str(path))
        return [var.name for var in model.getVars()]
    except Exception:
        return None


def reorder_semantic_features_to_solver_order(instance_path, payload, values):
    variable_names = payload.get("model_metadata", {}).get("variable_names")
    if not isinstance(variable_names, list) or len(variable_names) != values.shape[0]:
        return values

    solver_names = _solver_variable_names(instance_path)
    if solver_names is None or len(solver_names) != values.shape[0]:
        return values

    name_to_index = {str(name): idx for idx, name in enumerate(variable_names)}
    try:
        reorder_indices = [name_to_index[str(name)] for name in solver_names]
    except KeyError:
        return values
    return values[np.asarray(reorder_indices, dtype=np.int64)]


def align_semantic_feature_columns(payload, values, semantic_dim):
    """Map JSON semantic columns by name and append solver-order index."""
    semantic_dim = int(semantic_dim)
    if values.ndim != 2:
        return None
    target_names = SEMANTIC_FEATURE_NAMES[:semantic_dim]
    source_names = payload.get("model_metadata", {}).get("semantic_feature_names")
    if isinstance(source_names, list) and len(source_names) == values.shape[1]:
        aligned = np.zeros((values.shape[0], semantic_dim), dtype=np.float32)
        target_pos = {name: i for i, name in enumerate(target_names)}
        for src_idx, name in enumerate(source_names):
            dst_idx = target_pos.get(str(name))
            if dst_idx is not None:
                aligned[:, dst_idx] = values[:, src_idx]
    elif values.shape[1] == semantic_dim:
        aligned = values.astype(np.float32, copy=False)
    else:
        return None

    if "solver_index" in target_names and aligned.shape[0] > 1:
        aligned[:, target_names.index("solver_index")] = np.linspace(
            0.0, 1.0, aligned.shape[0], dtype=np.float32
        )
    return aligned


def load_instance_semantic_features(instance_path, feature_cache, semantic_dim):
    instance_path = str(instance_path)
    cache_key = (instance_path, int(semantic_dim))
    if cache_key in feature_cache:
        return feature_cache[cache_key]

    json_path = _instance_metadata_path(instance_path)
    semantic_features = None
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                payload = json.load(f)
            raw = payload.get("model_metadata", {}).get("semantic_variable_features")
            if raw is not None:
                values = _sanitize_array(np.asarray(raw, dtype=np.float32))
                values = align_semantic_feature_columns(payload, values, semantic_dim)
                if values is not None:
                    semantic_features = reorder_semantic_features_to_solver_order(instance_path, payload, values)
                    target_names = SEMANTIC_FEATURE_NAMES[: int(semantic_dim)]
                    if "solver_index" in target_names and semantic_features.shape[0] > 1:
                        semantic_features[:, target_names.index("solver_index")] = np.linspace(
                            0.0, 1.0, semantic_features.shape[0], dtype=np.float32
                        )
        except Exception:
            semantic_features = None

    feature_cache[cache_key] = semantic_features
    return semantic_features


def augment_variable_features_for_policy(variable_features, instance_path, policy):
    variable_features = _sanitize_array(variable_features)
    if not getattr(policy, "use_semantic_features", False):
        return variable_features

    semantic_dim = int(getattr(policy, "semantic_var_nfeats", SEMANTIC_VAR_NFEATS))
    feature_cache = getattr(policy, "semantic_feature_cache", None)
    if feature_cache is None:
        feature_cache = {}
        policy.semantic_feature_cache = feature_cache

    semantic_features = load_instance_semantic_features(instance_path, feature_cache, semantic_dim)
    if semantic_features is None or semantic_features.shape[0] != variable_features.shape[0]:
        semantic_features = np.zeros((variable_features.shape[0], semantic_dim), dtype=np.float32)
    return np.concatenate([variable_features, semantic_features], axis=-1)


def load_gnn_policy(train_script_path: str, model_path: str, device: torch.device):
    train_script = pathlib.Path(train_script_path)
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    module_name = "rh_train_gnn_module"
    spec = importlib.util.spec_from_file_location(module_name, str(train_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import training script: {train_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    run_dir = pathlib.Path(model_path).parent
    config_path = run_dir / "config.json"
    emb_size = 64
    norm_type = "prenorm"
    config = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
            emb_size = config.get("emb_size", 64)
            norm_type = config.get("norm_type", "prenorm")
    mp_rounds = int(config.get("mp_rounds", 1))
    share_mp_weights = bool(config.get("share_mp_weights", False))
    semantic_features = bool(config.get("semantic_features", False))
    semantic_dim = int(config.get("semantic_var_nfeats", SEMANTIC_VAR_NFEATS))
    var_nfeats = config.get("var_nfeats")
    if var_nfeats is None:
        var_nfeats = BASE_VAR_NFEATS + (semantic_dim if semantic_features else 0)
    var_nfeats = int(var_nfeats)
    policy = module.GNNPolicy(
        emb_size=emb_size,
        norm_type=norm_type,
        var_nfeats=var_nfeats,
        mp_rounds=mp_rounds,
        share_mp_weights=share_mp_weights,
    ).to(device)
    policy.use_semantic_features = semantic_features
    policy.semantic_var_nfeats = max(0, var_nfeats - BASE_VAR_NFEATS) if semantic_features else 0
    policy.var_nfeats = var_nfeats
    policy.mp_rounds = mp_rounds
    policy.share_mp_weights = share_mp_weights
    policy.candidate_scheme = config.get("candidate_scheme", "raw")
    policy.semantic_feature_cache = {}

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def run_gnn_episode(
    env,
    policy,
    instance_path: str,
    device: torch.device,
    *,
    heuristics_off=True,
    pyscipopt_heuristics_off=False,
):
    timing = {
        "reset_wall": 0.0,
        "tensor_wall": 0.0,
        "forward_wall": 0.0,
        "select_wall": 0.0,
        "env_step_wall": 0.0,
        "total_wall": 0.0,
        "n_decisions": 0,
    }
    t_episode = time.perf_counter()
    t0 = time.perf_counter()
    observation, action_set, _, done, info = env.reset(instance_path)
    if heuristics_off and pyscipopt_heuristics_off:
        _disable_heuristics_via_pyscipopt(env)
    timing["reset_wall"] = time.perf_counter() - t0
    while not done:
        timing["n_decisions"] += 1
        with torch.no_grad():
            t0 = time.perf_counter()
            variable_features_np = augment_variable_features_for_policy(
                observation.variable_features,
                instance_path,
                policy,
            )
            semantic_features = None
            semantic_dim = int(getattr(policy, "semantic_var_nfeats", SEMANTIC_VAR_NFEATS))
            if (
                getattr(policy, "use_semantic_features", False)
                and getattr(policy, "candidate_scheme", "raw") != "raw"
                and variable_features_np.shape[1] >= BASE_VAR_NFEATS + semantic_dim
            ):
                semantic_features = variable_features_np[:, BASE_VAR_NFEATS : BASE_VAR_NFEATS + semantic_dim]
            obs = (
                torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
                torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device),
                torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(device),
                torch.from_numpy(variable_features_np).to(device),
            )
            timing["tensor_wall"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            logits = policy(*obs)
            timing["forward_wall"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            candidate_idx = collapse_policy_action_set(
                action_set,
                semantic_features,
                candidate_scheme=getattr(policy, "candidate_scheme", "raw"),
                semantic_dim=semantic_dim,
            )
            candidate_idx_t = torch.as_tensor(candidate_idx, dtype=torch.long, device=logits.device)
            action = int(candidate_idx[logits[candidate_idx_t].argmax().item()])
            timing["select_wall"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            observation, action_set, _, done, info = env.step(action)
            timing["env_step_wall"] += time.perf_counter() - t0
    timing["total_wall"] = time.perf_counter() - t_episode
    return info, timing


def run_scip_default_episode(
    env,
    instance_path: str,
    *,
    heuristics_off=True,
    pyscipopt_heuristics_off=False,
):
    timing = {
        "reset_wall": 0.0,
        "env_step_wall": 0.0,
        "total_wall": 0.0,
    }
    t_episode = time.perf_counter()
    t0 = time.perf_counter()
    env.reset(instance_path)
    if heuristics_off and pyscipopt_heuristics_off:
        _disable_heuristics_via_pyscipopt(env)
    timing["reset_wall"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, _, _, _, info = env.step({})
    timing["env_step_wall"] = time.perf_counter() - t0
    timing["total_wall"] = time.perf_counter() - t_episode
    return info, timing


def safe_gain(baseline, compared):
    if baseline == 0:
        if compared == 0:
            return 0.0
        return np.nan
    return 100.0 * (1.0 - compared / baseline)


def fmt_gain(value):
    if np.isnan(value):
        return "   n/a  "
    return f"{value: >8.2f}%"


def append_text_block(path, lines):
    payload = "\n".join(lines).rstrip() + "\n"
    with open(path, "a", encoding="utf-8") as f:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            f.write("\n")
        f.write(payload)


def run_four_way_evaluation(
    args,
    device,
    instance_paths,
    instance_dir,
    result_path,
    timing_log_path,
):
    """Gasse2019_vanilla-style four-way comparison: VSB / FSB / SCIP DEFAULT / GNN.

    Runs all four branchers on each instance using matched SCIP parameter profiles.
    Outputs a compact console report and a CSV whose columns are dedicated
    to each method (vsb_*, fsb_*, default_*, gnn_*) plus per-baseline gains.
    """
    policy = load_gnn_policy(args.train_script, args.model_path, device)
    if args.candidate_scheme is not None:
        policy.candidate_scheme = args.candidate_scheme
    if getattr(policy, "candidate_scheme", "raw") not in CANDIDATE_SCHEMES:
        raise ValueError(f"Unsupported candidate_scheme: {getattr(policy, 'candidate_scheme', None)}")
    print(
        f"[{datetime.datetime.now()}] [four_way] model var_nfeats: "
        f"{getattr(policy, 'var_nfeats', BASE_VAR_NFEATS)} "
        f"semantic_features={int(getattr(policy, 'use_semantic_features', False))} "
        f"candidate_scheme={getattr(policy, 'candidate_scheme', 'raw')} "
        f"mp_rounds={getattr(policy, 'mp_rounds', 1)} "
        f"share_mp_weights={int(getattr(policy, 'share_mp_weights', False))}"
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    gnn_params = build_gasse_gnn_params(args.time_limit)
    vsb_params = build_gasse_vsb_params(args.time_limit)
    fsb_params = build_gasse_fsb_params(args.time_limit)
    default_params = build_gasse_default_params(args.time_limit)

    info_func = {
        "nb_nodes": ecole.reward.NNodes().cumsum(),
        "time": ecole.reward.SolvingTime().cumsum(),
    }

    gnn_env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        information_function=info_func,
        scip_params=gnn_params,
        pseudo_candidates=bool(args.pseudo_candidates),
    )
    vsb_env = ecole.environment.Configuring(
        observation_function=None, information_function=info_func, scip_params=vsb_params,
    )
    fsb_env = ecole.environment.Configuring(
        observation_function=None, information_function=info_func, scip_params=fsb_params,
    )
    default_env = ecole.environment.Configuring(
        observation_function=None, information_function=info_func, scip_params=default_params,
    )
    gnn_env.seed(args.seed)
    vsb_env.seed(args.seed)
    fsb_env.seed(args.seed)
    default_env.seed(args.seed)

    timing_lines = [
        "Four-way (VSB/FSB/DEFAULT/GNN) timing breakdown log",
        f"timestamp={datetime.datetime.now().isoformat(timespec='seconds')}",
        (
            f"instance_dir={instance_dir} n_instances={len(instance_paths)} "
            f"subset_mode={args.subset_mode} subset_seed={args.subset_seed}"
        ),
        (
            f"time_limit={args.time_limit} pseudo_candidates={int(args.pseudo_candidates)} "
            f"device={device} model={args.model_path} "
            f"model_var_nfeats={getattr(policy, 'var_nfeats', BASE_VAR_NFEATS)} "
            f"semantic_features={int(getattr(policy, 'use_semantic_features', False))} "
            f"candidate_scheme={getattr(policy, 'candidate_scheme', 'raw')} "
            f"mp_rounds={getattr(policy, 'mp_rounds', 1)} "
            f"share_mp_weights={int(getattr(policy, 'share_mp_weights', False))}"
        ),
        f"min_scip_nodes={args.min_scip_nodes} max_filtered_instances={args.max_filtered_instances}",
        "",
    ]

    rows = []
    skipped_by_scip_nodes = 0
    screened_count = 0

    for idx, path in enumerate(instance_paths):
        if args.max_filtered_instances is not None and len(rows) >= args.max_filtered_instances:
            break
        path_str = str(path)

        vsb_info, vsb_timing = run_scip_default_episode(
            vsb_env, path_str, heuristics_off=False, pyscipopt_heuristics_off=False,
        )
        fsb_info, fsb_timing = run_scip_default_episode(
            fsb_env, path_str, heuristics_off=False, pyscipopt_heuristics_off=False,
        )
        default_info, default_timing = run_scip_default_episode(
            default_env, path_str, heuristics_off=False, pyscipopt_heuristics_off=False,
        )
        screened_count += 1

        screen_nodes = int(default_info["nb_nodes"])
        if screen_nodes < args.min_scip_nodes:
            skipped_by_scip_nodes += 1
            timing_lines.append(
                f"instance={idx:03d} path={path_str} default_nodes={screen_nodes} "
                f"skipped=1 reason=min_scip_nodes"
            )
            continue

        gnn_info, gnn_timing = run_gnn_episode(
            gnn_env, policy, path_str, device,
            heuristics_off=False, pyscipopt_heuristics_off=False,
        )

        vsb_nodes = int(vsb_info["nb_nodes"])
        fsb_nodes = int(fsb_info["nb_nodes"])
        default_nodes = int(default_info["nb_nodes"])
        gnn_nodes = int(gnn_info["nb_nodes"])
        vsb_time = float(vsb_info["time"])
        fsb_time = float(fsb_info["time"])
        default_time = float(default_info["time"])
        gnn_time = float(gnn_info["time"])

        gain_vsb_nodes = safe_gain(vsb_nodes, gnn_nodes)
        gain_fsb_nodes = safe_gain(fsb_nodes, gnn_nodes)
        gain_default_nodes = safe_gain(default_nodes, gnn_nodes)
        gain_vsb_time = safe_gain(vsb_time, gnn_time)
        gain_fsb_time = safe_gain(fsb_time, gnn_time)
        gain_default_time = safe_gain(default_time, gnn_time)

        print(f"  ")
        print(
            f"Instance {idx: >3} | SCIP VSB nodes     {vsb_nodes: >10d} | "
            f"SCIP VSB time     {vsb_time: >9.2f} "
        )
        print(
            f"             | SCIP FSB nodes     {fsb_nodes: >10d} | "
            f"SCIP FSB time     {fsb_time: >9.2f} "
        )
        print(
            f"             | SCIP DEFAULT nodes {default_nodes: >10d} | "
            f"SCIP DEFAULT time {default_time: >9.2f} "
        )
        print(
            f"             | GNN nodes          {gnn_nodes: >10d} | "
            f"GNN time          {gnn_time: >9.2f} "
        )
        print(f"             | Gain VSB           {fmt_gain(gain_vsb_nodes)} | Gain VSB         {fmt_gain(gain_vsb_time)}")
        print(f"             | Gain FSB           {fmt_gain(gain_fsb_nodes)} | Gain FSB         {fmt_gain(gain_fsb_time)}")
        print(
            f"             | Gain DEFAULT       {fmt_gain(gain_default_nodes)} | "
            f"Gain DEFAULT     {fmt_gain(gain_default_time)}"
        )

        timing_lines.append(
            f"instance={idx:03d} path={path_str} "
            f"vsb_nodes={vsb_nodes} vsb_time={vsb_time:.6f} vsb_wall={vsb_timing['total_wall']:.6f} "
            f"fsb_nodes={fsb_nodes} fsb_time={fsb_time:.6f} fsb_wall={fsb_timing['total_wall']:.6f} "
            f"default_nodes={default_nodes} default_time={default_time:.6f} "
            f"default_wall={default_timing['total_wall']:.6f} "
            f"gnn_nodes={gnn_nodes} gnn_time={gnn_time:.6f} gnn_wall={gnn_timing['total_wall']:.6f} "
            f"gnn_decisions={gnn_timing['n_decisions']} "
            f"gnn_tensor={gnn_timing['tensor_wall']:.6f} gnn_forward={gnn_timing['forward_wall']:.6f} "
            f"gnn_select={gnn_timing['select_wall']:.6f} gnn_step={gnn_timing['env_step_wall']:.6f}"
        )

        rows.append({
            "instance_id": idx,
            "instance_path": path_str,
            "vsb_nodes": vsb_nodes,
            "vsb_time": vsb_time,
            "vsb_status": extract_status_from_info(vsb_info),
            "vsb_wall_time": float(vsb_timing["total_wall"]),
            "fsb_nodes": fsb_nodes,
            "fsb_time": fsb_time,
            "fsb_status": extract_status_from_info(fsb_info),
            "fsb_wall_time": float(fsb_timing["total_wall"]),
            "default_nodes": default_nodes,
            "default_time": default_time,
            "default_status": extract_status_from_info(default_info),
            "default_wall_time": float(default_timing["total_wall"]),
            "gnn_nodes": gnn_nodes,
            "gnn_time": gnn_time,
            "gnn_status": extract_status_from_info(gnn_info),
            "gnn_wall_time": float(gnn_timing["total_wall"]),
            "gnn_tensor_wall": float(gnn_timing["tensor_wall"]),
            "gnn_forward_wall": float(gnn_timing["forward_wall"]),
            "gnn_select_wall": float(gnn_timing["select_wall"]),
            "gnn_env_step_wall": float(gnn_timing["env_step_wall"]),
            "gnn_decisions": int(gnn_timing["n_decisions"]),
            "gain_vsb_nodes_pct": gain_vsb_nodes,
            "gain_vsb_time_pct": gain_vsb_time,
            "gain_fsb_nodes_pct": gain_fsb_nodes,
            "gain_fsb_time_pct": gain_fsb_time,
            "gain_default_nodes_pct": gain_default_nodes,
            "gain_default_time_pct": gain_default_time,
        })

    fieldnames = [
        "instance_id", "instance_path",
        "vsb_nodes", "vsb_time", "vsb_status", "vsb_wall_time",
        "fsb_nodes", "fsb_time", "fsb_status", "fsb_wall_time",
        "default_nodes", "default_time", "default_status", "default_wall_time",
        "gnn_nodes", "gnn_time", "gnn_status", "gnn_wall_time",
        "gnn_tensor_wall", "gnn_forward_wall", "gnn_select_wall", "gnn_env_step_wall",
        "gnn_decisions",
        "gain_vsb_nodes_pct", "gain_vsb_time_pct",
        "gain_fsb_nodes_pct", "gain_fsb_time_pct",
        "gain_default_nodes_pct", "gain_default_time_pct",
    ]
    with open(result_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSummary (four_way)")
    print(f"  Screened inst.   : {screened_count}")
    print(f"  Skipped by node  : {skipped_by_scip_nodes} (min_scip_nodes={args.min_scip_nodes})")
    print(f"  Evaluated inst.  : {len(rows)}")
    if rows:
        vsb_n = np.asarray([r["vsb_nodes"] for r in rows], dtype=np.float64)
        fsb_n = np.asarray([r["fsb_nodes"] for r in rows], dtype=np.float64)
        def_n = np.asarray([r["default_nodes"] for r in rows], dtype=np.float64)
        gnn_n = np.asarray([r["gnn_nodes"] for r in rows], dtype=np.float64)
        vsb_t = np.asarray([r["vsb_time"] for r in rows], dtype=np.float64)
        fsb_t = np.asarray([r["fsb_time"] for r in rows], dtype=np.float64)
        def_t = np.asarray([r["default_time"] for r in rows], dtype=np.float64)
        gnn_t = np.asarray([r["gnn_time"] for r in rows], dtype=np.float64)
        print(f"  VSB     avg nodes/time : {vsb_n.mean():>8.2f} / {vsb_t.mean():>6.2f}s")
        print(f"  FSB     avg nodes/time : {fsb_n.mean():>8.2f} / {fsb_t.mean():>6.2f}s")
        print(f"  DEFAULT avg nodes/time : {def_n.mean():>8.2f} / {def_t.mean():>6.2f}s")
        print(f"  GNN     avg nodes/time : {gnn_n.mean():>8.2f} / {gnn_t.mean():>6.2f}s")
        gv_nodes = np.asarray([r["gain_vsb_nodes_pct"] for r in rows], dtype=np.float64)
        gf_nodes = np.asarray([r["gain_fsb_nodes_pct"] for r in rows], dtype=np.float64)
        gd_nodes = np.asarray([r["gain_default_nodes_pct"] for r in rows], dtype=np.float64)
        gv_time = np.asarray([r["gain_vsb_time_pct"] for r in rows], dtype=np.float64)
        gf_time = np.asarray([r["gain_fsb_time_pct"] for r in rows], dtype=np.float64)
        gd_time = np.asarray([r["gain_default_time_pct"] for r in rows], dtype=np.float64)
        print(f"  Mean gain GNN vs VSB     (nodes / time): {np.nanmean(gv_nodes):>7.2f}% / {np.nanmean(gv_time):>7.2f}%")
        print(f"  Mean gain GNN vs FSB     (nodes / time): {np.nanmean(gf_nodes):>7.2f}% / {np.nanmean(gf_time):>7.2f}%")
        print(f"  Mean gain GNN vs DEFAULT (nodes / time): {np.nanmean(gd_nodes):>7.2f}% / {np.nanmean(gd_time):>7.2f}%")
        print(f"  Time win-rate GNN<VSB     : {100.0 * np.mean(gnn_t < vsb_t):.2f}%")
        print(f"  Time win-rate GNN<FSB     : {100.0 * np.mean(gnn_t < fsb_t):.2f}%")
        print(f"  Time win-rate GNN<DEFAULT : {100.0 * np.mean(gnn_t < def_t):.2f}%")

        timing_lines.extend([
            "",
            "summary (four_way):",
            f"screened_instances={screened_count}",
            f"skipped_by_scip_nodes={skipped_by_scip_nodes}",
            f"evaluated_instances={len(rows)}",
            f"vsb_nodes_avg={vsb_n.mean():.6f} vsb_time_avg={vsb_t.mean():.6f}",
            f"fsb_nodes_avg={fsb_n.mean():.6f} fsb_time_avg={fsb_t.mean():.6f}",
            f"default_nodes_avg={def_n.mean():.6f} default_time_avg={def_t.mean():.6f}",
            f"gnn_nodes_avg={gnn_n.mean():.6f} gnn_time_avg={gnn_t.mean():.6f}",
            f"gain_vsb_nodes_mean={np.nanmean(gv_nodes):.6f} gain_vsb_time_mean={np.nanmean(gv_time):.6f}",
            f"gain_fsb_nodes_mean={np.nanmean(gf_nodes):.6f} gain_fsb_time_mean={np.nanmean(gf_time):.6f}",
            f"gain_default_nodes_mean={np.nanmean(gd_nodes):.6f} gain_default_time_mean={np.nanmean(gd_time):.6f}",
        ])

    append_text_block(timing_log_path, timing_lines)
    print(f"\nSaved results: {result_path}")
    print(f"Saved timing log: {timing_log_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RH MILP: SCIP vanilla FSB vs trained GNN.")
    parser.add_argument("--instance_dir", type=str, default="data/instances/rh_milp/test_mixed")
    parser.add_argument("--model_path", type=str, default="trained_models/rh_milp/gnn/0/best_params.pkl")
    parser.add_argument("--train_script", type=str, default="03_train_gnn.py")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA GPU id (-1 for CPU).")
    parser.add_argument("--seed", type=int, default=0, help="Ecole + torch seed.")
    parser.add_argument("--n_instances", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--time_limit", type=float, default=3600.0, help="SCIP time limit (seconds).")
    parser.add_argument(
        "--scip_profile",
        type=str,
        choices=["branching_only", "example"],
        default="branching_only",
        help=(
            "SCIP parameter profile. 'branching_only' is the RH controlled profile; "
            "'example' uses the light deployment SCIP settings."
        ),
    )
    parser.add_argument(
        "--baseline_policy",
        type=str,
        choices=["vanilla_fullstrong", "default"],
        default="vanilla_fullstrong",
        help="SCIP baseline policy: vanilla full strong branching or SCIP's default brancher.",
    )
    parser.add_argument(
        "--compare_example_scip",
        action="store_true",
        help=(
            "Also evaluate SCIP's default brancher with the light SCIP profile "
            "(separating/maxrounds=0, presolving/maxrestarts=0) on the same kept instances."
        ),
    )
    parser.add_argument(
        "--four_way",
        action="store_true",
        help=(
            "Run Gasse2019_vanilla.py-style four-way comparison on each instance: "
            "VSB (vanillafullstrong), FSB (fullstrong), SCIP DEFAULT, and the trained GNN. "
            "SCIP parameter profiles are matched across the four baselines. "
            "Overrides --baseline_policy, --compare_example_scip, --scip_profile, --heuristics_off, "
            "and --fullstrong_* flags for the duration of this run."
        ),
    )
    parser.add_argument(
        "--no_presolve",
        action="store_true",
        help="Deprecated: the branching-only profile already disables presolve/propagation rounds.",
    )
    heuristics_group = parser.add_mutually_exclusive_group()
    heuristics_group.add_argument(
        "--heuristics_off",
        dest="heuristics_off",
        action="store_true",
        help="Disable SCIP primal heuristics through heuristics/*/freq=-1 SCIP parameters (default).",
    )
    heuristics_group.add_argument(
        "--no_heuristics_off",
        dest="heuristics_off",
        action="store_false",
        help="Leave SCIP primal heuristics at SCIP defaults.",
    )
    parser.set_defaults(heuristics_off=True)
    parser.add_argument(
        "--pyscipopt_heuristics_off",
        dest="pyscipopt_heuristics_off",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    pyscipopt_group = parser.add_mutually_exclusive_group()
    pyscipopt_group.add_argument(
        "--pyscipopt_scip_off",
        dest="pyscipopt_heuristics_off",
        action="store_true",
        help=(
            "Also apply the reference PySCIPOpt OFF calls before solving: "
            "setPresolve(OFF), setSeparating(OFF), setHeuristics(OFF), and propagation rounds 0. "
            "Disabled by default because this Ecole/PySCIPOpt build can segfault at as_pyscipopt()."
        ),
    )
    pyscipopt_group.add_argument(
        "--no_pyscipopt_scip_off",
        dest="pyscipopt_heuristics_off",
        action="store_false",
        help="Do not use PySCIPOpt OFF calls; rely only on SCIP parameter overrides.",
    )
    parser.set_defaults(pyscipopt_heuristics_off=False)
    scoreall_group = parser.add_mutually_exclusive_group()
    scoreall_group.add_argument(
        "--fullstrong_scoreall",
        dest="fullstrong_scoreall",
        action="store_true",
        help="Set branching/vanillafullstrong/scoreall=True (default; reference-compatible).",
    )
    scoreall_group.add_argument(
        "--no_fullstrong_scoreall",
        dest="fullstrong_scoreall",
        action="store_false",
        help="Leave branching/vanillafullstrong/scoreall at SCIP default.",
    )
    parser.set_defaults(fullstrong_scoreall=True)
    donotbranch_group = parser.add_mutually_exclusive_group()
    donotbranch_group.add_argument(
        "--fullstrong_donotbranch",
        dest="fullstrong_donotbranch",
        action="store_true",
        help="Set branching/vanillafullstrong/donotbranch=True (default; reference-compatible).",
    )
    donotbranch_group.add_argument(
        "--no_fullstrong_donotbranch",
        dest="fullstrong_donotbranch",
        action="store_false",
        help="Leave branching/vanillafullstrong/donotbranch at SCIP default.",
    )
    parser.set_defaults(fullstrong_donotbranch=True)
    pseudo_group = parser.add_mutually_exclusive_group()
    pseudo_group.add_argument(
        "--pseudo_candidates",
        dest="pseudo_candidates",
        action="store_true",
        help="Expose pseudo candidates to the GNN policy.",
    )
    pseudo_group.add_argument(
        "--no_pseudo_candidates",
        dest="pseudo_candidates",
        action="store_false",
        help=(
            "Use only SCIP's LP branch candidates for GNN decisions (default). "
            "This matches finite-candidate training samples; pseudo candidates can have NaN strong scores."
        ),
    )
    parser.set_defaults(pseudo_candidates=False)
    parser.add_argument(
        "--candidate_scheme",
        type=str,
        choices=CANDIDATE_SCHEMES,
        default=None,
        help=(
            "Override the model candidate scheme. By default this is read from the training config; "
            "use obstacle_group to collapse side binaries with the same vehicle/time/obstacle key."
        ),
    )
    parser.add_argument("--subset_size", type=int, default=100, help="Subset size for evaluation.")
    parser.add_argument("--subset_seed", type=int, default=0, help="Seed used to pick a fixed random subset.")
    parser.add_argument(
        "--subset_mode",
        type=str,
        choices=["random_fixed", "all"],
        default="random_fixed",
        help="Subset selection mode.",
    )
    parser.add_argument(
        "--instance_list_out",
        type=str,
        default=None,
        help="Optional file path to store selected instance list.",
    )
    parser.add_argument(
        "--min_scip_nodes",
        type=int,
        default=2,
        help=(
            "Evaluate GNN only on instances where SCIP vanilla FSB opens at least this many nodes. "
            "Use 2 to skip root-solved instances."
        ),
    )
    parser.add_argument(
        "--max_filtered_instances",
        type=int,
        default=None,
        help="Optional cap on number of instances kept after SCIP-node filtering.",
    )
    parser.add_argument("--csv_path", type=str, default=None, help="Optional output CSV path.")
    parser.add_argument(
        "--timing_log_path",
        type=str,
        default=None,
        help="Optional output txt path for timing breakdown logs.",
    )
    args = parser.parse_args()

    if args.gpu == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    instance_dir = pathlib.Path(args.instance_dir)
    if not instance_dir.exists():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")
    instance_paths = sorted(instance_dir.glob("*.lp"), key=instance_sort_key)
    if not instance_paths:
        raise FileNotFoundError(f"No .lp instances found in: {instance_dir}")
    if args.n_instances is not None:
        args.subset_size = args.n_instances
    if args.min_scip_nodes < 1:
        raise ValueError("--min_scip_nodes must be >= 1")
    if args.max_filtered_instances is not None and args.max_filtered_instances < 1:
        raise ValueError("--max_filtered_instances must be >= 1")
    instance_paths = select_instance_subset(
        instance_paths,
        subset_mode=args.subset_mode,
        subset_size=args.subset_size,
        subset_seed=args.subset_seed,
    )

    result_path = args.csv_path
    if result_path is None:
        os.makedirs("results", exist_ok=True)
        result_path = f"results/rh_milp_eval_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    else:
        os.makedirs(pathlib.Path(result_path).parent, exist_ok=True)
    result_path_obj = pathlib.Path(result_path)

    timing_log_path = args.timing_log_path
    if timing_log_path is None:
        timing_log_path = str(result_path_obj.with_suffix(result_path_obj.suffix + ".timing.txt"))
    else:
        os.makedirs(pathlib.Path(timing_log_path).parent, exist_ok=True)

    instance_list_out = args.instance_list_out
    if instance_list_out is None:
        os.makedirs("results", exist_ok=True)
        instance_list_out = f"results/rh_eval_subset_seed{args.subset_seed}_n{len(instance_paths)}.txt"
    else:
        os.makedirs(pathlib.Path(instance_list_out).parent, exist_ok=True)

    print(f"[{datetime.datetime.now()}] device: {device}")
    print(f"[{datetime.datetime.now()}] seed: {args.seed}")
    print(
        f"[{datetime.datetime.now()}] instances: {len(instance_paths)} from {instance_dir} "
        f"(mode={args.subset_mode}, subset_seed={args.subset_seed})"
    )
    print(f"[{datetime.datetime.now()}] model: {args.model_path}")
    print(f"[{datetime.datetime.now()}] csv: {result_path}")
    print(f"[{datetime.datetime.now()}] timing log: {timing_log_path}")
    print(f"[{datetime.datetime.now()}] instance list: {instance_list_out}")
    print(f"[{datetime.datetime.now()}] scip_profile: {args.scip_profile}")
    print(f"[{datetime.datetime.now()}] baseline_policy: {args.baseline_policy}")
    print(f"[{datetime.datetime.now()}] compare_example_scip: {args.compare_example_scip}")
    print(
        f"[{datetime.datetime.now()}] SCIP compat: "
        f"pyscipopt_scip_off={int(args.pyscipopt_heuristics_off)} "
        f"fullstrong_scoreall={int(args.fullstrong_scoreall)} "
        f"fullstrong_donotbranch={int(args.fullstrong_donotbranch)}"
    )
    print(f"[{datetime.datetime.now()}] min_scip_nodes filter: {args.min_scip_nodes}")
    if args.max_filtered_instances is not None:
        print(f"[{datetime.datetime.now()}] max_filtered_instances: {args.max_filtered_instances}")
    instance_lines = [
        f"# timestamp={datetime.datetime.now().isoformat(timespec='seconds')}",
        (
            f"# instance_dir={instance_dir} mode={args.subset_mode} "
            f"subset_seed={args.subset_seed} subset_size={args.subset_size}"
        ),
    ]
    instance_lines.extend(str(path) for path in instance_paths)
    append_text_block(instance_list_out, instance_lines)

    if args.four_way:
        print(
            f"[{datetime.datetime.now()}] four_way enabled: "
            "running VSB / FSB / SCIP DEFAULT / GNN comparison "
            "with Gasse2019_vanilla.py SCIP profiles. "
            "Flags --baseline_policy, --compare_example_scip, --scip_profile, "
            "--heuristics_off, --pyscipopt_scip_off, and --fullstrong_* are ignored in this mode."
        )
        run_four_way_evaluation(
            args=args,
            device=device,
            instance_paths=instance_paths,
            instance_dir=instance_dir,
            result_path=result_path,
            timing_log_path=timing_log_path,
        )
        return

    policy = load_gnn_policy(args.train_script, args.model_path, device)
    if args.candidate_scheme is not None:
        policy.candidate_scheme = args.candidate_scheme
    if getattr(policy, "candidate_scheme", "raw") not in CANDIDATE_SCHEMES:
        raise ValueError(f"Unsupported candidate_scheme: {getattr(policy, 'candidate_scheme', None)}")
    print(
        f"[{datetime.datetime.now()}] model var_nfeats: {getattr(policy, 'var_nfeats', BASE_VAR_NFEATS)} "
        f"semantic_features={int(getattr(policy, 'use_semantic_features', False))} "
        f"candidate_scheme={getattr(policy, 'candidate_scheme', 'raw')} "
        f"mp_rounds={getattr(policy, 'mp_rounds', 1)} "
        f"share_mp_weights={int(getattr(policy, 'share_mp_weights', False))}"
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.scip_profile == "example":
        gnn_scip_parameters = build_example_scip_params(
            args.time_limit,
            fullstrong=False,
            fullstrong_scoreall=bool(args.fullstrong_scoreall),
            fullstrong_donotbranch=bool(args.fullstrong_donotbranch),
        )
        baseline_scip_parameters = build_example_scip_params(
            args.time_limit,
            fullstrong=args.baseline_policy == "vanilla_fullstrong",
            fullstrong_scoreall=bool(args.fullstrong_scoreall),
            fullstrong_donotbranch=bool(args.fullstrong_donotbranch),
        )
    else:
        gnn_scip_parameters = build_branching_only_scip_params(
            args.time_limit,
            fullstrong=False,
            heuristics_off=bool(args.heuristics_off),
            fullstrong_scoreall=bool(args.fullstrong_scoreall),
            fullstrong_donotbranch=bool(args.fullstrong_donotbranch),
        )
        baseline_scip_parameters = build_branching_only_scip_params(
            args.time_limit,
            fullstrong=args.baseline_policy == "vanilla_fullstrong",
            heuristics_off=bool(args.heuristics_off),
            fullstrong_scoreall=bool(args.fullstrong_scoreall),
            fullstrong_donotbranch=bool(args.fullstrong_donotbranch),
        )
    baseline_label = "SCIP-FSB" if args.baseline_policy == "vanilla_fullstrong" else "SCIP-Default"
    example_scip_parameters = build_example_scip_params(args.time_limit, fullstrong=False)

    gnn_env = ecole.environment.Branching(
        observation_function=HeuristicsOffObservation(
            ecole.observation.NodeBipartite(),
            heuristics_off=bool(args.heuristics_off),
            pyscipopt_heuristics_off=bool(args.pyscipopt_heuristics_off),
        ),
        information_function={
            "nb_nodes": ecole.reward.NNodes().cumsum(),
            "time": ecole.reward.SolvingTime().cumsum(),
        },
        scip_params=gnn_scip_parameters,
        pseudo_candidates=bool(args.pseudo_candidates),
    )
    scip_env = ecole.environment.Configuring(
        observation_function=None,
        information_function={
            "nb_nodes": ecole.reward.NNodes().cumsum(),
            "time": ecole.reward.SolvingTime().cumsum(),
        },
        scip_params=baseline_scip_parameters,
    )
    example_scip_env = None
    if args.compare_example_scip:
        example_scip_env = ecole.environment.Configuring(
            observation_function=None,
            information_function={
                "nb_nodes": ecole.reward.NNodes().cumsum(),
                "time": ecole.reward.SolvingTime().cumsum(),
            },
            scip_params=example_scip_parameters,
        )
    gnn_env.seed(args.seed)
    scip_env.seed(args.seed)
    if example_scip_env is not None:
        example_scip_env.seed(args.seed)

    rows = []
    timing_lines = [
        f"Timing breakdown log",
        f"timestamp={datetime.datetime.now().isoformat(timespec='seconds')}",
        (
            f"instance_dir={instance_dir} n_instances={len(instance_paths)} "
            f"subset_mode={args.subset_mode} subset_seed={args.subset_seed}"
        ),
        (
            f"time_limit={args.time_limit} scip_profile={args.scip_profile} "
            f"baseline_policy={args.baseline_policy} heuristics_off={int(args.heuristics_off)} "
            f"compare_example_scip={int(args.compare_example_scip)} "
            f"pyscipopt_scip_off={int(args.pyscipopt_heuristics_off)} "
            f"fullstrong_scoreall={int(args.fullstrong_scoreall)} "
            f"fullstrong_donotbranch={int(args.fullstrong_donotbranch)} "
            f"pseudo_candidates={int(args.pseudo_candidates)} device={device} model={args.model_path} "
            f"model_var_nfeats={getattr(policy, 'var_nfeats', BASE_VAR_NFEATS)} "
            f"semantic_features={int(getattr(policy, 'use_semantic_features', False))} "
            f"candidate_scheme={getattr(policy, 'candidate_scheme', 'raw')} "
            f"mp_rounds={getattr(policy, 'mp_rounds', 1)} "
            f"share_mp_weights={int(getattr(policy, 'share_mp_weights', False))}"
        ),
        (
            f"min_scip_nodes={args.min_scip_nodes} "
            f"max_filtered_instances={args.max_filtered_instances}"
        ),
        "",
    ]
    skipped_by_scip_nodes = 0
    screened_count = 0
    for idx, path in enumerate(instance_paths):
        if args.max_filtered_instances is not None and len(rows) >= args.max_filtered_instances:
            break
        path_str = str(path)

        scip_info, scip_timing = run_scip_default_episode(
            scip_env,
            path_str,
            heuristics_off=bool(args.heuristics_off),
            pyscipopt_heuristics_off=bool(args.pyscipopt_heuristics_off),
        )
        scip_nodes = int(scip_info["nb_nodes"])
        scip_time = float(scip_info["time"])
        scip_status = extract_status_from_info(scip_info)
        scip_wall = float(scip_timing["total_wall"])
        scip_py_overhead = max(0.0, scip_wall - scip_time)
        screened_count += 1

        if scip_nodes < args.min_scip_nodes:
            skipped_by_scip_nodes += 1
            timing_lines.append(
                (
                    f"instance={idx:03d} path={path_str} "
                    f"scip_nodes={scip_nodes} scip_time={scip_time:.6f} scip_wall={scip_wall:.6f} "
                    f"scip_reset={scip_timing['reset_wall']:.6f} scip_step={scip_timing['env_step_wall']:.6f} "
                    f"scip_py_overhead={scip_py_overhead:.6f} skipped=1 reason=min_scip_nodes"
                )
            )
            continue

        example_scip_info = None
        example_scip_timing = None
        example_scip_nodes = np.nan
        example_scip_time = np.nan
        example_scip_status = ""
        example_scip_wall = np.nan
        example_scip_py_overhead = np.nan
        if example_scip_env is not None:
            example_scip_info, example_scip_timing = run_scip_default_episode(
                example_scip_env,
                path_str,
                heuristics_off=False,
                pyscipopt_heuristics_off=False,
            )
            example_scip_nodes = int(example_scip_info["nb_nodes"])
            example_scip_time = float(example_scip_info["time"])
            example_scip_status = extract_status_from_info(example_scip_info)
            example_scip_wall = float(example_scip_timing["total_wall"])
            example_scip_py_overhead = max(0.0, example_scip_wall - example_scip_time)

        gnn_info, gnn_timing = run_gnn_episode(
            gnn_env,
            policy,
            path_str,
            device,
            heuristics_off=bool(args.heuristics_off),
            pyscipopt_heuristics_off=bool(args.pyscipopt_heuristics_off),
        )
        gnn_nodes = int(gnn_info["nb_nodes"])
        gnn_time = float(gnn_info["time"])
        gnn_status = extract_status_from_info(gnn_info)
        gnn_wall = float(gnn_timing["total_wall"])
        gnn_py_overhead = max(0.0, gnn_wall - gnn_time)

        node_gain = safe_gain(scip_nodes, gnn_nodes)
        time_gain = safe_gain(scip_time, gnn_time)
        example_node_gain = (
            safe_gain(example_scip_nodes, gnn_nodes) if example_scip_env is not None else np.nan
        )
        example_time_gain = (
            safe_gain(example_scip_time, gnn_time) if example_scip_env is not None else np.nan
        )

        print(
            f"Instance {idx: >3} | {baseline_label} nodes   {scip_nodes: >4d}  | "
            f"{baseline_label} time {scip_time: >6.2f} "
        )
        if example_scip_env is not None:
            print(
                f"             | Example SCIP nodes {example_scip_nodes: >4d}  | "
                f"Example SCIP time {example_scip_time: >6.2f} "
            )
        print(
            f"             | GNN  nb nodes    {gnn_nodes: >4d}  | GNN  time   {gnn_time: >6.2f} "
        )
        print(
            f"             | Gain         {fmt_gain(node_gain)} | Gain      {fmt_gain(time_gain)}"
        )
        if example_scip_env is not None:
            print(
                f"             | Ex. Gain     {fmt_gain(example_node_gain)} | "
                f"Ex. Gain  {fmt_gain(example_time_gain)}"
            )
        print(
            f"             | GNN  wall {gnn_wall: >6.2f} (tensor {gnn_timing['tensor_wall']: >5.2f}, "
            f"fwd {gnn_timing['forward_wall']: >5.2f}, step {gnn_timing['env_step_wall']: >5.2f})"
        )
        print(
            f"             | FSB  wall {scip_wall: >6.2f} (step {scip_timing['env_step_wall']: >5.2f})"
        )
        if example_scip_env is not None:
            print(
                f"             | ExSC wall {example_scip_wall: >6.2f} "
                f"(step {example_scip_timing['env_step_wall']: >5.2f})"
            )

        timing_line = (
            f"instance={idx:03d} path={path_str} "
            f"scip_nodes={scip_nodes} scip_time={scip_time:.6f} scip_wall={scip_wall:.6f} "
            f"scip_reset={scip_timing['reset_wall']:.6f} scip_step={scip_timing['env_step_wall']:.6f} "
            f"scip_py_overhead={scip_py_overhead:.6f} "
        )
        if example_scip_env is not None:
            timing_line += (
                f"example_scip_nodes={example_scip_nodes} example_scip_time={example_scip_time:.6f} "
                f"example_scip_wall={example_scip_wall:.6f} "
                f"example_scip_reset={example_scip_timing['reset_wall']:.6f} "
                f"example_scip_step={example_scip_timing['env_step_wall']:.6f} "
                f"example_scip_py_overhead={example_scip_py_overhead:.6f} "
            )
        timing_line += (
            f"gnn_nodes={gnn_nodes} gnn_time={gnn_time:.6f} gnn_wall={gnn_wall:.6f} "
            f"gnn_reset={gnn_timing['reset_wall']:.6f} gnn_tensor={gnn_timing['tensor_wall']:.6f} "
            f"gnn_forward={gnn_timing['forward_wall']:.6f} gnn_select={gnn_timing['select_wall']:.6f} "
            f"gnn_step={gnn_timing['env_step_wall']:.6f} gnn_decisions={gnn_timing['n_decisions']} "
            f"gnn_py_overhead={gnn_py_overhead:.6f}"
        )
        timing_lines.append(timing_line)

        row = {
            "instance_id": idx,
            "instance_path": path_str,
            "baseline_policy": args.baseline_policy,
            "scip_nodes": scip_nodes,
            "scip_time": scip_time,
            "scip_status": scip_status,
            "gnn_nodes": gnn_nodes,
            "gnn_time": gnn_time,
            "gnn_status": gnn_status,
            "node_gain_pct": node_gain,
            "time_gain_pct": time_gain,
            "scip_wall_time": scip_wall,
            "scip_reset_wall": float(scip_timing["reset_wall"]),
            "scip_env_step_wall": float(scip_timing["env_step_wall"]),
            "scip_python_overhead": scip_py_overhead,
            "gnn_wall_time": gnn_wall,
            "gnn_reset_wall": float(gnn_timing["reset_wall"]),
            "gnn_tensor_wall": float(gnn_timing["tensor_wall"]),
            "gnn_forward_wall": float(gnn_timing["forward_wall"]),
            "gnn_select_wall": float(gnn_timing["select_wall"]),
            "gnn_env_step_wall": float(gnn_timing["env_step_wall"]),
            "gnn_decisions": int(gnn_timing["n_decisions"]),
            "gnn_python_overhead": gnn_py_overhead,
        }
        if example_scip_env is not None:
            row.update(
                {
                    "example_scip_nodes": example_scip_nodes,
                    "example_scip_time": example_scip_time,
                    "example_scip_status": example_scip_status,
                    "example_node_gain_pct": example_node_gain,
                    "example_time_gain_pct": example_time_gain,
                    "example_scip_wall_time": example_scip_wall,
                    "example_scip_reset_wall": float(example_scip_timing["reset_wall"]),
                    "example_scip_env_step_wall": float(example_scip_timing["env_step_wall"]),
                    "example_scip_python_overhead": example_scip_py_overhead,
                }
            )
        rows.append(row)

    if not rows:
        print(
            f"\nNo instances passed filter: min_scip_nodes={args.min_scip_nodes} "
            f"(screened={screened_count}, skipped={skipped_by_scip_nodes})."
        )
        timing_lines.extend(
            [
                "",
                "summary:",
                f"screened_instances={screened_count}",
                f"skipped_by_scip_nodes={skipped_by_scip_nodes}",
                "evaluated_instances=0",
            ]
        )
        fieldnames = [
            "instance_id",
            "instance_path",
            "baseline_policy",
            "scip_nodes",
            "scip_time",
            "scip_status",
            "gnn_nodes",
            "gnn_time",
            "gnn_status",
            "node_gain_pct",
            "time_gain_pct",
            "scip_wall_time",
            "scip_reset_wall",
            "scip_env_step_wall",
            "scip_python_overhead",
            "gnn_wall_time",
            "gnn_reset_wall",
            "gnn_tensor_wall",
            "gnn_forward_wall",
            "gnn_select_wall",
            "gnn_env_step_wall",
            "gnn_decisions",
            "gnn_python_overhead",
        ]
        if args.compare_example_scip:
            fieldnames[6:6] = [
                "example_scip_nodes",
                "example_scip_time",
                "example_scip_status",
                "example_node_gain_pct",
                "example_time_gain_pct",
                "example_scip_wall_time",
                "example_scip_reset_wall",
                "example_scip_env_step_wall",
                "example_scip_python_overhead",
            ]
        with open(result_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
        print(f"\nSaved results: {result_path}")
        append_text_block(timing_log_path, timing_lines)
        print(f"Saved timing log: {timing_log_path}")
        return

    scip_nodes_all = np.asarray([r["scip_nodes"] for r in rows], dtype=np.float64)
    gnn_nodes_all = np.asarray([r["gnn_nodes"] for r in rows], dtype=np.float64)
    scip_time_all = np.asarray([r["scip_time"] for r in rows], dtype=np.float64)
    gnn_time_all = np.asarray([r["gnn_time"] for r in rows], dtype=np.float64)
    node_gain_all = np.asarray([r["node_gain_pct"] for r in rows], dtype=np.float64)
    time_gain_all = np.asarray([r["time_gain_pct"] for r in rows], dtype=np.float64)
    scip_wall_all = np.asarray([r["scip_wall_time"] for r in rows], dtype=np.float64)
    gnn_wall_all = np.asarray([r["gnn_wall_time"] for r in rows], dtype=np.float64)
    gnn_tensor_all = np.asarray([r["gnn_tensor_wall"] for r in rows], dtype=np.float64)
    gnn_forward_all = np.asarray([r["gnn_forward_wall"] for r in rows], dtype=np.float64)
    gnn_select_all = np.asarray([r["gnn_select_wall"] for r in rows], dtype=np.float64)
    gnn_decisions_all = np.asarray([r["gnn_decisions"] for r in rows], dtype=np.float64)
    gnn_overhead_all = np.asarray([r["gnn_python_overhead"] for r in rows], dtype=np.float64)
    scip_overhead_all = np.asarray([r["scip_python_overhead"] for r in rows], dtype=np.float64)
    if args.compare_example_scip:
        example_scip_nodes_all = np.asarray([r["example_scip_nodes"] for r in rows], dtype=np.float64)
        example_scip_time_all = np.asarray([r["example_scip_time"] for r in rows], dtype=np.float64)
        example_node_gain_all = np.asarray([r["example_node_gain_pct"] for r in rows], dtype=np.float64)
        example_time_gain_all = np.asarray([r["example_time_gain_pct"] for r in rows], dtype=np.float64)
        example_scip_wall_all = np.asarray([r["example_scip_wall_time"] for r in rows], dtype=np.float64)
        example_scip_overhead_all = np.asarray(
            [r["example_scip_python_overhead"] for r in rows],
            dtype=np.float64,
        )

    print("\nSummary")
    print(f"  {baseline_label} avg nodes: {scip_nodes_all.mean():.2f}")
    if args.compare_example_scip:
        print(f"  Example SCIP avg nodes: {example_scip_nodes_all.mean():.2f}")
    print(f"  GNN  avg nodes   : {gnn_nodes_all.mean():.2f}")
    print(f"  {baseline_label} avg time : {scip_time_all.mean():.2f}")
    if args.compare_example_scip:
        print(f"  Example SCIP avg time : {example_scip_time_all.mean():.2f}")
    print(f"  GNN  avg time    : {gnn_time_all.mean():.2f}")
    print(f"  Screened inst.   : {screened_count}")
    print(f"  Skipped by node  : {skipped_by_scip_nodes} (min_scip_nodes={args.min_scip_nodes})")
    print(f"  Evaluated inst.  : {len(rows)}")
    print(f"  Mean node gain   : {np.nanmean(node_gain_all):.2f}%")
    print(f"  Median node gain : {np.nanmedian(node_gain_all):.2f}%")
    print(f"  Mean time gain   : {np.nanmean(time_gain_all):.2f}%")
    print(f"  Median time gain : {np.nanmedian(time_gain_all):.2f}%")
    print(f"  Time win-rate    : {100.0 * np.mean(gnn_time_all < scip_time_all):.2f}%")
    if args.compare_example_scip:
        print(f"  Mean ex. node gain   : {np.nanmean(example_node_gain_all):.2f}%")
        print(f"  Median ex. node gain : {np.nanmedian(example_node_gain_all):.2f}%")
        print(f"  Mean ex. time gain   : {np.nanmean(example_time_gain_all):.2f}%")
        print(f"  Median ex. time gain : {np.nanmedian(example_time_gain_all):.2f}%")
        print(f"  Ex. time win-rate    : {100.0 * np.mean(gnn_time_all < example_scip_time_all):.2f}%")
    print(f"  GNN  avg wall    : {gnn_wall_all.mean():.4f}s")
    print(f"  {baseline_label} avg wall : {scip_wall_all.mean():.4f}s")
    if args.compare_example_scip:
        print(f"  Example SCIP avg wall : {example_scip_wall_all.mean():.4f}s")
    print(f"  GNN  avg py ovh  : {gnn_overhead_all.mean():.4f}s")
    print(f"  FSB  avg py ovh  : {scip_overhead_all.mean():.4f}s")
    if args.compare_example_scip:
        print(f"  Example SCIP avg py ovh : {example_scip_overhead_all.mean():.4f}s")
    if np.sum(gnn_decisions_all) > 0:
        print(f"  GNN avg decisions: {gnn_decisions_all.mean():.2f}")
        print(f"  GNN tensor/dec   : {1000.0 * gnn_tensor_all.sum() / gnn_decisions_all.sum():.3f} ms")
        print(f"  GNN fwd/dec      : {1000.0 * gnn_forward_all.sum() / gnn_decisions_all.sum():.3f} ms")
        print(f"  GNN select/dec   : {1000.0 * gnn_select_all.sum() / gnn_decisions_all.sum():.3f} ms")

    timing_lines.extend(
        [
            "",
            "summary:",
            f"screened_instances={screened_count}",
            f"skipped_by_scip_nodes={skipped_by_scip_nodes}",
            f"evaluated_instances={len(rows)}",
            f"scip_time_avg={scip_time_all.mean():.6f}",
            f"gnn_time_avg={gnn_time_all.mean():.6f}",
            f"scip_wall_avg={scip_wall_all.mean():.6f}",
            f"gnn_wall_avg={gnn_wall_all.mean():.6f}",
            f"scip_py_overhead_avg={scip_overhead_all.mean():.6f}",
            f"gnn_py_overhead_avg={gnn_overhead_all.mean():.6f}",
            f"gnn_decisions_avg={gnn_decisions_all.mean():.6f}",
        ]
    )
    if args.compare_example_scip:
        timing_lines.extend(
            [
                f"example_scip_time_avg={example_scip_time_all.mean():.6f}",
                f"example_scip_wall_avg={example_scip_wall_all.mean():.6f}",
                f"example_scip_py_overhead_avg={example_scip_overhead_all.mean():.6f}",
                f"example_time_gain_mean={np.nanmean(example_time_gain_all):.6f}",
                f"example_time_gain_median={np.nanmedian(example_time_gain_all):.6f}",
            ]
        )
    if np.sum(gnn_decisions_all) > 0:
        timing_lines.extend(
            [
                f"gnn_tensor_ms_per_decision={1000.0 * gnn_tensor_all.sum() / gnn_decisions_all.sum():.6f}",
                f"gnn_forward_ms_per_decision={1000.0 * gnn_forward_all.sum() / gnn_decisions_all.sum():.6f}",
                f"gnn_select_ms_per_decision={1000.0 * gnn_select_all.sum() / gnn_decisions_all.sum():.6f}",
            ]
        )

    fieldnames = [
        "instance_id",
        "instance_path",
        "baseline_policy",
        "scip_nodes",
        "scip_time",
        "scip_status",
        "gnn_nodes",
        "gnn_time",
        "gnn_status",
        "node_gain_pct",
        "time_gain_pct",
        "scip_wall_time",
        "scip_reset_wall",
        "scip_env_step_wall",
        "scip_python_overhead",
        "gnn_wall_time",
        "gnn_reset_wall",
        "gnn_tensor_wall",
        "gnn_forward_wall",
        "gnn_select_wall",
        "gnn_env_step_wall",
        "gnn_decisions",
        "gnn_python_overhead",
    ]
    if args.compare_example_scip:
        fieldnames[6:6] = [
            "example_scip_nodes",
            "example_scip_time",
            "example_scip_status",
            "example_node_gain_pct",
            "example_time_gain_pct",
            "example_scip_wall_time",
            "example_scip_reset_wall",
            "example_scip_env_step_wall",
            "example_scip_python_overhead",
        ]
    with open(result_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results: {result_path}")
    append_text_block(timing_log_path, timing_lines)
    print(f"Saved timing log: {timing_log_path}")


if __name__ == "__main__":
    main()
