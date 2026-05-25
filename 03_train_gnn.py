"""
03_train_gnn.py
Train a GNN branching policy on RH MILP samples generated with Ecole.

Expected sample format (from 02_generate_dataset.py):
    {
        "episode": int,
        "instance": str,
        "seed": int,
        "data": [node_observation, action, action_set, scores],
    }

    node_observation = (
        constraint_features,                 # (n_constraints, 5)
        (edge_indices, edge_values),        # (2, n_edges), (n_edges,)
        variable_features,                  # (n_variables, 19), optionally concat sidecar semantics
    )
"""

import argparse
import datetime
import gzip
import json
import os
import pathlib
import pickle
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric

try:
    from pyscipopt import Model as PySCIPModel
except Exception:  # pragma: no cover - optional outside the Ecole environment
    PySCIPModel = None

FEATURE_CLIP_VALUE = 1e3
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

try:
    GraphNorm = torch_geometric.nn.GraphNorm
except AttributeError:  # pragma: no cover - compatibility for older torch_geometric
    from torch_geometric.nn.norm import GraphNorm


class PreNormException(Exception):
    pass


class PreNormLayer(torch.nn.Module):
    def __init__(self, n_units, shift=True, scale=True):
        super().__init__()
        assert shift or scale
        self.register_buffer("shift", torch.zeros(n_units) if shift else None)
        self.register_buffer("scale", torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input_ = input_ + self.shift

        if self.scale is not None:
            input_ = input_ * self.scale

        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_):
        assert self.n_units == 1 or input_.shape[-1] == self.n_units
        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size()) / self.n_units

        delta = sample_avg - self.avg
        self.m2 = (
            self.var * self.count
            + sample_var * sample_count
            + delta**2 * self.count * sample_count / (self.count + sample_count)
        )
        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg

        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False


def _apply_norm(norm_layer, features, batch_index=None):
    if isinstance(norm_layer, GraphNorm):
        if batch_index is None:
            batch_index = torch.zeros(features.size(0), dtype=torch.long, device=features.device)
        return norm_layer(features, batch_index)
    return norm_layer(features)


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    def __init__(self, emb_size=64, norm_type="prenorm"):
        super().__init__("add")
        if norm_type not in {"prenorm", "graphnorm", "layernorm"}:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.feature_module_left = torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size))
        self.feature_module_edge = torch.nn.Sequential(torch.nn.Linear(1, emb_size, bias=False))
        self.feature_module_right = torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size, bias=False))
        if norm_type == "prenorm":
            message_norm = PreNormLayer(1, shift=False)
            post_conv_norm = PreNormLayer(1, shift=False)
        elif norm_type == "layernorm":
            message_norm = torch.nn.LayerNorm(emb_size)
            post_conv_norm = torch.nn.LayerNorm(emb_size)
        else:
            message_norm = torch.nn.Identity()
            post_conv_norm = torch.nn.Identity()
        self.feature_module_final = torch.nn.Sequential(
            message_norm,
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )
        self.post_conv_module = torch.nn.Sequential(post_conv_norm)
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        return self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )


class BaseModel(torch.nn.Module):
    def pre_train_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pre_train_next(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer) and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def pre_train(self, *args):
        try:
            with torch.no_grad():
                self.forward(*args)
            return False
        except PreNormException:
            return True


class GNNPolicy(BaseModel):
    def __init__(
        self,
        emb_size=64,
        norm_type="graphnorm",
        var_nfeats=BASE_VAR_NFEATS,
        mp_rounds=1,
        share_mp_weights=False,
    ):
        super().__init__()
        if norm_type not in {"prenorm", "graphnorm", "layernorm"}:
            raise ValueError(f"Unsupported norm_type: {norm_type}")
        if int(mp_rounds) < 1:
            raise ValueError("mp_rounds must be >= 1")
        self.norm_type = norm_type
        self.mp_rounds = int(mp_rounds)
        self.share_mp_weights = bool(share_mp_weights)
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = int(var_nfeats)

        if norm_type == "prenorm":
            self.cons_input_norm = PreNormLayer(cons_nfeats)
            self.edge_input_norm = PreNormLayer(edge_nfeats)
            self.var_input_norm = PreNormLayer(var_nfeats)
            self.cons_post_conv_norm = torch.nn.Identity()
            self.var_post_conv_norm = torch.nn.Identity()
        elif norm_type == "layernorm":
            self.cons_input_norm = torch.nn.LayerNorm(cons_nfeats)
            self.edge_input_norm = torch.nn.LayerNorm(edge_nfeats)
            self.var_input_norm = torch.nn.LayerNorm(var_nfeats)
            self.cons_post_conv_norm = torch.nn.Identity()
            self.var_post_conv_norm = torch.nn.Identity()
        else:
            self.cons_input_norm = GraphNorm(cons_nfeats)
            self.edge_input_norm = GraphNorm(edge_nfeats)
            self.var_input_norm = GraphNorm(var_nfeats)
            self.cons_post_conv_norm = GraphNorm(emb_size)
            self.var_post_conv_norm = GraphNorm(emb_size)

        self.cons_embedding = torch.nn.Sequential(
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )
        self.edge_embedding = torch.nn.Identity()
        self.var_embedding = torch.nn.Sequential(
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )
        if self.mp_rounds == 1 or self.share_mp_weights:
            self.conv_v_to_c = BipartiteGraphConvolution(emb_size=emb_size, norm_type=norm_type)
            self.conv_c_to_v = BipartiteGraphConvolution(emb_size=emb_size, norm_type=norm_type)
        else:
            self.conv_v_to_c_layers = torch.nn.ModuleList(
                [
                    BipartiteGraphConvolution(emb_size=emb_size, norm_type=norm_type)
                    for _ in range(self.mp_rounds)
                ]
            )
            self.conv_c_to_v_layers = torch.nn.ModuleList(
                [
                    BipartiteGraphConvolution(emb_size=emb_size, norm_type=norm_type)
                    for _ in range(self.mp_rounds)
                ]
            )
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def _mp_layers(self, round_idx):
        if self.mp_rounds == 1 or self.share_mp_weights:
            return self.conv_v_to_c, self.conv_c_to_v
        return self.conv_v_to_c_layers[round_idx], self.conv_c_to_v_layers[round_idx]

    def forward(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        constraint_batch=None,
        variable_batch=None,
    ):
        if constraint_batch is None:
            constraint_batch = torch.zeros(
                constraint_features.size(0),
                dtype=torch.long,
                device=constraint_features.device,
            )
        if variable_batch is None:
            variable_batch = torch.zeros(
                variable_features.size(0),
                dtype=torch.long,
                device=variable_features.device,
            )
        edge_batch = constraint_batch[edge_indices[0]]
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        constraint_features = _apply_norm(self.cons_input_norm, constraint_features, constraint_batch)
        edge_features = _apply_norm(self.edge_input_norm, edge_features, edge_batch)
        variable_features = _apply_norm(self.var_input_norm, variable_features, variable_batch)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        for round_idx in range(self.mp_rounds):
            conv_v_to_c, conv_c_to_v = self._mp_layers(round_idx)
            constraint_features = conv_v_to_c(
                variable_features,
                reversed_edge_indices,
                edge_features,
                constraint_features,
            )
            constraint_features = _apply_norm(self.cons_post_conv_norm, constraint_features, constraint_batch)
            variable_features = conv_c_to_v(
                constraint_features,
                edge_indices,
                edge_features,
                variable_features,
            )
            variable_features = _apply_norm(self.var_post_conv_norm, variable_features, variable_batch)
        return self.output_module(variable_features).squeeze(-1)


def log(msg, logfile=None):
    text = f"[{datetime.datetime.now()}] {msg}"
    print(text)
    if logfile is not None:
        with open(logfile, "a", encoding="utf-8") as f:
            print(text, file=f)


def _load_pickle_auto(path):
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def _sanitize_array(values, clip=FEATURE_CLIP_VALUE):
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=clip, neginf=-clip)
    values = np.clip(values, -clip, clip)
    return values


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


def load_instance_semantic_features(instance_path, feature_cache, semantic_dim=SEMANTIC_VAR_NFEATS):
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


def augment_variable_features(
    variable_features,
    instance_path,
    use_semantic_features,
    feature_cache,
    semantic_dim=SEMANTIC_VAR_NFEATS,
):
    variable_features = _sanitize_array(variable_features)
    if not use_semantic_features:
        return variable_features

    semantic_dim = int(semantic_dim)
    semantic_features = None
    if instance_path is not None:
        semantic_features = load_instance_semantic_features(instance_path, feature_cache, semantic_dim)
    if semantic_features is None or semantic_features.shape[0] != variable_features.shape[0]:
        semantic_features = np.zeros((variable_features.shape[0], semantic_dim), dtype=np.float32)
    return np.concatenate([variable_features, semantic_features], axis=-1)


def variable_feature_dim(use_semantic_features, semantic_dim=SEMANTIC_VAR_NFEATS):
    return BASE_VAR_NFEATS + (int(semantic_dim) if use_semantic_features else 0)


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


def collapse_candidate_groups(
    candidates,
    action,
    scores,
    semantic_features,
    *,
    candidate_scheme="raw",
    semantic_dim=SEMANTIC_VAR_NFEATS,
):
    candidates = np.asarray(candidates, dtype=np.int64)
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-1e8, posinf=1e8, neginf=-1e8)
    if candidates.size == 0:
        return candidates, int(action), np.asarray([], dtype=np.float32)
    raw_candidate_scores = scores[candidates]
    if candidate_scheme == "raw" or semantic_features is None:
        return candidates, int(action), raw_candidate_scores
    if candidate_scheme != "obstacle_group":
        raise ValueError(f"Unsupported candidate_scheme: {candidate_scheme}")

    groups = {}
    group_order = []
    action_group_key = None
    action = int(action)
    for var_idx in candidates:
        var_idx = int(var_idx)
        group_key = _obstacle_group_key(var_idx, semantic_features, semantic_dim)
        if group_key is None:
            group_key = ("raw", var_idx)
        if group_key not in groups:
            groups[group_key] = []
            group_order.append(group_key)
        groups[group_key].append(var_idx)
        if var_idx == action:
            action_group_key = group_key

    collapsed_candidates = []
    collapsed_scores = []
    representative_by_group = {}
    for group_key in group_order:
        members = np.asarray(groups[group_key], dtype=np.int64)
        member_scores = scores[members]
        if group_key[0] == "obstacle_group":
            representative = int(np.min(members))
            group_score = float(np.max(member_scores))
        else:
            representative = int(members[0])
            group_score = float(member_scores[0])
        representative_by_group[group_key] = representative
        collapsed_candidates.append(representative)
        collapsed_scores.append(group_score)

    if action_group_key is not None:
        collapsed_action = int(representative_by_group[action_group_key])
    else:
        collapsed_action = action

    return (
        np.asarray(collapsed_candidates, dtype=np.int64),
        collapsed_action,
        np.asarray(collapsed_scores, dtype=np.float32),
    )


def _is_sample_finite(path):
    sample = _load_pickle_auto(path)
    sample_observation, sample_action, sample_action_set, sample_scores = sample["data"]
    constraint_features, (_, edge_features), variable_features = sample_observation
    candidates = np.asarray(sample_action_set, dtype=np.int64)
    scores = np.asarray(sample_scores)
    if candidates.size == 0:
        return False
    if np.any(candidates < 0) or np.any(candidates >= scores.shape[0]):
        return False
    candidate_scores = np.asarray(scores, dtype=np.float64)[candidates]
    if not np.isfinite(candidate_scores).all():
        return False
    arrays = [
        _sanitize_array(constraint_features),
        _sanitize_array(edge_features),
        _sanitize_array(variable_features),
        candidate_scores,
    ]
    return all(np.isfinite(arr).all() for arr in arrays)


def filter_finite_samples(sample_files):
    kept = []
    dropped = []
    for path in sample_files:
        if _is_sample_finite(path):
            kept.append(path)
        else:
            dropped.append(path)
    return kept, dropped


def _sample_best_tie_count(
    path,
    *,
    use_semantic_features=False,
    semantic_dim=SEMANTIC_VAR_NFEATS,
    candidate_scheme="raw",
    semantic_feature_cache=None,
):
    sample = _load_pickle_auto(path)
    _, sample_action, sample_action_set, sample_scores = sample["data"]
    action_set = np.asarray(sample_action_set, dtype=np.int64)
    scores = np.asarray(sample_scores, dtype=np.float64)
    if action_set.size == 0 or np.any(action_set < 0) or np.any(action_set >= scores.shape[0]):
        return None

    semantic_features = None
    if use_semantic_features and candidate_scheme != "raw":
        if semantic_feature_cache is None:
            semantic_feature_cache = {}
        semantic_features = load_instance_semantic_features(
            sample.get("instance"),
            semantic_feature_cache,
            semantic_dim,
        )
    _, _, candidate_scores = collapse_candidate_groups(
        action_set,
        sample_action,
        scores,
        semantic_features,
        candidate_scheme=candidate_scheme,
        semantic_dim=semantic_dim,
    )
    if candidate_scores.size == 0 or not np.isfinite(candidate_scores).all():
        return None
    best_score = candidate_scores.max()
    return int(np.isclose(candidate_scores, best_score, rtol=1e-6, atol=1e-12).sum())


def filter_samples_by_best_tie_count(
    sample_files,
    max_best_tie_count,
    *,
    use_semantic_features=False,
    semantic_dim=SEMANTIC_VAR_NFEATS,
    candidate_scheme="raw",
):
    if max_best_tie_count <= 0:
        return list(sample_files), []

    kept = []
    dropped = []
    semantic_feature_cache = {}
    for path in sample_files:
        best_tie_count = _sample_best_tie_count(
            path,
            use_semantic_features=use_semantic_features,
            semantic_dim=semantic_dim,
            candidate_scheme=candidate_scheme,
            semantic_feature_cache=semantic_feature_cache,
        )
        if best_tie_count is not None and best_tie_count <= max_best_tie_count:
            kept.append(path)
        else:
            dropped.append(path)
    return kept, dropped


def build_inverse_instance_sampling_probs(sample_files):
    instance_ids = []
    for path in sample_files:
        sample = _load_pickle_auto(path)
        instance_ids.append(sample.get("instance", ""))
    instance_counts = Counter(instance_ids)
    weights = np.asarray([1.0 / max(instance_counts[inst], 1) for inst in instance_ids], dtype=np.float64)
    weights_sum = weights.sum()
    if weights_sum <= 0:
        raise RuntimeError("Invalid sampling weights: sum is not positive.")
    weights /= weights_sum
    return weights, instance_counts


def summarize_label_quality(
    sample_files,
    *,
    use_semantic_features=False,
    semantic_dim=SEMANTIC_VAR_NFEATS,
    candidate_scheme="raw",
):
    stats = {
        "samples": 0,
        "unique_instances": 0,
        "exact_tie_rate": 0.0,
        "tieaware_tie_rate": 0.0,
        "avg_candidates": 0.0,
        "avg_exact_best_tie_count": 0.0,
        "avg_tieaware_best_tie_count": 0.0,
        "max_tieaware_best_tie_count": 0,
        "best_at_clip_ceiling_rate": 0.0,
    }
    instance_counts = Counter()
    exact_ties = 0
    tieaware_ties = 0
    best_at_clip_ceiling = 0
    candidate_counts = []
    exact_best_tie_counts = []
    tieaware_best_tie_counts = []
    semantic_feature_cache = {}

    for path in sample_files:
        sample = _load_pickle_auto(path)
        _, sample_action, sample_action_set, sample_scores = sample["data"]
        action_set = np.asarray(sample_action_set, dtype=np.int64)
        scores = np.asarray(sample_scores, dtype=np.float64)
        instance_counts[sample.get("instance", "")] += 1
        if action_set.size == 0 or np.any(action_set < 0) or np.any(action_set >= scores.shape[0]):
            continue
        semantic_features = None
        if use_semantic_features and candidate_scheme != "raw":
            semantic_features = load_instance_semantic_features(
                sample.get("instance"),
                semantic_feature_cache,
                semantic_dim,
            )
        action_set, _, candidate_scores = collapse_candidate_groups(
            action_set,
            sample_action,
            scores,
            semantic_features,
            candidate_scheme=candidate_scheme,
            semantic_dim=semantic_dim,
        )
        if not np.isfinite(candidate_scores).all():
            continue

        best_score = candidate_scores.max()
        exact_best_count = int(np.sum(candidate_scores == best_score))
        tieaware_best_count = int(np.isclose(candidate_scores, best_score, rtol=1e-6, atol=1e-12).sum())
        exact_ties += int(exact_best_count > 1)
        tieaware_ties += int(tieaware_best_count > 1)
        best_at_clip_ceiling += int(np.isclose(best_score, FEATURE_CLIP_VALUE, rtol=0.0, atol=0.0))
        candidate_counts.append(int(action_set.size))
        exact_best_tie_counts.append(exact_best_count)
        tieaware_best_tie_counts.append(tieaware_best_count)

    checked = len(candidate_counts)
    stats["samples"] = checked
    stats["unique_instances"] = len(instance_counts)
    if checked > 0:
        stats["exact_tie_rate"] = exact_ties / checked
        stats["tieaware_tie_rate"] = tieaware_ties / checked
        stats["avg_candidates"] = float(np.mean(candidate_counts))
        stats["avg_exact_best_tie_count"] = float(np.mean(exact_best_tie_counts))
        stats["avg_tieaware_best_tie_count"] = float(np.mean(tieaware_best_tie_counts))
        stats["max_tieaware_best_tie_count"] = int(np.max(tieaware_best_tie_counts))
        stats["best_at_clip_ceiling_rate"] = best_at_clip_ceiling / checked
    return stats


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack(
        [F.pad(slice_, (0, max_pad_size - slice_.size(0)), "constant", pad_value) for slice_ in output],
        dim=0,
    )
    return output


def build_batch_indices(counts):
    counts = counts.reshape(-1).to(dtype=torch.long)
    return torch.arange(counts.size(0), device=counts.device, dtype=torch.long).repeat_interleave(counts)


class BipartiteNodeData(torch_geometric.data.Data):
    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        candidates,
        nb_candidates,
        nb_constraints,
        nb_variables,
        candidate_choice,
        candidate_scores,
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = nb_candidates
        self.nb_constraints = nb_constraints
        self.nb_variables = nb_variables
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores

    def __inc__(self, key, value, store, *args, **kwargs):
        if key == "edge_index":
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        if key == "candidates":
            return self.variable_features.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(
        self,
        sample_files,
        use_semantic_features=False,
        semantic_dim=SEMANTIC_VAR_NFEATS,
        candidate_scheme="raw",
    ):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.use_semantic_features = bool(use_semantic_features)
        self.semantic_dim = int(semantic_dim)
        if candidate_scheme not in CANDIDATE_SCHEMES:
            raise ValueError(f"Unsupported candidate_scheme: {candidate_scheme}")
        self.candidate_scheme = candidate_scheme
        self.semantic_feature_cache = {}

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        sample = _load_pickle_auto(self.sample_files[index])
        sample_observation, sample_action, sample_action_set, sample_scores = sample["data"]

        constraint_features, (edge_indices, edge_features), variable_features = sample_observation
        constraint_features = torch.FloatTensor(_sanitize_array(constraint_features))
        edge_indices = torch.LongTensor(edge_indices.astype(np.int64, copy=False))
        edge_features = torch.FloatTensor(np.expand_dims(_sanitize_array(edge_features), axis=-1))
        variable_features_np = augment_variable_features(
            variable_features,
            instance_path=sample.get("instance"),
            use_semantic_features=self.use_semantic_features,
            feature_cache=self.semantic_feature_cache,
            semantic_dim=self.semantic_dim,
        )
        semantic_features = None
        if (
            self.use_semantic_features
            and self.candidate_scheme != "raw"
            and variable_features_np.shape[1] >= BASE_VAR_NFEATS + self.semantic_dim
        ):
            semantic_features = variable_features_np[:, BASE_VAR_NFEATS : BASE_VAR_NFEATS + self.semantic_dim]
        variable_features = torch.FloatTensor(variable_features_np)

        candidates_np = np.array(sample_action_set, dtype=np.int64)
        sample_scores = np.nan_to_num(np.asarray(sample_scores, dtype=np.float32), nan=-1e8, posinf=1e8, neginf=-1e8)
        candidates_np, sample_action, candidate_scores_np = collapse_candidate_groups(
            candidates_np,
            sample_action,
            sample_scores,
            semantic_features,
            candidate_scheme=self.candidate_scheme,
            semantic_dim=self.semantic_dim,
        )
        candidates = torch.LongTensor(candidates_np)
        choice_matches = np.where(candidates_np == int(sample_action))[0]
        if choice_matches.size == 0:
            raise ValueError(f"Action {sample_action} not in action_set for sample {self.sample_files[index]}")
        candidate_choice = torch.tensor(int(choice_matches[0]), dtype=torch.long)
        candidate_scores = torch.FloatTensor(candidate_scores_np)

        graph = BipartiteNodeData(
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,
            candidates,
            torch.tensor([len(candidates_np)], dtype=torch.long),
            torch.tensor([constraint_features.shape[0]], dtype=torch.long),
            torch.tensor([variable_features.shape[0]], dtype=torch.long),
            candidate_choice,
            candidate_scores,
        )
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        return graph


def pretrain(policy, pretrain_loader, device):
    policy.pre_train_init()
    n_layers = 0
    while True:
        for batch in pretrain_loader:
            batch = batch.to(device)
            constraint_batch = build_batch_indices(batch.nb_constraints)
            variable_batch = build_batch_indices(batch.nb_variables)
            if not policy.pre_train(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                constraint_batch,
                variable_batch,
            ):
                break
        if policy.pre_train_next() is None:
            break
        n_layers += 1
    return n_layers


def process_epoch(policy, data_loader, device, top_k, entropy_bonus=0.0, optimizer=None, loss_mode="ce"):
    mean_loss = 0.0
    mean_kacc = np.zeros(len(top_k), dtype=np.float64)
    mean_entropy = 0.0
    n_samples_processed = 0
    skipped_batches = 0

    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(device)
            constraint_batch = build_batch_indices(batch.nb_constraints)
            variable_batch = build_batch_indices(batch.nb_variables)
            logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                constraint_batch,
                variable_batch,
            )
            if not torch.isfinite(logits).all():
                skipped_batches += 1
                continue
            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            valid_mask = (
                torch.arange(logits.size(-1), device=logits.device).unsqueeze(0)
                < batch.nb_candidates.unsqueeze(1)
            )
            masked_true_scores = true_scores.masked_fill(~valid_mask, -torch.inf)
            true_bestscore = masked_true_scores.max(dim=-1, keepdim=True).values

            if loss_mode == "tie":
                tie_mask = torch.isclose(masked_true_scores, true_bestscore, rtol=1e-6, atol=1e-12) & valid_mask
                tie_count = tie_mask.sum(dim=-1, keepdim=True).clamp(min=1)
                target_distribution = tie_mask.float() / tie_count.float()
                log_probs = F.log_softmax(logits, dim=-1)
                supervised_loss = -(target_distribution * log_probs).sum(dim=-1).mean()
            elif loss_mode == "bce":
                tie_mask = torch.isclose(masked_true_scores, true_bestscore, rtol=1e-6, atol=1e-12) & valid_mask
                targets = tie_mask.float()
                valid_targets = targets[valid_mask]
                pos = valid_targets.sum().clamp(min=1.0)
                neg = (valid_mask.sum().float() - pos).clamp(min=1.0)
                pos_weight = (neg / pos).clamp(min=1.0, max=20.0)
                supervised_loss = F.binary_cross_entropy_with_logits(
                    logits[valid_mask],
                    valid_targets,
                    pos_weight=pos_weight,
                    reduction="mean",
                )
            elif loss_mode == "ce":
                supervised_loss = F.cross_entropy(logits, batch.candidate_choices, reduction="mean")
            else:
                raise ValueError(f"Unknown loss_mode: {loss_mode}")

            if not torch.isfinite(supervised_loss):
                skipped_batches += 1
                continue
            entropy = (-F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            loss = supervised_loss - entropy_bonus * entropy

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                grads_finite = True
                for param in policy.parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        grads_finite = False
                        break
                if not grads_finite:
                    optimizer.zero_grad()
                    skipped_batches += 1
                    continue
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

            kacc = []
            metric_logits = logits.masked_fill(~valid_mask, -torch.inf)
            for k in top_k:
                k_eff = min(int(k), logits.size(-1))
                pred_top_k = metric_logits.topk(k_eff).indices
                if loss_mode in {"tie", "bce"}:
                    accuracy = tie_mask.gather(1, pred_top_k).any(dim=-1).float().mean().item()
                else:
                    accuracy = (
                        pred_top_k == batch.candidate_choices.unsqueeze(-1)
                    ).any(dim=-1).float().mean().item()
                kacc.append(accuracy)
            kacc = np.asarray(kacc)

            batch_size = batch.num_graphs
            mean_loss += supervised_loss.item() * batch_size
            mean_entropy += entropy.item() * batch_size
            mean_kacc += kacc * batch_size
            n_samples_processed += batch_size

    if n_samples_processed == 0:
        raise RuntimeError(f"No valid batches were processed (skipped={skipped_batches}).")

    mean_loss /= n_samples_processed
    mean_entropy /= n_samples_processed
    mean_kacc /= n_samples_processed
    return mean_loss, mean_kacc, mean_entropy


def main():
    parser = argparse.ArgumentParser(description="Train GNN policy on RH MILP Ecole samples.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA GPU id (-1 for CPU).")
    parser.add_argument(
        "--sample_root",
        type=str,
        default="data/samples/rh_milp",
        help="Root sample directory containing train/valid/test folders.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output run directory. Default: trained_models/rh_milp/gnn/<seed>.",
    )
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pretrain_batch_size", type=int, default=128)
    parser.add_argument("--valid_batch_size", type=int, default=128)
    parser.add_argument("--epoch_size", type=int, default=12000, help="Number of sampled train files per epoch.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--early_stopping", type=int, default=10)
    parser.add_argument("--entropy_bonus", type=float, default=0.0)
    parser.add_argument(
        "--loss_mode",
        type=str,
        choices=["ce", "tie", "bce"],
        default="tie",
        help=(
            "Supervised loss: standard CE on the collected expert action, "
            "tie-aware soft CE, or multi-label BCE over best-score candidates."
        ),
    )
    parser.add_argument(
        "--tie_aware_loss",
        action="store_true",
        help="Alias for --loss_mode tie.",
    )
    parser.add_argument(
        "--show_entropy",
        action="store_true",
        help="If set, include entropy values in training/validation logs.",
    )
    parser.add_argument(
        "--test_eval",
        type=str,
        choices=["none", "final", "each_epoch"],
        default="final",
        help=(
            "Test split evaluation mode. "
            "'none': no test evaluation, "
            "'final': evaluate test once with best-valid checkpoint, "
            "'each_epoch': evaluate test at every epoch + final best checkpoint."
        ),
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        choices=["loss", "acc"],
        default="loss",
        help="Validation metric used to select best_params.pkl. Default keeps prior loss-based behavior.",
    )
    parser.add_argument(
        "--emb_size",
        type=int,
        default=64,
        choices=[32, 64],
        help="GNN embedding size. 32 = lighter/faster per node, 64 = default.",
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        choices=["prenorm", "graphnorm", "layernorm"],
        default="graphnorm",
        help="Normalization mode in GNN blocks (default: graphnorm; layernorm matches the reference architecture more closely).",
    )
    parser.add_argument(
        "--mp_rounds",
        type=int,
        default=1,
        help=(
            "Number of variable-constraint-variable message-passing rounds. "
            "The reference architecture uses 1; use 2+ to test deeper WL-style hops."
        ),
    )
    parser.add_argument(
        "--share_mp_weights",
        action="store_true",
        help="Reuse the same message-passing layers across all rounds to limit model size.",
    )
    parser.add_argument(
        "--semantic_features",
        action="store_true",
        help="Concatenate RH sidecar semantic variable features to Ecole variable features.",
    )
    parser.add_argument(
        "--candidate_scheme",
        type=str,
        choices=CANDIDATE_SCHEMES,
        default="raw",
        help=(
            "Branch-candidate supervision granularity. "
            "'raw' keeps SCIP variables; 'obstacle_group' collapses side binaries with the same "
            "vehicle/time/obstacle semantic key into one canonical candidate."
        ),
    )
    parser.add_argument(
        "--max_best_tie_count",
        type=int,
        default=0,
        help=(
            "Drop samples whose best expert score is shared by more than this many candidates "
            "after candidate_scheme processing. Disabled by default; use 1 for unique expert labels."
        ),
    )
    args = parser.parse_args()
    if args.tie_aware_loss:
        args.loss_mode = "tie"
    if args.max_best_tie_count < 0:
        raise ValueError("--max_best_tie_count must be >= 0")
    if args.mp_rounds < 1:
        raise ValueError("--mp_rounds must be >= 1")

    top_k = [1, 3, 5, 10]
    semantic_dim = SEMANTIC_VAR_NFEATS
    var_nfeats = variable_feature_dim(args.semantic_features, semantic_dim)

    if args.gpu == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_dir = args.out_dir or f"trained_models/rh_milp/gnn/{args.seed}"
    os.makedirs(run_dir, exist_ok=True)
    logfile = os.path.join(run_dir, "train_log.txt")
    if os.path.exists(logfile) and os.path.getsize(logfile) > 0:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write("\n")
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "emb_size": args.emb_size,
                "norm_type": args.norm_type,
                "mp_rounds": int(args.mp_rounds),
                "share_mp_weights": bool(args.share_mp_weights),
                "semantic_features": bool(args.semantic_features),
                "semantic_var_nfeats": int(semantic_dim),
                "semantic_feature_names": list(SEMANTIC_FEATURE_NAMES),
                "var_nfeats": int(var_nfeats),
                "candidate_scheme": args.candidate_scheme,
                "selection_metric": args.selection_metric,
                "max_best_tie_count": int(args.max_best_tie_count),
            },
            f,
            indent=2,
        )

    train_dir = pathlib.Path(args.sample_root) / "train"
    valid_dir = pathlib.Path(args.sample_root) / "valid"
    test_dir = pathlib.Path(args.sample_root) / "test"
    train_files = sorted(str(p) for p in train_dir.glob("sample_*.pkl"))
    valid_files = sorted(str(p) for p in valid_dir.glob("sample_*.pkl"))
    test_files = sorted(str(p) for p in test_dir.glob("sample_*.pkl")) if test_dir.exists() else []

    if not train_files:
        raise FileNotFoundError(f"No training samples found in {train_dir}")
    if not valid_files:
        raise FileNotFoundError(f"No validation samples found in {valid_dir}")

    train_files, dropped_train = filter_finite_samples(train_files)
    valid_files, dropped_valid = filter_finite_samples(valid_files)
    dropped_test = []
    if test_files:
        test_files, dropped_test = filter_finite_samples(test_files)
    train_files, dropped_train_tie = filter_samples_by_best_tie_count(
        train_files,
        args.max_best_tie_count,
        use_semantic_features=args.semantic_features,
        semantic_dim=semantic_dim,
        candidate_scheme=args.candidate_scheme,
    )
    valid_files, dropped_valid_tie = filter_samples_by_best_tie_count(
        valid_files,
        args.max_best_tie_count,
        use_semantic_features=args.semantic_features,
        semantic_dim=semantic_dim,
        candidate_scheme=args.candidate_scheme,
    )
    dropped_test_tie = []
    if test_files:
        test_files, dropped_test_tie = filter_samples_by_best_tie_count(
            test_files,
            args.max_best_tie_count,
            use_semantic_features=args.semantic_features,
            semantic_dim=semantic_dim,
            candidate_scheme=args.candidate_scheme,
        )
    if not train_files:
        if args.max_best_tie_count > 0:
            raise RuntimeError(
                "No training samples left after tie filtering. Regenerate samples with "
                "02_generate_dataset.py using a less strict --max_best_tie_count, or pass "
                "--max_best_tie_count 0 to disable the tie filter."
            )
        raise RuntimeError("All training samples were dropped due to non-finite values.")
    if not valid_files:
        if args.max_best_tie_count > 0:
            raise RuntimeError(
                "No validation samples left after tie filtering. Regenerate samples with "
                "02_generate_dataset.py using a less strict --max_best_tie_count, or pass "
                "--max_best_tie_count 0 to disable the tie filter."
            )
        raise RuntimeError("All validation samples were dropped due to non-finite values.")
    train_sampling_probs, train_instance_counts = build_inverse_instance_sampling_probs(train_files)
    train_file_array = np.asarray(train_files, dtype=object)

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    log(f"seed: {args.seed}", logfile)
    log(f"device: {device}", logfile)
    log(f"train samples: {len(train_files)}", logfile)
    log(f"valid samples: {len(valid_files)}", logfile)
    log(f"test samples: {len(test_files)}", logfile)
    log(f"dropped train samples (non-finite): {len(dropped_train)}", logfile)
    log(f"dropped valid samples (non-finite): {len(dropped_valid)}", logfile)
    log(f"dropped test samples (non-finite): {len(dropped_test)}", logfile)
    if args.max_best_tie_count > 0:
        log(f"dropped train samples (best-tie>{args.max_best_tie_count}): {len(dropped_train_tie)}", logfile)
        log(f"dropped valid samples (best-tie>{args.max_best_tie_count}): {len(dropped_valid_tie)}", logfile)
        log(f"dropped test samples (best-tie>{args.max_best_tie_count}): {len(dropped_test_tie)}", logfile)
    else:
        log("best-tie filter: disabled", logfile)
    log(f"max_epochs: {args.max_epochs}", logfile)
    log(f"epoch_size: {args.epoch_size}", logfile)
    log(f"batch_size: {args.batch_size}", logfile)
    log(f"pretrain_batch_size: {args.pretrain_batch_size}", logfile)
    log(f"valid_batch_size: {args.valid_batch_size}", logfile)
    log(f"lr: {args.lr}", logfile)
    log(f"entropy_bonus: {args.entropy_bonus}", logfile)
    log(f"loss_mode: {args.loss_mode}", logfile)
    log(f"selection_metric: {args.selection_metric}", logfile)
    log(f"show_entropy: {args.show_entropy}", logfile)
    log(f"test_eval: {args.test_eval}", logfile)
    log(f"emb_size: {args.emb_size}", logfile)
    log(f"norm_type: {args.norm_type}", logfile)
    log(f"mp_rounds: {args.mp_rounds}", logfile)
    log(f"share_mp_weights: {args.share_mp_weights}", logfile)
    log(f"semantic_features: {int(args.semantic_features)}", logfile)
    log(f"candidate_scheme: {args.candidate_scheme}", logfile)
    log(f"max_best_tie_count: {args.max_best_tie_count}", logfile)
    log(f"var_nfeats: {var_nfeats}", logfile)
    log(f"top_k: {top_k}", logfile)
    log(f"train unique instances: {len(train_instance_counts)}", logfile)
    log(
        f"train instance sample count min/max: {min(train_instance_counts.values())}/"
        f"{max(train_instance_counts.values())}",
        logfile,
    )
    for split_name, split_files in (("train", train_files), ("valid", valid_files), ("test", test_files)):
        if not split_files:
            continue
        label_stats = summarize_label_quality(
            split_files,
            use_semantic_features=args.semantic_features,
            semantic_dim=semantic_dim,
            candidate_scheme=args.candidate_scheme,
        )
        log(
            f"{split_name.upper()} LABELS ({args.candidate_scheme}): samples={label_stats['samples']} "
            f"unique_instances={label_stats['unique_instances']} "
            f"exact_tie_rate={label_stats['exact_tie_rate']:.4f} "
            f"tieaware_tie_rate={label_stats['tieaware_tie_rate']:.4f} "
            f"avg_exact_best_tie={label_stats['avg_exact_best_tie_count']:.2f} "
            f"avg_tieaware_best_tie={label_stats['avg_tieaware_best_tie_count']:.2f} "
            f"max_tieaware_best_tie={label_stats['max_tieaware_best_tie_count']} "
            f"avg_candidates={label_stats['avg_candidates']:.2f} "
            f"best_at_clip_ceiling={label_stats['best_at_clip_ceiling_rate']:.4f}",
            logfile,
        )

    policy = GNNPolicy(
        emb_size=args.emb_size,
        norm_type=args.norm_type,
        var_nfeats=var_nfeats,
        mp_rounds=args.mp_rounds,
        share_mp_weights=args.share_mp_weights,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.patience,
        factor=0.2,
    )

    pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]
    if not pretrain_files:
        pretrain_files = train_files[: min(len(train_files), args.pretrain_batch_size)]
    pretrain_data = GraphDataset(
        pretrain_files,
        use_semantic_features=args.semantic_features,
        semantic_dim=semantic_dim,
        candidate_scheme=args.candidate_scheme,
    )
    pretrain_loader = torch_geometric.loader.DataLoader(
        pretrain_data,
        batch_size=args.pretrain_batch_size,
        shuffle=False,
    )

    valid_data = GraphDataset(
        valid_files,
        use_semantic_features=args.semantic_features,
        semantic_dim=semantic_dim,
        candidate_scheme=args.candidate_scheme,
    )
    valid_loader = torch_geometric.loader.DataLoader(
        valid_data,
        batch_size=args.valid_batch_size,
        shuffle=False,
    )
    run_test_each_epoch = args.test_eval == "each_epoch" and len(test_files) > 0
    run_test_final = args.test_eval in ("final", "each_epoch") and len(test_files) > 0
    if run_test_each_epoch or run_test_final:
        test_data = GraphDataset(
            test_files,
            use_semantic_features=args.semantic_features,
            semantic_dim=semantic_dim,
            candidate_scheme=args.candidate_scheme,
        )
        test_loader = torch_geometric.loader.DataLoader(
            test_data,
            batch_size=args.valid_batch_size,
            shuffle=False,
        )
    else:
        test_loader = None
        if args.test_eval != "none" and test_dir.exists():
            log(f"No usable test samples found in {test_dir}; final test evaluation will be skipped.", logfile)
        elif args.test_eval != "none":
            log(f"Test directory not found at {test_dir}; final test evaluation will be skipped.", logfile)

    best_loss = float("inf")
    best_acc = -float("inf")
    best_epoch = None
    plateau_count = 0

    for epoch in range(args.max_epochs + 1):
        log(f"EPOCH {epoch}", logfile)

        if epoch == 0:
            if args.norm_type == "prenorm":
                n_layers = pretrain(policy, pretrain_loader, device)
            else:
                n_layers = 0
            log(f"PRETRAINED LAYERS: {n_layers}", logfile)
        else:
            draw_n = max(args.epoch_size, args.batch_size)
            epoch_train_files = rng.choice(train_file_array, draw_n, replace=True, p=train_sampling_probs)
            train_data = GraphDataset(
                list(epoch_train_files),
                use_semantic_features=args.semantic_features,
                semantic_dim=semantic_dim,
                candidate_scheme=args.candidate_scheme,
            )
            train_loader = torch_geometric.loader.DataLoader(
                train_data,
                batch_size=args.batch_size,
                shuffle=True,
            )
            train_loss, train_kacc, train_entropy = process_epoch(
                policy,
                train_loader,
                device,
                top_k=top_k,
                entropy_bonus=args.entropy_bonus,
                optimizer=optimizer,
                loss_mode=args.loss_mode,
            )
            train_acc = float(train_kacc[0])
            topk_msg = " ".join([f"acc@{k}: {acc:.3f}" for k, acc in zip(top_k, train_kacc)])
            if args.show_entropy:
                log(
                    f"TRAIN LOSS: {train_loss:.4f} TRAIN ACC: {train_acc:.3f} "
                    f"ENTROPY: {train_entropy:.4f} {topk_msg}",
                    logfile,
                )
            else:
                log(
                    f"TRAIN LOSS: {train_loss:.4f} TRAIN ACC: {train_acc:.3f} {topk_msg}",
                    logfile,
                )

        valid_loss, valid_kacc, valid_entropy = process_epoch(
            policy,
            valid_loader,
            device,
            top_k=top_k,
            entropy_bonus=args.entropy_bonus,
            optimizer=None,
            loss_mode=args.loss_mode,
        )
        valid_acc = float(valid_kacc[0])
        valid_topk_msg = " ".join([f"acc@{k}: {acc:.3f}" for k, acc in zip(top_k, valid_kacc)])
        if args.show_entropy:
            log(
                f"VALID LOSS: {valid_loss:.4f} VALID ACC: {valid_acc:.3f} "
                f"ENTROPY: {valid_entropy:.4f} {valid_topk_msg}",
                logfile,
            )
        else:
            log(
                f"VALID LOSS: {valid_loss:.4f} VALID ACC: {valid_acc:.3f} {valid_topk_msg}",
                logfile,
            )
        if run_test_each_epoch and test_loader is not None:
            test_loss, test_kacc, test_entropy = process_epoch(
                policy,
                test_loader,
                device,
                top_k=top_k,
                entropy_bonus=args.entropy_bonus,
                optimizer=None,
                loss_mode=args.loss_mode,
            )
            test_acc = float(test_kacc[0])
            test_topk_msg = " ".join([f"acc@{k}: {acc:.3f}" for k, acc in zip(top_k, test_kacc)])
            if args.show_entropy:
                log(
                    f"TEST LOSS: {test_loss:.4f} TEST ACC: {test_acc:.3f} "
                    f"ENTROPY: {test_entropy:.4f} {test_topk_msg}",
                    logfile,
                )
            else:
                log(
                    f"TEST LOSS: {test_loss:.4f} TEST ACC: {test_acc:.3f} {test_topk_msg}",
                    logfile,
                )

        scheduler.step(valid_loss)

        if args.selection_metric == "loss":
            improved = valid_loss < best_loss
        else:
            improved = valid_acc > best_acc or (valid_acc == best_acc and valid_loss < best_loss)

        if improved:
            best_loss = valid_loss
            best_acc = valid_acc
            best_epoch = epoch
            plateau_count = 0
            torch.save(policy.state_dict(), os.path.join(run_dir, "best_params.pkl"))
            log(f"best model so far ({args.selection_metric})", logfile)
        else:
            plateau_count += 1
            if plateau_count >= args.early_stopping:
                log(f"early stopping at epoch {epoch}", logfile)
                break

    best_path = os.path.join(run_dir, "best_params.pkl")
    if not os.path.exists(best_path):
        raise RuntimeError("No best model checkpoint was saved.")

    try:
        state_dict = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(best_path, map_location=device)
    policy.load_state_dict(state_dict)
    final_valid_loss, final_valid_kacc, final_valid_entropy = process_epoch(
        policy,
        valid_loader,
        device,
        top_k=top_k,
        entropy_bonus=args.entropy_bonus,
        optimizer=None,
        loss_mode=args.loss_mode,
    )
    final_valid_acc = float(final_valid_kacc[0])
    final_topk_msg = " ".join([f"acc@{k}: {acc:.3f}" for k, acc in zip(top_k, final_valid_kacc)])
    if args.show_entropy:
        log(
            f"BEST VALID (epoch={best_epoch}) LOSS: {final_valid_loss:.4f} "
            f"VALID ACC: {final_valid_acc:.3f} ENTROPY: {final_valid_entropy:.4f} {final_topk_msg}",
            logfile,
        )
    else:
        log(
            f"BEST VALID (epoch={best_epoch}) LOSS: {final_valid_loss:.4f} "
            f"VALID ACC: {final_valid_acc:.3f} {final_topk_msg}",
            logfile,
        )
    if run_test_final and test_loader is not None:
        final_test_loss, final_test_kacc, final_test_entropy = process_epoch(
            policy,
            test_loader,
            device,
            top_k=top_k,
            entropy_bonus=args.entropy_bonus,
            optimizer=None,
            loss_mode=args.loss_mode,
        )
        final_test_acc = float(final_test_kacc[0])
        final_test_topk_msg = " ".join([f"acc@{k}: {acc:.3f}" for k, acc in zip(top_k, final_test_kacc)])
        if args.show_entropy:
            log(
                f"BEST TEST (epoch={best_epoch}) LOSS: {final_test_loss:.4f} "
                f"TEST ACC: {final_test_acc:.3f} ENTROPY: {final_test_entropy:.4f} {final_test_topk_msg}",
                logfile,
            )
        else:
            log(
                f"BEST TEST (epoch={best_epoch}) LOSS: {final_test_loss:.4f} "
                f"TEST ACC: {final_test_acc:.3f} {final_test_topk_msg}",
                logfile,
            )
    log(f"Saved checkpoint: {best_path}", logfile)


if __name__ == "__main__":
    main()
