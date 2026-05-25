"""
02_generate_dataset.py
Collect branching-decision samples from RH MILP instances using Ecole.

Each saved sample follows the Ecole/learn2branch-ecole schema:
    {
        "episode": int,
        "instance": str,
        "seed": int,
        "data": [node_observation, action, action_set, scores],
    }

where:
    node_observation = (
        row_features,
        (edge_indices, edge_values),
        variable_features,
    )
"""

import argparse
import glob
import gzip
import json
import multiprocessing as mp
import os
import pickle
import queue
import re
import shutil
import time as _time
from collections import Counter

import numpy as np

try:
    import ecole
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Ecole is not installed in this Python environment. "
        "Activate the correct environment first, e.g.:\n"
        "  conda activate ecole_gpu"
    ) from exc

try:
    from pyscipopt import SCIP_PARAMSETTING
except Exception:  # pragma: no cover - PySCIPOpt is environment-dependent
    SCIP_PARAMSETTING = None


UNSUPPORTED_HEURISTIC_PARAMS = {
    # These appear in some PySCIPOpt builds but not in the SCIP build used by Ecole.
    "heuristics/dks/freq",
    "heuristics/indicatordiving/freq",
    "heuristics/scheduler/freq",
}


def _load_heuristics_off_overrides():
    """Load safe SCIP parameter overrides that disable primal heuristics."""
    path = os.path.join("tools", "scip_heuristics_off_params.json")
    if not os.path.exists(path):
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
    fullstrong=True,
    heuristics_off=True,
    fullstrong_scoreall=True,
    fullstrong_donotbranch=True,
):
    """SCIP profile used for branching-only imitation experiments."""
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
    """Light SCIP profile used for the deployment-style branching baseline.

    Only the three core settings are fixed: separating/maxrounds=0,
    presolving/maxrestarts=0, limits/time. Heuristics, propagation, presolving
    rounds, and non-root separating stay enabled to match the --four_way
    evaluation environment.
    """
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


class AlwaysStrongBranch:
    """Return full strong branching scores at every branching decision."""

    def __init__(self, *, heuristics_off=True, pyscipopt_heuristics_off=False):
        self.heuristics_off = bool(heuristics_off)
        self.pyscipopt_heuristics_off = bool(pyscipopt_heuristics_off)
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        if self.heuristics_off and self.pyscipopt_heuristics_off:
            _disable_heuristics_via_pyscipopt(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        return self.strong_branching_function.extract(model, done), True


def _clean_strong_branching_scores(scores, transform="raw"):
    """Return finite scores while preserving finite candidate ordering.

    The previous implementation clipped all finite scores above 1e8. RH MILPs can
    produce many valid strong-branching scores beyond that value, and clipping
    collapses them into artificial best-score ties. A monotone transform is safe
    because downstream training only uses the ordering/ties of candidate scores.
    """
    scores64 = np.asarray(scores, dtype=np.float64)
    finite_mask = np.isfinite(scores64)
    cleaned = np.zeros_like(scores64, dtype=np.float64)

    if finite_mask.any():
        finite_scores = scores64[finite_mask]
        if transform == "raw":
            finite_clean = finite_scores
        elif transform == "signed_log1p":
            finite_clean = np.sign(finite_scores) * np.log1p(np.abs(finite_scores))
        else:
            raise ValueError(f"Unknown score transform: {transform}")

        cleaned[finite_mask] = finite_clean
        finite_min = float(np.min(finite_clean))
        finite_max = float(np.max(finite_clean))
        span = max(1.0, abs(finite_min), abs(finite_max))
        margin = span * 1e-6
        nan_fill = finite_min - margin
        posinf_fill = finite_max + margin
        neginf_fill = finite_min - margin
    else:
        nan_fill = 0.0
        posinf_fill = 1.0
        neginf_fill = -1.0

    cleaned[np.isnan(scores64)] = nan_fill
    cleaned[np.isposinf(scores64)] = posinf_fill
    cleaned[np.isneginf(scores64)] = neginf_fill
    return cleaned


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _load_pickle_auto(path):
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def _max_sample_index_in_dir(out_dir):
    max_idx = 0
    for path in glob.glob(os.path.join(out_dir, "sample_*.pkl")):
        match = re.search(r"sample_(\d+)\.pkl$", os.path.basename(path))
        if match is not None:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx


def _extract_instance_index(path):
    match = re.search(r"instance_(\d+)\.lp$", os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1))


def _load_latest_append_range(instance_dir):
    range_path = os.path.join(instance_dir, "latest_append_range.json")
    if not os.path.exists(range_path):
        raise FileNotFoundError(
            f"--only_new_instances was set but range file was not found: {range_path}"
        )
    with open(range_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_instances = int(data.get("new_instances", 0))
    start = data.get("new_instance_start")
    end = data.get("new_instance_end")
    if new_instances <= 0 or start is None or end is None:
        raise RuntimeError(
            f"--only_new_instances requested, but range in {range_path} is empty "
            f"(new_instances={new_instances}, start={start}, end={end})."
        )
    start = int(start)
    end = int(end)
    if end < start:
        raise RuntimeError(f"Invalid range in {range_path}: start={start}, end={end}")

    return {
        "path": range_path,
        "start": start,
        "end": end,
        "new_instances": new_instances,
    }


def _filter_instances_by_index_range(lp_files, start_idx, end_idx):
    filtered = []
    for path in lp_files:
        idx = _extract_instance_index(path)
        if idx is None:
            continue
        if start_idx <= idx <= end_idx:
            filtered.append(path)
    return filtered


class ExploreThenStrongBranch:
    """Randomly return pseudocost or strong-branching scores."""

    def __init__(self, expert_probability, *, heuristics_off=True, pyscipopt_heuristics_off=False):
        if not 0.0 <= expert_probability <= 1.0:
            raise ValueError("expert_probability must be in [0, 1]")
        self.expert_probability = expert_probability
        self.heuristics_off = bool(heuristics_off)
        self.pyscipopt_heuristics_off = bool(pyscipopt_heuristics_off)
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        if self.heuristics_off and self.pyscipopt_heuristics_off:
            _disable_heuristics_via_pyscipopt(model)
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        probabilities = [1.0 - self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return self.strong_branching_function.extract(model, done), True
        return self.pseudocosts_function.extract(model, done), False


def send_orders(
    orders_queue,
    instances,
    seed,
    stop_flag,
    dispatch_mode="random",
):
    """Continuously dispatch sampling episodes to workers."""
    rng = np.random.RandomState(seed)
    instances = list(instances)
    episode = 0
    cycle_order = []
    cycle_index = 0
    while not stop_flag.is_set():
        if dispatch_mode == "cycle":
            if cycle_index >= len(cycle_order):
                cycle_order = list(instances)
                rng.shuffle(cycle_order)
                cycle_index = 0
            instance = cycle_order[cycle_index]
            cycle_index += 1
        else:
            instance = rng.choice(instances)

        episode_seed = int(rng.randint(2**32))
        order = [episode, instance, episode_seed]
        try:
            orders_queue.put(order, timeout=1.0)
            episode += 1
        except queue.Full:
            continue


def make_samples(
    in_queue,
    out_queue,
    stop_flag,
    query_expert_prob,
    time_limit,
    out_dir,
    compress,
    error_log_path,
    finite_candidates_only,
    min_candidates,
    fullstrong,
    heuristics_off,
    pyscipopt_heuristics_off,
    vanillafullstrong_scip_params,
    fullstrong_scoreall,
    fullstrong_donotbranch,
    pseudo_candidates,
    score_transform,
    max_samples_per_episode=0,
    max_best_tie_count=0,
    worker_logs=False,
    example_scip_params=False,
):
    """Worker loop: solve one episode and emit sample events."""
    total_sample_counter = 0
    if example_scip_params:
        # Light deployment profile: the same core SCIP settings used by the
        # default/GNN four-way evaluation.
        scip_parameters = build_example_scip_params(
            time_limit,
            fullstrong=bool(fullstrong or vanillafullstrong_scip_params),
            fullstrong_scoreall=bool(fullstrong_scoreall),
            fullstrong_donotbranch=bool(fullstrong_donotbranch),
        )
    else:
        scip_parameters = build_branching_only_scip_params(
            time_limit,
            fullstrong=bool(fullstrong or vanillafullstrong_scip_params),
            heuristics_off=bool(heuristics_off),
            fullstrong_scoreall=bool(fullstrong_scoreall),
            fullstrong_donotbranch=bool(fullstrong_donotbranch),
        )
    scores_observation = (
        AlwaysStrongBranch(
            heuristics_off=heuristics_off,
            pyscipopt_heuristics_off=pyscipopt_heuristics_off,
        )
        if fullstrong
        else ExploreThenStrongBranch(
            expert_probability=query_expert_prob,
            heuristics_off=heuristics_off,
            pyscipopt_heuristics_off=pyscipopt_heuristics_off,
        )
    )
    observation_function = {
        "scores": scores_observation,
        "node_observation": ecole.observation.NodeBipartite(),
    }
    env = ecole.environment.Branching(
        observation_function=observation_function,
        scip_params=scip_parameters,
        pseudo_candidates=bool(pseudo_candidates),
    )

    while not stop_flag.is_set():
        try:
            episode, instance, seed = in_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        episode_sample_counter = 0
        if worker_logs:
            print(
                f"[w {os.getpid()}] start ep={episode} seed={seed} "
                f"instance='{os.path.basename(instance)}'"
            )
        out_queue.put(
            {
                "type": "start",
                "episode": episode,
                "instance": instance,
                "seed": seed,
            }
        )

        env.seed(seed)
        skipped_small_candidate_set = 0
        skipped_nonfinite_candidates = 0
        skipped_action_not_in_set = 0
        skipped_best_tie = 0
        try:
            observation, action_set, _, done, _ = env.reset(instance)
            if heuristics_off and pyscipopt_heuristics_off:
                _disable_heuristics_via_pyscipopt(env)
            while not done and not stop_flag.is_set():
                scores, scores_are_expert = observation["scores"]

                action_set_all = np.asarray(action_set, dtype=np.int32)
                raw_scores_array = np.asarray(scores, dtype=np.float64)
                scores_array = _clean_strong_branching_scores(raw_scores_array, transform=score_transform)
                candidate_scores = scores_array[action_set_all]
                finite_candidate_mask = np.isfinite(raw_scores_array[action_set_all])
                if finite_candidate_mask.any():
                    finite_action_set = action_set_all[finite_candidate_mask]
                    finite_candidate_scores = candidate_scores[finite_candidate_mask]
                    action = int(finite_action_set[np.argmax(finite_candidate_scores)])
                else:
                    action = int(action_set_all[0])

                if scores_are_expert and not stop_flag.is_set():
                    if finite_candidates_only:
                        sample_action_set = action_set_all[finite_candidate_mask]
                    else:
                        sample_action_set = action_set_all

                    if sample_action_set.size < min_candidates:
                        skipped_small_candidate_set += 1
                    else:
                        sample_scores = scores_array[sample_action_set]
                        if not np.isfinite(sample_scores).all():
                            skipped_nonfinite_candidates += 1
                        elif np.where(sample_action_set == action)[0].size == 0:
                            skipped_action_not_in_set += 1
                        elif (
                            max_best_tie_count > 0
                            and int(
                                np.isclose(
                                    sample_scores,
                                    sample_scores.max(),
                                    rtol=1e-6,
                                    atol=1e-12,
                                ).sum()
                            )
                            > max_best_tie_count
                        ):
                            skipped_best_tie += 1
                        else:
                            node_observation = observation["node_observation"]
                            node_observation = (
                                node_observation.row_features,
                                (
                                    node_observation.edge_features.indices,
                                    node_observation.edge_features.values,
                                ),
                                node_observation.variable_features,
                            )
                            # Keep tensor shape stable while preventing NaN/inf payloads.
                            with np.errstate(over="ignore", invalid="ignore"):
                                scores_clean = scores_array.astype(np.float32, copy=False)
                            if not np.isfinite(scores_clean[sample_action_set]).all():
                                skipped_nonfinite_candidates += 1
                                continue
                            data = [node_observation, int(action), sample_action_set, scores_clean]
                            filename = f"{out_dir}/sample_{episode}_{episode_sample_counter}.pkl"
                            payload = {
                                "episode": episode,
                                "instance": instance,
                                "seed": seed,
                                "data": data,
                            }
                            if compress:
                                with gzip.open(filename, "wb") as f:
                                    pickle.dump(payload, f)
                            else:
                                with open(filename, "wb") as f:
                                    pickle.dump(payload, f)

                            out_queue.put(
                                {
                                    "type": "sample",
                                    "episode": episode,
                                    "instance": instance,
                                    "seed": seed,
                                    "filename": filename,
                                }
                            )
                            episode_sample_counter += 1
                            total_sample_counter += 1
                            if (
                                max_samples_per_episode > 0
                                and episode_sample_counter >= max_samples_per_episode
                            ):
                                break

                observation, action_set, _, done, _ = env.step(action)
        except Exception as exc:
            with open(error_log_path, "a", encoding="utf-8") as log_f:
                log_f.write(f"Error occurred solving {instance} with seed {seed}\n")
                log_f.write(f"{type(exc).__name__}: {exc}\n")

        if worker_logs:
            print(
                f"[w {os.getpid()}] done ep={episode} "
                f"episode_samples={episode_sample_counter} "
                f"skipped_small_candidate_set={skipped_small_candidate_set} "
                f"skipped_nonfinite_candidates={skipped_nonfinite_candidates} "
                f"skipped_action_not_in_set={skipped_action_not_in_set} "
                f"skipped_best_tie={skipped_best_tie} "
                f"worker_total={total_sample_counter}"
            )
        out_queue.put(
            {
                "type": "done",
                "episode": episode,
                "instance": instance,
                "seed": seed,
                "episode_samples": episode_sample_counter,
            }
        )


def validate_split_samples(out_dir, min_candidates):
    sample_files = sorted(glob.glob(os.path.join(out_dir, "sample_*.pkl")))
    stats = {
        "total_samples": len(sample_files),
        "action_not_in_action_set": 0,
        "nonfinite_candidate_scores": 0,
        "below_min_candidates": 0,
        "bad_candidate_indices": 0,
        "tie_rate": 0.0,
        "all_tied_rate": 0.0,
        "avg_candidate_count": 0.0,
        "avg_best_tie_count": 0.0,
        "max_best_tie_count": 0,
        "avg_unique_scores": 0.0,
        "avg_score_span": 0.0,
        "unique_instances": 0,
        "instance_counts": Counter(),
    }
    tie_count = 0
    all_tied_count = 0
    checked = 0
    candidate_counts = []
    best_tie_counts = []
    unique_score_counts = []
    score_spans = []

    for path in sample_files:
        sample = _load_pickle_auto(path)
        _, sample_action, sample_action_set, sample_scores = sample["data"]
        action_set = np.asarray(sample_action_set, dtype=np.int64)
        scores = np.asarray(sample_scores)
        stats["instance_counts"][sample["instance"]] += 1

        if action_set.size < min_candidates:
            stats["below_min_candidates"] += 1
        if np.where(action_set == int(sample_action))[0].size == 0:
            stats["action_not_in_action_set"] += 1
        if action_set.size == 0 or np.any(action_set < 0) or np.any(action_set >= scores.shape[0]):
            stats["bad_candidate_indices"] += 1
            continue

        candidate_scores = scores[action_set]
        if not np.isfinite(candidate_scores).all():
            stats["nonfinite_candidate_scores"] += 1
            continue
        best_score = candidate_scores.max()
        best_tie_count = int(np.isclose(candidate_scores, best_score, rtol=1e-6, atol=1e-12).sum())
        tie_count += int(best_tie_count > 1)
        all_tied_count += int(best_tie_count == action_set.size)
        candidate_counts.append(int(action_set.size))
        best_tie_counts.append(best_tie_count)
        unique_score_counts.append(int(np.unique(candidate_scores).size))
        score_spans.append(float(candidate_scores.max() - candidate_scores.min()))
        checked += 1

    stats["tie_rate"] = tie_count / checked if checked > 0 else 0.0
    stats["all_tied_rate"] = all_tied_count / checked if checked > 0 else 0.0
    stats["avg_candidate_count"] = float(np.mean(candidate_counts)) if candidate_counts else 0.0
    stats["avg_best_tie_count"] = float(np.mean(best_tie_counts)) if best_tie_counts else 0.0
    stats["max_best_tie_count"] = int(np.max(best_tie_counts)) if best_tie_counts else 0
    stats["avg_unique_scores"] = float(np.mean(unique_score_counts)) if unique_score_counts else 0.0
    stats["avg_score_span"] = float(np.mean(score_spans)) if score_spans else 0.0
    stats["unique_instances"] = len(stats["instance_counts"])
    return stats


def _compress_one_file(item):
    """Worker: compress a single .pkl file in place. item = (path, compresslevel)."""
    path, compresslevel = item
    with open(path, "rb") as src_f:
        magic = src_f.read(2)
    if magic == b"\x1f\x8b":
        return path, 0
    tmp_path = f"{path}.tmp.gz"
    try:
        with open(path, "rb") as src_f, gzip.open(tmp_path, "wb", compresslevel=compresslevel) as dst_f:
            shutil.copyfileobj(src_f, dst_f, length=1024 * 1024)
        os.replace(tmp_path, path)
        return path, 1
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def gzip_split_samples_in_place(out_dir, compresslevel=1, log_every=2000, n_jobs=1):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    sample_files = sorted(glob.glob(os.path.join(out_dir, "sample_*.pkl")))
    total = len(sample_files)
    if total == 0:
        return {"total_files": 0, "compressed_files": 0}

    if n_jobs <= 1:
        # Original single-threaded path
        compressed = 0
        for i, path in enumerate(sample_files, start=1):
            with open(path, "rb") as src_f:
                magic = src_f.read(2)
            if magic == b"\x1f\x8b":
                continue
            tmp_path = f"{path}.tmp.gz"
            try:
                with open(path, "rb") as src_f, gzip.open(tmp_path, "wb", compresslevel=compresslevel) as dst_f:
                    shutil.copyfileobj(src_f, dst_f, length=1024 * 1024)
                os.replace(tmp_path, path)
                compressed += 1
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            if log_every > 0 and (compressed == 1 or compressed % log_every == 0 or i == total):
                print(f"[m {os.getpid()}] post-compressed {compressed}/{total} files")
        return {"total_files": total, "compressed_files": compressed}

    # Parallel path
    work = [(p, compresslevel) for p in sample_files]
    compressed = 0
    done = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(_compress_one_file, w): w[0] for w in work}
        for fut in as_completed(futures):
            _, n = fut.result()
            compressed += n
            done += 1
            if log_every > 0 and (done == 1 or done % log_every == 0 or done == total):
                print(f"[m {os.getpid()}] post-compressed {done}/{total} files ({compressed} new)")
    return {"total_files": total, "compressed_files": compressed}


def collect_samples(
    instances,
    out_dir,
    rng,
    n_samples,
    n_jobs,
    query_expert_prob,
    time_limit,
    mp_ctx,
    compress=True,
    log_every=25,
    worker_logs=False,
    example_logs=False,
    finite_candidates_only=True,
    min_candidates=2,
    append_samples=False,
    fullstrong=True,
    heuristics_off=True,
    pyscipopt_heuristics_off=False,
    vanillafullstrong_scip_params=True,
    fullstrong_scoreall=True,
    fullstrong_donotbranch=True,
    pseudo_candidates=False,
    score_transform="raw",
    dispatch_mode="random",
    max_samples_per_episode=0,
    max_best_tie_count=0,
    example_scip_params=False,
):
    """Collect *n_samples* Ecole-format samples from a set of instances."""
    os.makedirs(out_dir, exist_ok=True)
    existing_before = _max_sample_index_in_dir(out_dir)
    if not append_samples:
        for old_sample in glob.glob(os.path.join(out_dir, "sample_*.pkl")):
            os.remove(old_sample)
        existing_before = 0

    orders_queue = mp_ctx.Queue(maxsize=max(2 * n_jobs, 1))
    answers_queue = mp_ctx.SimpleQueue()

    tmp_samples_dir = f"{out_dir}/tmp"
    os.makedirs(tmp_samples_dir, exist_ok=True)
    error_log_path = os.path.join(out_dir, "error_log.txt")

    dispatcher_stop_flag = mp_ctx.Event()
    dispatcher = mp_ctx.Process(
        target=send_orders,
        args=(
            orders_queue,
            instances,
            int(rng.randint(2**32)),
            dispatcher_stop_flag,
            dispatch_mode,
        ),
        daemon=True,
    )
    dispatcher.start()

    workers = []
    workers_stop_flag = mp_ctx.Event()
    for _ in range(n_jobs):
        worker = mp_ctx.Process(
            target=make_samples,
            args=(
                orders_queue,
                answers_queue,
                workers_stop_flag,
                query_expert_prob,
                time_limit,
                tmp_samples_dir,
                compress,
                error_log_path,
                finite_candidates_only,
                min_candidates,
                fullstrong,
                heuristics_off,
                pyscipopt_heuristics_off,
                vanillafullstrong_scip_params,
                fullstrong_scoreall,
                fullstrong_donotbranch,
                pseudo_candidates,
                score_transform,
                max_samples_per_episode,
                max_best_tie_count,
                worker_logs,
                example_scip_params,
            ),
            daemon=True,
        )
        workers.append(worker)
        worker.start()

    sample_index = 0
    instance_written = Counter()
    stream_episode = None
    while sample_index < n_samples:
        message = answers_queue.get()
        msg_type = message["type"]

        if msg_type == "done":
            continue
        if msg_type == "start":
            if example_logs:
                episode = message["episode"]
                if stream_episode is None:
                    stream_episode = episode
                elif episode != stream_episode:
                    print(f"Episode {stream_episode}, {sample_index} samples collected so far")
                    stream_episode = episode
            continue

        sample_index += 1
        destination = f"{out_dir}/sample_{existing_before + sample_index}.pkl"
        os.replace(message["filename"], destination)
        instance_written[message["instance"]] += 1

        if log_every > 0 and (
            sample_index == 1
            or sample_index == n_samples
            or sample_index % log_every == 0
        ):
            print(
                f"[m {os.getpid()}] wrote {existing_before + sample_index}/{existing_before + n_samples} "
                f"(+{sample_index}/{n_samples}) "
                f"from ep={message['episode']} instance='{os.path.basename(message['instance'])}'"
            )

    if example_logs and stream_episode is not None:
        print(f"Episode {stream_episode}, {sample_index} samples collected so far")

    dispatcher_stop_flag.set()
    workers_stop_flag.set()

    dispatcher.join(timeout=10)
    if dispatcher.is_alive():
        dispatcher.terminate()
        dispatcher.join(timeout=5)
    for worker in workers:
        worker.join(timeout=10)
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)

    print(f"Done collecting samples for {out_dir}")
    shutil.rmtree(tmp_samples_dir, ignore_errors=True)

    return {
        "n_collected": sample_index,
        "instance_written": instance_written,
        "n_existing_before": existing_before,
        "n_total_after": existing_before + sample_index,
    }


def main():
    default_jobs = os.cpu_count()
    if default_jobs is None or default_jobs < 1:
        default_jobs = 4
    parser = argparse.ArgumentParser(
        description="Collect branching-decision samples from RH MILP instances using Ecole."
    )
    parser.add_argument("--seed", type=int, default=0)

    # Run only one split (for parallel pipeline: run 3 processes with --split train/valid/test).
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid", "test"],
        default=None,
        help="If set, process only this split; otherwise process train, valid, test in sequence.",
    )

    # New split-size interface (default for recovery plan).
    parser.add_argument("--samples_per_split_train", type=int, default=100000)
    parser.add_argument("--samples_per_split_valid", type=int, default=20000)
    parser.add_argument("--samples_per_split_test", type=int, default=20000)
    parser.add_argument(
        "--instance_base",
        type=str,
        default=os.path.join("data", "instances", "rh_milp"),
        help="Base directory containing train_mixed/valid_mixed/test_mixed instance folders.",
    )
    parser.add_argument(
        "--sample_base",
        type=str,
        default=os.path.join("data", "samples", "rh_milp"),
        help="Base directory where train/valid/test sample folders are written.",
    )

    # Backward-compatible aliases.
    parser.add_argument("--train_size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--valid_size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--test_size", type=int, default=None, help=argparse.SUPPRESS)

    parser.add_argument(
        "--expert_prob",
        type=float,
        default=0.05,
        help=(
            "Probability of querying expert per node in --no_fullstrong legacy mode "
            "(default 0.05). Ignored by the default fullstrong mode."
        ),
    )
    fullstrong_group = parser.add_mutually_exclusive_group()
    fullstrong_group.add_argument(
        "--fullstrong",
        dest="fullstrong",
        action="store_true",
        help="Collect labels on a vanilla full strong branching trajectory (default).",
    )
    fullstrong_group.add_argument(
        "--no_fullstrong",
        dest="fullstrong",
        action="store_false",
        help="Use the legacy explore-then-strong-branch collection scheme.",
    )
    parser.set_defaults(fullstrong=True)
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
    vfsb_params_group = parser.add_mutually_exclusive_group()
    vfsb_params_group.add_argument(
        "--vanillafullstrong_scip_params",
        dest="vanillafullstrong_scip_params",
        action="store_true",
        help=(
            "Install vanilla fullstrong SCIP brancher parameters even for explore-then-strongbranch "
            "collection. This matches the reference scripts (default)."
        ),
    )
    vfsb_params_group.add_argument(
        "--no_vanillafullstrong_scip_params",
        dest="vanillafullstrong_scip_params",
        action="store_false",
        help="Only install vanilla fullstrong SCIP brancher parameters when --fullstrong is active.",
    )
    parser.set_defaults(vanillafullstrong_scip_params=True)
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
        help="Expose pseudo candidates to the branching policy.",
    )
    pseudo_group.add_argument(
        "--no_pseudo_candidates",
        dest="pseudo_candidates",
        action="store_false",
        help="Use only SCIP's LP branch candidates (default).",
    )
    parser.set_defaults(pseudo_candidates=False)
    parser.add_argument(
        "--time_limit",
        type=float,
        default=3600.0,
        help="Per-instance SCIP time limit in seconds (default 3600).",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=None,
        help=f"Number of parallel worker processes (default: CPU count = {default_jobs}).",
    )
    parser.add_argument(
        "--sb_itlim",
        type=int,
        default=100_000,
        help="Unused with Ecole API, kept for CLI compatibility.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        choices=["none", "worker", "post"],
        default="post",
        help=(
            "Compression mode for sample payloads: "
            "'none' = plain pickle, "
            "'worker' = gzip during collection, "
            "'post' = collect plain and gzip at the end of each split (default)."
        ),
    )
    parser.add_argument(
        "--post_compress_level",
        type=int,
        default=1,
        help="Gzip level used in --compression=post (0..9, default 1 for speed).",
    )
    parser.add_argument(
        "--post_compress_log_every",
        type=int,
        default=2000,
        help="Post-compression progress log frequency in files (default 2000).",
    )
    parser.add_argument(
        "--post_compress_jobs",
        type=int,
        default=None,
        help=f"Number of parallel workers for post-compression (default: CPU count = {default_jobs}). 1 = single-threaded.",
    )
    parser.add_argument(
        "--append_samples",
        action="store_true",
        help="Append new samples to existing sample_*.pkl files in split directories.",
    )
    parser.add_argument(
        "--only_new_instances",
        action="store_true",
        help=(
            "Use only the latest appended instance range from "
            "<split>/latest_append_range.json (written by 01_generate_rh_instances.py)."
        ),
    )
    parser.add_argument("--no_compress", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Write manager progress log every N samples (default 50, 0 disables).",
    )
    parser.add_argument(
        "--dispatch_mode",
        type=str,
        choices=["random", "cycle"],
        default="random",
        help=(
            "Episode dispatch order. 'cycle' shuffles then visits each instance once "
            "before repeating, improving instance coverage (default random)."
        ),
    )
    parser.add_argument(
        "--max_samples_per_episode",
        type=int,
        default=0,
        help=(
            "If >0, stop an episode after writing this many samples. "
            "Useful with --dispatch_mode cycle to avoid one instance filling the split."
        ),
    )
    parser.add_argument(
        "--max_best_tie_count",
        type=int,
        default=0,
        help=(
            "If >0, save only samples whose best-score tie count is at most this value. "
            "This filters uninformative cutoff-style strong-branching ties. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--worker_logs",
        action="store_true",
        help="Enable per-episode worker start/done logs (off by default).",
    )
    parser.add_argument(
        "--example_logs",
        action="store_true",
        help="Print notebook-style progress log on episode switches.",
    )
    parser.add_argument(
        "--finite_candidates_only",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Save only finite-scored candidates in action_set (default True).",
    )
    parser.add_argument(
        "--min_candidates",
        type=int,
        default=2,
        help="Minimum number of candidates required for a saved sample (default 2).",
    )
    parser.add_argument(
        "--score_transform",
        type=str,
        choices=["raw", "signed_log1p"],
        default="raw",
        help=(
            "Monotone transform applied to expert scores before saving. "
            "'raw' preserves finite scores without clipping; 'signed_log1p' compresses large magnitudes "
            "while preserving ordering."
        ),
    )
    parser.add_argument(
        "--example_scip_params",
        action="store_true",
        help=(
            "Use the light deployment SCIP profile during sample "
            "collection (only separating/maxrounds=0, presolving/maxrestarts=0, limits/time). "
            "This LEAVES heuristics, propagation, presolving rounds, and non-root cuts ON so the "
            "imitation distribution matches the --four_way eval profile. Strongly recommended "
            "for new four-way runs. Implicitly overrides "
            "--heuristics_off (the env profile is what matters; heuristics_off via JSON is unused)."
        ),
    )
    start_methods = mp.get_all_start_methods()
    default_start_method = "fork" if "fork" in start_methods else start_methods[0]
    parser.add_argument(
        "--mp_start_method",
        type=str,
        choices=start_methods,
        default=default_start_method,
        help=f"Multiprocessing start method (default {default_start_method}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "For splits that already have sample_*.pkl: skip collection, only run "
            "post-compression (parallel). Use to finish train compression and then "
            "produce valid/test without touching existing train samples."
        ),
    )
    args = parser.parse_args()

    if args.njobs is None:
        args.njobs = default_jobs
    if args.post_compress_jobs is None:
        args.post_compress_jobs = default_jobs
    if args.njobs < 1:
        raise ValueError("--njobs must be >= 1")
    if args.min_candidates < 1:
        raise ValueError("--min_candidates must be >= 1")
    if not 0 <= args.post_compress_level <= 9:
        raise ValueError("--post_compress_level must be in [0, 9]")
    if args.post_compress_jobs < 1:
        raise ValueError("--post_compress_jobs must be >= 1")
    if args.max_samples_per_episode < 0:
        raise ValueError("--max_samples_per_episode must be >= 0")
    if args.max_best_tie_count < 0:
        raise ValueError("--max_best_tie_count must be >= 0")
    if args.fullstrong:
        args.expert_prob = 1.0

    train_samples = args.train_size if args.train_size is not None else args.samples_per_split_train
    valid_samples = args.valid_size if args.valid_size is not None else args.samples_per_split_valid
    test_samples = args.test_size if args.test_size is not None else args.samples_per_split_test
    compression_mode = "none" if args.no_compress else args.compression
    mp_ctx = mp.get_context(args.mp_start_method)

    instance_base = args.instance_base
    sample_base = args.sample_base

    splits = [
        ("train", "train_mixed", train_samples, args.seed),
        ("valid", "valid_mixed", valid_samples, args.seed + 1),
        ("test", "test_mixed", test_samples, args.seed + 2),
    ]
    if args.split is not None:
        splits = [s for s in splits if s[0] == args.split]
        if not splits:
            raise ValueError(f"--split {args.split} did not match any split")

    for split_name, inst_dir_name, n_samples, seed in splits:
        inst_dir = os.path.join(instance_base, inst_dir_name)
        out_dir = os.path.join(sample_base, split_name)

        lp_files_all = sorted(glob.glob(os.path.join(inst_dir, "instance_*.lp")))
        lp_files = lp_files_all
        range_info = None
        if args.only_new_instances:
            range_info = _load_latest_append_range(inst_dir)
            lp_files = _filter_instances_by_index_range(
                lp_files_all,
                range_info["start"],
                range_info["end"],
            )

        if not lp_files:
            if args.only_new_instances:
                raise RuntimeError(
                    f"{split_name}: no instances matched latest append range in {inst_dir}."
                )
            print(f"\n[SKIP] {split_name}: no instance files in {inst_dir}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Split: {split_name}  ({n_samples} samples from {len(lp_files)} instances)")
        print(f"{'=' * 60}")
        if range_info is not None:
            print(
                "Instance source filter: only_new_instances=1 "
                f"range=[{range_info['start']}, {range_info['end']}] "
                f"from {range_info['path']} "
                f"(selected={len(lp_files)} / total={len(lp_files_all)})"
            )
        if compression_mode == "worker":
            payload_desc = "gzip-compressed pickle (.pkl extension)"
        elif compression_mode == "none":
            payload_desc = "plain pickle (.pkl)"
        else:
            payload_desc = "plain pickle during collection, gzip-compressed in post-process"
        print("Output payload: " + payload_desc)
        print(
            f"Collection config: finite_candidates_only={args.finite_candidates_only}, "
            f"min_candidates={args.min_candidates}, pseudo_candidates={bool(args.pseudo_candidates)}, "
            f"score_transform={args.score_transform}, "
            f"parallel_backend=mp/{args.mp_start_method}, compression={compression_mode}, "
            f"append_samples={int(args.append_samples)}, "
            f"only_new_instances={int(args.only_new_instances)}, "
            f"dispatch_mode={args.dispatch_mode}, "
            f"max_samples_per_episode={args.max_samples_per_episode}, "
            f"max_best_tie_count={args.max_best_tie_count}"
        )
        print(
            "SCIP profile: "
            f"fullstrong={int(args.fullstrong)}, heuristics_off={int(args.heuristics_off)}, "
            f"pyscipopt_scip_off={int(args.pyscipopt_heuristics_off)}, "
            f"vanillafullstrong_scip_params={int(args.vanillafullstrong_scip_params)}, "
            f"fullstrong_scoreall={int(args.fullstrong_scoreall)}, "
            f"fullstrong_donotbranch={int(args.fullstrong_donotbranch)}, "
            f"expert_prob={args.expert_prob}, pseudo_candidates={int(args.pseudo_candidates)}, "
            "cuts/presolve/propagation rounds disabled"
        )

        existing_samples = sorted(glob.glob(os.path.join(out_dir, "sample_*.pkl")))
        resume_compress_only = args.resume and len(existing_samples) > 0

        if resume_compress_only:
            print(f"[RESUME] {split_name}: {len(existing_samples)} existing samples — skipping collection, running post-compress only.")

        t0 = _time.time()
        if resume_compress_only:
            report = {
                "n_collected": 0,
                "n_existing_before": len(existing_samples),
                "n_total_after": len(existing_samples),
            }
        else:
            rng = np.random.RandomState(seed)
            report = collect_samples(
                lp_files,
                out_dir,
                rng,
                n_samples,
                args.njobs,
                query_expert_prob=args.expert_prob,
                time_limit=args.time_limit,
                mp_ctx=mp_ctx,
                compress=(compression_mode == "worker"),
                log_every=args.log_every,
                worker_logs=args.worker_logs,
                example_logs=args.example_logs,
                finite_candidates_only=args.finite_candidates_only,
                min_candidates=args.min_candidates,
                append_samples=args.append_samples,
                fullstrong=args.fullstrong,
                heuristics_off=args.heuristics_off,
                pyscipopt_heuristics_off=args.pyscipopt_heuristics_off,
                vanillafullstrong_scip_params=args.vanillafullstrong_scip_params,
                fullstrong_scoreall=args.fullstrong_scoreall,
                fullstrong_donotbranch=args.fullstrong_donotbranch,
                pseudo_candidates=args.pseudo_candidates,
                score_transform=args.score_transform,
                dispatch_mode=args.dispatch_mode,
                max_samples_per_episode=args.max_samples_per_episode,
                max_best_tie_count=args.max_best_tie_count,
                example_scip_params=args.example_scip_params,
            )
        elapsed = _time.time() - t0

        if not resume_compress_only:
            print(
                f"  >> +{report['n_collected']} samples collected in {elapsed:.1f}s "
                f"(before={report['n_existing_before']}, after={report['n_total_after']})"
            )
        else:
            print(f"  >> kept {report['n_existing_before']} existing samples (no collection).")
        print(f"  >> saved to {out_dir}/")

        split_validation = validate_split_samples(out_dir, args.min_candidates)
        counts = list(split_validation["instance_counts"].values())
        if counts:
            print(
                f"  >> instance coverage: {split_validation['unique_instances']}/{len(lp_files)} "
                f"(min/median/max per-instance = {int(np.min(counts))}/"
                f"{int(np.median(counts))}/{int(np.max(counts))})"
            )

        if compression_mode == "post":
            t_comp = _time.time()
            comp_stats = gzip_split_samples_in_place(
                out_dir,
                compresslevel=args.post_compress_level,
                log_every=args.post_compress_log_every,
                n_jobs=args.post_compress_jobs,
            )
            comp_elapsed = _time.time() - t_comp
            print(
                f"  >> post-compress: {comp_stats['compressed_files']}/{comp_stats['total_files']} files "
                f"in {comp_elapsed:.1f}s (gzip level={args.post_compress_level}, jobs={args.post_compress_jobs})"
            )
        print(
            "  >> validation: "
            f"action_not_in_action_set={split_validation['action_not_in_action_set']}, "
            f"nonfinite_candidate_scores={split_validation['nonfinite_candidate_scores']}, "
            f"below_min_candidates={split_validation['below_min_candidates']}, "
            f"bad_candidate_indices={split_validation['bad_candidate_indices']}, "
            f"tie_rate={split_validation['tie_rate']:.4f}, "
            f"all_tied_rate={split_validation['all_tied_rate']:.4f}, "
            f"avg_best_tie_count={split_validation['avg_best_tie_count']:.2f}, "
            f"max_best_tie_count={split_validation['max_best_tie_count']}, "
            f"avg_candidates={split_validation['avg_candidate_count']:.2f}, "
            f"avg_unique_scores={split_validation['avg_unique_scores']:.2f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
