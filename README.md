# GNN-MILP Path Planning

Graph neural network branching policies for mixed-integer receding-horizon path planning.

This repository builds a compact research pipeline around multi-vehicle path planning MILPs: generate receding-horizon instances, collect expert branching decisions with SCIP/Ecole, train a bipartite GNN policy, and evaluate it against standard SCIP branching baselines.

The project targets the practical question behind learned branching: can a lightweight learned policy reduce solve time on structured path-planning MILPs, even when it does not always minimize the branch-and-bound node count?

## Highlights

- Schouwenaars-style receding-horizon MILP formulation for 2D double-integrator vehicles.
- Rectangular obstacle avoidance with Big-M disjunctions and vehicle-vehicle separation constraints.
- Ecole-based full-strong-branching supervision for imitation learning.
- PyTorch Geometric GNN policy over the MILP bipartite graph.
- Four-way evaluation against VSB, FSB, SCIP DEFAULT, and the trained GNN brancher.
- Generated data, trained checkpoints, and result CSVs are intentionally excluded from the repository.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `01_generate_rh_instances.py` | Generate receding-horizon MILP `.lp` instances. |
| `02_generate_dataset.py` | Collect branching samples with Ecole. |
| `03_train_gnn.py` | Train the GNN branching policy. |
| `04_evaluate.py` | Compare SCIP baselines and the learned policy. |
| `true_receding_horizon.py` | Core MILP formulation and receding-horizon utilities. |
| `visualize_rh_scenario.py` | Scenario visualization helper. |
| `tools/_per_instance_root_report.py` | Per-instance result breakdown. |
| `tools/_quick_report.py` | Fast timing/node summary. |

## Setup

The solver stack is easiest to run from a conda environment with SCIP, PySCIPOpt, Ecole, PyTorch, and PyTorch Geometric installed.

```bash
conda activate ecole_gpu
pip install -r requirements.txt
```

If Ecole/PySCIPOpt are not already available, install them through the same SCIP-compatible conda environment before running the numbered scripts.

## Quick Start

Use a short experiment tag and keep it consistent across instances, samples, model output, and results.

```bash
TAG=demo

python 01_generate_rh_instances.py \
  --fixed_n_vehicles 3 \
  --horizon_time 4.0 \
  --map_mode random \
  --seed 0 \
  --base_dir data/instances/rh_milp_${TAG}

python 02_generate_dataset.py \
  --samples_per_split_train 1000 \
  --samples_per_split_valid 200 \
  --samples_per_split_test 200 \
  --example_scip_params \
  --no_fullstrong \
  --no_vanillafullstrong_scip_params \
  --no_heuristics_off \
  --expert_prob 0.05 \
  --instance_base data/instances/rh_milp_${TAG} \
  --sample_base data/samples/rh_milp_${TAG}

python 03_train_gnn.py \
  --sample_root data/samples/rh_milp_${TAG} \
  --out_dir trained_models/rh_milp/gnn/${TAG}_seed0 \
  --max_epochs 50 \
  --early_stopping 10

python 04_evaluate.py \
  --four_way \
  --time_limit 3600 \
  --instance_dir data/instances/rh_milp_${TAG}/test_mixed \
  --model_path trained_models/rh_milp/gnn/${TAG}_seed0/best_params.pkl \
  --csv_path results/${TAG}_four_way.csv

python tools/_per_instance_root_report.py results/${TAG}_four_way.csv
```

## Notes

This repository contains source code only. Generated LP instances, Ecole samples, model checkpoints, logs, local notes, papers, notebooks, and evaluation CSVs are ignored by default.

The implementation follows the standard learned-branching workflow introduced for SCIP/Ecole-style GNN branchers and adapts it to receding-horizon multi-vehicle path-planning MILPs.
