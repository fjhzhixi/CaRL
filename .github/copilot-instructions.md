# Project Guidelines

## Architecture
- This repository is a multi-project research workspace. Keep changes scoped to one stack unless the task explicitly spans multiple stacks.
- Main boundaries:
  - `CARLA/`: RL training and evaluation on CARLA leaderboard (primary CaRL codepath).
  - `nuPlan/`: RL training and simulation on nuPlan data with separate env and Hydra configs.
  - `PlanT/`: imitation learning baseline (PyTorch Lightning) for CARLA Longest6 v2.
  - `Think2Drive/`: DreamerV3-based world-model baseline with its own env/scripts.
- CARLA training/evaluation entry points are usually under `CARLA/team_code/` and leaderboard evaluators under `CARLA/custom_leaderboard/` or `CARLA/original_leaderboard/`.

## Build And Run
- Root has no single build/test command; run commands from the relevant subproject.
- CARLA setup and run:
  - Setup docs: `CARLA/README.md`
  - Typical setup: `cd CARLA && conda env create --file=environment.yml && conda activate carl`
  - Training orchestration: `python team_code/train_parallel.py ...`
  - Evaluation orchestration: `python evaluate_routes_slurm.py ...`
- nuPlan setup and run:
  - Setup docs: `nuPlan/README.md`
  - Typical setup: `cd nuPlan && conda env create --name carl_nuplan -f environment.yml && pip install -e .`
  - Training scripts: `nuPlan/scripts/gym/`
  - Simulation/eval scripts: `nuPlan/scripts/simulation/`
- PlanT setup and run:
  - Setup docs: `PlanT/README.md`
  - Typical setup: `cd PlanT && conda env create -f environment.yml && conda activate PlanTUpdate`
  - Train: `python lit_train.py`
  - Eval: `python evaluate_routes_slurm.py ...`
- Think2Drive setup and run:
  - Setup docs: `Think2Drive/README.md`
  - Train: `bash Think2Drive/training.sh`
  - Eval: `bash Think2Drive/evaluation.sh`

## Agent Operating Conventions
- Prefer read-only investigation first: inspect README files and existing shell scripts before changing commands.
- Do not start long-running jobs (CARLA simulator, distributed training, SLURM submissions, large downloads) unless explicitly requested.
- Before running any Python command, activate the matching conda environment for that subproject.
- Preserve the existing script-first workflow: when a `*.sh` launcher exists, prefer updating/using it over inventing ad-hoc command lines.
- For CARLA work, verify environment variables expected by scripts (`CARLA_ROOT`, `WORK_DIR`, `SCENARIO_RUNNER_ROOT`, `LEADERBOARD_ROOT`, `PYTHONPATH`) before debugging runtime failures.
- Keep changes minimal and local. Avoid cross-project refactors across `CARLA/`, `nuPlan/`, `PlanT/`, and `Think2Drive/` unless requested.

## Project-Specific Pitfalls
- CARLA codepaths are sensitive to port allocation and thread settings; avoid changing defaults casually in training/eval scripts.
- nuPlan workflows depend on dataset layout, Hydra search paths, and cache paths; treat path/config edits as high-risk.
- PlanT and Think2Drive use separate environments and dependencies; do not mix them with the `carl` or `carl_nuplan` environments.
- Many workflows assume checkpoint directories include config plus model artifacts; preserve expected file names when editing load/save logic.