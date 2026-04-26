# Public Release

This folder is the GitHub-ready public subset extracted from the private paper-aligned codebase.

What is included:
- paper-aligned local observation and context summary
- hard-routed route selection with feasible-mask enforcement
- bounded four-objective reward computation
- clean federated clipping and aggregation utilities
- a minimal runnable demo for auditing the pipeline end to end

What is intentionally not included:
- the full private paper-aligned training loop
- internal research-only implementation details
- local virtual environments, caches, and generated artifacts

## Structure

- `config.py`: paper-style defaults
- `schemas.py`: task, node, observation, and action data structures
- `environment.py`: privacy-aware edge-offloading environment
- `policy.py`: clean MNDDPG hard-routing scaffold
- `federated.py`: clipping and aggregation
- `metrics.py`: hypervolume and reporting helpers
- `runner.py`: minimal multi-client demo
- `main.py`: command-line entry point

## Run

```powershell
pip install -r requirements.txt
python main.py --rounds 3 --steps 24
```

The demo writes a JSON summary to `outputs/demo_summary.json`.

## GitHub upload

Recommended repository root contents:
- `README.md`
- `requirements.txt`
- `config.py`
- `environment.py`
- `federated.py`
- `main.py`
- `metrics.py`
- `policy.py`
- `runner.py`
- `schemas.py`
- `__init__.py`
- optional example output in `outputs/demo_summary.json`

Do not upload:
- `__pycache__/`
- local virtual environments
- private training code
- research-only internal folders

## Note

The public runner is intentionally lightweight. The full paper-aligned training and federated update pipeline remains private and is not meant for GitHub upload.
