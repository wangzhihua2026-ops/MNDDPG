from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from src.utils.paper_config import PaperTrainingConfig, paper_training_config


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config root must be a mapping: {config_path}")
        return loaded
    except ModuleNotFoundError:
        return _load_simple_yaml(text)


def build_paper_config(
    mapping: Mapping[str, Any],
    *,
    rounds: int | None = None,
    local_steps: int | None = None,
    evaluation_steps: int | None = None,
) -> PaperTrainingConfig:
    federated = _section(mapping, "federated")
    agent = _section(mapping, "agent")
    ablation = _section(mapping, "ablation")
    evaluation = _section(mapping, "evaluation")

    config = paper_training_config(
        rounds=rounds if rounds is not None else _optional_int(federated, "rounds"),
        local_steps=(
            local_steps if local_steps is not None else _optional_int(federated, "local_steps")
        ),
        local_updates=_optional_int(federated, "local_updates"),
        num_clients=_optional_int(federated, "num_clients"),
        batch_size=_optional_int(agent, "batch_size"),
        evaluation_steps=(
            evaluation_steps
            if evaluation_steps is not None
            else _optional_int(evaluation, "steps")
        ),
        scenario_limit=_optional_int(evaluation, "scenario_limit"),
        bootstrap_resamples=_optional_int(evaluation, "bootstrap_resamples"),
    )

    clip_norm = _optional_float(federated, "clip_norm")
    if clip_norm is not None:
        config = replace(
            config,
            federated=replace(config.federated, clip_norm=clip_norm),
        )

    federated_updates: dict[str, Any] = {}
    use_federated_clipping = _optional_bool(ablation, "use_federated_clipping")
    if use_federated_clipping is not None:
        federated_updates["use_clipping"] = use_federated_clipping
    if federated_updates:
        config = replace(config, federated=replace(config.federated, **federated_updates))

    agent_updates: dict[str, Any] = {}
    for key in ("gating_temperature", "route_temperature"):
        value = _optional_float(agent, key)
        if value is not None:
            agent_updates[key] = value
    for key in ("shared_dim", "actor_hidden_dim", "critic_hidden_dim"):
        value = _optional_int(agent, key)
        if value is not None:
            agent_updates[key] = value
    ablation_agent_keys = {
        "use_proxy_weights": "use_proxy_weights",
        "use_context_gating": "use_context_gating",
        "use_hard_routing": "use_hard_routing",
    }
    for source_key, target_key in ablation_agent_keys.items():
        value = _optional_bool(ablation, source_key)
        if value is not None:
            agent_updates[target_key] = value
    if agent_updates:
        config = replace(config, agent=replace(config.agent, **agent_updates))

    use_privacy_mask = _optional_bool(ablation, "use_privacy_mask")
    if use_privacy_mask is not None:
        config = replace(
            config,
            environment=replace(
                config.environment,
                enforce_privacy_mask=use_privacy_mask,
            ),
        )

    return config


def output_dirs(mapping: Mapping[str, Any]) -> dict[str, Path]:
    output = _section(mapping, "output")
    dirs = {
        "log_dir": Path(str(output.get("log_dir", "outputs/logs"))),
        "checkpoint_dir": Path(str(output.get("checkpoint_dir", "outputs/checkpoints"))),
        "result_dir": Path(str(output.get("result_dir", "outputs/results"))),
        "figure_dir": Path(str(output.get("figure_dir", "outputs/figures"))),
        "table_dir": Path(str(output.get("table_dir", "outputs/tables"))),
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def _section(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _optional_int(mapping: Mapping[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    return None if value is None else int(value)


def _optional_float(mapping: Mapping[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    return None if value is None else float(value)


def _optional_bool(mapping: Mapping[str, Any], key: str) -> bool | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported config line: {raw_line}")

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if raw_value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(raw_value)

    return root


def _parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("'\"")
