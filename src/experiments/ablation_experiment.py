from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from src.training.federated import PaperAlignedFederatedRunner
from src.utils.config import build_paper_config
from src.utils.config import load_config, output_dirs
from src.utils.io import write_csv
from src.utils.visualization import write_markdown_table


ABLATION_GROUPS = (
    "Full MNDDPG",
    "w/o Proxy Weights",
    "w/o Context Gating",
    "Soft Routing",
    "w/o Privacy Mask",
    "w/o Federated Clipping",
    "w/o Proxy + w/o Gating",
    "w/o Proxy + Soft Routing",
    "w/o Gating + Soft Routing",
    "w/o Privacy Mask + w/o Clipping",
)


ABLATION_SETTINGS: dict[str, dict[str, bool]] = {
    "Full MNDDPG": {
        "use_proxy_weights": True,
        "use_context_gating": True,
        "use_hard_routing": True,
        "use_privacy_mask": True,
        "use_federated_clipping": True,
    },
    "w/o Proxy Weights": {"use_proxy_weights": False},
    "w/o Context Gating": {"use_context_gating": False},
    "Soft Routing": {"use_hard_routing": False},
    "w/o Privacy Mask": {"use_privacy_mask": False},
    "w/o Federated Clipping": {"use_federated_clipping": False},
    "w/o Proxy + w/o Gating": {
        "use_proxy_weights": False,
        "use_context_gating": False,
    },
    "w/o Proxy + Soft Routing": {
        "use_proxy_weights": False,
        "use_hard_routing": False,
    },
    "w/o Gating + Soft Routing": {
        "use_context_gating": False,
        "use_hard_routing": False,
    },
    "w/o Privacy Mask + w/o Clipping": {
        "use_privacy_mask": False,
        "use_federated_clipping": False,
    },
}


def run_ablation_experiment(
    config_path: str | Path,
    *,
    groups: Iterable[str] | None = None,
    rounds: int | None = None,
    steps: int = 24,
) -> list[dict[str, Any]]:
    config_mapping = load_config(config_path)
    dirs = output_dirs(config_mapping)
    selected_groups = tuple(groups) if groups is not None else ABLATION_GROUPS
    rows = []

    for index, group in enumerate(selected_groups):
        if group not in ABLATION_SETTINGS:
            raise ValueError(f"Unknown ablation group: {group}")
        group_config = _ablation_config(config_mapping, group)
        paper_config = build_paper_config(group_config, rounds=rounds)
        runner = PaperAlignedFederatedRunner(paper_config)
        runner.run_rounds(rounds=paper_config.federated.rounds)
        stats = runner.evaluate(mode="seen", num_steps=steps)
        rows.append(
            {
                "seed": paper_config.protocol.independent_run_seeds[
                    index % len(paper_config.protocol.independent_run_seeds)
                ],
                "ablation_group": group,
                "HV": stats["normalized_hv"],
                "privacy_match": stats["privacy_match"],
                "violation_rate": stats["violation_rate"],
            }
        )

    csv_path = write_csv(
        dirs["result_dir"] / "combined_ablation.csv",
        rows,
        ["seed", "ablation_group", "HV", "privacy_match", "violation_rate"],
    )
    write_markdown_table(
        dirs["table_dir"] / "ablation_table.md",
        ["Ablation Group", "HV", "Privacy Match", "Violation Rate"],
        [
            (
                row["ablation_group"],
                f"{float(row['HV']):.4f}",
                f"{float(row['privacy_match']):.4f}",
                f"{float(row['violation_rate']):.4f}",
            )
            for row in rows
        ],
    )
    return rows


def write_ablation_plan(config_path: str | Path) -> Path:
    config_mapping = load_config(config_path)
    dirs = output_dirs(config_mapping)
    rows = [
        {"seed": "", "ablation_group": group, "HV": "", "privacy_match": "", "violation_rate": ""}
        for group in ABLATION_GROUPS
    ]
    csv_path = write_csv(
        dirs["result_dir"] / "combined_ablation.csv",
        rows,
        ["seed", "ablation_group", "HV", "privacy_match", "violation_rate"],
    )
    write_markdown_table(
        dirs["table_dir"] / "ablation_table.md",
        ["Ablation Group", "HV", "Privacy Match", "Violation Rate"],
        [(group, "", "", "") for group in ABLATION_GROUPS],
    )
    return csv_path


def _ablation_config(config_mapping: dict[str, Any], group: str) -> dict[str, Any]:
    group_config = deepcopy(config_mapping)
    base = dict(ABLATION_SETTINGS["Full MNDDPG"])
    base.update(ABLATION_SETTINGS[group])
    group_config["ablation"] = {
        **group_config.get("ablation", {}),
        "group": group,
        **base,
    }
    return group_config
