from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from src.utils.config import load_config, output_dirs
from src.utils.io import write_csv


@dataclass(frozen=True)
class BaselineCatalogEntry:
    method: str
    layer: str
    role: str
    implementation_status: str
    reference_key: str
    source_url: str
    adapter_module: str
    adaptation_rule: str
    notes: str


CATALOG_FIELDS = [
    "method",
    "layer",
    "role",
    "implementation_status",
    "reference_key",
    "source_url",
    "adapter_module",
    "adaptation_rule",
    "notes",
]


def core_manuscript_methods() -> set[str]:
    return {
        "MNDDPG",
        "PM-Soft-MoE",
        "MO-DDPG",
        "MO-SAC",
        "CAPQL",
        "PGMORL",
        "FL-MO-SAC",
        "FedMORL",
    }


def baseline_status_by_method() -> dict[str, str]:
    statuses = {
        entry.method: entry.implementation_status
        for entry in baseline_catalog_entries()
    }
    statuses.update(
        {
            "RandomFeasible": "local-implemented",
            "LocalOnly": "local-implemented",
            "EdgeOnly": "local-implemented",
            "GreedyMinLatency": statuses.get("Greedy-MinLatency", "local-implemented"),
            "GreedyMinEnergy": statuses.get("Greedy-MinEnergy", "local-implemented"),
        }
    )
    return statuses


def baseline_catalog_entries() -> list[BaselineCatalogEntry]:
    return [
        BaselineCatalogEntry(
            method="MNDDPG",
            layer="core-main",
            role="Proposed hard-routed federated multi-objective controller.",
            implementation_status="local-implemented",
            reference_key="proposed",
            source_url="",
            adapter_module="src.models.proposed_model.MNDDPGAgent",
            adaptation_rule="Native shared route-resource action, feasible mask, projection, and clipped federated aggregation.",
            notes="Current smoke implementation is migrated from paper_aligned_private.",
        ),
        BaselineCatalogEntry(
            method="PM-Soft-MoE",
            layer="core-main",
            role="Soft-routed structural comparator for the proposed hard routing mechanism.",
            implementation_status="local-structural-adapt",
            reference_key="proposed_ablation",
            source_url="",
            adapter_module="src.models.baseline_algorithms.PmSoftMoeAdapter",
            adaptation_rule="Reuse MNDDPG experts and proxy weights, but replace hard expert selection with soft action mixing before projection.",
            notes="Needed for the hard-vs-soft routing comparison in the manuscript.",
        ),
        BaselineCatalogEntry(
            method="MO-DDPG",
            layer="core-main",
            role="Non-federated deterministic actor-critic MORL control.",
            implementation_status="public-code-adapt",
            reference_key="573696096e3b12023e51cb6b",
            source_url="https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html",
            adapter_module="src.models.baseline_algorithms.MoDdpgAdapter",
            adaptation_rule="Keep DDPG actor-critic, replay, target update, and noise pattern; replace scalar reward with four normalized objectives and validation-selected scalarization.",
            notes="A compact trainable actor-critic smoke implementation is available; CleanRL DDPG can be used as a cross-check.",
        ),
        BaselineCatalogEntry(
            method="MO-SAC",
            layer="core-main",
            role="Non-federated entropy-regularized MORL control.",
            implementation_status="public-code-adapt",
            reference_key="haarnoja2018softactorcritic",
            source_url="https://stable-baselines3.readthedocs.io/en/master/modules/sac.html",
            adapter_module="src.models.baseline_algorithms.MoSacAdapter",
            adaptation_rule="Keep SAC critics, stochastic actor, replay, target update, and temperature handling; train through the same four-objective scalarization and route-resource decoder.",
            notes="A compact trainable twin-critic smoke implementation is available; use the same seeds, mask, projection, and validation rule as MNDDPG for manuscript-scale runs.",
        ),
        BaselineCatalogEntry(
            method="CAPQL",
            layer="core-main",
            role="Preference-conditioned MORL reference policy.",
            implementation_status="public-code-adapt",
            reference_key="shianifar2026hindsightpreference",
            source_url="https://github.com/LucasAlegre/morl-baselines",
            adapter_module="src.models.baseline_algorithms.CapqlAdapter",
            adaptation_rule="Train preference-conditioned policy on four normalized objectives; decode the selected preference action through the shared feasible route-resource template.",
            notes="MORL-Baselines provides the reference implementation family used for adaptation.",
        ),
        BaselineCatalogEntry(
            method="PGMORL",
            layer="core-main",
            role="Pareto-front MORL reference policy.",
            implementation_status="public-code-adapt",
            reference_key="5ede0553e06a4c1b26a83eb1",
            source_url="https://github.com/LucasAlegre/morl-baselines",
            adapter_module="src.models.baseline_algorithms.PgmorlAdapter",
            adaptation_rule="Generate Pareto or population policies, select by validation HV and tie-breakers, then decode through the shared feasible route-resource interface.",
            notes="MORL-Baselines includes PGMORL as a multi-policy algorithm.",
        ),
        BaselineCatalogEntry(
            method="FL-MO-SAC",
            layer="core-main",
            role="Federated version of MO-SAC under the manuscript client partition.",
            implementation_status="federated-adapt-required",
            reference_key="haarnoja2018softactorcritic",
            source_url="https://stable-baselines3.readthedocs.io/en/master/modules/sac.html",
            adapter_module="src.models.baseline_algorithms.FlMoSacAdapter",
            adaptation_rule="Wrap MO-SAC local updates with six-client partitioning, synchronization interval, clipping threshold, and secure-aggregation byte accounting.",
            notes="Used to isolate the contribution of federation under shared objectives.",
        ),
        BaselineCatalogEntry(
            method="FedMORL",
            layer="core-main",
            role="Direct reviewer-requested federated multi-objective RL reference.",
            implementation_status="paper-adapt-required",
            reference_key="zhao2023fedmorl",
            source_url="https://doi.org/10.1016/j.ins.2022.12.083",
            adapter_module="src.models.baseline_algorithms.FedMorlAdapter",
            adaptation_rule="Implement the paper method under the six-client split when the action interface can be matched; otherwise report unavailable components and adapted decoder boundaries.",
            notes="No directly reusable public code was confirmed during this pass.",
        ),
        BaselineCatalogEntry(
            method="DDPG",
            layer="supplementary-basic-rl",
            role="Single-objective lower-bound actor-critic reference.",
            implementation_status="public-code-reference",
            reference_key="573696096e3b12023e51cb6b",
            source_url="https://docs.cleanrl.dev/rl-algorithms/ddpg/",
            adapter_module="src.models.baseline_algorithms.DdpgAdapter",
            adaptation_rule="Use scalar reward with the same observation, mask, projection, client partition, and test seeds.",
            notes="Reserved for supplementary or focused analyses.",
        ),
        BaselineCatalogEntry(
            method="SAC",
            layer="supplementary-basic-rl",
            role="Single-objective stochastic actor-critic reference.",
            implementation_status="public-code-reference",
            reference_key="haarnoja2018softactorcritic",
            source_url="https://docs.cleanrl.dev/rl-algorithms/sac/",
            adapter_module="src.models.baseline_algorithms.SacAdapter",
            adaptation_rule="Use scalar reward with the same observation, mask, projection, client partition, and test seeds.",
            notes="Reserved for supplementary or focused analyses.",
        ),
        BaselineCatalogEntry(
            method="FL-DDPG",
            layer="supplementary-federated-rl",
            role="Federated single-objective DDPG reference.",
            implementation_status="federated-adapt-required",
            reference_key="573696096e3b12023e51cb6b",
            source_url="https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html",
            adapter_module="src.training.federated",
            adaptation_rule="Wrap DDPG local updates with the same client split, clipping, synchronization, and accounting used by MNDDPG.",
            notes="Useful for communication-efficiency and non-IID diagnostics.",
        ),
        BaselineCatalogEntry(
            method="Greedy-MinLatency",
            layer="supplementary-engineering",
            role="Latency-oriented engineering sanity check.",
            implementation_status="local-implemented",
            reference_key="engineering_heuristic",
            source_url="",
            adapter_module="src.models.baseline.GreedyMinLatencyPolicy",
            adaptation_rule="Select feasible edge or cloud route with aggressive resource allocation.",
            notes="Already exported by baseline_results.csv.",
        ),
        BaselineCatalogEntry(
            method="Greedy-MinEnergy",
            layer="supplementary-engineering",
            role="Energy-oriented engineering sanity check.",
            implementation_status="local-implemented",
            reference_key="engineering_heuristic",
            source_url="",
            adapter_module="src.models.baseline.GreedyMinEnergyPolicy",
            adaptation_rule="Prefer local execution and conservative transmit power under the feasible mask.",
            notes="Already exported by baseline_results.csv.",
        ),
        BaselineCatalogEntry(
            method="RandomFeasible",
            layer="supplementary-engineering",
            role="Random feasible route sanity check.",
            implementation_status="local-implemented",
            reference_key="engineering_heuristic",
            source_url="",
            adapter_module="src.models.baseline.RandomFeasiblePolicy",
            adaptation_rule="Sample one feasible route and bounded continuous resources.",
            notes="Already exported by baseline_results.csv.",
        ),
    ]


def write_baseline_catalog_files(config_path: str | Path) -> dict[str, Path]:
    dirs = output_dirs(load_config(config_path))
    entries = baseline_catalog_entries()
    csv_path = write_csv(
        dirs["result_dir"] / "baseline_implementation_catalog.csv",
        (asdict(entry) for entry in entries),
        CATALOG_FIELDS,
    )
    markdown_path = dirs["table_dir"] / "baseline_implementation_catalog.md"
    markdown_path.write_text(_to_markdown(entries), encoding="utf-8")
    return {"markdown": markdown_path, "csv": csv_path}


def _to_markdown(entries: list[BaselineCatalogEntry]) -> str:
    header = [
        "# Baseline Implementation Catalog",
        "",
        "| Method | Layer | Status | Source | Adapter | Adaptation rule |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    rows = [
        "| {method} | {layer} | {status} | {source} | {adapter} | {rule} |".format(
            method=_escape_markdown(entry.method),
            layer=_escape_markdown(entry.layer),
            status=_escape_markdown(entry.implementation_status),
            source=_escape_markdown(entry.source_url or entry.reference_key),
            adapter=_escape_markdown(entry.adapter_module),
            rule=_escape_markdown(entry.adaptation_rule),
        )
        for entry in entries
    ]
    return "\n".join(header + rows) + "\n"


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|")
