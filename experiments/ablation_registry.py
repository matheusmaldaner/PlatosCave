
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    name: str
    level: str
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StudySpec:
    study_id: int
    short_name: str
    title: str
    objective: str
    cache_dependency: str
    variants: List[VariantSpec]


METRICS = [
    "credibility",
    "relevance",
    "evidence_strength",
    "method_rigor",
    "reproducibility",
    "citation_support",
]

GRAPH_COMPONENTS = [
    "bridge_coverage",
    "best_path",
    "redundancy",
    "fragility",
    "coherence",
    "coverage",
]

ROLES = [
    "Hypothesis",
    "Claim",
    "Evidence",
    "Method",
    "Result",
    "Conclusion",
    "Assumption",
    "Counterevidence",
    "Limitation",
    "Context",
]


def build_registry() -> dict[int, StudySpec]:
    reg: dict[int, StudySpec] = {}

    # 1) Node metric-level ablations
    vars1: List[VariantSpec] = [
        VariantSpec("base", "Baseline (recompute from cache)", "node-metric", "baseline", {}),
    ]
    for m in METRICS:
        vars1.append(
            VariantSpec(
                f"lomo_{m}",
                f"Leave-one-metric-out: {m}",
                "node-metric",
                "node_metric_lomo",
                {"metric": m},
            )
        )
    vars1.extend([
        VariantSpec(
            "metric_only_cred_rel",
            "Only credibility+relevance",
            "node-metric",
            "node_metric_keep_only",
            {"keep_metrics": ["credibility", "relevance"]},
        ),
        VariantSpec(
            "metric_only_method_reprod",
            "Only method rigor+reproducibility",
            "node-metric",
            "node_metric_keep_only",
            {"keep_metrics": ["method_rigor", "reproducibility"]},
        ),
    ])
    reg[1] = StudySpec(
        1,
        "node_metric_ablation",
        "Node-level metric fusion ablations",
        "Quantify dependence on each verifier metric and small metric subsets while keeping DAG structure fixed.",
        "Uses cached DAG JSON + cached node_scores only; no LLM calls.",
        vars1,
    )

    # 2) Node role-level / semantic unit ablations
    vars2: List[VariantSpec] = [VariantSpec("base", "Baseline", "node-role", "baseline", {})]
    for role in ["Evidence", "Method", "Result", "Claim", "Counterevidence", "Limitation", "Context"]:
        vars2.append(
            VariantSpec(
                f"drop_role_{role.lower()}",
                f"Drop all {role} nodes",
                "node-role",
                "drop_role_nodes",
                {"role": role, "reconnect": "bridge_through_removed"},
            )
        )
    vars2.extend([
        VariantSpec(
            "drop_low_quality_quintile",
            "Drop lowest-quality 20% nodes (per trial)",
            "node-role",
            "drop_nodes_by_trial_quality_quantile",
            {"quantile": 0.2, "which": "lowest", "reconnect": "none"},
        ),
        VariantSpec(
            "drop_high_quality_quintile",
            "Drop highest-quality 20% nodes (per trial)",
            "node-role",
            "drop_nodes_by_trial_quality_quantile",
            {"quantile": 0.2, "which": "highest", "reconnect": "none"},
        ),
    ])
    reg[2] = StudySpec(
        2,
        "node_role_ablation",
        "Node-level semantic role / unit ablations",
        "Probe which role categories and which quality strata dominate graph-level scoring.",
        "Uses cached DAG JSON + cached node_scores; graph re-built offline per variant.",
        vars2,
    )

    # 3) Edge feature and prior ablations
    vars3: List[VariantSpec] = [VariantSpec("base", "Baseline", "edge-feature", "baseline", {})]
    for feat in ["role_prior", "alignment", "synergy", "parent_quality", "child_quality"]:
        vars3.append(
            VariantSpec(
                f"zero_{feat}",
                f"Zero edge feature weight: {feat}",
                "edge-feature",
                "edge_weight_zero",
                {"feature": feat},
            )
        )
    vars3.extend([
        VariantSpec(
            "uniform_role_prior_05",
            "Uniform role prior = 0.5",
            "edge-feature",
            "uniform_role_prior",
            {"value": 0.5},
        ),
        VariantSpec(
            "uniform_role_prior_00",
            "Uniform role prior = 0.0",
            "edge-feature",
            "uniform_role_prior",
            {"value": 0.0},
        ),
        VariantSpec(
            "no_pair_synergy_table",
            "Disable pair-specific synergy table",
            "edge-feature",
            "empty_pair_synergy",
            {},
        ),
    ])
    reg[3] = StudySpec(
        3,
        "edge_feature_ablation",
        "Edge-level confidence feature ablations",
        "Measure sensitivity to role priors, lexical alignment, and pair synergy in edge confidence.",
        "Uses cached DAG JSON + cached node_scores; recomputes edge confidences offline.",
        vars3,
    )

    # 4) Propagation / edge-topology trust ablations
    vars4: List[VariantSpec] = [VariantSpec("base", "Baseline", "edge-propagation", "baseline", {})]
    vars4.extend([
        VariantSpec("penalty_off", "Disable trust propagation penalty", "edge-propagation", "penalty_toggle", {"enabled": False}),
        VariantSpec("agg_mean", "Propagation aggregate = mean", "edge-propagation", "penalty_mode", {"agg": "mean"}),
        VariantSpec("agg_softmin", "Propagation aggregate = softmin", "edge-propagation", "penalty_mode", {"agg": "softmin"}),
        VariantSpec("agg_dampmin", "Propagation aggregate = dampmin", "edge-propagation", "penalty_mode", {"agg": "dampmin"}),
        VariantSpec("alpha_05", "Propagation alpha = 0.5", "edge-propagation", "penalty_param", {"alpha": 0.5}),
        VariantSpec("alpha_20", "Propagation alpha = 2.0", "edge-propagation", "penalty_param", {"alpha": 2.0}),
        VariantSpec("eta_100", "Trust gate eta = 1.0 (no floor attenuation)", "edge-propagation", "penalty_param", {"eta": 1.0}),
        VariantSpec("eta_070", "Trust gate eta = 0.7", "edge-propagation", "penalty_param", {"eta": 0.7}),
        VariantSpec("drop_edges_low_alignment", "Drop edges with alignment < 0.05", "edge-propagation", "drop_edges_by_feature_threshold", {"feature": "alignment", "op": "<", "threshold": 0.05}),
    ])
    reg[4] = StudySpec(
        4,
        "propagation_ablation",
        "Trust propagation / edge topology ablations",
        "Test how much graph scores depend on weakest-link aggregation and trust gating mechanics.",
        "Uses cached DAG JSON + cached node_scores; modifies scorer hyperparameters and/or edge set offline.",
        vars4,
    )

    # 5) Graph-head / component ablations
    vars5: List[VariantSpec] = [VariantSpec("base", "Baseline", "graph-head", "baseline", {})]
    for comp in GRAPH_COMPONENTS:
        vars5.append(
            VariantSpec(
                f"zero_{comp}",
                f"Zero graph component weight: {comp}",
                "graph-head",
                "graph_weight_zero",
                {"component": comp},
            )
        )
    for comp in ["best_path", "bridge_coverage", "coherence", "coverage"]:
        vars5.append(
            VariantSpec(
                f"only_{comp}",
                f"{comp} only",
                "graph-head",
                "graph_weight_single_component",
                {"component": comp},
            )
        )
    reg[5] = StudySpec(
        5,
        "graph_component_ablation",
        "Graph-level aggregation ablations",
        "Identify which graph-level components drive final score and ranking behavior.",
        "Uses cached DAG JSON + cached node_scores; recomputes graph score offline.",
        vars5,
    )

    # 6) Final study: hierarchical variance + stability
    vars6 = [
        VariantSpec("base_variance_decomp", "Hierarchical variance decomposition", "final", "variance_decomposition", {}),
        VariantSpec("dag_vs_node_stability", "DAG-vs-node resample stability summary", "final", "resample_stability", {}),
    ]
    reg[6] = StudySpec(
        6,
        "final_stability_variance",
        "Final study: hierarchical variance and stability",
        "Decompose total variability into between-DAG and within-DAG (node verification) components and summarize ranking stability.",
        "Primarily uses cached graph_scores.csv from the original factorized run; no rescoring required.",
        vars6,
    )

    return reg


REGISTRY = build_registry()
