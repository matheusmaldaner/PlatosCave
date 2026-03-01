
from __future__ import annotations

import csv
import json
import math
import statistics
import hashlib
import copy
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    # Script mode (python experiments/ablation_studies_cli.py)
    from ablation_registry import REGISTRY, METRICS
except Exception:
    # Module mode (python -m experiments.ablation_studies_cli)
    from .ablation_registry import REGISTRY, METRICS


def _clip01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if math.isnan(x) or math.isinf(x):
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _mean(xs: List[float]) -> float:
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return sum(vals) / len(vals) if vals else float("nan")


def _std(xs: List[float]) -> float:
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if len(vals) < 2:
        return 0.0 if vals else float("nan")
    return statistics.pstdev(vals)


def _quantile(xs: List[float], q: float) -> float:
    vals = sorted(float(x) for x in xs)
    if not vals:
        return float("nan")
    q = min(1.0, max(0.0, float(q)))
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    w = pos - lo
    return (1.0 - w) * vals[lo] + w * vals[hi]


def _rankdata_avg(xs: List[float]) -> List[float]:
    # Average ranks for ties, 1-indexed
    pairs = sorted(enumerate(xs), key=lambda t: t[1])
    ranks = [0.0] * len(xs)
    i = 0
    n = len(xs)
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    rx = _rankdata_avg(xs)
    ry = _rankdata_avg(ys)
    mx = _mean(rx)
    my = _mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    den = denx * deny
    return num / den if den > 0 else float("nan")


def _kendall_tau_a(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2 or len(ys) != n:
        return float("nan")
    conc = 0
    disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            s = dx * dy
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    den = n * (n - 1) / 2
    return (conc - disc) / den if den else float("nan")


def _auc_good_bad(labels01: List[int], scores: List[float]) -> float:
    # Mann-Whitney U / AUC; labels01: 1=good,0=bad
    pairs = [(l, s) for l, s in zip(labels01, scores) if l in (0, 1) and not math.isnan(float(s))]
    pos = [s for l, s in pairs if l == 1]
    neg = [s for l, s in pairs if l == 0]
    if not pos or not neg:
        return float("nan")
    # Average ranks for ties on all scores
    all_scores = pos + neg
    ranks = _rankdata_avg(all_scores)
    sum_pos = sum(ranks[: len(pos)])
    n1, n0 = len(pos), len(neg)
    u1 = sum_pos - n1 * (n1 + 1) / 2
    return u1 / (n1 * n0)


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _csv_write(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _csv_read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class AblationError(RuntimeError):
    pass


def _import_scorer():
    import sys
    # Expect script executed from repo root or with repo root already on path
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    try:
        from graph_app import kg_realtime_scoring as kgr
    except Exception as e:
        raise AblationError(
            "Could not import graph_app.kg_realtime_scoring. "
            "Run this from the Plato's Cave repo root after copying the ablation scripts."
        ) from e
    return kgr


def _meta_pick(meta: Dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        if key in meta:
            val = meta.get(key)
            if val is None:
                continue
            sval = str(val).strip()
            if sval:
                return sval
    return default


def _load_paper_meta(paper_dir: Path) -> Dict[str, Any]:
    # Current pipeline writes record.json; keep compatibility with older metadata.json too.
    for name in ("metadata.json", "record.json"):
        p = paper_dir / name
        if not p.exists():
            continue
        try:
            obj = _json_load(p)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def _paper_id_from_meta_or_dir(meta: Dict[str, Any], paper_dir: Path, paper_key: str) -> str:
    paper_id = _meta_pick(meta, "paper_id", "paperId", "id", default="")
    if paper_id:
        return paper_id

    row_index = meta.get("row_index", None)
    if row_index is not None:
        try:
            return f"{int(float(row_index)):03d}"
        except Exception:
            pass

    stem = paper_dir.name.split("__", 1)[0].strip()
    if stem:
        return stem
    return paper_key


def _paper_key_from_dir(paper_dir: Path, runs_root: Path) -> str:
    try:
        rel = paper_dir.relative_to(runs_root).as_posix().strip("/")
        key = rel.replace("/", "__")
    except Exception:
        key = paper_dir.name
    if key:
        return key
    return hashlib.sha1(str(paper_dir).encode()).hexdigest()[:12]


def _discover_papers(runs_root: Path) -> List[Dict[str, Any]]:
    papers: List[Dict[str, Any]] = []
    seen_dirs: set[str] = set()
    for summary_path in runs_root.rglob("summary.json"):
        paper_dir = summary_path.parent
        paper_dir_key = str(paper_dir.resolve())
        if paper_dir_key in seen_dirs:
            continue
        seen_dirs.add(paper_dir_key)

        dag_dir = paper_dir / "dag"
        node_dir = paper_dir / "node_scores"
        graph_csv = paper_dir / "graph_scores.csv"
        if not dag_dir.exists() or not node_dir.exists():
            continue

        meta = _load_paper_meta(paper_dir)

        # discover dag/node trials
        dag_paths = sorted(dag_dir.glob("dag_k*.json"))
        trial_index: List[Tuple[int, int, Path, Path]] = []
        for dag_path in dag_paths:
            stem = dag_path.stem  # dag_k000
            try:
                k = int(stem.split("k")[-1])
            except Exception:
                continue
            per_k = node_dir / f"dag_k{k:03d}"
            if not per_k.exists():
                continue
            for ns_path in sorted(per_k.glob("node_scores_m*.json")):
                try:
                    m = int(ns_path.stem.split("m")[-1])
                except Exception:
                    continue
                trial_index.append((k, m, dag_path, ns_path))

        if not trial_index:
            continue

        key = _paper_key_from_dir(paper_dir, runs_root)
        sheet_default = paper_dir.parent.name if paper_dir.parent != runs_root else ""
        title_default = paper_dir.name
        if "__" in paper_dir.name:
            title_default = paper_dir.name.split("__", 1)[1]
        paper_id = _paper_id_from_meta_or_dir(meta, paper_dir, key)

        papers.append(
            {
                "paper_dir": paper_dir,
                "paper_key": key,
                "sheet": _meta_pick(meta, "sheet", "Sheet", default=sheet_default),
                "paper_id": paper_id,
                "title": _meta_pick(meta, "title", "paper_title", "Paper Title", default=title_default),
                "rating": _meta_pick(meta, "rating", "Rating", default=""),
                "graph_scores_csv": graph_csv if graph_csv.exists() else None,
                "trial_index": trial_index,
                "n_trials_indexed": len(trial_index),
            }
        )

    papers.sort(key=lambda d: (d.get("sheet", ""), d.get("paper_id", ""), d.get("paper_key", "")))
    return papers


def _filter_papers(
    papers: List[Dict[str, Any]],
    *,
    paper_ids: Optional[set[str]] = None,
    num_shards: int = 1,
    shard_index: int = 0,
    max_papers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in papers:
        if paper_ids and p["paper_id"] not in paper_ids and p["paper_key"] not in paper_ids:
            continue
        if num_shards > 1:
            h = int(hashlib.sha1(p["paper_key"].encode()).hexdigest(), 16)
            if (h % num_shards) != shard_index:
                continue
        out.append(p)
    if max_papers is not None:
        out = out[: int(max_papers)]
    return out


def _canonical_edges_from_nodes(nodes: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    edges = set()
    node_ids = {str(n.get("id")) for n in nodes}
    for n in nodes:
        v = str(n.get("id"))
        for u in n.get("parents", []) or []:
            u = str(u)
            if u in node_ids and v in node_ids and u != v:
                edges.add((u, v))
        for c in n.get("children", []) or []:
            c = str(c)
            if c in node_ids and v in node_ids and c != v:
                edges.add((v, c))
    return sorted(edges)


def _rebuild_children_from_parents(nodes: List[Dict[str, Any]]) -> None:
    byid = {str(n.get("id")): n for n in nodes}
    for n in nodes:
        n["children"] = []
        n["parents"] = [str(x) for x in (n.get("parents") or []) if str(x) in byid and str(x) != str(n.get("id"))]
    for n in nodes:
        v = str(n.get("id"))
        for u in n["parents"]:
            byid[u].setdefault("children", [])
            if v not in byid[u]["children"]:
                byid[u]["children"].append(v)
    for n in nodes:
        n["children"] = [str(x) for x in (n.get("children") or []) if str(x) in byid and str(x) != str(n.get("id"))]
        # stable unique
        n["children"] = sorted(set(n["children"]))
        n["parents"] = sorted(set(n["parents"]))


def _dag_copy(dag_json: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(dag_json)


def _node_results_copy(node_results: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(node_results)


def _baseline_node_quality_from_metrics(metrics: Dict[str, float]) -> float:
    vals = [_clip01(metrics.get(k, 0.0)) for k in METRICS]
    return sum(vals) / len(vals) if vals else 0.0


def _apply_node_metric_ablation(
    node_results: Dict[str, Dict[str, Any]],
    variant_kind: str,
    params: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    out = _node_results_copy(node_results)
    if variant_kind == "baseline":
        return out

    if variant_kind == "node_metric_lomo":
        m = str(params["metric"])
        for nid, d in out.items():
            if isinstance(d, dict):
                d[m] = 0.0
        return out

    if variant_kind == "node_metric_keep_only":
        keep = set(params.get("keep_metrics", []))
        for nid, d in out.items():
            if not isinstance(d, dict):
                continue
            for m in METRICS:
                if m not in keep:
                    d[m] = 0.0
        return out

    return out


def _drop_role_nodes(
    dag_json: Dict[str, Any],
    node_results: Dict[str, Dict[str, Any]],
    *,
    role: str,
    reconnect: str = "none",
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    dag = _dag_copy(dag_json)
    nodes = dag.get("nodes") or []
    role = str(role)
    remove_ids = {str(n.get("id")) for n in nodes if str(n.get("role")) == role}
    if not remove_ids:
        return dag, _node_results_copy(node_results)

    keep_nodes = [copy.deepcopy(n) for n in nodes if str(n.get("id")) not in remove_ids]
    byid_old = {str(n.get("id")): n for n in nodes}

    # Optional bridge-through reconnection: connect parents(removed) -> children(removed)
    bridge_edges = set()
    if reconnect == "bridge_through_removed":
        for rid in remove_ids:
            rn = byid_old.get(rid) or {}
            parents = [str(x) for x in (rn.get("parents") or []) if str(x) not in remove_ids]
            children = [str(x) for x in (rn.get("children") or []) if str(x) not in remove_ids]
            for u in parents:
                for v in children:
                    if u != v:
                        bridge_edges.add((u, v))

    keep_ids = {str(n.get("id")) for n in keep_nodes}
    # start from kept nodes' existing parent references (filtered)
    for n in keep_nodes:
        nid = str(n.get("id"))
        n["parents"] = [str(u) for u in (n.get("parents") or []) if str(u) in keep_ids and str(u) != nid]
        n["children"] = [str(v) for v in (n.get("children") or []) if str(v) in keep_ids and str(v) != nid]

    if bridge_edges:
        byid_keep = {str(n.get("id")): n for n in keep_nodes}
        for u, v in bridge_edges:
            if u in byid_keep and v in byid_keep and u != v:
                byid_keep[v].setdefault("parents", [])
                byid_keep[v]["parents"].append(u)

    _rebuild_children_from_parents(keep_nodes)
    dag["nodes"] = keep_nodes

    out_scores = _node_results_copy(node_results)
    for rid in list(out_scores.keys()):
        if str(rid) in remove_ids:
            out_scores.pop(rid, None)
    return dag, out_scores


def _drop_nodes_by_trial_quality_quantile(
    dag_json: Dict[str, Any],
    node_results: Dict[str, Dict[str, Any]],
    *,
    quantile: float,
    which: str,
    reconnect: str = "none",
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    dag = _dag_copy(dag_json)
    nodes = dag.get("nodes") or []
    scored = []
    for n in nodes:
        nid = str(n.get("id"))
        d = node_results.get(nid) or node_results.get(int(nid)) if nid.isdigit() else node_results.get(nid)
        if not isinstance(d, dict):
            d = node_results.get(nid)
        if not isinstance(d, dict):
            continue
        q = _baseline_node_quality_from_metrics({k: _safe_float(d.get(k), 0.5) for k in METRICS})
        scored.append((nid, q))
    if not scored:
        return dag, _node_results_copy(node_results)

    qs = [q for _, q in scored]
    thr = _quantile(qs, quantile)
    if which == "lowest":
        remove = {nid for nid, q in scored if q <= thr}
    else:
        remove = {nid for nid, q in scored if q >= thr}

    # avoid degenerate empty graph when possible
    if len(remove) >= max(0, len(nodes) - 1):
        # keep one node with median-ish quality
        scored_sorted = sorted(scored, key=lambda t: t[1])
        keep_nid = scored_sorted[len(scored_sorted)//2][0]
        remove.discard(keep_nid)

    # implement via role-agnostic drop path
    tmp_scores = _node_results_copy(node_results)
    tmp_dag = _dag_copy(dag_json)
    tmp_nodes = tmp_dag.get("nodes") or []
    keep_nodes = [copy.deepcopy(n) for n in tmp_nodes if str(n.get("id")) not in remove]
    byid_old = {str(n.get("id")): n for n in tmp_nodes}
    bridge_edges = set()
    if reconnect == "bridge_through_removed":
        for rid in remove:
            rn = byid_old.get(rid) or {}
            parents = [str(x) for x in (rn.get("parents") or []) if str(x) not in remove]
            children = [str(x) for x in (rn.get("children") or []) if str(x) not in remove]
            for u in parents:
                for v in children:
                    if u != v:
                        bridge_edges.add((u, v))
    keep_ids = {str(n.get("id")) for n in keep_nodes}
    for n in keep_nodes:
        nid = str(n.get("id"))
        n["parents"] = [str(u) for u in (n.get("parents") or []) if str(u) in keep_ids and str(u) != nid]
        n["children"] = [str(v) for v in (n.get("children") or []) if str(v) in keep_ids and str(v) != nid]
    if bridge_edges:
        byid_keep = {str(n.get("id")): n for n in keep_nodes}
        for u, v in bridge_edges:
            if u in byid_keep and v in byid_keep and u != v:
                byid_keep[v].setdefault("parents", [])
                byid_keep[v]["parents"].append(u)
    _rebuild_children_from_parents(keep_nodes)
    tmp_dag["nodes"] = keep_nodes
    for rid in list(tmp_scores.keys()):
        if str(rid) in remove:
            tmp_scores.pop(rid, None)
    return tmp_dag, tmp_scores


def _apply_dag_and_scores_ablation(
    dag_json: Dict[str, Any],
    node_results: Dict[str, Dict[str, Any]],
    variant_kind: str,
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    # Node metric-only changes
    if variant_kind in {"baseline", "node_metric_lomo", "node_metric_keep_only"}:
        return _dag_copy(dag_json), _apply_node_metric_ablation(node_results, variant_kind, params)

    if variant_kind == "drop_role_nodes":
        return _drop_role_nodes(dag_json, node_results, role=str(params["role"]), reconnect=str(params.get("reconnect", "none")))

    if variant_kind == "drop_nodes_by_trial_quality_quantile":
        return _drop_nodes_by_trial_quality_quantile(
            dag_json,
            node_results,
            quantile=float(params.get("quantile", 0.2)),
            which=str(params.get("which", "lowest")),
            reconnect=str(params.get("reconnect", "none")),
        )

    # Scorer-only variants leave DAG and node scores unchanged here
    return _dag_copy(dag_json), _node_results_copy(node_results)


def _build_kg_from_dag(kgr, dag_json: Dict[str, Any], reconcile: str = "prefer_parents"):
    kg, rep = kgr.DAGValidation.validate_and_build_from_json(
        dag_json,
        reconcile=reconcile,
        strict_roles=True,
        expect_roots=True,
    )
    rep_dict = {
        "errors": list(getattr(rep, "errors", []) or []),
        "warnings": list(getattr(rep, "warnings", []) or []),
        "edge_mismatches": list(getattr(rep, "edge_mismatches", []) or []),
        "stats": dict(getattr(rep, "stats", {}) or {}),
    }
    return kg, rep_dict


def _apply_node_metrics_to_kg(kg, node_results: Dict[str, Dict[str, Any]]) -> None:
    for node_id, result in (node_results or {}).items():
        nid = str(node_id)
        if nid not in kg.nodes:
            continue
        if not isinstance(result, dict):
            continue
        vals = {m: _clip01(_safe_float(result.get(m, 0.5), 0.5)) for m in METRICS}
        kg.update_node_metrics(node_id=nid, **vals)


def _apply_scorer_variant(kgr, kg, variant_kind: str, params: Dict[str, Any]) -> None:
    if variant_kind == "baseline":
        return

    if variant_kind == "edge_weight_zero":
        feat = str(params["feature"])
        if not hasattr(kg.edge_w, feat):
            raise AblationError(f"Unknown edge weight feature: {feat}")
        setattr(kg.edge_w, feat, 0.0)
        kg.recompute_all_confidences()
        return

    if variant_kind == "uniform_role_prior":
        val = float(params.get("value", 0.5))
        # keep table but force default and explicit table same value to isolate role prior content
        kg.role_prior.default_value = val
        try:
            table = dict(getattr(kg.role_prior, "table", {}) or {})
            for k in list(table.keys()):
                table[k] = val
            kg.role_prior.table = table
        except Exception:
            pass
        kg.recompute_all_confidences()
        return

    if variant_kind == "empty_pair_synergy":
        kg.pair_syn.per_pair = {}
        kg.recompute_all_confidences()
        return

    if variant_kind == "penalty_toggle":
        kg.penalty.enabled = bool(params.get("enabled", True))
        kg.recompute_all_confidences()
        return

    if variant_kind == "penalty_mode":
        kg.penalty.agg = str(params["agg"])
        kg.recompute_all_confidences()
        return

    if variant_kind == "penalty_param":
        for k, v in params.items():
            if hasattr(kg.penalty, k):
                setattr(kg.penalty, k, float(v))
        kg.recompute_all_confidences()
        return

    if variant_kind == "drop_edges_by_feature_threshold":
        feat = str(params["feature"])
        op = str(params.get("op", "<"))
        thr = float(params.get("threshold", 0.0))
        # Ensure features computed
        kg.recompute_all_confidences()
        to_remove = []
        for u, v in list(kg.G.edges()):
            f = kg.G[u][v].get("features", {}) or {}
            x = _safe_float(f.get(feat), float("nan"))
            if math.isnan(x):
                continue
            cond = (x < thr) if op == "<" else (x > thr)
            if cond:
                to_remove.append((u, v))
        if to_remove:
            kg.G.remove_edges_from(to_remove)
            kg._mark_topology_dirty()
            kg.recompute_all_confidences()
        return

    if variant_kind == "graph_weight_zero":
        comp = str(params["component"])
        if not hasattr(kg.graph_w, comp):
            raise AblationError(f"Unknown graph component weight: {comp}")
        setattr(kg.graph_w, comp, 0.0)
        return

    if variant_kind == "graph_weight_single_component":
        comp = str(params["component"])
        for k in ["bridge_coverage","best_path","redundancy","fragility","coherence","coverage"]:
            setattr(kg.graph_w, k, 1.0 if k == comp else 0.0)
        return

    # node-metric variants already applied prior to building kg
    if variant_kind in {"node_metric_lomo", "node_metric_keep_only"}:
        return
    if variant_kind in {"drop_role_nodes", "drop_nodes_by_trial_quality_quantile"}:
        return

    # final study variants handled elsewhere
    if variant_kind in {"variance_decomposition", "resample_stability"}:
        return

    raise AblationError(f"Unhandled variant kind: {variant_kind}")


def _edge_summary(kg) -> Dict[str, float]:
    try:
        feats = kg.export_edge_features()
    except Exception:
        feats = []
    if not feats:
        return {
            "n_edges": float(kg.G.number_of_edges()),
            "edge_conf_mean": float("nan"),
            "edge_conf_raw_mean": float("nan"),
            "alignment_mean": float("nan"),
            "role_prior_mean": float("nan"),
            "synergy_mean": float("nan"),
            "trust_parent_mean": float("nan"),
        }
    def mean_key(k: str) -> float:
        vals = [_safe_float(d.get(k), float("nan")) for d in feats]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")
    return {
        "n_edges": float(kg.G.number_of_edges()),
        "edge_conf_mean": mean_key("confidence"),
        "edge_conf_raw_mean": mean_key("confidence_raw"),
        "alignment_mean": mean_key("alignment"),
        "role_prior_mean": mean_key("role_prior"),
        "synergy_mean": mean_key("synergy"),
        "trust_parent_mean": mean_key("trust_parent"),
    }


def _node_role_counts(kg) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for nid in kg.G.nodes():
        role = str(kg.nodes[nid].role)
        out[role] = out.get(role, 0) + 1
    return out


def _score_single_trial(
    *,
    kgr,
    dag_json: Dict[str, Any],
    node_results: Dict[str, Dict[str, Any]],
    variant_kind: str,
    params: Dict[str, Any],
    reconcile: str,
) -> Dict[str, Any]:
    dag2, scores2 = _apply_dag_and_scores_ablation(dag_json, node_results, variant_kind, params)
    kg, rep = _build_kg_from_dag(kgr, dag2, reconcile=reconcile)
    if rep.get("errors"):
        return {
            "success": 0,
            "graph_score": float("nan"),
            "error_stage": "dag_validation",
            "error_message": "; ".join(str(e) for e in rep["errors"])[:500],
            "n_nodes": len((dag2 or {}).get("nodes") or []),
            "n_edges": float("nan"),
        }

    _apply_node_metrics_to_kg(kg, scores2)
    _apply_scorer_variant(kgr, kg, variant_kind, params)

    score, details = kg.graph_score()
    edge_stats = _edge_summary(kg)
    role_counts = _node_role_counts(kg)

    row: Dict[str, Any] = {
        "success": 1,
        "graph_score": _safe_float(score),
        "n_nodes": int(kg.G.number_of_nodes()),
        "n_edges": int(kg.G.number_of_edges()),
        "n_hypothesis": int(role_counts.get("Hypothesis", 0)),
        "n_conclusion": int(role_counts.get("Conclusion", 0)),
    }
    for k, v in details.items():
        row[f"component__{k}"] = _safe_float(v)
    for k, v in edge_stats.items():
        row[f"edge__{k}"] = _safe_float(v)
    return row


def _summarize_trial_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [_safe_float(r.get("graph_score"), float("nan")) for r in rows if str(r.get("success")) in {"1", "True", "true"}]
    scores = [s for s in scores if not math.isnan(s)]
    out: Dict[str, Any] = {
        "n_rows": len(rows),
        "n_success": len(scores),
        "graph_score_mean": _mean(scores),
        "graph_score_std": _std(scores),
        "graph_score_min": min(scores) if scores else float("nan"),
        "graph_score_max": max(scores) if scores else float("nan"),
    }
    # component means
    comp_keys = sorted({k for r in rows for k in r.keys() if k.startswith("component__")})
    for ck in comp_keys:
        vals = [_safe_float(r.get(ck), float("nan")) for r in rows]
        vals = [v for v in vals if not math.isnan(v)]
        out[f"{ck}__mean"] = _mean(vals) if vals else float("nan")
    return out


def _flatten_summary_for_csv(summary: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for k, v in summary.items():
        if isinstance(v, (dict, list)):
            flat[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[k] = v
    return flat


def _list_cached_trials_for_paper(paper: Dict[str, Any]) -> List[Tuple[int, int, Path, Path]]:
    return list(paper["trial_index"])


def _paper_dir_for_variant(out_root: Path, study_id: int, study_short: str, variant_id: str) -> Path:
    return out_root / f"study_{study_id:02d}_{study_short}" / f"variant_{variant_id}"


def _load_trial_inputs(dag_path: Path, ns_path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    dag_json = _json_load(dag_path)
    node_results = _json_load(ns_path)
    if not isinstance(node_results, dict):
        raise AblationError(f"node_scores JSON is not a dict: {ns_path}")
    # Coerce keys to str for consistency (preserve values)
    nr2 = {str(k): v for k, v in node_results.items()}
    return dag_json, nr2


def run_rescoring_study(
    *,
    runs_root: str,
    out_root: str,
    study_id: int,
    paper_ids: Optional[set[str]] = None,
    num_shards: int = 1,
    shard_index: int = 0,
    max_papers: Optional[int] = None,
    reconcile: str = "prefer_parents",
    reuse_cache: bool = True,
    force: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    if study_id not in REGISTRY:
        raise AblationError(f"Unknown study_id={study_id}")
    spec = REGISTRY[study_id]
    if study_id == 6:
        return run_final_study(
            runs_root=runs_root,
            out_root=out_root,
            study_id=study_id,
            paper_ids=paper_ids,
            num_shards=num_shards,
            shard_index=shard_index,
            max_papers=max_papers,
            reuse_cache=reuse_cache,
            force=force,
        )

    kgr = _import_scorer()
    runs_root_p = Path(runs_root)
    out_root_p = Path(out_root)
    all_papers = _discover_papers(runs_root_p)
    papers = _filter_papers(
        all_papers,
        paper_ids=paper_ids,
        num_shards=int(num_shards),
        shard_index=int(shard_index),
        max_papers=max_papers,
    )

    study_dir = out_root_p / f"study_{spec.study_id:02d}_{spec.short_name}"
    study_dir.mkdir(parents=True, exist_ok=True)

    study_manifest = {
        "study_id": spec.study_id,
        "short_name": spec.short_name,
        "title": spec.title,
        "objective": spec.objective,
        "cache_dependency": spec.cache_dependency,
        "runs_root": str(runs_root_p.resolve()),
        "reconcile": reconcile,
        "num_papers_discovered": len(all_papers),
        "num_papers_selected": len(papers),
        "num_shards": num_shards,
        "shard_index": shard_index,
    }
    _json_dump(study_dir / "study_manifest.json", study_manifest)

    variant_summaries: List[Dict[str, Any]] = []

    for variant in spec.variants:
        vdir = _paper_dir_for_variant(out_root_p, spec.study_id, spec.short_name, variant.variant_id)
        papers_dir = vdir / "papers"
        papers_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"[study {study_id}] variant={variant.variant_id} papers={len(papers)}")

        merged_rows: List[Dict[str, Any]] = []
        per_paper_summary_rows: List[Dict[str, Any]] = []

        for paper in papers:
            pdir = papers_dir / paper["paper_key"]
            pdir.mkdir(parents=True, exist_ok=True)
            trials_csv = pdir / "trials.csv"
            summary_json = pdir / "summary.json"

            if reuse_cache and not force and trials_csv.exists() and summary_json.exists():
                paper_rows = _csv_read_rows(trials_csv)
                # cast numerics lazily later
                merged_rows.extend(paper_rows)
                try:
                    ps = _json_load(summary_json)
                except Exception:
                    ps = {}
                per_paper_summary_rows.append({
                    "paper_key": paper["paper_key"],
                    "paper_id": paper["paper_id"],
                    "sheet": paper["sheet"],
                    "title": paper["title"],
                    "rating": paper["rating"],
                    **_flatten_summary_for_csv(ps),
                })
                continue

            rows: List[Dict[str, Any]] = []
            for k, m, dag_path, ns_path in _list_cached_trials_for_paper(paper):
                try:
                    dag_json, node_results = _load_trial_inputs(dag_path, ns_path)
                    row = _score_single_trial(
                        kgr=kgr,
                        dag_json=dag_json,
                        node_results=node_results,
                        variant_kind=variant.kind,
                        params=variant.params,
                        reconcile=reconcile,
                    )
                except Exception as e:
                    row = {
                        "success": 0,
                        "graph_score": float("nan"),
                        "error_stage": "rescoring_exception",
                        "error_message": f"{type(e).__name__}: {e}"[:500],
                    }
                row.update(
                    {
                        "paper_key": paper["paper_key"],
                        "paper_id": paper["paper_id"],
                        "sheet": paper["sheet"],
                        "title": paper["title"],
                        "rating": paper["rating"],
                        "k": int(k),
                        "m": int(m),
                        "dag_path": str(dag_path.relative_to(paper["paper_dir"])),
                        "node_scores_path": str(ns_path.relative_to(paper["paper_dir"])),
                        "variant_id": variant.variant_id,
                        "variant_name": variant.name,
                        "variant_kind": variant.kind,
                        "study_id": spec.study_id,
                        "study_short_name": spec.short_name,
                    }
                )
                rows.append(row)

            # write per-paper outputs
            if rows:
                fieldnames = sorted({k for r in rows for k in r.keys()})
                _csv_write(trials_csv, rows, fieldnames)
            psum = _summarize_trial_rows(rows)
            psum["paper_key"] = paper["paper_key"]
            psum["paper_id"] = paper["paper_id"]
            psum["sheet"] = paper["sheet"]
            psum["title"] = paper["title"]
            psum["rating"] = paper["rating"]
            psum["variant_id"] = variant.variant_id
            psum["variant_name"] = variant.name
            _json_dump(summary_json, psum)
            merged_rows.extend(rows)
            per_paper_summary_rows.append(_flatten_summary_for_csv(psum))

        # write merged outputs for variant
        if merged_rows:
            merged_fields = sorted({k for r in merged_rows for k in r.keys()})
            _csv_write(vdir / "all_trials.csv", merged_rows, merged_fields)
        if per_paper_summary_rows:
            summary_fields = sorted({k for r in per_paper_summary_rows for k in r.keys()})
            _csv_write(vdir / "per_paper_summary.csv", per_paper_summary_rows, summary_fields)

        vsummary = {
            "study_id": spec.study_id,
            "study_short_name": spec.short_name,
            "variant_id": variant.variant_id,
            "variant_name": variant.name,
            "variant_kind": variant.kind,
            "level": variant.level,
            "objective": spec.objective,
            "num_papers_selected": len(papers),
            "variant_params": dict(variant.params),
        }
        if merged_rows:
            vsummary["aggregate"] = _summarize_trial_rows(merged_rows)
            vsummary["rating_eval"] = _rating_eval_from_rows(merged_rows)
        else:
            vsummary["aggregate"] = {"n_rows": 0, "n_success": 0}
            vsummary["rating_eval"] = {}
        _json_dump(vdir / "variant_summary.json", vsummary)
        variant_summaries.append(vsummary)

    # study-wide summary table
    summary_table_rows = []
    for v in variant_summaries:
        agg = v.get("aggregate", {}) or {}
        rev = v.get("rating_eval", {}) or {}
        summary_table_rows.append({
            "study_id": v["study_id"],
            "study_short_name": v["study_short_name"],
            "variant_id": v["variant_id"],
            "variant_name": v["variant_name"],
            "variant_kind": v["variant_kind"],
            "level": v["level"],
            "n_rows": agg.get("n_rows", ""),
            "n_success": agg.get("n_success", ""),
            "graph_score_mean": agg.get("graph_score_mean", ""),
            "graph_score_std": agg.get("graph_score_std", ""),
            "spearman_good_neutral_bad": rev.get("spearman_rating_ordinal", ""),
            "auc_good_vs_bad": rev.get("auc_good_vs_bad", ""),
        })
    if summary_table_rows:
        _csv_write(study_dir / "study_variant_summary.csv", summary_table_rows, list(summary_table_rows[0].keys()))
    _json_dump(study_dir / "study_variant_summary.json", variant_summaries)

    return {
        "study_id": spec.study_id,
        "study_short_name": spec.short_name,
        "num_variants": len(spec.variants),
        "num_papers_selected": len(papers),
        "out_dir": str(study_dir),
    }


def _rating_eval_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Aggregate to per-paper mean score first
    by_paper: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if str(r.get("success")) not in {"1", "True", "true"}:
            continue
        pk = str(r.get("paper_key"))
        score = _safe_float(r.get("graph_score"), float("nan"))
        if math.isnan(score):
            continue
        rec = by_paper.setdefault(pk, {"scores": [], "rating": str(r.get("rating", ""))})
        rec["scores"].append(score)
        if not rec.get("rating"):
            rec["rating"] = str(r.get("rating", ""))
    if len(by_paper) < 2:
        return {"n_papers_eval": len(by_paper)}

    rating_ord_map = {"bad": 0, "neutral": 1, "good": 2}
    xs_ord = []
    ys = []
    auc_labels = []
    auc_scores = []
    for pk, rec in by_paper.items():
        if not rec["scores"]:
            continue
        mean_s = sum(rec["scores"]) / len(rec["scores"])
        rating = str(rec.get("rating", "")).strip().lower()
        ys.append(mean_s)
        if rating in rating_ord_map:
            xs_ord.append(rating_ord_map[rating])
        else:
            xs_ord.append(float("nan"))
        if rating in {"good", "bad"}:
            auc_labels.append(1 if rating == "good" else 0)
            auc_scores.append(mean_s)

    # filter NaNs for spearman
    xs2 = []
    ys2 = []
    for x, y in zip(xs_ord, ys):
        if isinstance(x, float) and math.isnan(x):
            continue
        xs2.append(float(x))
        ys2.append(float(y))

    out = {"n_papers_eval": len(by_paper)}
    if len(xs2) >= 2:
        out["spearman_rating_ordinal"] = _spearman(xs2, ys2)
    if len(auc_labels) >= 2 and len(set(auc_labels)) == 2:
        out["auc_good_vs_bad"] = _auc_good_bad(auc_labels, auc_scores)
    return out


def _load_cached_graph_scores_for_paper(paper: Dict[str, Any]) -> List[Dict[str, Any]]:
    csv_path = paper.get("graph_scores_csv")
    if not csv_path or not Path(csv_path).exists():
        return []
    rows = _csv_read_rows(Path(csv_path))
    out = []
    for r in rows:
        try:
            success = str(r.get("success", "1")) in {"1", "True", "true"}
            if not success:
                continue
            out.append({
                "k": int(r.get("k")),
                "m": int(r.get("m")),
                "graph_score": _safe_float(r.get("graph_score"), float("nan")),
            })
        except Exception:
            continue
    return out


def _variance_decomposition_rows(papers: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    per_paper = []
    between_ratios = []
    within_ratios = []
    for p in papers:
        rows = _load_cached_graph_scores_for_paper(p)
        rows = [r for r in rows if not math.isnan(r["graph_score"])]
        if len(rows) < 2:
            continue
        scores = [r["graph_score"] for r in rows]
        total_var = statistics.pvariance(scores) if len(scores) > 1 else 0.0

        by_k: Dict[int, List[float]] = {}
        for r in rows:
            by_k.setdefault(r["k"], []).append(r["graph_score"])
        k_means = [sum(v)/len(v) for v in by_k.values() if v]
        # Law of total variance approximations under finite samples
        between = statistics.pvariance(k_means) if len(k_means) > 1 else 0.0
        within_means = [statistics.pvariance(v) if len(v) > 1 else 0.0 for v in by_k.values()]
        within = sum(within_means) / len(within_means) if within_means else 0.0

        # decomposition mismatch due unequal M and finite sample; record both
        br = (between / total_var) if total_var > 0 else float("nan")
        wr = (within / total_var) if total_var > 0 else float("nan")
        between_ratios.append(br) if not math.isnan(br) else None
        within_ratios.append(wr) if not math.isnan(wr) else None

        per_paper.append({
            "paper_key": p["paper_key"],
            "paper_id": p["paper_id"],
            "sheet": p["sheet"],
            "title": p["title"],
            "rating": p["rating"],
            "n_trials": len(rows),
            "n_unique_k": len(by_k),
            "total_var": total_var,
            "between_dag_var_of_means": between,
            "avg_within_dag_var": within,
            "between_over_total": br,
            "within_over_total": wr,
            "score_mean": sum(scores)/len(scores),
            "score_std": math.sqrt(total_var),
        })

    overall = {
        "n_papers": len(per_paper),
        "median_between_over_total": _quantile([x for x in between_ratios if not math.isnan(x)], 0.5) if between_ratios else float("nan"),
        "median_within_over_total": _quantile([x for x in within_ratios if not math.isnan(x)], 0.5) if within_ratios else float("nan"),
        "mean_between_over_total": _mean([x for x in between_ratios if not math.isnan(x)]),
        "mean_within_over_total": _mean([x for x in within_ratios if not math.isnan(x)]),
    }
    return per_paper, overall


def _resample_stability_summary(papers: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    per_paper = []
    paper_means = []
    paper_stds = []
    cv_vals = []
    for p in papers:
        rows = _load_cached_graph_scores_for_paper(p)
        vals = [r["graph_score"] for r in rows if not math.isnan(r["graph_score"])]
        if len(vals) < 2:
            continue
        mu = sum(vals)/len(vals)
        sd = statistics.pstdev(vals)
        cv = sd / abs(mu) if abs(mu) > 1e-12 else float("nan")
        by_k = {}
        for r in rows:
            if math.isnan(r["graph_score"]):
                continue
            by_k.setdefault(r["k"], []).append(r["graph_score"])
        k_means = [sum(v)/len(v) for v in by_k.values() if v]
        k_range = (max(k_means)-min(k_means)) if k_means else float("nan")
        per_paper.append({
            "paper_key": p["paper_key"],
            "paper_id": p["paper_id"],
            "sheet": p["sheet"],
            "title": p["title"],
            "rating": p["rating"],
            "n_trials": len(vals),
            "score_mean": mu,
            "score_std": sd,
            "score_cv": cv,
            "k_mean_range": k_range,
            "n_unique_k": len(by_k),
        })
        paper_means.append(mu)
        paper_stds.append(sd)
        if not math.isnan(cv):
            cv_vals.append(cv)
    overall = {
        "n_papers": len(per_paper),
        "mean_score_mean": _mean(paper_means),
        "mean_score_std": _mean(paper_stds),
        "median_score_cv": _quantile(cv_vals, 0.5) if cv_vals else float("nan"),
        "mean_score_cv": _mean(cv_vals),
    }
    # ranking stability by k means if possible
    # compute pairwise Kendall/Spearman across K-specific rankings over common papers
    # Build paper->kmean matrix
    all_ks = sorted({r["k"] for p in papers for r in _load_cached_graph_scores_for_paper(p)})
    pmap = {}
    for p in papers:
        rows = _load_cached_graph_scores_for_paper(p)
        by_k = {}
        for r in rows:
            if math.isnan(r["graph_score"]):
                continue
            by_k.setdefault(r["k"], []).append(r["graph_score"])
        if by_k:
            pmap[p["paper_key"]] = {k: sum(v)/len(v) for k, v in by_k.items()}
    kt_vals = []
    sp_vals = []
    for i in range(len(all_ks)):
        for j in range(i+1, len(all_ks)):
            k1, k2 = all_ks[i], all_ks[j]
            xs, ys = [], []
            for pk, km in pmap.items():
                if k1 in km and k2 in km:
                    xs.append(km[k1]); ys.append(km[k2])
            if len(xs) >= 3:
                kt = _kendall_tau_a(xs, ys)
                sp = _spearman(xs, ys)
                if not math.isnan(kt):
                    kt_vals.append(kt)
                if not math.isnan(sp):
                    sp_vals.append(sp)
    if kt_vals:
        overall["mean_kendall_tau_across_k_rankings"] = _mean(kt_vals)
        overall["mean_spearman_across_k_rankings"] = _mean(sp_vals)
    return per_paper, overall


def run_final_study(
    *,
    runs_root: str,
    out_root: str,
    study_id: int = 6,
    paper_ids: Optional[set[str]] = None,
    num_shards: int = 1,
    shard_index: int = 0,
    max_papers: Optional[int] = None,
    reuse_cache: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    spec = REGISTRY[study_id]
    runs_root_p = Path(runs_root)
    out_root_p = Path(out_root)
    all_papers = _discover_papers(runs_root_p)
    papers = _filter_papers(
        all_papers, paper_ids=paper_ids, num_shards=num_shards, shard_index=shard_index, max_papers=max_papers
    )
    study_dir = out_root_p / f"study_{spec.study_id:02d}_{spec.short_name}"
    study_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(
        study_dir / "study_manifest.json",
        {
            "study_id": spec.study_id,
            "short_name": spec.short_name,
            "title": spec.title,
            "objective": spec.objective,
            "runs_root": str(runs_root_p.resolve()),
            "num_papers_discovered": len(all_papers),
            "num_papers_selected": len(papers),
            "cache_dependency": spec.cache_dependency,
        },
    )

    for variant in spec.variants:
        vdir = _paper_dir_for_variant(out_root_p, spec.study_id, spec.short_name, variant.variant_id)
        vdir.mkdir(parents=True, exist_ok=True)
        summary_json = vdir / "variant_summary.json"
        if reuse_cache and not force and summary_json.exists():
            continue

        if variant.kind == "variance_decomposition":
            per_paper_rows, overall = _variance_decomposition_rows(papers)
        elif variant.kind == "resample_stability":
            per_paper_rows, overall = _resample_stability_summary(papers)
        else:
            raise AblationError(f"Unknown final-study variant kind: {variant.kind}")

        if per_paper_rows:
            _csv_write(vdir / "per_paper_summary.csv", per_paper_rows, list(per_paper_rows[0].keys()))
        summary = {
            "study_id": spec.study_id,
            "study_short_name": spec.short_name,
            "variant_id": variant.variant_id,
            "variant_name": variant.name,
            "variant_kind": variant.kind,
            "num_papers_selected": len(papers),
            "aggregate": overall,
        }
        _json_dump(summary_json, summary)

    # Build study summary
    rows = []
    for variant in spec.variants:
        vdir = _paper_dir_for_variant(out_root_p, spec.study_id, spec.short_name, variant.variant_id)
        sj = vdir / "variant_summary.json"
        if not sj.exists():
            continue
        data = _json_load(sj)
        agg = data.get("aggregate", {}) or {}
        row = {
            "study_id": spec.study_id,
            "study_short_name": spec.short_name,
            "variant_id": variant.variant_id,
            "variant_name": variant.name,
            "variant_kind": variant.kind,
        }
        row.update({k: v for k, v in agg.items() if not isinstance(v, (dict, list))})
        rows.append(row)
    if rows:
        _csv_write(study_dir / "study_variant_summary.csv", rows, list(rows[0].keys()))
    _json_dump(study_dir / "study_variant_summary.json", rows)

    return {
        "study_id": spec.study_id,
        "study_short_name": spec.short_name,
        "num_variants": len(spec.variants),
        "num_papers_selected": len(papers),
        "out_dir": str(study_dir),
    }


def list_studies() -> List[Dict[str, Any]]:
    out = []
    for sid in sorted(REGISTRY):
        s = REGISTRY[sid]
        out.append(
            {
                "study_id": s.study_id,
                "short_name": s.short_name,
                "title": s.title,
                "objective": s.objective,
                "cache_dependency": s.cache_dependency,
                "variants": [
                    {
                        "variant_id": v.variant_id,
                        "name": v.name,
                        "level": v.level,
                        "kind": v.kind,
                        "params": dict(v.params),
                    }
                    for v in s.variants
                ],
            }
        )
    return out
