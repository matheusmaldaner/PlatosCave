from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from ablation_studies import (
        AblationError,
        _csv_write,
        _discover_papers,
        _filter_papers,
        _json_dump,
        _json_load,
        _load_trial_inputs,
        _safe_float,
    )
    from grid_search import (
        SUPPORTED_PARAM_PATTERNS,
        _aggregate_config_rows,
        _import_scorer,
        _paper_result_fieldnames,
        _score_trial_for_config,
        _search_space_hash,
        _summarize_trial_rows_compact,
        _constraint_holds,
        _validate_constraints,
    )
except Exception:
    from .ablation_studies import (
        AblationError,
        _csv_write,
        _discover_papers,
        _filter_papers,
        _json_dump,
        _json_load,
        _load_trial_inputs,
        _safe_float,
    )
    from .grid_search import (
        SUPPORTED_PARAM_PATTERNS,
        _aggregate_config_rows,
        _import_scorer,
        _paper_result_fieldnames,
        _score_trial_for_config,
        _search_space_hash,
        _summarize_trial_rows_compact,
        _constraint_holds,
        _validate_constraints,
    )


def _import_optuna():
    try:
        import optuna  # type: ignore
    except Exception as e:
        raise AblationError(
            "Optuna is not installed. Install it in the experiment environment, e.g. "
            "'.venv-exp/bin/pip install optuna'."
        ) from e
    return optuna


def _load_optuna_space(path: str) -> Dict[str, Any]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise AblationError("Optuna search space must be a JSON object.")
    params = obj.get("params")
    if not isinstance(params, dict) or not params:
        raise AblationError("Optuna search space must define a non-empty 'params' object.")

    norm_params: Dict[str, Dict[str, Any]] = {}
    for name, spec in params.items():
        norm_params[str(name)] = _normalize_distribution_spec(str(name), spec)

    constraints = obj.get("constraints", [])
    if constraints is None:
        constraints = []
    if not isinstance(constraints, list):
        raise AblationError("'constraints' must be a list when provided.")
    # validate only names; numeric checks happen after suggestion
    _validate_constraints({k: [0] for k in norm_params.keys()}, constraints)

    out = {
        "name": str(obj.get("name", "optuna_search")).strip() or "optuna_search",
        "description": str(obj.get("description", "") or "").strip(),
        "objective_metric": str(obj.get("objective_metric", "auc_good_vs_bad")).strip() or "auc_good_vs_bad",
        "direction": str(obj.get("direction", "maximize")).strip().lower() or "maximize",
        "normalize_metric_weights": bool(obj.get("normalize_metric_weights", False)),
        "include_component_means": bool(obj.get("include_component_means", True)),
        "sampler": obj.get("sampler", {}) or {},
        "pruner": obj.get("pruner", {}) or {},
        "params": norm_params,
        "constraints": constraints,
        "save_top_k_per_paper": int(obj.get("save_top_k_per_paper", 3) or 0),
    }
    if out["direction"] not in {"maximize", "minimize"}:
        raise AblationError("direction must be 'maximize' or 'minimize'.")
    valid_metrics = {"auc_good_vs_bad", "spearman_good_neutral_bad", "graph_score_mean"}
    if out["objective_metric"] not in valid_metrics:
        raise AblationError(f"Unsupported objective_metric '{out['objective_metric']}'.")
    return out


def _normalize_distribution_spec(name: str, spec: Any) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise AblationError(f"Param '{name}' must map to an object.")
    dtype = str(spec.get("type", "")).strip().lower()
    if dtype == "float":
        if "low" not in spec or "high" not in spec:
            raise AblationError(f"Float param '{name}' must include low/high.")
        out = {"type": "float", "low": float(spec["low"]), "high": float(spec["high"])}
        if "step" in spec and spec["step"] is not None:
            out["step"] = float(spec["step"])
        if "log" in spec:
            out["log"] = bool(spec["log"])
        return out
    if dtype == "int":
        if "low" not in spec or "high" not in spec:
            raise AblationError(f"Int param '{name}' must include low/high.")
        out = {"type": "int", "low": int(spec["low"]), "high": int(spec["high"])}
        if "step" in spec and spec["step"] is not None:
            out["step"] = int(spec["step"])
        if "log" in spec:
            out["log"] = bool(spec["log"])
        return out
    if dtype == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AblationError(f"Categorical param '{name}' must include non-empty choices.")
        return {"type": "categorical", "choices": list(choices)}
    if dtype == "fixed":
        if "value" not in spec:
            raise AblationError(f"Fixed param '{name}' must include value.")
        return {"type": "fixed", "value": spec["value"]}
    raise AblationError(f"Param '{name}' has unsupported type '{dtype}'.")


def _build_sampler(optuna, spec: Dict[str, Any]):
    cfg = dict(spec.get("sampler", {}) or {})
    stype = str(cfg.get("type", "tpe")).strip().lower()
    seed = cfg.get("seed", None)
    if stype == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=int(cfg.get("n_startup_trials", 10)),
            multivariate=bool(cfg.get("multivariate", True)),
            constant_liar=bool(cfg.get("constant_liar", False)),
        )
    if stype == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise AblationError(f"Unsupported sampler type '{stype}'.")


def _build_pruner(optuna, spec: Dict[str, Any]):
    cfg = dict(spec.get("pruner", {}) or {})
    ptype = str(cfg.get("type", "median")).strip().lower()
    if ptype in {"", "none", "nop"}:
        return optuna.pruners.NopPruner()
    if ptype == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(cfg.get("n_startup_trials", 5)),
            n_warmup_steps=int(cfg.get("n_warmup_steps", 5)),
            interval_steps=int(cfg.get("interval_steps", 1)),
        )
    raise AblationError(f"Unsupported pruner type '{ptype}'.")


def _suggest_value(optuna_trial, name: str, spec: Dict[str, Any]) -> Any:
    dtype = spec["type"]
    if dtype == "float":
        kwargs = {"low": float(spec["low"]), "high": float(spec["high"])}
        if "step" in spec:
            kwargs["step"] = float(spec["step"])
        elif "log" in spec:
            kwargs["log"] = bool(spec["log"])
        return optuna_trial.suggest_float(name, **kwargs)
    if dtype == "int":
        kwargs = {"low": int(spec["low"]), "high": int(spec["high"])}
        if "step" in spec:
            kwargs["step"] = int(spec["step"])
        elif "log" in spec:
            kwargs["log"] = bool(spec["log"])
        return optuna_trial.suggest_int(name, **kwargs)
    if dtype == "categorical":
        return optuna_trial.suggest_categorical(name, list(spec["choices"]))
    if dtype == "fixed":
        return spec["value"]
    raise AblationError(f"Unhandled distribution type '{dtype}'.")


def _paper_cache_key(paper: Dict[str, Any]) -> str:
    return str(paper["paper_dir"].resolve())


def _make_trial_index_signature(paper: Dict[str, Any]) -> Tuple[Tuple[int, int, str, str], ...]:
    out = []
    for k, m, dag_path, ns_path in paper["trial_index"]:
        out.append((int(k), int(m), str(dag_path), str(ns_path)))
    return tuple(out)


@lru_cache(maxsize=8)
def _load_paper_trial_inputs_cached(sig: Tuple[Tuple[int, int, str, str], ...]):
    out = []
    for k, m, dag_path_str, ns_path_str in sig:
        dag_json, node_results = _load_trial_inputs(Path(dag_path_str), Path(ns_path_str))
        out.append((k, m, dag_json, node_results))
    return tuple(out)


def _evaluate_config_on_papers(
    *,
    papers: List[Dict[str, Any]],
    config_meta: Dict[str, Any],
    search_space: Dict[str, Any],
    reconcile: str,
    trial=None,
):
    kgr = _evaluate_config_on_papers._kgr  # type: ignore[attr-defined]
    include_component_means = bool(search_space.get("include_component_means", True))
    normalize_metric_weights = bool(search_space.get("normalize_metric_weights", False))
    per_paper_rows: List[Dict[str, Any]] = []

    for idx, paper in enumerate(papers, 1):
        trial_inputs = _load_paper_trial_inputs_cached(_make_trial_index_signature(paper))
        trial_rows = []
        for _, _, dag_json, node_results in trial_inputs:
            try:
                row = _score_trial_for_config(
                    kgr=kgr,
                    dag_json=dag_json,
                    node_results=node_results,
                    config_params=config_meta["params"],
                    reconcile=reconcile,
                    normalize_metric_weights=normalize_metric_weights,
                )
            except Exception:
                row = {"success": 0, "graph_score": float("nan")}
            trial_rows.append(row)
        summary = _summarize_trial_rows_compact(
            trial_rows,
            include_component_means=include_component_means,
        )
        summary.update(
            {
                "config_id": config_meta["config_id"],
                "paper_key": paper["paper_key"],
                "paper_id": paper["paper_id"],
                "sheet": paper["sheet"],
                "title": paper["title"],
                "rating": paper["rating"],
            }
        )
        per_paper_rows.append(summary)

        if trial is not None:
            partial = _aggregate_config_rows(
                per_paper_rows,
                config_index={config_meta["config_id"]: config_meta},
                search_space=search_space,
            )[0]
            metric = _safe_float(partial.get(search_space["objective_metric"]), float("nan"))
            if not math.isnan(metric):
                trial.report(metric, step=idx)
                if trial.should_prune():
                    optuna = _import_optuna()
                    raise optuna.TrialPruned()

    summary = _aggregate_config_rows(
        per_paper_rows,
        config_index={config_meta["config_id"]: config_meta},
        search_space=search_space,
    )[0]
    return summary, per_paper_rows


def _trial_row_from_frozen(trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "trial_number": trial.number,
        "state": str(trial.state.name),
        "value": trial.value,
    }
    summary = dict(trial.user_attrs.get("summary", {}) or {})
    row.update(summary)
    params = dict(trial.params)
    row["params_json"] = json.dumps(params, sort_keys=True, ensure_ascii=False)
    for pname in sorted(search_space["params"].keys()):
        row[f"param__{pname.replace('.', '_')}"] = params.get(pname, "")
    return row


def run_optuna_search(
    *,
    runs_root: str,
    out_root: str,
    search_space_path: str,
    paper_ids: Optional[set[str]] = None,
    num_shards: int = 1,
    shard_index: int = 0,
    max_papers: Optional[int] = None,
    reconcile: str = "prefer_parents",
    n_trials: int = 50,
    n_jobs: int = 1,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    reuse_cache: bool = True,
    force: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    optuna = _import_optuna()
    runs_root_p = Path(runs_root)
    out_root_p = Path(out_root)
    search_space = _load_optuna_space(search_space_path)
    search_name = study_name or search_space["name"]
    search_dir = out_root_p / search_name
    search_dir.mkdir(parents=True, exist_ok=True)
    search_hash = _search_space_hash(search_space)

    manifest_path = search_dir / "study_manifest.json"
    if reuse_cache and manifest_path.exists():
        manifest = _json_load(manifest_path)
        prev_hash = str(manifest.get("search_space_hash", ""))
        if prev_hash and prev_hash != search_hash and not force:
            raise AblationError(
                f"Output directory {search_dir} already contains a different Optuna search space. "
                "Use --force or a different --out-root/study name."
            )

    all_papers = _discover_papers(runs_root_p)
    papers = _filter_papers(
        all_papers,
        paper_ids=paper_ids,
        num_shards=int(num_shards),
        shard_index=int(shard_index),
        max_papers=max_papers,
    )

    # Import scorer once in this process; worker threads share it.
    _evaluate_config_on_papers._kgr = _import_scorer()

    storage_url = storage or f"sqlite:///{(search_dir / 'optuna_study.sqlite3').resolve()}"
    sampler = _build_sampler(optuna, search_space)
    pruner = _build_pruner(optuna, search_space)
    study = optuna.create_study(
        study_name=search_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        direction=search_space["direction"],
        load_if_exists=True,
    )

    _json_dump(
        manifest_path,
        {
            "study_name": search_name,
            "runs_root": str(runs_root_p.resolve()),
            "search_space_path": str(Path(search_space_path).resolve()),
            "search_space_hash": search_hash,
            "num_papers_discovered": len(all_papers),
            "num_papers_selected": len(papers),
            "num_shards": int(num_shards),
            "shard_index": int(shard_index),
            "n_trials_requested": int(n_trials),
            "n_jobs": int(n_jobs),
            "storage": storage_url,
            "reconcile": reconcile,
            "search_space": search_space,
        },
    )

    def objective(trial):
        params = {name: _suggest_value(trial, name, spec) for name, spec in search_space["params"].items()}
        if search_space["constraints"]:
            for constraint in search_space["constraints"]:
                if not _constraint_holds(params, constraint):
                    raise optuna.TrialPruned()

        config_id = f"trial_{trial.number:06d}"
        config_meta = {
            "config_index": trial.number,
            "config_id": config_id,
            "config_hash": f"trial_{trial.number:06d}",
            "params": params,
        }
        summary, _ = _evaluate_config_on_papers(
            papers=papers,
            config_meta=config_meta,
            search_space=search_space,
            reconcile=reconcile,
            trial=trial,
        )
        value = _safe_float(summary.get(search_space["objective_metric"]), float("nan"))
        if math.isnan(value):
            raise optuna.TrialPruned()
        trial.set_user_attr("summary", summary)
        return value

    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=int(n_trials), n_jobs=int(n_jobs))

    trials = list(study.trials)
    trial_rows = [_trial_row_from_frozen(t, search_space) for t in trials]
    if trial_rows:
        fields = sorted({k for row in trial_rows for k in row.keys()})
        preferred = ["trial_number", "state", "value", "n_papers", "n_rows", "n_success", "graph_score_mean", "graph_score_std", "spearman_good_neutral_bad", "auc_good_vs_bad", "params_json"]
        ordered = preferred + [f for f in fields if f not in preferred]
        _csv_write(search_dir / "trials.csv", trial_rows, ordered)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_sorted = sorted(
        completed,
        key=lambda t: float(t.value if t.value is not None else (-math.inf if search_space["direction"] == "maximize" else math.inf)),
        reverse=(search_space["direction"] == "maximize"),
    )

    top_trials_json = []
    for t in completed_sorted[: max(1, search_space.get("save_top_k_per_paper", 3))]:
        top_trials_json.append(
            {
                "trial_number": t.number,
                "value": t.value,
                "params": dict(t.params),
                "summary": dict(t.user_attrs.get("summary", {}) or {}),
            }
        )
    _json_dump(search_dir / "top_trials.json", top_trials_json)

    if completed_sorted:
        best = completed_sorted[0]
        best_payload = {
            "trial_number": best.number,
            "value": best.value,
            "params": dict(best.params),
            "summary": dict(best.user_attrs.get("summary", {}) or {}),
        }
        _json_dump(search_dir / "best_trial.json", best_payload)

        top_k = max(0, int(search_space.get("save_top_k_per_paper", 0)))
        if top_k > 0:
            per_paper_dir = search_dir / "top_trial_per_paper"
            per_paper_dir.mkdir(parents=True, exist_ok=True)
            for t in completed_sorted[:top_k]:
                config_meta = {
                    "config_index": t.number,
                    "config_id": f"trial_{t.number:06d}",
                    "config_hash": f"trial_{t.number:06d}",
                    "params": dict(t.params),
                }
                _, per_paper_rows = _evaluate_config_on_papers(
                    papers=papers,
                    config_meta=config_meta,
                    search_space=search_space,
                    reconcile=reconcile,
                    trial=None,
                )
                if per_paper_rows:
                    _csv_write(
                        per_paper_dir / f"trial_{t.number:06d}_per_paper.csv",
                        per_paper_rows,
                        _paper_result_fieldnames(search_space),
                    )

    return {
        "study_name": search_name,
        "num_trials_total": len(trials),
        "num_trials_complete": len(completed),
        "num_papers_selected": len(papers),
        "objective_metric": search_space["objective_metric"],
        "direction": search_space["direction"],
        "storage": storage_url,
        "out_dir": str(search_dir),
    }


def supported_parameter_patterns() -> Dict[str, str]:
    return dict(SUPPORTED_PARAM_PATTERNS)
