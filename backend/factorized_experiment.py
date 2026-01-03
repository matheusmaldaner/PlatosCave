"""backend.factorized_experiment

Factorized resampling experiment driver (offline/batch).

Implements the pipeline structure you described:
  1) Systematic paper collection (external; we read a spreadsheet)
  2) DAG extraction & validation (K resamples per paper)
  3) Node scoring pipeline (M resamples per DAG)
  4) Graph scoring pipeline (score for each of the K*M trials)

Key properties:
  - Factorized resampling: K DAGs, and for each DAG, M independent node-score
    resamples.
  - Writes intermediate artifacts for auditability.
  - Produces per-DAG and global summary statistics.

NOTE: This module deliberately avoids importing backend/main.py to prevent
stdout UI protocol noise.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import csv
import logging
import traceback
from collections import Counter
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import LLMClient, LLMConfig
from .paper_io import extract_text_from_pdf
from .prompts import (
    build_fact_dag_prompt,
    build_claim_verification_prompt_exa,
    build_claim_verification_prompt_llm_only,
    parse_verification_result,
    build_json_repair_prompt,
)


REQUIRED_METRICS = (
    "credibility",
    "relevance",
    "evidence_strength",
    "method_rigor",
    "reproducibility",
    "citation_support",
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


async def _complete_json(llm: LLMClient, prompt: str) -> str:
    """
    Best-effort JSON-mode completion. Falls back to plain completion if the client
    doesn't support structured output.
    """
    fn = getattr(llm, "complete_json", None)
    if callable(fn):
        return await fn(prompt)
    return await llm.complete(prompt)


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


class LLMJsonParseError(ValueError):
    def __init__(self, message: str, *, last_content: str, last_error: str):
        super().__init__(message)
        self.last_content = last_content
        self.last_error = last_error


def _strip_code_fences(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        # drop opening fence line
        if lines:
            lines = lines[1:]
        # drop closing fence line if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _normalize_jsonish_text(s: str) -> str:
    if not s:
        return s
    s = s.replace("\ufeff", "")  # BOM
    # common “smart quotes” that break JSON
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return s


def _json_extract_object(text: str) -> str:
    """Extract the first balanced JSON object substring.

    More robust than (first '{', last '}'), and avoids trailing text / extra braces.
    """
    if not text:
        raise ValueError("Empty response")

    text = _strip_code_fences(_normalize_jsonish_text(text))

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object start found")

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Unbalanced JSON braces (no complete object found)")


def _summ_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    if len(values) == 1:
        v = float(values[0])
        return {"n": 1, "mean": v, "std": 0.0, "min": v, "max": v}
    return {
        "n": len(values),
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _default_metrics_for_role(role: str) -> Dict[str, float]:
    # Convention used in backend/main.py: hypotheses are not verified; they start as neutral.
    if (role or "").lower() == "hypothesis":
        return {k: 0.5 for k in REQUIRED_METRICS}
    return {k: 0.5 for k in REQUIRED_METRICS}


async def exa_retrieve(claim: str, *, k: int = 6) -> str:
    """Exa retrieval wrapper (optional).

    If k <= 0, retrieval is disabled and this returns an empty context string.
    """
    if int(k) <= 0:
        return ""

    from exa_py import Exa

    exa_api_key = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        raise RuntimeError("EXA_API_KEY is not set (required only when retrieval_mode='exa' and exa_k>0)")
    exa = Exa(api_key=exa_api_key)

    def _run() -> str:
        res = exa.search_and_contents(
            claim,
            num_results=int(k),
            text={"max_characters": 1200},
        )
        lines: List[str] = []
        for i, r in enumerate(getattr(res, "results", []) or [], 1):
            title = getattr(r, "title", "") or ""
            url = getattr(r, "url", "") or ""
            snippet = ""
            if getattr(r, "text", None):
                snippet = (r.text or "")[:400].replace("\n", " ")
            lines.append(f"[EXA{i}] {title}\n{url}\nSnippet: {snippet}\n")
        return "\n".join(lines)

    return await asyncio.to_thread(_run)


async def _llm_json_or_repair(
    llm: LLMClient,
    *,
    prompt: str,
    repair_prompt_builder,
    max_repairs: int = 1,
) -> Dict[str, Any]:
    """Ask LLM for JSON and attempt a single repair on parse failure."""

    last_err: Optional[Exception] = None
    content = await _complete_json(llm, prompt)
    for attempt in range(max_repairs + 1):
        try:
            cleaned = _strip_code_fences(_normalize_jsonish_text(content))
            candidate = _json_extract_object(cleaned)

            # Remove trailing commas before } or ] (common LLM mistake)
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

            obj = json.loads(candidate)
            if not isinstance(obj, dict):
                raise ValueError("Parsed JSON is not an object")
            return obj

        except Exception as e:
            last_err = e
            if attempt >= max_repairs:
                break
            content = await _complete_json(llm, repair_prompt_builder(content, str(e)))
    raise LLMJsonParseError(
        f"Failed to parse JSON after repair: {last_err}",
        last_content=str(content),
        last_error=str(last_err),
    )


def _build_dag_repair_prompt(malformed: str, error: str) -> str:
    return (
        "The following response contains malformed JSON that failed to parse.\n\n"
        f"ERROR: {error}\n\n"
        "Please output ONLY a valid JSON object matching this schema:\n"
        '{"nodes": [{"id": 0, "text": "...", "role": "Hypothesis", "parents": null, "children": [1]}]}\n\n'
        "Rules:\n"
        "- Output ONLY the JSON object (no markdown).\n"
        "- Use key 'nodes' only.\n"
        "- Each node must have keys: id, text, role, parents, children.\n"
        "- id starts at 0 and is sequential.\n"
        "- Node 0 MUST be role Hypothesis with parents=null.\n\n"
        "ORIGINAL RESPONSE (may be truncated):\n"
        + malformed[:4000]
    )


async def extract_dag_once(
    llm: LLMClient,
    *,
    raw_text: str,
    max_nodes: int,
    max_json_repairs: int = 1,
) -> Dict[str, Any]:
    prompt = build_fact_dag_prompt(raw_text, max_nodes=max_nodes)
    return await _llm_json_or_repair(
        llm,
        prompt=prompt,
        repair_prompt_builder=_build_dag_repair_prompt,
        max_repairs=max(0, int(max_json_repairs)),
    )


async def score_single_node_once(
    llm: LLMClient,
    *,
    node_id: str,
    node_text: str,
    node_role: str,
    exa_k: int,
    retrieval_mode: str = "llm",  # "exa" | "llm"
    max_json_repairs: int = 1,
) -> Dict[str, Any]:

    """Return the full verification JSON for a node (includes required metrics)."""

    # Hypothesis nodes are not verified; keep a neutral prior.
    if (node_role or "").lower() == "hypothesis":
        base = _default_metrics_for_role(node_role)
        return {**base, "sources_checked": [], "verification_summary": "hypothesis_not_verified", "confidence_level": "n/a"}

    max_json_repairs = max(0, int(max_json_repairs))

    retrieval_mode = (retrieval_mode or "exa").lower().strip()

    if retrieval_mode == "exa":
        ctx = await exa_retrieve(node_text, k=exa_k)
        if not ctx:
            # If Exa retrieval is disabled/empty, degrade gracefully but make it explicit.
            # (You can also raise here if you prefer strict behavior.)
            prompt = build_claim_verification_prompt_llm_only(node_text, node_role)
        else:
            prompt = build_claim_verification_prompt_exa(node_text, node_role, claim_context=ctx)

    elif retrieval_mode == "llm":
        prompt = build_claim_verification_prompt_llm_only(node_text, node_role)

    else:
        raise ValueError(f"Unknown retrieval_mode={retrieval_mode!r}. Expected 'exa' or 'llm'.")

    # Primary parse
    content = await _complete_json(llm, prompt)
    parsed = parse_verification_result(content)
    if parsed is not None:
        for k in REQUIRED_METRICS:
            parsed[k] = _clip01(float(parsed[k]))
        return parsed

    # Iterative repair passes (each pass gets the last faulty output + last parse/validation error).
    last_text = content
    last_err = "parse_verification_result returned None"

    for attempt in range(1, max_json_repairs + 1):
        repair_prompt = build_json_repair_prompt(last_text, f"{last_err} (repair attempt {attempt}/{max_json_repairs})")
        last_text = await _complete_json(llm, repair_prompt)

        parsed2 = parse_verification_result(last_text)
        if parsed2 is not None:
            for k in REQUIRED_METRICS:
                parsed2[k] = _clip01(float(parsed2[k]))
            return parsed2

        # Best-effort error message for the next repair attempt.
        try:
            _ = json.loads(_json_extract_object(last_text))
            last_err = "JSON parsed but failed verification schema/constraints (missing keys and/or out-of-range metrics)"
        except Exception as e:
            last_err = str(e)

    # Hard fallback: neutral metrics but keep traceability.
    base = _default_metrics_for_role(node_role)
    return {
        **base,
        "sources_checked": [],
        "verification_summary": "verification_parse_failed",
        "confidence_level": "failed",
        "raw_response": content[:2000],
    }


class NodeScoringError(Exception):
    def __init__(self, node_id: str, role: str, original: BaseException):
        super().__init__(f"node_id={node_id} role={role} err={type(original).__name__}: {original}")
        self.node_id = node_id
        self.role = role
        self.original = original


class NodeScoringBatchError(Exception):
    def __init__(self, errors: List[NodeScoringError]):
        self.errors = errors
        preview = "; ".join(str(e) for e in errors[:3])
        more = "" if len(errors) <= 3 else f" (+{len(errors)-3} more)"
        super().__init__(f"{len(errors)} node scoring errors: {preview}{more}")


async def score_nodes_once(
    llm: LLMClient,
    *,
    dag_json: Dict[str, Any],
    exa_k: int,
    retrieval_mode: str = "llm",
    node_concurrency: int = 3,
    max_json_repairs: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """Score all nodes once (one resample) and return a dict keyed by node_id.

    IMPORTANT: this function now isolates per-node exceptions and aggregates them
    so debugging doesn't depend on whichever node failed first.
    """
    nodes = dag_json.get("nodes") or []
    if not isinstance(nodes, list):
        raise ValueError("dag_json['nodes'] must be a list")

    sem = asyncio.Semaphore(max(1, int(node_concurrency)))

    async def _run_one(n: Dict[str, Any]):
        nid = str(n.get("id"))
        text = str(n.get("text", ""))
        role = str(n.get("role", ""))
        async with sem:
            try:
                res = await score_single_node_once(
                    llm,
                    node_id=nid,
                    node_text=text,
                    node_role=role,
                    exa_k=exa_k,
                    retrieval_mode=retrieval_mode,
                    max_json_repairs=max_json_repairs,
                )
                return nid, res
            except Exception as e:
                raise NodeScoringError(nid, role, e) from e

    results = await asyncio.gather(*[_run_one(n) for n in nodes], return_exceptions=True)

    out: Dict[str, Dict[str, Any]] = {}
    errs: List[NodeScoringError] = []
    for r in results:
        if isinstance(r, Exception):
            if isinstance(r, NodeScoringError):
                errs.append(r)
            else:
                # Should be rare, but keep it visible.
                errs.append(NodeScoringError(node_id="unknown", role="unknown", original=r))
        else:
            nid, res = r
            out[str(nid)] = res

    if errs:
        raise NodeScoringBatchError(errs)

    return out


def compute_graph_score(
    *,
    dag_json: Dict[str, Any],
    node_results: Dict[str, Dict[str, Any]],
    reconcile: str = "prefer_parents",
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    """Compute KGScorer.graph_score() given a validated DAG and per-node metrics.

    Returns: (score, graph_details, validation_report_as_dict)
    """

    from graph_app.kg_realtime_scoring import DAGValidation

    kg, rep = DAGValidation.validate_and_build_from_json(
        dag_json,
        reconcile=reconcile,
        strict_roles=True,
        expect_roots=True,
    )

    rep_dict = {
        "errors": rep.errors,
        "warnings": rep.warnings,
        "edge_mismatches": rep.edge_mismatches,
        "stats": rep.stats,
    }

    if rep.errors:
        return float("nan"), {"error": 1.0}, rep_dict

    # Apply metrics
    for node_id, result in node_results.items():
        # DAGValidation canonicalizes IDs to strings
        nid = str(node_id)
        if nid not in kg.nodes:
            continue
        kg.update_node_metrics(
            node_id=nid,
            credibility=_clip01(float(result.get("credibility", 0.5))),
            relevance=_clip01(float(result.get("relevance", 0.5))),
            evidence_strength=_clip01(float(result.get("evidence_strength", 0.5))),
            method_rigor=_clip01(float(result.get("method_rigor", 0.5))),
            reproducibility=_clip01(float(result.get("reproducibility", 0.5))),
            citation_support=_clip01(float(result.get("citation_support", 0.5))),
        )

    score, details = kg.graph_score()
    return float(score), {k: float(v) for k, v in details.items()}, rep_dict

@dataclass
class TrialResult:
    k: int
    m: int
    success: bool
    graph_score: float
    graph_details: Dict[str, float]
    validation_report: Dict[str, Any]

    # Failure diagnostics (populated when success=False)
    error_stage: Optional[str] = None          # e.g., "dag_extraction", "node_scoring", "graph_scoring"
    error_type: Optional[str] = None           # exception class name or sentinel
    error_message: Optional[str] = None        # short message (CSV-safe)
    traceback_file: Optional[str] = None       # relative path under out_dir (only in debug mode)


async def run_factorized_resampling_for_pdf(
    *,
    pdf_path: str,
    out_dir: str,
    llm_cfg: LLMConfig,
    retrieval_mode: str = "llm",
    k_dags: int,
    m_node_resamples: int,
    max_nodes: int = 10,
    exa_k: int = 6,
    reconcile: str = "prefer_parents",
    node_concurrency: int = 3,
    max_json_repairs: int = 1,
    reuse_cached: bool = False,
    llm: LLMClient | None = None,
    paper_id: str | None = None,
    logger: Optional[logging.Logger] = None,
    debug: bool = False,
) -> Dict[str, Any]:

    """Run the K×M factorized resampling experiment for one PDF.

    Writes:
      - extracted_text.txt
      - dag/dag_kXXX.json (+ validation report)
      - node_scores/dag_kXXX/node_scores_mYYY.json
      - graph_scores.csv
      - summary.json
    """
    owned_llm = llm is None
    if owned_llm:
        llm = LLMClient(llm_cfg)
    assert llm is not None

    # Best-effort paper_id derivation (keeps outputs traceable even if caller forgets).
    if not paper_id:
        # Prefer out_dir name prefix like "003__my-title-slug"
        base = Path(out_dir).name
        if "__" in base:
            paper_id = base.split("__", 1)[0]
        else:
            paper_id = Path(pdf_path).stem[:32]  # fallback, bounded length

    def _tb_write(stage: str, k: int, m: int, exc: BaseException) -> str:
        if not debug:
            return ""
        errors_dir = out / "errors"
        _ensure_dir(errors_dir)
        rel = f"errors/trial_k{k:03d}_m{m:03d}_{stage}.txt"
        (out / rel).write_text(traceback.format_exc(), encoding="utf-8")
        return rel

    def _fail_trial(
        k: int,
        m: int,
        *,
        stage: str,
        exc: Optional[BaseException],
        rep: Dict[str, Any],
        score: float = float("nan"),
        details: Optional[Dict[str, float]] = None,
    ) -> TrialResult:
        tb_file = _tb_write(stage, k, m, exc) if exc else ""

        # Prefer actionable messages in CSV/summary even when we didn't raise.
        if exc is None:
            rep_errors = rep.get("errors")
            if isinstance(rep_errors, list) and rep_errors:
                msg = "; ".join(str(x) for x in rep_errors)
            else:
                msg = stage
            err_type = stage
        else:
            msg = str(exc)
            err_type = type(exc).__name__

        return TrialResult(
            k=k,
            m=m,
            success=False,
            graph_score=float(score),
            graph_details=details or {},
            validation_report=rep,
            error_stage=stage,
            error_type=err_type,
            error_message=(msg[:800] if msg else stage),
            traceback_file=(tb_file or None),
        )



    try:
        out = Path(out_dir)
        _ensure_dir(out)
        dag_dir = out / "dag"
        node_dir = out / "node_scores"
        _ensure_dir(dag_dir)
        _ensure_dir(node_dir)

        # 0) Extract text (cache)
        text_path = out / "extracted_text.txt"
        if reuse_cached and text_path.exists():
            raw_text = text_path.read_text(encoding="utf-8", errors="ignore")
        else:
            raw_text = extract_text_from_pdf(pdf_path)
            text_path.write_text(raw_text, encoding="utf-8")

        trials: List[TrialResult] = []
        per_dag_scores: Dict[int, List[float]] = {k: [] for k in range(int(k_dags))}

        # 1) DAG resamples
        for k in range(int(k_dags)):
            dag_path = dag_dir / f"dag_k{k:03d}.json"
            dag_val_path = dag_dir / f"dag_k{k:03d}_validation.json"

            dag_json: Dict[str, Any] | None = None

            # Load cached DAG if requested
            if reuse_cached and dag_path.exists():
                try:
                    dag_json = json.loads(dag_path.read_text(encoding="utf-8"))
                    if not (isinstance(dag_json, dict) and isinstance(dag_json.get("nodes"), list)):
                        raise ValueError("cached DAG JSON is not an object with a 'nodes' list")
                except Exception as e:
                    (dag_dir / f"dag_k{k:03d}_cache_invalid.txt").write_text(
                        f"Invalid cached DAG; will regenerate.\n\n{type(e).__name__}: {e}\n",
                        encoding="utf-8",
                    )
                    try:
                        dag_path.unlink()
                    except Exception:
                        pass
                    dag_json = None

            # Generate DAG
            if dag_json is None:
                try:
                    dag_json = await extract_dag_once(
                        llm,
                        raw_text=raw_text,
                        max_nodes=int(max_nodes),
                        max_json_repairs=int(max_json_repairs),
                    )
                    dag_path.write_text(json.dumps(dag_json, indent=2, ensure_ascii=False), encoding="utf-8")
                except LLMJsonParseError as e:
                    (dag_dir / f"dag_k{k:03d}_error.txt").write_text(str(e), encoding="utf-8")
                    rep_dict = {"errors": [f"dag_extraction_failed: {e}"], "warnings": [], "edge_mismatches": [], "stats": {}, "reconciled_edges": []}
                    dag_val_path.write_text(json.dumps(rep_dict, indent=2, ensure_ascii=False), encoding="utf-8")
                    for m in range(int(m_node_resamples)):
                        trials.append(_fail_trial(k, m, stage="dag_extraction_failed", exc=e, rep=rep_dict))
                    continue
                except Exception as e:
                    (dag_dir / f"dag_k{k:03d}_error.txt").write_text(f"{type(e).__name__}: {e}", encoding="utf-8")
                    rep_dict = {"errors": [f"dag_extraction_exception:{type(e).__name__}: {e}"], "warnings": [], "edge_mismatches": [], "stats": {}, "reconciled_edges": []}
                    dag_val_path.write_text(json.dumps(rep_dict, indent=2, ensure_ascii=False), encoding="utf-8")
                    for m in range(int(m_node_resamples)):
                        trials.append(_fail_trial(k, m, stage="dag_extraction_exception", exc=e, rep=rep_dict))
                    continue

            # Validate once (and store report)
            from graph_app.kg_realtime_scoring import DAGValidation

            try:
                _, rep = DAGValidation.validate_and_build_from_json(
                    dag_json,
                    reconcile=reconcile,
                    strict_roles=True,
                    expect_roots=True,
                )
                rep_dict = {
                    "errors": rep.errors,
                    "warnings": rep.warnings,
                    "edge_mismatches": rep.edge_mismatches,
                    "stats": rep.stats,
                    "reconciled_edges": sorted([list(e) for e in rep.reconciled_edges]),
                }
            except Exception as e:
                rep_dict = {
                    "errors": [f"validate_and_build_from_json raised {type(e).__name__}: {e}"],
                    "warnings": [],
                    "edge_mismatches": [],
                    "stats": {},
                    "reconciled_edges": [],
                }

            dag_val_path.write_text(json.dumps(rep_dict, indent=2, ensure_ascii=False), encoding="utf-8")

            if rep_dict.get("errors"):
                for m in range(int(m_node_resamples)):
                    trials.append(_fail_trial(k, m, stage="dag_validation_failed", exc=None, rep=rep_dict))
                continue

            # 2) Node-score resamples for this DAG
            per_k_node_dir = node_dir / f"dag_k{k:03d}"
            _ensure_dir(per_k_node_dir)

            for m in range(int(m_node_resamples)):
                ns_path = per_k_node_dir / f"node_scores_m{m:03d}.json"

                # Load cached node scores if requested
                if reuse_cached and ns_path.exists():
                    try:
                        node_results = json.loads(ns_path.read_text(encoding="utf-8"))
                        if not isinstance(node_results, dict):
                            raise ValueError("cached node_scores is not a dict")
                    except Exception as e:
                        (per_k_node_dir / f"node_scores_m{m:03d}_cache_invalid.txt").write_text(
                            f"Invalid cached node scores; will regenerate.\n\n{type(e).__name__}: {e}\n",
                            encoding="utf-8",
                        )
                        try:
                            ns_path.unlink()
                        except Exception:
                            pass
                        node_results = None
                else:
                    node_results = None

                if node_results is None:
                    try:
                        node_results = await score_nodes_once(
                            llm,
                            dag_json=dag_json,
                            exa_k=int(exa_k),
                            retrieval_mode=str(retrieval_mode),
                            node_concurrency=int(node_concurrency),
                            max_json_repairs=int(max_json_repairs),
                        )
                    except NodeScoringBatchError as e:
                        # Persist per-node errors for debugging
                        per_k_node_dir = node_dir / f"dag_k{k:03d}"
                        _ensure_dir(per_k_node_dir)
                        err_json = {
                            "stage": "node_scoring",
                            "k": k,
                            "m": m,
                            "num_node_errors": len(e.errors),
                            "node_errors": [
                                {"node_id": er.node_id, "role": er.role, "error": str(er)}
                                for er in e.errors
                            ],
                        }
                        (per_k_node_dir / f"node_scores_m{m:03d}_error.json").write_text(
                            json.dumps(err_json, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        trials.append(_fail_trial(k, m, stage="node_scoring", exc=e, rep=rep_dict))
                        continue
                    except Exception as e:
                        trials.append(_fail_trial(k, m, stage="node_scoring", exc=e, rep=rep_dict))
                        continue


                # Graph score
                try:
                    score, details, rep2 = compute_graph_score(
                        dag_json=dag_json,
                        node_results=node_results,
                        reconcile=reconcile,
                    )

                    if math.isnan(score) or math.isinf(score):
                        trials.append(
                            _fail_trial(
                                k,
                                m,
                                stage="graph_scoring",
                                exc=ValueError("graph_score_nan_or_inf"),
                                rep=rep2,
                                score=float(score),
                                details=details,
                            )
                        )
                        continue

                    trials.append(
                        TrialResult(
                            k=k,
                            m=m,
                            success=True,
                            graph_score=float(score),
                            graph_details=details,
                            validation_report=rep2,
                            # error_* left as None by default
                        )
                    )
                    per_dag_scores[k].append(float(score))

                except Exception as e:
                    trials.append(_fail_trial(k, m, stage="graph_scoring", exc=e, rep=rep_dict))
                    continue


        # 3) Write CSV + summary
        csv_path = out / "graph_scores.csv"
        fieldnames = [
            "k",
            "m",
            "success",
            "graph_score",
            "error_stage",
            "error_type",
            "error_message",
            "traceback_file",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()
            for t in trials:
                wr.writerow(
                    {
                        "k": t.k,
                        "m": t.m,
                        "success": "1" if t.success else "0",
                        "graph_score": "" if (t.graph_score is None or math.isnan(t.graph_score)) else f"{t.graph_score:.8f}",
                        "error_stage": t.error_stage or "",
                        "error_type": t.error_type or "",
                        "error_message": t.error_message or "",
                        "traceback_file": t.traceback_file or "",
                    }
                )


        global_scores = [t.graph_score for t in trials if t.success and not math.isnan(t.graph_score)]

        dag_summaries = {f"k{k:03d}": _summ_stats(per_dag_scores.get(k, [])) for k in range(int(k_dags))}

        failures = [t for t in trials if not t.success]
        first_failure = {}
        if failures:
            f0 = failures[0]
            first_failure = {
                "stage": f0.error_stage or "",
                "type": f0.error_type or "",
                "message": f0.error_message or "",
                "k": f0.k,
                "m": f0.m,
                "traceback_file": f0.traceback_file or "",
            }

        failure_counts = Counter([t.error_stage or "unknown" for t in failures])

        summary = {
            "paper_id": str(paper_id),
            "pdf_path": str(pdf_path),
            "k_dags": int(k_dags),
            "m_node_resamples": int(m_node_resamples),
            "max_nodes": int(max_nodes),
            "exa_k": int(exa_k),
            "retrieval_mode": str(retrieval_mode),
            "reconcile": reconcile,
            "llm": {"model": llm_cfg.model, "temperature": float(llm_cfg.temperature)},
            "global": _summ_stats(global_scores),
            "per_dag": dag_summaries,
            "node_concurrency": int(node_concurrency),
            "max_json_repairs": int(max_json_repairs),
            "diagnostics": {
                "text_chars": int(len(raw_text or "")),
                "num_trials": int(len(trials)),
                "num_success": int(sum(1 for t in trials if t.success)),
                "failure_counts": dict(failure_counts),
                "first_failure": first_failure,
            },
        }

        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    finally:
        if owned_llm:
            await llm.aclose()


