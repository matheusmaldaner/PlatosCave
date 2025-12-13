"""
Verification Pipeline for PaperParser

Handles graph-based scoring of verification results:
1. Converts DAG JSON to KGScorer instance
2. Accepts pre-computed verification results from agents
3. Updates node metrics in real-time
4. Streams progress to frontend via WebSocket
5. Calculates edge and graph scores
"""

import json
import os
import sys
from typing import Dict, Any, List, Optional, Callable

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from graph_app.kg_realtime_scoring import (
    KGScorer, Node, DAGValidation, ValidationReport,
    EdgeCombineWeights, NodeQualityWeights, RoleTransitionPrior,
    PairSynergyWeights, GraphScoreWeights
)

def send_metric_update(node_id: str, node_text: str, metrics: Dict[str, float], verification_data: Dict = None, flush: bool = True):
    """Send node metric update to frontend"""
    update_message = json.dumps({
        "type": "METRIC_UPDATE",
        "node_id": node_id,
        "node_text": node_text[:100] + "..." if len(node_text) > 100 else node_text,
        "metrics": metrics,
        "verification_summary": verification_data.get("verification_summary", "") if verification_data else "",
        "confidence_level": verification_data.get("confidence_level", "") if verification_data else "",
        "sources_checked": verification_data.get("sources_checked", []) if verification_data else [],
        "red_flags": verification_data.get("red_flags", []) if verification_data else []
    })
    print(update_message, flush=flush)


def send_verification_progress(current: int, total: int, node_text: str, flush: bool = True):
    """Send verification progress update to frontend"""
    progress_message = json.dumps({
        "type": "VERIFICATION_PROGRESS",
        "current": current,
        "total": total,
        "node_text": node_text[:100] + "..." if len(node_text) > 100 else node_text
    })
    print(progress_message, flush=flush)


def send_edge_update(source: str, target: str, confidence: float, features: Dict[str, float], flush: bool = True):
    """Send edge confidence update to frontend"""
    edge_message = json.dumps({
        "type": "EDGE_UPDATE",
        "source": source,
        "target": target,
        "confidence": confidence,
        "features": features
    })
    print(edge_message, flush=flush)


def send_graph_score(score: float, details: Dict[str, float], flush: bool = True):
    """Send final graph score to frontend"""
    score_message = json.dumps({
        "type": "GRAPH_SCORE",
        "score": score,
        "details": details
    })
    print(score_message, flush=flush)


def dag_json_to_kg_scorer(dag_json: Dict[str, Any]) -> tuple[KGScorer, ValidationReport]:
    """Convert DAG JSON to KGScorer instance"""
    kg_scorer, validation_report = DAGValidation.validate_and_build_from_json(
        dag_json,
        reconcile="prefer_parents",
        strict_roles=True,
        expect_roots=True,
        forbid_self_loops=True
    )
    return kg_scorer, validation_report


def run_verification_pipeline(
    dag_json: Dict[str, Any],
    verification_results: Dict[str, Dict[str, float]],
    send_update_fn: Optional[Callable] = None
) -> tuple[KGScorer, Dict[str, Any]]:
    """
    Run verification pipeline with pre-computed results.

    Args:
        dag_json: DAG JSON structure from main.py
        verification_results: Dict mapping node_id -> verification metrics
        send_update_fn: Optional custom update function (for testing)

    Returns:
        Tuple of (KGScorer with metrics, verification summary dict)
    """
    send_update("Organizing Agents", "Converting DAG to knowledge graph scorer...")

    kg_scorer, validation_report = dag_json_to_kg_scorer(dag_json)

    if not validation_report.ok():
        error_msg = f"DAG validation failed: {validation_report.errors}"
        send_update("Organizing Agents", error_msg)
        raise ValueError(error_msg)

    send_update("Organizing Agents",
                f"Knowledge graph initialized: {validation_report.stats['num_nodes']} nodes, "
                f"{validation_report.stats['num_edges']} edges")

    # Set default scores for Hypothesis nodes
    hypothesis_nodes = [node for node in kg_scorer.nodes.values() if node.role == "Hypothesis"]
    for hyp_node in hypothesis_nodes:
        kg_scorer.update_node_metrics(
            node_id=hyp_node.id,
            credibility=0.75,
            relevance=1.0,
            evidence_strength=0.5,
            method_rigor=0.5,
            reproducibility=0.5,
            citation_support=0.5
        )

    total_nodes = len(verification_results)
    send_update("Organizing Agents", f"Processing {total_nodes} verified claims...")

    def edge_callback(u: str, v: str, confidence: float, features: Dict[str, float]):
        try:
            send_edge_update(u, v, confidence, features)
        except Exception as e:
            print(f"Error in edge callback for ({u}, {v}): {e}", file=sys.stderr)

    kg_scorer.register_edge_update_callback(edge_callback)

    send_update("Compiling Evidence", "Applying verification results to knowledge graph...")

    for node_id, result in verification_results.items():
        node = kg_scorer.nodes.get(node_id)
        if node is None:
            print(f"Node {node_id} not found, skipping", file=sys.stderr)
            continue

        kg_scorer.update_node_metrics(
            node_id=node_id,
            credibility=result["credibility"],
            relevance=result["relevance"],
            evidence_strength=result["evidence_strength"],
            method_rigor=result["method_rigor"],
            reproducibility=result["reproducibility"],
            citation_support=result["citation_support"]
        )

        send_metric_update(node_id, node.text, {
            "credibility": result["credibility"],
            "relevance": result["relevance"],
            "evidence_strength": result["evidence_strength"],
            "method_rigor": result["method_rigor"],
            "reproducibility": result["reproducibility"],
            "citation_support": result["citation_support"]
        }, verification_data=result)

    send_update("Compiling Evidence", "All verification results applied. Evidence compiled.")

    send_update("Evaluating Integrity", "Calculating graph-level integrity score...")

    try:
        graph_score, graph_details = kg_scorer.graph_score()
        send_graph_score(graph_score, graph_details)
        send_update("Evaluating Integrity", f"Final integrity score: {graph_score:.2f}")
    except Exception as e:
        error_msg = f"Error calculating graph score: {str(e)}"
        print(error_msg, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

        send_update("Evaluating Integrity", f"ERROR: {error_msg}")
        graph_score = 0.0
        graph_details = {"error": error_msg, "calculation_failed": True}
        send_graph_score(graph_score, graph_details)

    summary = {
        "total_nodes_verified": total_nodes,
        "graph_score": graph_score,
        "graph_details": graph_details,
        "validation_report": {
            "errors": validation_report.errors,
            "warnings": validation_report.warnings,
            "stats": validation_report.stats
        },
        "verification_results": verification_results
    }

    return kg_scorer, summary


def send_update(stage: str, text: str, flush: bool = True):
    """Helper for stage updates"""
    update_message = json.dumps({"type": "UPDATE", "stage": stage, "text": text})
    print(update_message, flush=flush)