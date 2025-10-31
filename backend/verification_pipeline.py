"""
Verification Pipeline for PaperParser

This module handles graph-based scoring of verification results by:
1. Converting DAG JSON to KGScorer instance
2. Accepting pre-computed verification results from agents
3. Updating node metrics in real-time
4. Streaming progress to the frontend via WebSocket
5. Calculating edge and graph scores
"""

import json
import os
import sys
from typing import Dict, Any, List, Optional, Callable

# Import our graph scoring system
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from graph_app.kg_realtime_scoring import (
    KGScorer, Node, DAGValidation, ValidationReport,
    EdgeCombineWeights, NodeQualityWeights, RoleTransitionPrior,
    PairSynergyWeights, GraphScoreWeights
)

def send_metric_update(node_id: str, node_text: str, metrics: Dict[str, float], verification_data: Dict = None, flush: bool = True):
    """Send node metric update to frontend via WebSocket"""
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
    print(f"[BACKEND DEBUG] ========== METRIC_UPDATE SENT ==========", file=sys.stderr, flush=True)
    print(f"[BACKEND DEBUG] Node ID: {node_id}", file=sys.stderr, flush=True)
    print(f"[BACKEND DEBUG] Node Text: {node_text[:50]}...", file=sys.stderr, flush=True)
    print(f"[BACKEND DEBUG] Metrics: {metrics}", file=sys.stderr, flush=True)
    print(update_message, flush=flush)


def send_verification_progress(current: int, total: int, node_text: str, flush: bool = True):
    """Send verification progress update to frontend"""
    progress_message = json.dumps({
        "type": "VERIFICATION_PROGRESS",
        "current": current,
        "total": total,
        "node_text": node_text[:100] + "..." if len(node_text) > 100 else node_text
    })
    print(f"[BACKEND DEBUG] ========== VERIFICATION_PROGRESS SENT ==========", file=sys.stderr, flush=True)
    print(f"[BACKEND DEBUG] Progress: {current}/{total}", file=sys.stderr, flush=True)
    print(f"[BACKEND DEBUG] Node Text: {node_text[:50]}...", file=sys.stderr, flush=True)
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
    print(f"[BACKEND DEBUG] ========== EDGE_UPDATE SENT ==========", file=sys.stderr, flush=True)
    print(f"[BACKEND DEBUG] Edge: {source} -> {target}", file=sys.stderr, flush=True)
    print(f"[BACKEND DEBUG] Confidence: {confidence:.4f}", file=sys.stderr, flush=True)
    print(f"[BACKEND DEBUG] Features: {features}", file=sys.stderr, flush=True)
    print(edge_message, flush=flush)


def send_graph_score(score: float, details: Dict[str, float], flush: bool = True):
    """Send final graph score to frontend"""
    score_message = json.dumps({
        "type": "GRAPH_SCORE",
        "score": score,
        "details": details
    })
    print(f"[BACKEND DEBUG] ========== GRAPH_SCORE SENT ==========", file=sys.stderr, flush=True)
    print(f"[BACKEND DEBUG] Final Score: {score:.4f}", file=sys.stderr, flush=True)
    print(f"[BACKEND DEBUG] Details: {details}", file=sys.stderr, flush=True)
    print(score_message, flush=flush)


def dag_json_to_kg_scorer(dag_json: Dict[str, Any]) -> tuple[KGScorer, ValidationReport]:
    """
    Convert DAG JSON (from main.py) to KGScorer instance.

    Args:
        dag_json: The DAG JSON with nodes array containing id, text, role, parents, children

    Returns:
        Tuple of (KGScorer instance, ValidationReport)
    """
    # Use the DAGValidation utility from kg_realtime_scoring
    kg_scorer, validation_report = DAGValidation.validate_and_build_from_json(
        dag_json,
        reconcile="prefer_parents",  # Use parent edges as authoritative
        strict_roles=True,           # Enforce valid roles
        expect_roots=True,           # Expect Hypothesis root
        forbid_self_loops=True       # No self-loops allowed
    )

    return kg_scorer, validation_report


def run_verification_pipeline(
    dag_json: Dict[str, Any],
    verification_results: Dict[str, Dict[str, float]],
    send_update_fn: Optional[Callable] = None
) -> tuple[KGScorer, Dict[str, Any]]:
    """
    Run the verification pipeline with pre-computed verification results.

    This function handles pure data processing:
    1. Converts DAG JSON to KGScorer
    2. Accepts pre-computed verification results from agents
    3. Updates node metrics in real-time
    4. Calculates edge and graph scores
    5. Streams updates to frontend

    Args:
        dag_json: The DAG JSON structure from main.py
        verification_results: Dict mapping node_id -> verification metrics dict
                              Each metrics dict should contain:
                              - credibility, relevance, evidence_strength,
                                method_rigor, reproducibility, citation_support
        send_update_fn: Optional custom update function (for testing)

    Returns:
        Tuple of (KGScorer with all metrics, verification summary dict)
    """
    # Step 1: Convert DAG to KGScorer
    print(f"[BACKEND DEBUG] ========== PIPELINE STARTED ==========", file=sys.stderr, flush=True)
    print(f"[BACKEND DEBUG] Converting DAG to KGScorer...", file=sys.stderr, flush=True)
    send_update("Organizing Agents", "Converting DAG to knowledge graph scorer...")

    kg_scorer, validation_report = dag_json_to_kg_scorer(dag_json)
    print(f"[BACKEND DEBUG] DAG converted successfully", file=sys.stderr, flush=True)

    if not validation_report.ok():
        error_msg = f"DAG validation failed: {validation_report.errors}"
        send_update("Organizing Agents", error_msg)
        raise ValueError(error_msg)

    send_update("Organizing Agents",
                f"Knowledge graph initialized: {validation_report.stats['num_nodes']} nodes, "
                f"{validation_report.stats['num_edges']} edges")

    # Step 2: Set default scores for Hypothesis nodes (they don't need verification)
    hypothesis_nodes = [node for node in kg_scorer.nodes.values() if node.role == "Hypothesis"]
    print(f"[BACKEND DEBUG] Setting default scores for {len(hypothesis_nodes)} hypothesis nodes", file=sys.stderr, flush=True)
    for hyp_node in hypothesis_nodes:
        # Give hypothesis moderate-to-high default scores since it's the research question
        print(f"[BACKEND DEBUG] Hypothesis node {hyp_node.id}: {hyp_node.text[:50]}...", file=sys.stderr, flush=True)
        kg_scorer.update_node_metrics(
            node_id=hyp_node.id,
            credibility=0.75,      # Assume hypothesis is well-formed
            relevance=1.0,         # Hypothesis is always 100% relevant to itself
            evidence_strength=0.5, # Neutral - to be determined by children
            method_rigor=0.5,      # Neutral
            reproducibility=0.5,   # Neutral
            citation_support=0.5   # Neutral
        )

    # Get total nodes count (including those already verified)
    total_nodes = len(verification_results)
    print(f"[BACKEND DEBUG] Total nodes verified: {total_nodes}", file=sys.stderr, flush=True)
    send_update("Organizing Agents", f"Processing {total_nodes} verified claims...")

    # Register edge update callback to stream edge scores (with error handling)
    def edge_callback(u: str, v: str, confidence: float, features: Dict[str, float]):
        try:
            send_edge_update(u, v, confidence, features)
        except Exception as e:
            print(f"⚠️ Error in edge update callback for edge ({u}, {v}): {e}", file=sys.stderr)
            # Don't crash the entire pipeline for a callback error - just log and continue

    kg_scorer.register_edge_update_callback(edge_callback)

    # Step 3: Apply pre-computed verification results to nodes
    send_update("Compiling Evidence", "Applying verification results to knowledge graph...")

    for node_id, result in verification_results.items():
        print(f"[BACKEND DEBUG] ========== UPDATING NODE {node_id} ==========", file=sys.stderr, flush=True)
        print(f"[BACKEND DEBUG] Node ID: {node_id}", file=sys.stderr, flush=True)

        # Get the node from KGScorer to access its text
        node = kg_scorer.nodes.get(node_id)
        if node is None:
            print(f"⚠️ Node {node_id} not found in KGScorer, skipping", file=sys.stderr, flush=True)
            continue

        print(f"[BACKEND DEBUG] Node Role: {node.role}", file=sys.stderr, flush=True)
        print(f"[BACKEND DEBUG] Node Text: {node.text[:100]}...", file=sys.stderr, flush=True)
        print(f"[BACKEND DEBUG] Verification result: {result}", file=sys.stderr, flush=True)

        # Update node metrics in KGScorer (this triggers edge recalculation)
        kg_scorer.update_node_metrics(
            node_id=node_id,
            credibility=result["credibility"],
            relevance=result["relevance"],
            evidence_strength=result["evidence_strength"],
            method_rigor=result["method_rigor"],
            reproducibility=result["reproducibility"],
            citation_support=result["citation_support"]
        )
        print(f"[BACKEND DEBUG] Metrics updated for node {node_id}", file=sys.stderr, flush=True)

        # Send metric update to frontend
        send_metric_update(node_id, node.text, {
            "credibility": result["credibility"],
            "relevance": result["relevance"],
            "evidence_strength": result["evidence_strength"],
            "method_rigor": result["method_rigor"],
            "reproducibility": result["reproducibility"],
            "citation_support": result["citation_support"]
        }, verification_data=result)

    send_update("Compiling Evidence", "All verification results applied. Evidence compiled.")

    # Step 4: Calculate graph-level score
    print(f"[BACKEND DEBUG] ========== CALCULATING GRAPH SCORE ==========", file=sys.stderr, flush=True)
    send_update("Evaluating Integrity", "Calculating graph-level integrity score...")

    try:
        print(f"[BACKEND DEBUG] Calling kg_scorer.graph_score()...", file=sys.stderr, flush=True)
        graph_score, graph_details = kg_scorer.graph_score()
        print(f"[BACKEND DEBUG] Graph score calculated: {graph_score:.4f}", file=sys.stderr, flush=True)
        print(f"[BACKEND DEBUG] Graph details: {graph_details}", file=sys.stderr, flush=True)
        send_graph_score(graph_score, graph_details)
        send_update("Evaluating Integrity", f"Final integrity score: {graph_score:.2f}")
    except Exception as e:
        error_msg = f"Error calculating graph score: {str(e)}"
        print(f"{error_msg}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

        # Send error to frontend with clear indication this is a calculation error, not a low score
        send_update("Evaluating Integrity", f"ERROR: {error_msg}")
        graph_score = 0.0
        graph_details = {"error": error_msg, "calculation_failed": True}
        # Still send graph score message so frontend knows analysis is complete (but failed)
        send_graph_score(graph_score, graph_details)

    # Step 5: Build summary
    print(f"[BACKEND DEBUG] ========== BUILDING SUMMARY ==========", file=sys.stderr, flush=True)
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
    print(f"[BACKEND DEBUG] Summary built: {total_nodes} nodes verified, score: {graph_score:.4f}", file=sys.stderr, flush=True)
    print(f"[BACKEND DEBUG] ========== PIPELINE COMPLETED ==========", file=sys.stderr, flush=True)

    return kg_scorer, summary


def send_update(stage: str, text: str, flush: bool = True):
    """Helper function for stage updates (matches main.py format)"""
    update_message = json.dumps({"type": "UPDATE", "stage": stage, "text": text})
    print(f"[BACKEND DEBUG] ========== UPDATE SENT ==========", file=sys.stderr, flush=True)
    print(f"[BACKEND DEBUG] Stage: {stage}", file=sys.stderr, flush=True)
    print(f"[BACKEND DEBUG] Text: {text}", file=sys.stderr, flush=True)
    print(update_message, flush=flush)
