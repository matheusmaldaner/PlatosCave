"""
Verification Pipeline for PaperParser

This module orchestrates sequential claim verification by:
1. Converting DAG JSON to KGScorer instance
2. Running browser-use agents to verify each claim
3. Updating node metrics in real-time
4. Streaming progress to the frontend via WebSocket
5. Calculating edge and graph scores
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, List, Optional, Callable

from browser_use import Agent, ChatBrowserUse, Browser
from browser_use.llm.messages import UserMessage

# Import our graph scoring system
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from graph_app.kg_realtime_scoring import (
    KGScorer, Node, DAGValidation, ValidationReport,
    EdgeCombineWeights, NodeQualityWeights, RoleTransitionPrior,
    PairSynergyWeights, GraphScoreWeights
)

from prompts import build_claim_verification_prompt, parse_verification_result


def send_metric_update(node_id: str, node_text: str, metrics: Dict[str, float], flush: bool = True):
    """Send node metric update to frontend via WebSocket"""
    update_message = json.dumps({
        "type": "METRIC_UPDATE",
        "node_id": node_id,
        "node_text": node_text[:100] + "..." if len(node_text) > 100 else node_text,
        "metrics": metrics
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


async def verify_node_with_agent(
    node: Node,
    llm: ChatBrowserUse,
    browser: Optional[Browser] = None,
    max_steps: int = 30
) -> Optional[Dict[str, Any]]:
    """
    Verify a single node using browser-use agent.

    Args:
        node: The Node instance to verify
        llm: The LLM instance for the agent
        browser: Optional browser instance (reuse across agents)
        max_steps: Maximum agent steps for verification

    Returns:
        Verification result dict with metrics, or None if verification fails
    """
    # Build verification prompt
    verification_prompt = build_claim_verification_prompt(
        claim_text=node.text,
        claim_role=node.role,
        claim_context=""  # Could add parent node context here in future
    )

    # Create agent
    agent_kwargs = {
        'task': verification_prompt,
        'llm': llm,
        'vision_detail_level': 'low',  # Lower detail for faster verification
        'generate_gif': False,          # Don't generate GIFs for each verification
        'use_vision': False             # Text-only verification is faster
    }

    if browser is not None:
        agent_kwargs['browser'] = browser

    agent = Agent(**agent_kwargs)

    try:
        # Run agent verification
        history = await agent.run(max_steps=max_steps)

        # Extract result from agent
        result_text = history.final_result()

        # Parse verification result
        verification_result = parse_verification_result(result_text)

        return verification_result

    except Exception as e:
        print(f"Error verifying node {node.id}: {e}", file=sys.stderr)
        return None


async def run_verification_pipeline(
    dag_json: Dict[str, Any],
    llm: ChatBrowserUse,
    browser: Optional[Browser] = None,
    send_update_fn: Optional[Callable] = None
) -> tuple[KGScorer, Dict[str, Any]]:
    """
    Run the complete verification pipeline.

    This is the main orchestration function that:
    1. Converts DAG JSON to KGScorer
    2. Iterates through all nodes sequentially
    3. Verifies each node with a browser agent
    4. Updates metrics in real-time
    5. Calculates edge and graph scores
    6. Streams updates to frontend

    Args:
        dag_json: The DAG JSON structure from main.py
        llm: The LLM instance to use for agents
        browser: Optional browser instance to reuse
        send_update_fn: Optional custom update function (for testing)

    Returns:
        Tuple of (KGScorer with all metrics, verification summary dict)
    """
    # Stage update function (use custom or default)
    update_fn = send_update_fn or send_verification_progress

    # Step 1: Convert DAG to KGScorer
    send_update("Organizing Agents", "Converting DAG to knowledge graph scorer...")
    kg_scorer, validation_report = dag_json_to_kg_scorer(dag_json)

    if not validation_report.ok():
        error_msg = f"DAG validation failed: {validation_report.errors}"
        send_update("Organizing Agents", error_msg)
        raise ValueError(error_msg)

    send_update("Organizing Agents",
                f"Knowledge graph initialized: {validation_report.stats['num_nodes']} nodes, "
                f"{validation_report.stats['num_edges']} edges")

    # Step 2: Set default scores for Hypothesis nodes (they don't need verification)
    hypothesis_nodes = [node for node in kg_scorer.nodes.values() if node.role == "Hypothesis"]
    for hyp_node in hypothesis_nodes:
        # Give hypothesis moderate-to-high default scores since it's the research question
        kg_scorer.update_node_metrics(
            node_id=hyp_node.id,
            credibility=0.75,      # Assume hypothesis is well-formed
            relevance=1.0,         # Hypothesis is always 100% relevant to itself
            evidence_strength=0.5, # Neutral - to be determined by children
            method_rigor=0.5,      # Neutral
            reproducibility=0.5,   # Neutral
            citation_support=0.5   # Neutral
        )

    # Collect all nodes that need verification (skip Hypothesis)
    nodes_to_verify = [
        node for node in kg_scorer.nodes.values()
        if node.role != "Hypothesis"  # Hypothesis is the research question, doesn't need web verification
    ]

    total_nodes = len(nodes_to_verify)
    send_update("Organizing Agents", f"Preparing to verify {total_nodes} claims...")

    # Register edge update callback to stream edge scores
    def edge_callback(u: str, v: str, confidence: float, features: Dict[str, float]):
        send_edge_update(u, v, confidence, features)

    kg_scorer.register_edge_update_callback(edge_callback)

    # Step 3: Sequential verification
    send_update("Compiling Evidence", "Starting sequential claim verification...")

    verification_results = {}

    for idx, node in enumerate(nodes_to_verify, start=1):
        # Progress update
        update_fn(idx, total_nodes, node.text)

        # Verify the node
        verification_result = await verify_node_with_agent(
            node=node,
            llm=llm,
            browser=browser,
            max_steps=30  # Limit steps per verification to avoid timeouts
        )

        if verification_result is None:
            # Verification failed - use default low scores
            print(f"⚠️ Verification failed for node {node.id}, using default scores", file=sys.stderr)
            verification_result = {
                "credibility": 0.5,
                "relevance": 0.5,
                "evidence_strength": 0.5,
                "method_rigor": 0.5,
                "reproducibility": 0.5,
                "citation_support": 0.5,
                "verification_summary": "Verification failed or timed out",
                "confidence_level": "low"
            }

        # Update node metrics in KGScorer (this triggers edge recalculation)
        kg_scorer.update_node_metrics(
            node_id=node.id,
            credibility=verification_result["credibility"],
            relevance=verification_result["relevance"],
            evidence_strength=verification_result["evidence_strength"],
            method_rigor=verification_result["method_rigor"],
            reproducibility=verification_result["reproducibility"],
            citation_support=verification_result["citation_support"]
        )

        # Send metric update to frontend
        send_metric_update(node.id, node.text, {
            "credibility": verification_result["credibility"],
            "relevance": verification_result["relevance"],
            "evidence_strength": verification_result["evidence_strength"],
            "method_rigor": verification_result["method_rigor"],
            "reproducibility": verification_result["reproducibility"],
            "citation_support": verification_result["citation_support"]
        })

        # Store full verification result
        verification_results[node.id] = verification_result

        # Small delay between verifications (avoid overwhelming the system)
        await asyncio.sleep(0.5)

    send_update("Compiling Evidence", "All claims verified. Evidence compiled.")

    # Step 4: Calculate graph-level score
    send_update("Evaluating Integrity", "Calculating graph-level integrity score...")

    try:
        graph_score, graph_details = kg_scorer.graph_score()
        send_graph_score(graph_score, graph_details)
        send_update("Evaluating Integrity", f"Final integrity score: {graph_score:.2f}")
    except Exception as e:
        print(f"Error calculating graph score: {e}", file=sys.stderr)
        graph_score = 0.0
        graph_details = {}

    # Step 5: Build summary
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
    """Helper function for stage updates (matches main.py format)"""
    update_message = json.dumps({"type": "UPDATE", "stage": stage, "text": text})
    print(update_message, flush=flush)
