# Quick Start: KG Real-Time Scoring & Service Adapter API

This folder contains the backend logic for scoring knowledge-graph (KG) edges as an agent fills node metrics, plus a thin service adapter that exposes an easy API for a frontend/agent loop.

## Files

- `kg_realtime_scoring.py` — Core library:
  - Validate KG JSON into a DAG (`DAGValidation.validate_and_build_from_json`)
  - Maintain nodes, recompute **edge confidences** when ready (`KGScorer.update_node_metrics`)
  - Compute **graph-level score** (`KGScorer.graph_score`)
  - Export **embeddings/features** for meta-analysis:
    - `export_edge_features()`
    - `export_node_feature_matrix(...)`
    - `random_walk_corpus(...)` (node2vec)
    - `paper_fingerprint()`

- `service_adapter.py` — Small API over `KGScorer` for your frontend/agent:
  - `KGSession(graph_json)` — build + validate
  - `validation_report()` — errors/warnings/stats
  - `current()` — node id to score next (BFS from Hypothesis roots)
  - `set_metrics_and_advance(node_id, metrics)` — update node metrics, receive
    `updated_edges` (u,v,confidence), and `next_node`
  - `graph_score()` — overall score + components
  - (Optional helpers you can add) `state()`, `snapshot()`, `reset()`

## JSON Schema (input)

```json
{
  "nodes": [
    { "id": 0, "text": "…", "role": "Hypothesis", "parents": [], "children": [1,2] },
    { "id": 1, "text": "…", "role": "Evidence",   "parents": [0], "children": [3] },
    { "id": 2, "text": "…", "role": "Method",     "parents": [0], "children": [3] },
    { "id": 3, "text": "…", "role": "Conclusion", "parents": [1,2], "children": [] }
  ]
}
```
## Minimal Usage (Agent Loop)

```
from service_adapter import KGSession

sess = KGSession(graph_json)
sess.set_penalty(enabled=True, agg="min", alpha=1.0, eta=0.05)  # optional
print(sess.validation_report())

nid = sess.current()
while nid is not None:
    # agent produces 6 metrics in [0,1]
    m = {
      "credibility": 0.8, "relevance": 0.7, "evidence_strength": 0.6,
      "method_rigor": 0.5, "reproducibility": 0.4, "citation_support": 0.3
    }
    out = sess.set_metrics_and_advance(nid, m)
    print("updated_edges:", out["updated_edges"])
    nid = out["next_node"]

print(sess.graph_score())
print(sess.trusts())
```
## Embeddings / Meta-Analysis

```
# Per-node features (role one-hot + metrics [+ optional text embedding])
node_feats = sess.node_feature_matrix()

# Edge features table (confidence + components)
edges = sess.edge_features()

# Random-walk corpus for node2vec (uses edge confidence as weight)
walks = sess.node2vec_corpus(num_walks=20, walk_length=12, p=1.0, q=2.0, min_conf=0.2)

# Graph-level fingerprint (for retrieval/clustering)
paper_vec = sess.paper_fingerprint()
```
## Hyper-Parameters

```
# Make evidence-related metrics count more *convexly* everywhere (node quality),
# leave synergy pair-specific (apply_to_synergy=False):
session.set_convex_metric_weights({
    "credibility": 1.0,
    "relevance": 0.9,
    "evidence_strength": 1.4,
    "method_rigor": 1.1,
    "reproducibility": 1.0,
    "citation_support": 1.2,
}, apply_to_synergy=False)
```
---

# Why This Knowledge Graph Is Novel (Nodes • Edges • Graph)

## 1) Nodes — Feature-Rich & Agent-Extracted

Each node carries six normalized, agent-scored metrics in [0,1]—**credibility, relevance, evidence_strength, method_rigor, reproducibility, citation_support**—plus a canonical **role** (Hypothesis, Evidence, Method, Result, Claim, Conclusion, etc.) and the node text. The agent fills these metrics **systematically in BFS order** (starting from Hypothesis roots), so downstream edge/graph computations are triggered deterministically as information arrives. We support both **global per-metric re-weighting** and a strict **convex mix** (∑wᵢ=1) that can be applied across all roles, and even mirrored into pair-synergy if desired. Features export cleanly for meta-analysis (per-node matrices with role one-hot, metrics, and optional text embeddings). 

**What’s special here**

* **Agent-structured extraction:** metrics are produced by an agent and committed node-by-node in a controlled traversal, ensuring readiness guarantees for edge scoring. 
* **Two weighting regimes:**

  * *Global per-metric weights* for multiplicative scaling while preserving [0,1].
  * *Convex weights* for principled, comparable blends across roles (optionally reused for pair synergy), then **full recomputation** of confidences. 
* **Analysis-ready exports:** role one-hots + six metrics (+ optional text embeddings) for ML pipelines. 

---

## 2) Edges — Role-Aware Confidence With Trust Gating

An edge (u→v) is scored **only when** (v) and **all** of its parents have complete metrics. The raw edge confidence is a **weighted combination** of interpretable features:

* **Role transition prior** (r(u→v))
* **Parent quality** (q(u)) (role-specific mix of the six metrics)
* **Child quality** (q(v))
* **Text alignment** (token Jaccard on content words)
* **Pairwise synergy** (role-pair-specific mixes of parent/child metrics)

Raw confidence is then **trust-gated** by parent trust to produce the final edge weight (defensive against weak upstream evidence). All feature components (including `confidence_raw`, `trust_parent`, `trust_child`) are stored per-edge for transparency/debugging and live UI updates via callbacks. 

**What’s special here**

* **Interpretable decomposition** (prior/quality/alignment/synergy) rather than a black-box score.
* **Readiness discipline:** edges update *only when* parents and child are ready—no stale mixes.
* **Trust gating:** a **propagation penalty** aggregates parent trust (min/mean/LSE with α, η knobs) before gating the raw confidence, stabilizing long causal chains. 
* **Realtime hooks:** edge updates trigger subscribed callbacks so the frontend can animate diffs instantly. 

---

## 3) Graph — Validated DAG, Deterministic Traversal, and Rich Scores/Embeddings

Input graphs are **validated and reconciled** (parents/children lists, self-loop bans, cycle detection, role sanity, optional Hypothesis roots & Conclusion leaves). After validation, we compute a stable **BFS order from Hypothesis roots** (fallbacks included) for the agent to follow. The graph exposes:

* **Graph-level score** combining: bridge coverage, best-path product, redundancy (max-flow), fragility (min-cut on (1-)confidence), coherence (role-prior compliance), and coverage (Method/Evidence/Result presence).
* **Embeddings for meta-analysis:** edge-feature tables; per-node feature matrices; **random-walk corpora** (node2vec-style with confidence-weighted transitions); and a deterministic **paper fingerprint** vector aggregating score components, role histograms, role-pair histograms, and confidence histograms. 

**What’s special here**

* **Strict DAG discipline** with edge-list reconciliation modes (prefer_parents/children, union/intersection) and detailed validation reports. 
* **Deterministic agent loop** exposed via a tiny session API (`current()`, `set_metrics_and_advance(...)`, `graph_score()`) for clean frontend ↔ agent orchestration. 
* **End-to-end ML hooks** (node/edge features, walks, fingerprint) for clustering, retrieval, and downstream learning. 

---

## Agent Loop at a Glance (Deterministic)

1. **Validate + build** the DAG and compute BFS order from Hypothesis roots.
2. **Frontier node ⇒ agent**: agent emits the six metrics in [0,1].
3. **Commit metrics** → auto-recompute eligible incoming edges (with trust gating) → stream updated edge deltas to UI.
4. **Advance** to next node per BFS until complete; request **graph score**, features, walks, or fingerprint at any time. 

---

## Tuning & Extensibility

* **Metric weighting:** `set_metric_weights(..., normalize=...)` (global scale) or `set_convex_metric_weights(..., apply_to_synergy=...)` (principled blends across roles).  
* **Trust/propagation:** `set_penalty(enabled, agg, alpha, eta)` to adjust how upstream reliability gates edges. 
* **Priors & synergy:** editable role-transition prior table and role-pair synergy maps for domain-specific behavior. 

**Bottom line:** This KG is not just a static diagram—it’s a **live, agent-scored, role-aware, trust-propagating DAG** with first-class analytics and embeddings, designed for **reproducible evaluation** and **downstream learning**. 


