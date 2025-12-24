# Quick Start: KG Real-Time Scoring & Service Adapter API

This folder contains:

- **`kg_realtime_scoring.py`**: core library for validating a KG JSON into a DAG, maintaining node metrics, recomputing **edge confidences** when edges become eligible, and producing **graph-level scores** and **feature exports**.
- **`service_adapter.py`**: a thin session adapter (`KGSession`) suitable for a frontend/agent loop (deterministic BFS traversal + “streaming” edge updates).

---

## Files

### `kg_realtime_scoring.py` (core)

- **Validation / build**
  - `DAGValidation.validate_and_build_from_json(...)` → returns `(KGScorer, ValidationReport)`
- **Streaming scoring**
  - `KGScorer.update_node_metrics(node_id, **metrics)` → validates inputs; recomputes eligible incoming edges
  - `KGScorer.register_edge_update_callback(fn)` → subscribe to edge-confidence deltas `(u, v, confidence, features)`
  - `KGScorer.validate_scoring_state(strict=False)` → diagnostics to detect reliance on defaults/missing confidences
- **Graph score**
  - `KGScorer.graph_score()` → `(score, details)`; score is strictly in **(0,1)** via an epsilon squeeze
- **Exports**
  - `export_edge_features()`
  - `export_node_feature_matrix(text_embeddings=None)`
  - `random_walk_corpus(...)` (node2vec-style walks with confidence weighting)
  - `paper_fingerprint()`

### `service_adapter.py` (frontend/session API)

- `KGSession(graph_json)` — build + validate (deterministic BFS order, callback wiring)
- `validation_report()` — `ok/errors/warnings/stats`
- `scoring_validation(strict=False)` — frontend-friendly wrapper around `validate_scoring_state`
- `current()` — node id to score next (deterministic BFS order)
- `set_metrics_and_advance(node_id, metrics)` — updates node metrics (must match `current()`), returns
  - `updated_edges`: list of `{"u","v","confidence"}` deltas (rounded to 3 decimals)
  - `next_node`: next node id (or `None`)
- `graph_score()` — overall score + component breakdown
- `trusts()` — current trust values per node (computed lazily)
- `set_metric_weights(weights, normalize=False)` — set a **single global** metric weight vector (applies everywhere metrics are used)
- `node_feature_matrix(...)`, `edge_features()`, `node2vec_corpus(...)`, `paper_fingerprint()`
- `state()`, `snapshot()`, `reset()` — traversal and graph snapshot utilities

`__init__.py` re-exports `KGScorer`, `DAGValidation`, `KGSession`.

---

## JSON Schema (input)

```json
{
  "nodes": [
    { "id": "0", "text": "…", "role": "Hypothesis", "parents": [], "children": ["1","2"] },
    { "id": "1", "text": "…", "role": "Evidence",   "parents": ["0"], "children": ["3"] },
    { "id": "2", "text": "…", "role": "Method",     "parents": ["0"], "children": ["3"] },
    { "id": "3", "text": "…", "role": "Conclusion", "parents": ["1","2"], "children": [] }
  ]
}
```

### Notes

- **IDs** may be strings or integers in the JSON; they are canonicalized to **strings** internally.
- You may specify edges via `parents`, `children`, or both:
  - `DAGValidation.validate_and_build_from_json(..., reconcile="prefer_parents")` is the session default.
  - Other reconciliation modes include `"strict"`, `"prefer_children"`, `"union"`, `"intersection"`.

---

## Minimal Usage (Agent Loop)

```python
from service_adapter import KGSession

sess = KGSession(graph_json)

# 1) Validate (fail fast before you start prompting an agent)
print(sess.validation_report())

# Optional: surface readiness / missing-confidence issues for the UI
print(sess.scoring_validation(strict=False))

# 2) Optional: tune global metric weights (applies to node quality + synergy)
sess.set_metric_weights(
    {
        "credibility": 1.0,
        "relevance": 0.8,
        "evidence_strength": 1.2,
        "method_rigor": 1.1,
        "reproducibility": 0.9,
        "citation_support": 1.0,
    },
    normalize=False,  # if True, rescale so mean weight = 1.0
)

# 3) Optional: enable trust propagation penalty
# Supported agg modes: "min", "mean", "softmin", "dampmin"
sess.set_penalty(
    enabled=True,
    agg="softmin",
    alpha=1.0,         # strength of trust attenuation (>=0)
    eta=0.90,          # floor: eta=1 disables gating; eta=0 uses full gating
    softmin_beta=6.0,  # larger => closer to hard min
)

nid = sess.current()
while nid is not None:
    # Agent produces the 6 metrics in [0,1]
    m = {
        "credibility": 0.8,
        "relevance": 0.7,
        "evidence_strength": 0.6,
        "method_rigor": 0.5,
        "reproducibility": 0.4,
        "citation_support": 0.3,
    }

    out = sess.set_metrics_and_advance(nid, m)
    print("updated_edges:", out["updated_edges"])
    nid = out["next_node"]

print(sess.graph_score())
print(sess.trusts())
```

### Ordering constraint

`KGSession.set_metrics_and_advance(...)` enforces **in-order** updates: `node_id` must equal `current()`.  
If you need out-of-order updates, you can call `KGScorer.update_node_metrics(...)` directly or modify the adapter.

---

## Scoring Model (matches current implementation)

### Metrics

Each node has six metrics (all expected in **[0,1]**; 0–100 inputs are auto-scaled):

- `credibility`
- `relevance`
- `evidence_strength`
- `method_rigor`
- `reproducibility`
- `citation_support`

### Node quality

Let `m_v` be the metric vector for node `v`. A **single global** metric weight vector `w` is applied first:

- `m'_v[i] = clip01(w[i] * m_v[i])`

Then node quality is computed with role-specific weights (defaults live in `NodeQualityWeights`):

- `q_v = clip01( Σ_i a_i^{(role(v))} m'_v[i] / Σ_i |a_i^{(role(v))}| )`

### Edge readiness

An edge `u → v` becomes **eligible** (scored) when:

- `v` has all metrics, and
- every parent `p ∈ P(v)` has all metrics.

This ensures you do not compute “final” confidences before upstream information exists.

### Raw edge confidence

For each eligible edge `u → v`, a raw confidence is computed as a clipped weighted blend:

- role transition prior `r_{u→v}` (from `RoleTransitionPrior`)
- parent quality `q_u`
- child quality `q_v`
- (optional) text alignment `a_{u→v}` (defaults to 0 unless you provide alignment)
- pair synergy `s_{u→v}` (role-pair-specific, derived from metrics)

Formally:

- `c_raw(u,v) = clip01( λ_r r_{u→v} + λ_p q_u + λ_c q_v + λ_a a_{u→v} + λ_s s_{u→v} )`

### Trust propagation and final edge confidence (penalty)

If the propagation penalty is **disabled**, the edge confidence is simply:

- `c(u,v) = c_raw(u,v)`

If enabled, the system computes a **trust** value `t_v ∈ [0,1]` for each node:

- roots: `t_v = q_v`
- non-roots: `t_v = q_v · Agg( { (t_u)^α · c_raw(u,v) : u ∈ P(v) } )`

Where `Agg` is one of:

- `min` (hard minimum)
- `mean` (average)
- `softmin` (smooth approximation to minimum; controlled by `softmin_beta`)
- `dampmin` (convex mix: `(1-λ)·min + λ·mean`; controlled by `dampmin_lambda`)

Finally, each edge `u → v` is gated by the **parent trust**:

- `c(u,v) = clip01( c_raw(u,v) · ( η + (1-η) · (t_u)^α ) )`

Interpretation:
- `α` increases the penalty for low-trust parents.
- `η` is a *floor*; `η=1` disables gating, `η=0` applies full gating.

---

## Graph Score (strictly in (0,1))

`KGScorer.graph_score()` computes six interpretable components (all in [0,1]):

- `bridge_coverage`: fraction of nodes that lie on at least one Hypothesis→…→Conclusion path
- `best_path`: best Hypothesis→Conclusion path score (geometric mean of edge confidences)
- `redundancy`: edge-disjoint path multiplicity (soft-capped)
- `fragility`: normalized min-cut cost using capacities `(1 - confidence)`
- `coherence`: fraction of bridge edges whose role prior ≥ 0.5
- `coverage`: presence of key roles `{Method, Evidence, Result}` on the bridge

A weighted raw sum is formed (weights in `GraphScoreWeights`), then mapped into **[0,1]** using the
theoretical extrema implied by the weights, and finally “squeezed” into **(0,1)** with a small epsilon.

The returned `details` includes:
- `graph_score_raw`, `graph_score_01`, `graph_score`, plus all component metrics.

---

## Embeddings / Meta-Analysis

```python
# Per-node features (role one-hot + raw metrics [+ optional text embedding])
node_feats = sess.node_feature_matrix(text_embeddings=None)

# Edge features table (confidence + components + optional trust fields)
edges = sess.edge_features()

# Random-walk corpus for node2vec (uses edge confidence as weight)
walks = sess.node2vec_corpus(num_walks=20, walk_length=12, p=1.0, q=2.0, min_conf=0.2)

# Graph-level fingerprint (for retrieval/clustering)
paper_vec = sess.paper_fingerprint()
```

---

## Tunable Parameters (learnable or fixed)

- **Global metric weights**: `MetricWeights.weights` (via `KGSession.set_metric_weights(...)`)
- **Role-specific node quality weights**: `NodeQualityWeights.per_role`
- **Edge blend weights**: `EdgeCombineWeights` (role prior / parent quality / child quality / alignment / synergy)
- **Role transition prior**: `RoleTransitionPrior.matrix`
- **Pair synergy weights**: `PairSynergyWeights.pairs`
- **Propagation penalty**: `PropagationPenalty(enabled, agg, alpha, eta, softmin_beta, dampmin_lambda)`
- **Graph score weights**: `GraphScoreWeights`
- **Walk parameters**: node2vec-style `(p, q)` and confidence exponent / thresholds (see `random_walk_corpus`)

---

## Implementation Notes

- Inputs are validated and clipped to [0,1] at the API boundary.
- Recomputations are gated by **eligibility** and stream **edge deltas** to the frontend after each commit.
- `scoring_validation()` is intended to be shown in the UI to avoid silently depending on defaults.
