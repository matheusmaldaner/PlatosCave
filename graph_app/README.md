# Paper Rater: KG Real-Time Scoring & Service Adapter

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
