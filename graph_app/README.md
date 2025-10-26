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

# Technical Novelty: A Live, Learnable, Trust-Propagating KG

This system is a **validated DAG** whose **nodes** carry agent-extracted, normalized metrics; whose **edges** compute **role-aware, interpretable confidences** under **trust gating**; and whose **graph** exposes **deterministic traversal**, **learnable parameters**, and **analysis-ready embeddings**.

## Notation

* Nodes (v \in V), directed edges (u \to v \in E). Parents (P(v)={u:(u\to v)\in E}).
* Six node metrics (m_v \in [0,1]^6) in fixed order:
  (m_v = [\text{cred}, \text{rel}, \text{evid}, \text{rigor}, \text{repr}, \text{cites}]_v).
* Convex per-metric weights (w \in \Delta^5) (simplex): (w_i\ge 0,\ \sum_i w_i=1).
  (We also support role-specific (w^{(\rho)}) if desired.)

---

## 1) Nodes: Agent-Extracted Features + Convex Quality

**Definition (Node readiness).**
The agent fills metrics in a **BFS order** from Hypothesis roots. A node (v) is *ready* iff its six metrics are present:
[
\mathbb{1}*{\mathrm{ready}}(v)=\prod*{i=1}^{6}\mathbb{1}{m_{v,i}\text{ is present}}.
]

**Definition (Node quality).**
A node’s scalar quality is a convex blend of its metrics:
[
q_v ;=;\langle w,,m_v\rangle ;=; \sum_{i=1}^{6} w_i,m_{v,i},
\qquad w\in \Delta^5.
]
*Intuition:* convexity makes (q_v) **interpretable**, **scale-stable**, and **comparable** across roles and graphs. Missing metrics can be imputed or the weights renormalized over present components.

**Optional (Role-aware quality).**
For role (\rho(v)), use (q_v=\langle w^{(\rho(v))}, m_v\rangle) with (w^{(\rho)}\in\Delta^5).
*Intuition:* lets you emphasize (e.g.) *evidence_strength* more for **Evidence** than for **Method** nodes.

---

## 2) Edges: Interpretable Confidence with Trust Gating

Edges are scored **only when ready**:
[
\mathbb{1}*{\mathrm{ready}}(u!\to!v) ;=; \mathbb{1}*{\mathrm{ready}}(v)\cdot \prod_{p\in P(v)} \mathbb{1}_{\mathrm{ready}}(p).
]

We compose **five interpretable factors** in ([0,1]):

1. **Role transition prior** (r_{u\to v}\in[0,1])
   (table over (\rho(u)\to\rho(v)), encodes structure like **Hypothesis→Evidence** being likelier than **Claim→Method**).

2. **Parent quality** (q_u) and **Child quality** (q_v) (from above).

3. **Text alignment** (a_{u\to v}\in[0,1])
   (e.g., token Jaccard on content words; could be replaced by cosine on embeddings).

4. **Pairwise synergy** (s_{u\to v}\in[0,1])
   (role-pair-specific mix of parent/child metrics; e.g., how *method_rigor* in a **Method** informs *evidence_strength* in an **Evidence** child).

**Raw edge confidence (interpretable convex blend).**
[
c^{\mathrm{raw}}*{u\to v}
;=;
\lambda_r, r*{u\to v} ;+; \lambda_p, q_u ;+; \lambda_c, q_v ;+; \lambda_a, a_{u\to v} ;+; \lambda_s, s_{u\to v},
\quad \lambda\in\Delta^4.
]
*Intuition:* a **convex** mixture keeps scores in ([0,1]), is easy to calibrate, and each (\lambda_k) is directly **explainable** in the UI.

> Alternative (multiplicative): (c^{\mathrm{raw}}=\prod_k x_k^{\gamma_k}) with exponents (\gamma_k\ge 0).
> *Intuition:* multiplicative mixes penalize any weak factor more aggressively; choosing between the two is a modeling decision (and can be learned).

**Trust gate from parents.**
Aggregate parent trust with a selectable operator (\mathrm{Agg}\in{\min,\ \mathrm{mean},\ \mathrm{LSE}*\alpha}), where
[
\mathrm{LSE}*\alpha(S);=;\frac{1}{\alpha}\log!\sum_{x\in S}!e^{\alpha x}
\quad(\alpha>0\ \text{≈ soft-max; } \alpha<0\ \text{≈ soft-min}).
]
We then shape/gate via a sigmoid with cutoff (\eta):
[
\tau_{P(v)} ;=; \sigma!\big(\beta,[,\mathrm{Agg}({q_p: p\in P(v)}) - \eta,]\big),
\qquad \sigma(z)=\frac{1}{1+e^{-z}}.
]
*Intuition:* **noisy/weak parents** should **down-weight** downstream edges. (\beta) controls sharpness; (\eta) the minimum acceptable upstream quality.

**Final edge confidence.**
[
C_{u\to v}
;=; \mathbb{1}*{\mathrm{ready}}(u!\to!v);\cdot; \tau*{P(v)};\cdot; c^{\mathrm{raw}}_{u\to v}.
]
*Intuition:* readiness avoids stale scores; the gate (\tau) protects against **over-crediting** chains with weak/uncertain ancestors.

---

## 3) Graph: Scores, Walks, Fingerprints

**Best path reliability.**
Let (\Pi) be all root→leaf paths; define
[
S_{\text{path}} ;=; \max_{\pi\in\Pi}\ \prod_{(i\to j)\in\pi} C_{i\to j}.
]
*Intuition:* a paper is strong if **at least one** high-confidence causal chain survives.

**Coverage / coherence / redundancy / fragility.**

* **Coverage** over roles (\mathcal{R}): (S_{\text{cov}}=\sum_{\rho\in\mathcal{R}} \bar q_\rho) where (\bar q_\rho) is the mean (q_v) over nodes with role (\rho).
* **Coherence** (role-prior compliance): expected (r_{u\to v}) under edge distribution.
* **Redundancy**: mean of top-(k) disjoint path reliabilities.
* **Fragility**: (1-)min-cut on edge weights (C_{u\to v}) after mapping to costs (1-C_{u\to v}).

**Graph score (convex).**
[
S_{\text{graph}} ;=; \sum_{t\in{\text{path,cov,coh,red,frag}}} \mu_t, S_t,
\qquad \mu \in \Delta^{4}.
]
*Intuition:* one scalar with **decomposable components** for transparency and ablation.

**Confidence-weighted random walks (node2vec-style).**
Transition kernel (directed):
[
P(v!\to!w) ;=; \frac{C_{v\to w}^\kappa\ \cdot\ \phi_{p,q}(\text{2nd-order bias})}{\sum_{w':(v\to w')\in E} C_{v\to w'}^\kappa\ \cdot\ \phi_{p,q}(\cdot)},
]
with standard node2vec bias (\phi_{p,q}) (return/outward control) and temperature (\kappa>0).
*Intuition:* walks concentrate on **trustworthy subgraphs**, producing corpora for node/graph embeddings.

**Paper fingerprint.**
[
f(G) ;=; \big[,S_{\text{graph}},\ \text{hist}(C),\ \text{role counts},\ \text{role-pair counts},\ \bar m,\ \text{top-}k\ \text{path stats},\ldots\big].
]
*Intuition:* a fixed-length, stable vector for retrieval, clustering, or downstream learning.

---

## Intuition Recap (Why this is different)

* **Convex node quality** keeps interpretability and calibration simple.
* **Role-aware priors/synergy** encode the **schema** of scientific reasoning.
* **Trust gating** prevents weak evidence from inflating downstream claims.
* **Deterministic BFS readiness** yields reproducible, streaming updates.
* **Confidence-weighted walks** and **fingerprints** bridge into standard embedding and IR pipelines.

---

## Next Steps: Learnability, Embedding Alignment, and Recommendation

### A. Learn the Parameters from Human Annotations

We expose a **differentiable parameter surface** suitable for supervision:

* **Node weights:** (w) or (w^{(\rho)}).
* **Edge blend:** (\lambda) (or multiplicative (\gamma)).
* **Trust gate:** (\beta,\eta) and Agg selection (soft-min/mean via (\mathrm{LSE}_\alpha) makes this differentiable).
* **Role prior / synergy:** entries of (r_{u\to v}) and (s_{u\to v}) (constrained to ([0,1]) with, e.g., sigmoid parameterization).

**Targets & losses** (choose per label granularity):

* **Per-edge** confidence labels (\hat C_{u\to v}): MSE or calibrated cross-entropy on (\text{logit}(C)).
* **Per-node** quality labels (\hat q_v): MSE on (q_v).
* **Per-paper** ratings (\hat S): MSE on (S_{\text{graph}}) + rank losses for pairwise paper comparisons.
* **Regularization:** (\ell_2) on parameters, simplex constraints via softmax, monotonicity priors (e.g., enforce (\partial C/\partial q_u\ge 0)).

**Training loop sketch:**

1. Build graphs, collect annotations ((\hat C,\hat q,\hat S)).
2. Forward pass: compute (q, C, S) with current parameters.
3. Loss: weighted sum of (edge/node/graph) terms + regularizers.
4. Backprop (PyTorch/JAX), update ((w,\lambda,\beta,\eta,r,s,\ldots)).
5. Calibrate with Temperature Scaling/Platt as needed.

### B. Make Graph Embeddings Comparable to Content Embeddings

Graph gives **trust-aware** signals; content models (e.g., SBERT) give **semantic** signals. Combine them:

* **Early fusion:** ([,e_{\text{text}}(v)\ |\ X_v,]) and learn a projection.
* **Late fusion:** score (=) (\alpha,\cos(e_{\text{text}}(x), e_{\text{text}}(y))\ +\ (1-\alpha),\cos(f(G_x), f(G_y))).
* **Multiview alignment:** CCA / Deep CCA between ({f(G)}) and ({e_{\text{text}}(\cdot)}) over a corpus of papers; or contrastive loss that pushes *same-topic* pairs together with higher trust weight.

*Outcome:* a **hybrid embedding** where **content** and **trust-structure** live in the **same space**—perfect for retrieval and re-ranking.

### C. Build a Recommendation & Research-Companion Layer

* **Trust-aware retrieval:** index (f(G)) (plus content vectors) in FAISS; query with adjustable (\alpha) between semantics and trust.
* **Session-aware ranking:** boost edges/nodes overlapping user’s active graph; decay low-confidence chains.
* **Active learning:** pick the **next node to annotate** that maximizes expected information gain on (S_{\text{graph}}) or reduces uncertainty on critical paths.
* **Contradiction detection:** search for high (C) edges forming **conflicting** role patterns across papers (e.g., Evidence contradicting Result), surface to user.
* **Planning:** from a target Conclusion, backchain to missing **Method/Evidence** roles with low availability; propose reading list.

---

## Tunable Hyper-Parameters (Learnable or Fixed)

* Node: (w) (or (w^{(\rho)})).
* Edge: (\lambda) (or (\gamma)), synergy map (s_{u\to v}).
* Trust: (\beta,\eta,\alpha) (for (\mathrm{LSE}_\alpha)) and Agg choice (soft-min/mean).
* Graph: (\mu) for score components; (\kappa,p,q) for walks.
  All are either **convex** (simplex) or **smoothly parameterized** (sigmoid/softmax), enabling straightforward gradient-based learning.

---

## Implementation Notes

* All metric inputs are normalized to ([0,1]) (0–100 auto-scaled).
* We gate recomputation by **readiness**, and stream **edge deltas** to the UI after each agent commit.
* Random-walk corpora use (C^\kappa) as weights so paths through **reliable subgraphs** dominate learned embeddings.
* Validation enforces DAG, role sanity, and parent/child reconciliation.
