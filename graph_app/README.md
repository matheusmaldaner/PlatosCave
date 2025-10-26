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

* Nodes $v\in V$, directed edges $u\to v\in E$. Parents $P(v)={,u:(u\to v)\in E,}$.
* Six node metrics $m_v\in[0,1]^6$ in fixed order:
  $m_v=\big[\text{cred},,\text{rel},,\text{evid},,\text{rigor},,\text{repr},,\text{cites}\big]_v.$
* Convex per-metric weights $w\in\Delta_6:={w\in\mathbb R_{\ge 0}^6:\sum_i w_i=1}$.
  (Optionally role-specific $w^{(\rho)}$.)

---

## 1) Nodes: Agent-Extracted Features + Convex Quality

**Node readiness.** The agent fills metrics in **BFS order** from Hypothesis roots. A node is *ready* iff all six metrics are present:
$\mathbf 1_{\mathrm{ready}}(v)=\prod_{i=1}^{6}\mathbf 1_{,m_{v,i}\ \text{is present},}.$

**Node quality (convex).** A node’s scalar quality is a convex blend:
$q_v=\langle w,m_v\rangle=\sum_{i=1}^{6}w_i m_{v,i},\text{ with } w\in\Delta_6.$
**Intuition.** Convexity keeps $q_v$ in $[0,1]$, **interpretable**, and **comparable** across roles/graphs. If some metrics are missing, either impute or **renormalize** $w$ over present components.

**Role-aware option.** For role $\rho(v)$, use $q_v=\langle w^{(\rho(v))},m_v\rangle$ with $w^{(\rho)}\in\Delta_6$.
**Intuition.** Emphasize, e.g., `evidence_strength` more on **Evidence** than **Method**.

---

## 2) Edges: Interpretable Confidence with Trust Gating

Edges are scored **only when ready**:
$\mathbf 1_{\mathrm{ready}}(u!\to!v);=;\mathbf 1_{\mathrm{ready}}(v)\cdot\prod_{p\in P(v)}\mathbf 1_{\mathrm{ready}}(p).$

We combine **five interpretable factors** in $[0,1]$:

1. **Role transition prior** $r_{u\to v}$ (table over $\rho(u)!\to!\rho(v)$).
2. **Parent quality** $q_u$ and **Child quality** $q_v$ (from above).
3. **Text alignment** $a_{u\to v}$ (e.g., token Jaccard or cosine on embeddings).
4. **Pairwise synergy** $s_{u\to v}$ (role-pair-specific mix of parent/child metrics).

**Raw edge confidence (convex blend).**
$
c^{\mathrm{raw}}*{u\to v}
;=;
\lambda_r,r*{u\to v}
+\lambda_p,q_u
+\lambda_c,q_v
+\lambda_a,a_{u\to v}
+\lambda_s,s_{u\to v},
\qquad \lambda\in\Delta_5.
$
**Intuition.** A convex mixture keeps scores in $[0,1]$, eases calibration, and each $\lambda_k$ remains **explainable** in the UI.

> **Alternative (multiplicative).** \quad $c^{\mathrm{raw}}=\prod_k x_k^{\gamma_k}$ with $\gamma_k\ge 0$.
> **Intuition.** Penalizes weak factors more aggressively; learnable choice between additive vs. multiplicative.

**Trust gate from parents.** Aggregate parent trust with $\operatorname{Agg}\in{\min,\ \mathrm{mean},\ \mathrm{LSE}*\alpha}$ where
$
\mathrm{LSE}*\alpha(S);=;\tfrac{1}{\alpha}\log!\sum_{x\in S}e^{\alpha x}
\quad(\alpha>0\ \text{soft-max},\ \alpha<0\ \text{soft-min}).
$
Then apply a sigmoid gate with cutoff $\eta$ and sharpness $\beta$:
$
\tau_{P(v)};=;\sigma!\big(\beta,[,\operatorname{Agg}({q_p:p\in P(v)})-\eta,]\big),\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
$
**Intuition.** **Noisy/weak parents** should **down-weight** downstream edges. $\beta$ governs gate steepness; $\eta$ is the minimum acceptable upstream quality.

**Final edge confidence.**
$C_{u\to v};=;\mathbf 1_{\mathrm{ready}}(u!\to!v)\cdot\tau_{P(v)}\cdot c^{\mathrm{raw}}_{u\to v}.$
**Intuition.** Readiness avoids stale scores; the gate $\tau$ protects against **over-crediting** chains with weak ancestors.

---

## 3) Graph: Scores, Walks, Fingerprints

**Best-path reliability.** Let $\Pi$ be all root→leaf paths:
$S_{\text{path}}=\max_{\pi\in\Pi}\ \prod_{(i\to j)\in\pi} C_{i\to j}.$
**Intuition.** A paper is strong if **at least one** high-confidence causal chain survives.

**Coverage / coherence / redundancy / fragility.**

* **Coverage** over roles $\mathcal R$: $S_{\text{cov}}=\sum_{\rho\in\mathcal R}\bar q_\rho$, with $\bar q_\rho=\operatorname{mean}{q_v:\rho(v)=\rho}$.
* **Coherence** (role-prior compliance): $\mathbb E_{(u\to v)}[,r_{u\to v},]$.
* **Redundancy**: mean of top-$k$ disjoint path reliabilities.
* **Fragility**: $1-\text{min-cut}$ computed on costs $1-C_{u\to v}$.

**Graph score (convex).**
$S_{\text{graph}};=;\sum_{t\in{\text{path},\text{cov},\text{coh},\text{red},\text{frag}}}\mu_t,S_t,
\qquad \mu\in\Delta_5.$
**Intuition.** A single scalar with **decomposable** components for transparency/ablation.

**Confidence-weighted random walks (node2vec-style).** Directed transition kernel
$P(v!\to!w);=;\frac{C_{v\to w}^{,\kappa}\cdot\phi_{p,q}(\text{2nd-order bias})}{\sum_{w':(v\to w')\in E}C_{v\to w'}^{,\kappa}\cdot\phi_{p,q}(\cdot)},$
with standard node2vec bias $\phi_{p,q}$ (return/outward control) and temperature $\kappa>0$.
**Intuition.** Walks concentrate on **trustworthy subgraphs**, yielding corpora for node/graph embeddings.

**Paper fingerprint.**
$f(G)=\big[S_{\text{graph}},,\text{hist}(C),,\text{role counts},,\text{role-pair counts},,\overline m,,\text{top-}k\text{ path stats},\ldots\big].$
**Intuition.** A fixed-length, stable vector for retrieval, clustering, or downstream learning.

---

## Intuition Recap (Why this is different)

* **Convex node quality** keeps calibration simple and interpretable.
* **Role-aware priors/synergy** encode the **schema** of scientific reasoning.
* **Trust gating** prevents weak evidence from inflating downstream claims.
* **Deterministic BFS readiness** yields reproducible, streaming updates.
* **Confidence-weighted walks** and **fingerprints** bridge into standard embedding/IR pipelines.

---

## Next Steps: Learnability, Embedding Alignment, Recommendation

### A. Learn the Parameters from Human Annotations

We expose a **differentiable** parameter surface:

* **Node weights:** $w$ (or $w^{(\rho)}$).
* **Edge blend:** $\lambda$ (or multiplicative $\gamma$).
* **Trust gate:** $(\beta,\eta)$ and aggregator choice (use $\mathrm{LSE}_\alpha$ for a differentiable soft-min/mean).
* **Role prior / synergy:** entries of $r_{u\to v}$ and $s_{u\to v}$ (parametrize with sigmoid to keep $[0,1]$).

**Targets & losses (choose per label granularity).**

* **Per-edge** labels $\hat C_{u\to v}$: MSE or calibrated cross-entropy on $\operatorname{logit}(C)$.
* **Per-node** labels $\hat q_v$: MSE on $q_v$.
* **Per-paper** ratings $\hat S$: MSE on $S_{\text{graph}}$ + ranking losses for pairwise comparisons.
* **Regularization:** $\ell_2$, simplex via softmax, monotonicity priors (e.g., enforce $\partial C/\partial q_u\ge 0$).

**Training loop sketch.**

1. Build graphs; collect $(\hat C,\hat q,\hat S)$.
2. Forward: compute $(q,C,S)$ with current parameters.
3. Loss: weighted sum of edge/node/graph terms + regularizers.
4. Backprop (PyTorch/JAX), update $(w,\lambda,\beta,\eta,r,s,,\ldots)$.
5. Calibrate (temperature scaling / Platt) if needed.

### B. Align Graph Embeddings with Content Embeddings

Graph gives **trust-aware** signals; content models (e.g., SBERT) give **semantic** signals.

* **Early fusion:** concatenate $[,e_{\text{text}}(v)\ |\ X_v,]$ then learn a projection.
* **Late fusion:** $\text{score}=\alpha,\cos!\big(e_{\text{text}}(x),e_{\text{text}}(y)\big)+(1-\alpha),\cos!\big(f(G_x),f(G_y)\big)$.
* **Multiview alignment:** (Deep) CCA between ${f(G)}$ and ${e_{\text{text}}(\cdot)}$; or contrastive loss that pulls **same-topic & high-trust** pairs together.

**Outcome.** A **hybrid embedding** where **content** and **trust-structure** live in the **same space**—ideal for retrieval and re-ranking.

### C. Recommendation & Research-Companion Layer

* **Trust-aware retrieval:** index $f(G)$ (plus content vectors) in FAISS; query with adjustable $\alpha$ between semantics and trust.
* **Session-aware ranking:** boost edges/nodes overlapping the user’s active graph; decay low-confidence chains.
* **Active learning:** choose the **next node to annotate** that maximizes expected gain in $S_{\text{graph}}$ or reduces uncertainty on critical paths.
* **Contradiction mining:** surface high-$C$ edges forming **conflicting** role patterns across papers.
* **Planning:** from a target Conclusion, back-chain to missing **Method/Evidence** roles; propose a reading list.

---

## Tunable Hyper-Parameters (Learnable or Fixed)

* **Node:** $w$ (or $w^{(\rho)}$).
* **Edge:** $\lambda$ (or $\gamma$), synergy map $s_{u\to v}$.
* **Trust:** $(\beta,\eta,\alpha)$ for gating and $\mathrm{LSE}_\alpha$; aggregator choice (soft-min/mean).
* **Graph:** $\mu$ for score components; $(\kappa,p,q)$ for walks.

All are either **convex** (simplex) or **smoothly parameterized** (sigmoid/softmax), enabling straightforward gradient-based learning.

---

## Implementation Notes

* All metric inputs are normalized to $[0,1]$ (0–100 auto-scaled).
* Recomputations are **gated by readiness**, and we stream **edge deltas** to the UI after each agent commit.
* Random-walk corpora use $C^\kappa$ so walks dwell in **reliable subgraphs**.
* Validation enforces DAG, role sanity, and parent/child reconciliation.
