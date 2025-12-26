# service_adapter.py
from collections import deque
from typing import Dict, Any, List, Optional, Tuple

from kg_realtime_scoring import DAGValidation, KGScorer  # your module


# deterministic BFS (copy of the simple one in the demo)
def bfs_order(G, roots: List[str]) -> List[str]:
    roots = [r for r in roots if r in G]
    seen, order = set(), []
    dq = deque(roots)
    for r in roots:
        seen.add(r)
    while dq:
        u = dq.popleft()
        order.append(u)
        for v in sorted(G.successors(u), key=lambda z: str(z)):
            if v not in seen:
                seen.add(v)
                dq.append(v)
    # append any isolated/unreached nodes to guarantee coverage
    for n in sorted(G.nodes(), key=lambda z: str(z)):
        if n not in seen:
            order.append(n)
    return order


METRIC_KEYS: Tuple[str, ...] = (
    "credibility",
    "relevance",
    "evidence_strength",
    "method_rigor",
    "reproducibility",
    "citation_support",
)


class KGSession:
    def __init__(self, graph_json: Dict[str, Any]):
        self.kg, self.report = DAGValidation.validate_and_build_from_json(
            graph_json, reconcile="prefer_parents", strict_roles=True, expect_roots=True
        )
        if not self.report.errors:
            H = [
                n
                for n in self.kg.G.nodes()
                if self.kg.nodes[n].role == "Hypothesis" and self.kg.G.in_degree(n) == 0
            ]
            if not H:
                H = [n for n in self.kg.G.nodes() if self.kg.nodes[n].role == "Hypothesis"]
            self.order: List[str] = bfs_order(self.kg.G, H or list(self.kg.G.nodes())[:1])
        else:
            self.order = []
        self.i = 0
        self._updated_edges: List[Tuple[str, str, float]] = []
        self.kg.register_edge_update_callback(
            lambda u, v, w, feats: self._updated_edges.append((u, v, round(w, 3)))
        )

    def validation_report(self) -> Dict[str, Any]:
        return {
            "ok": len(self.report.errors) == 0,
            "errors": self.report.errors,
            "warnings": self.report.warnings,
            "stats": self.report.stats,
        }

    def scoring_validation(self, strict: bool = False) -> Dict[str, Any]:
        """
        Validate that the current scoring state is complete enough to produce stable scores.

        Intended to be surfaced to the frontend so it can warn when scoring is relying on
        defaults (e.g., eligible edges missing confidence/confidence_raw).
        """
        rep = self.kg.validate_scoring_state(strict=strict)
        ok = not rep.get("errors")
        return {"ok": ok, "report": rep}

    def current(self) -> Optional[str]:
        return None if self.i >= len(self.order) else self.order[self.i]

    def set_metrics_and_advance(self, node_id: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Front-end calls this after the agent returns the 6 metrics for the current node."""
        expected = self.current()
        if expected is None:
            raise ValueError("No current node: session is already at the end of the traversal order.")
        if node_id != expected:
            raise ValueError(
                f"Out-of-order metrics update: got node_id='{node_id}', expected current='{expected}'. "
                "If you need out-of-order updates, do not advance the pointer (or call reset())."
            )

        self._updated_edges.clear()

        # Validates keys & ranges [0,1]; raises on bad input (good for API 400s)
        self.kg.update_node_metrics(node_id, **metrics)  # recalculates incoming edges if eligible

        # advance pointer to next not-yet-complete node
        self.i += 1
        while self.i < len(self.order):
            nid = self.order[self.i]
            n = self.kg.nodes[nid]
            if all(getattr(n, k) is not None for k in METRIC_KEYS):
                self.i += 1
            else:
                break

        return {
            "updated_edges": [{"u": u, "v": v, "confidence": w} for (u, v, w) in self._updated_edges],
            "next_node": self.current(),
        }

    def graph_score(self) -> Dict[str, Any]:
        score, details = self.kg.graph_score()
        return {"score": score, "details": details}

    # --- Embedding helpers for the frontend ---

    def node_feature_matrix(self, text_embeddings: Dict[str, Any] | None = None):
        """
        Returns {'ids': [...], 'features': 2D list/array, 'names': [...]}
        text_embeddings: optional dict node_id -> vector (list or np.ndarray)
        """
        X, ids, names = self.kg.export_node_feature_matrix(text_embeddings=text_embeddings)
        # Convert np to Python list for JSON if needed
        try:
            X_out = X.tolist()
        except AttributeError:
            X_out = X
        return {"ids": ids, "features": X_out, "names": names}

    def edge_features(self):
        return self.kg.export_edge_features()

    def node2vec_corpus(self, num_walks=10, walk_length=8, p=1.0, q=1.0, min_conf=0.0):
        return self.kg.random_walk_corpus(
            num_walks=num_walks, walk_length=walk_length, bias_p=p, bias_q=q, min_conf=min_conf
        )

    def paper_fingerprint(self):
        vec, names = self.kg.paper_fingerprint()
        return {"vector": vec, "names": names}

    def state(self) -> Dict[str, Any]:
        return {"order": self.order, "index": self.i, "current": self.current()}

    def snapshot(self) -> Dict[str, Any]:
        nodes = []
        for nid in sorted(self.kg.nodes.keys(), key=str):
            n = self.kg.nodes[nid]

            raw = {k: getattr(n, k) for k in METRIC_KEYS}
            present = [raw[k] is not None for k in METRIC_KEYS]
            status = "complete" if all(present) else "partial" if any(present) else "empty"

            # Keep existing 'metrics' for backwards compatibility (metric_dict uses 0.0 for missing),
            # and add 'metrics_raw' so callers can distinguish None vs an actual 0.0.
            nodes.append(
                {
                    "id": nid,
                    "role": n.role,
                    "status": status,
                    "metrics": n.metric_dict(),
                    "metrics_raw": raw,
                }
            )

        edges = []
        for u, v in self.kg.G.edges():
            edges.append({"u": u, "v": v, "confidence": self.kg.G[u][v].get("confidence")})
        return {"nodes": nodes, "edges": edges, **self.state()}

    def reset(self):
        if self.report.errors:
            self.order, self.i = [], 0
            return self.state()
        H = [
            n
            for n in self.kg.G.nodes()
            if self.kg.nodes[n].role == "Hypothesis" and self.kg.G.in_degree(n) == 0
        ]
        if not H:
            H = [n for n in self.kg.G.nodes() if self.kg.nodes[n].role == "Hypothesis"]
        self.order = bfs_order(self.kg.G, H or list(self.kg.G.nodes())[:1])
        self.i = 0
        return self.state()

    def set_penalty(self, **kwargs):
        """e.g., set_penalty(enabled=True, agg='min'|'mean'|'lse', alpha=1.0, eta=0.05)"""
        for k, v in kwargs.items():
            if hasattr(self.kg.penalty, k):
                setattr(self.kg.penalty, k, v)

    def trusts(self):
        """Return current node trust map (computed lazily on next edge recompute)."""
        # ensure trusts exist for at least current frontier
        for nid in self.kg.G.nodes():
            if nid not in self.kg.trust and self.kg.nodes[nid].has_all_metrics():
                try:
                    self.kg._compute_node_trust(nid)  # safe: DAG + memoization
                except Exception:
                    pass
        return {nid: float(val) for nid, val in self.kg.trust.items()}

    def set_metric_weights(self, weights: Dict[str, float], normalize: bool = False):
        """
        Frontend/API entry point to set a single global weight vector over the six metrics.
        Example payload:
        {
            "credibility": 1.0,
            "relevance": 0.8,
            "evidence_strength": 1.2,
            "method_rigor": 1.1,
            "reproducibility": 0.9,
            "citation_support": 1.0
        }
        """
        self.kg.set_metric_weights(weights, normalize=normalize)
        return {"ok": True, "weights": dict(self.kg.metric_w.weights)}
