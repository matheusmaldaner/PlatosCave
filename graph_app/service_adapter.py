# service_adapter.py
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
from kg_realtime_scoring import DAGValidation, KGScorer  # your module

# deterministic BFS (copy of the simple one in the demo)
def bfs_order(G, roots: List[str]) -> List[str]:
    roots = [r for r in roots if r in G]
    seen, order = set(), []
    dq = deque(roots)
    for r in roots: seen.add(r)
    while dq:
        u = dq.popleft()
        order.append(u)
        for v in sorted(G.successors(u), key=lambda z: str(z)):
            if v not in seen:
                seen.add(v); dq.append(v)
    # append any isolated/unreached nodes to guarantee coverage
    for n in sorted(G.nodes(), key=lambda z: str(z)):
        if n not in seen:
            order.append(n)
    return order

class KGSession:
    def __init__(self, graph_json: Dict[str, Any]):
        self.kg, self.report = DAGValidation.validate_and_build_from_json(
            graph_json, reconcile="prefer_parents", strict_roles=True, expect_roots=True
        )
        if not self.report.errors:
            H = [n for n in self.kg.G.nodes() if self.kg.nodes[n].role == "Hypothesis" and self.kg.G.in_degree(n) == 0]
            if not H:
                H = [n for n in self.kg.G.nodes() if self.kg.nodes[n].role == "Hypothesis"]
            self.order: List[str] = bfs_order(self.kg.G, H or list(self.kg.G.nodes())[:1])
        else:
            self.order = []
        self.i = 0
        self._updated_edges: List[Tuple[str,str,float]] = []
        self.kg.register_edge_update_callback(lambda u,v,w,feats: self._updated_edges.append((u,v,round(w,3))))

    def validation_report(self) -> Dict[str, Any]:
        return {"ok": len(self.report.errors) == 0,
                "errors": self.report.errors,
                "warnings": self.report.warnings,
                "stats": self.report.stats}

    def current(self) -> Optional[str]:
        return None if self.i >= len(self.order) else self.order[self.i]

    def set_metrics_and_advance(self, node_id: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Front-end calls this after the agent returns the 6 metrics for current node."""
        self._updated_edges.clear()
        # Validates keys & ranges [0,1]; raises on bad input (good for API 400s)
        self.kg.update_node_metrics(node_id, **metrics)  # recalculates incoming edges if eligible
        # advance pointer to next not-yet-complete node
        self.i += 1
        while self.i < len(self.order):
            nid = self.order[self.i]
            n = self.kg.nodes[nid]
            if all(getattr(n,k) is not None for k in
                   ("credibility","relevance","evidence_strength",
                    "method_rigor","reproducibility","citation_support")):
                self.i += 1
            else:
                break
        return {
            "updated_edges": [{"u":u, "v":v, "confidence":w} for (u,v,w) in self._updated_edges],
            "next_node": self.current()
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
        return self.kg.random_walk_corpus(num_walks=num_walks, walk_length=walk_length,
                                          bias_p=p, bias_q=q, min_conf=min_conf)

    def paper_fingerprint(self):
        vec, names = self.kg.paper_fingerprint()
        return {"vector": vec, "names": names}

    def state(self) -> Dict[str, Any]:
        return {"order": self.order, "index": self.i, "current": self.current()}
    
    def snapshot(self) -> Dict[str, Any]:
        nodes = []
        for nid in sorted(self.kg.nodes.keys(), key=str):
            n = self.kg.nodes[nid]
            md = n.metric_dict()
            status = ("complete" if all(v > 0 for v in md.values())
                    else "partial" if any(v > 0 for v in md.values())
                    else "empty")
            nodes.append({"id": nid, "role": n.role, "status": status, "metrics": md})
        edges = []
        for u, v in self.kg.G.edges():
            edges.append({"u": u, "v": v, "confidence": self.kg.G[u][v].get("confidence")})
        return {"nodes": nodes, "edges": edges, **self.state()}

    def reset(self):
        if self.report.errors:
            self.order, self.i = [], 0
            return self.state()
        H = [n for n in self.kg.G.nodes() if self.kg.nodes[n].role == "Hypothesis" and self.kg.G.in_degree(n) == 0]
        if not H:
            H = [n for n in self.kg.G.nodes() if self.kg.nodes[n].role == "Hypothesis"]
        self.order = bfs_order(self.kg.G, H or list(self.kg.G.nodes())[:1])
        self.i = 0
        return self.state()


