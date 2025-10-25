#!/usr/bin/env python3
"""
Minimal DAG + BFS HTTP API with strict validation for LLM-friendly retries.

Endpoints:
  GET  /health
  POST /load   body: { graph: <obj>|(raw graph as body), paper_id?, title?, default_role?, default_relation?, strict_dag? }
  GET  /graph
  GET  /state
  POST /step   body: {"op":"next"|"prev"|"goto","target":"<id>"}
  POST /score  body: {"id":"<node_id>", <metrics 0..1>}

Return codes:
  200  ok
  400  invalid JSON body / malformed HTTP
  422  structurally valid JSON but graph invalid per requested strictness (use error_code to guide LLM)
"""

from __future__ import annotations
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
import argparse, json

import networkx as nx  # for extra sanity checks
# Reuse your repo’s builders (DAG repair + root + levels + BFS, minimal->PaperGraph)
from app.graph_build import minimal_to_papergraph, build_graph_and_bfs  # type: ignore

# ---------------- Validation helpers ----------------

def validate_raw_graph(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight JSON/schema validation before normalization/build."""
    diags: Dict[str, Any] = {
        "json_ok": True,
        "schema_ok": True,
        "errors": [],
        "warnings": [],
        "node_count": 0,
        "edge_count": 0,
        "duplicate_ids": [],
        "unknown_sources": [],
        "unknown_targets": [],
        "dangling_edges": [],
    }
    if not isinstance(raw, dict):
        diags["json_ok"] = False
        diags["errors"].append("body_not_object")
        return diags

    nodes = raw.get("nodes")
    edges = raw.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        diags["schema_ok"] = False
        if not isinstance(nodes, list): diags["errors"].append("missing_or_invalid_nodes")
        if not isinstance(edges, list): diags["errors"].append("missing_or_invalid_edges")
        return diags

    # ids and duplicates
    ids = []
    for n in nodes:
        if not isinstance(n, dict) or "id" not in n:
            diags["schema_ok"] = False
            diags["errors"].append("node_missing_id")
            continue
        ids.append(str(n["id"]))
    diags["node_count"] = len(nodes)

    dup = sorted([i for i in set(ids) if ids.count(i) > 1])
    if dup:
        diags["schema_ok"] = False
        diags["duplicate_ids"] = dup
        diags["errors"].append("duplicate_node_ids")

    idset = set(ids)

    # edges validity
    bad_src, bad_tgt, dangling = [], [], []
    for e in edges:
        if not isinstance(e, dict) or "source" not in e or "target" not in e:
            diags["schema_ok"] = False
            diags["errors"].append("edge_missing_source_or_target")
            continue
        s, t = str(e["source"]), str(e["target"])
        if s not in idset: bad_src.append(s)
        if t not in idset: bad_tgt.append(t)
        if s not in idset or t not in idset:
            dangling.append({"source": s, "target": t})
    diags["edge_count"] = len(edges)
    if bad_src: diags["unknown_sources"] = sorted(set(bad_src))
    if bad_tgt: diags["unknown_targets"] = sorted(set(bad_tgt))
    if dangling: diags["dangling_edges"] = dangling

    if bad_src or bad_tgt or dangling or dup:
        diags["schema_ok"] = False

    return diags

# ---------------- Controller (no agent loop) ----------------

class BFSController:
    METRIC_KEYS = [
        "credibility", "relevance", "evidence_strength",
        "method_rigor", "reproducibility", "citation_support",
    ]

    def __init__(self) -> None:
        self.G: Optional[nx.DiGraph] = None
        self.root: Optional[str] = None
        self.order: List[str] = []
        self.removed: List[Tuple[str, str]] = []
        self.index: int = 0
        self.paper_id: str = "unknown"
        self.title: str = "Untitled"

    def load(self, raw: Dict[str, Any], *, paper_id="unknown", title="Untitled",
             default_role="other", default_relation="supports", strict_dag=False) -> Dict[str, Any]:
        # 1) pre-validate the raw input
        pre = validate_raw_graph(raw)
        if not pre["json_ok"]:
            raise ValueError("invalid_json_body")
        if not pre["schema_ok"]:
            # return diagnostics allowing LLM to fix IDs/edges
            return {
                "ok": False,
                "error_code": "invalid_input_graph",
                "message": "Input graph failed schema checks (see diagnostics).",
                "diagnostics": pre,
            }

        # 2) normalize minimal -> full PaperGraph (preserves provided role/relation)
        nodes = raw.get("nodes", [])
        edges = raw.get("edges", [])
        has_role = any(isinstance(n, dict) and ("role" in n) for n in nodes)
        has_rel  = any(isinstance(e, dict) and ("relation" in e) for e in edges)
        try:
            if not has_role or not has_rel:
                pg = minimal_to_papergraph(
                    raw, paper_id=paper_id, title=title,
                    default_role=default_role, default_relation=default_relation
                )
                pg_dict = pg.model_dump()
            else:
                pg_dict = {**raw}
                pg_dict.setdefault("paper_id", paper_id)
                pg_dict.setdefault("title", title)
        except Exception as e:
            return {
                "ok": False,
                "error_code": "normalization_failed",
                "message": f"Could not normalize graph: {e}",
                "diagnostics": pre,
            }

        # 3) build DAG + BFS
        try:
            G, _map, root, order, removed = build_graph_and_bfs(pg_dict)
        except Exception as e:
            return {
                "ok": False,
                "error_code": "build_failed",
                "message": f"Graph build failed: {e}",
                "diagnostics": pre,
            }

        dag_ok = nx.is_directed_acyclic_graph(G)
        disconnected = nx.number_weakly_connected_components(G) if len(G) else 0
        graph_diags = {
            "dag_ok": dag_ok,
            "repaired": bool(removed),
            "removed_edges": [[u, v] for (u, v) in removed],
            "root_found": bool(root),
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "bfs_order_len": len(order),
            "disconnected_components": disconnected,
        }

        # strictness policy
        if not dag_ok:
            return {
                "ok": False,
                "error_code": "graph_not_dag",
                "message": "Input contains irreparable cycles.",
                "diagnostics": {**pre, **graph_diags},
            }
        if strict_dag and removed:
            return {
                "ok": False,
                "error_code": "dag_repair_required",
                "message": f"DAG repair would remove {len(removed)} edge(s).",
                "diagnostics": {**pre, **graph_diags},
            }
        if not root:
            return {
                "ok": False,
                "error_code": "root_not_found",
                "message": "No suitable root could be determined.",
                "diagnostics": {**pre, **graph_diags},
            }
        if not order:
            return {
                "ok": False,
                "error_code": "empty_bfs_order",
                "message": "BFS order is empty.",
                "diagnostics": {**pre, **graph_diags},
            }

        # 4) commit state and return
        self.G, self.root, self.order, self.removed = G, root, order, removed
        self.index = 0
        self.paper_id = pg_dict.get("paper_id", paper_id)
        self.title = pg_dict.get("title", title)

        return {"ok": True, "state": self.state(), "diagnostics": {**pre, **graph_diags}}

    # ---- accessors ----
    def current_id(self) -> str:
        if not self.order: return ""
        return self.order[max(0, min(self.index, len(self.order)-1))]

    def successors(self, nid: str) -> List[str]:
        if self.G is None or nid not in self.G.nodes: return []
        return sorted(self.G.successors(nid), key=str)

    def node_json(self, nid: str) -> Dict[str, Any]:
        if self.G is None or nid not in self.G.nodes: return {}
        d = self.G.nodes[nid]
        return {"id": str(nid), "text": d.get("text",""), "role": d.get("role","other"),
                "level": int(d.get("level",0)), "score": d.get("score")}

    # ---- API payloads ----
    def graph(self) -> Dict[str, Any]:
        if self.G is None:
            return {"paper_id": self.paper_id, "title": self.title,
                    "root": self.root, "nodes": [], "edges": []}
        nodes = [self.node_json(n) for n in self.G.nodes]
        edges = [{"source":u, "target":v, "relation": self.G.edges[u,v].get("relation","supports")}
                 for (u,v) in self.G.edges]
        return {"paper_id": self.paper_id, "title": self.title, "root": self.root,
                "nodes": nodes, "edges": edges}

    def state(self) -> Dict[str, Any]:
        cid = self.current_id()
        visited = self.order[: self.index+1] if self.order else []
        return {"paper_id": self.paper_id, "title": self.title, "root": self.root,
                "order": self.order, "index": self.index, "current": self.node_json(cid),
                "successors": self.successors(cid), "visited": visited,
                "removed_edges": [[u,v] for (u,v) in self.removed],
                "metric_keys": self.METRIC_KEYS}

    # ---- step / score ----
    def step(self, op="next", target: Optional[str]=None) -> Dict[str, Any]:
        if not self.order: return self.state()
        op = (op or "next").lower()
        if op == "next": self.index = min(self.index+1, len(self.order)-1)
        elif op == "prev": self.index = max(self.index-1, 0)
        elif op == "goto" and target in self.order: self.index = self.order.index(target)
        return self.state()

    def set_score(self, node_id: str, score: Dict[str, Any]) -> Dict[str, Any]:
        if self.G is None or node_id not in self.G.nodes:
            return {"ok": False, "error": "unknown node"}
        s = {}
        for k in self.METRIC_KEYS:
            try: s[k] = max(0.0, min(1.0, float(score.get(k,0) or 0.0)))
            except Exception: s[k] = 0.0
        s["composite"] = round(sum(s[k] for s_k in [k for k in self.METRIC_KEYS] for k in [s_k] and [k] and [k] and [] or [k]  # noqa
                                  ) / len(self.METRIC_KEYS), 4)  # simple average
        self.G.nodes[node_id]["score"] = s
        return {"ok": True, "id": str(node_id), "score": s}

# ---------------- HTTP server ----------------

_CTRL = BFSController()

class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: Dict[str, Any]):
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        # CORS
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Dict[str, Any]:
        try:
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n) if n > 0 else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return {}

    def do_OPTIONS(self):
        self._send(200, {"ok": True})

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health": self._send(200, {"ok": True}); return
        if path == "/graph":  self._send(200, _CTRL.graph()); return
        if path == "/state":  self._send(200, _CTRL.state()); return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        body = self._read_json()

        if path == "/load":
            graph_obj = body.get("graph") if isinstance(body.get("graph"), dict) else body
            paper_id = body.get("paper_id", "unknown")
            title = body.get("title", "Untitled")
            default_role = body.get("default_role", "other")
            default_relation = body.get("default_relation", "supports")
            strict_dag = bool(body.get("strict_dag", False))
            if not isinstance(graph_obj, dict):
                self._send(400, {"ok": False, "error_code": "invalid_json_body", "message": "Body must be an object"}); return
            res = _CTRL.load(graph_obj, paper_id=paper_id, title=title,
                             default_role=default_role, default_relation=default_relation,
                             strict_dag=strict_dag)
            # choose status code by ok flag
            if res.get("ok") is True:
                self._send(200, res)
            else:
                # validation/norm/build issues → 422
                self._send(422, res)
            return

        if path == "/step":
            op = body.get("op", "next"); target = body.get("target")
            self._send(200, _CTRL.step(op=op, target=target)); return

        if path == "/score":
            nid = body.get("id")
            if not nid:
                self._send(400, {"ok": False, "error_code": "missing_id", "message": "Missing node id"}); return
            self._send(200, _CTRL.set_score(str(nid), body)); return

        self._send(404, {"error": "not found"})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    httpd = HTTPServer((args.host, args.port), Handler)
    print(f"[bfs-api] listening on http://{args.host}:{args.port}")
    httpd.serve_forever()

if __name__ == "__main__":
    main()
