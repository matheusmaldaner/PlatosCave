#!/usr/bin/env python3
"""
Minimal DAG + BFS HTTP API (no UI, no agent loop)

Endpoints:
  GET  /health                 -> {"ok": true}
  POST /load                   -> load/normalize graph JSON, build DAG+BFS
  GET  /graph                  -> full graph snapshot (nodes+edges)
  GET  /state                  -> BFS runtime state (root, order, index, current, successors, visited, removed_edges)
  POST /step                   -> {"op": "next"|"prev"|"goto", "target": "<id>"}  -> updated /state
  POST /score                  -> {"id": "<node_id>", metrics...}  -> attach metrics in {0..1} to node

Notes:
  • Accepts "minimal" graph JSON (nodes[id,text,role?], edges[source,target,relation?]) or full PaperGraph.
  • Missing role/relation are normalized via app.graph_build.minimal_to_papergraph (role defaults to "other", relation to "supports").
  • DAG repair + root + levels + deterministic BFS order are computed by app.graph_build.build_graph_and_bfs.
  • CORS enabled for simple browser clients.

Run:
  python bfs_api_min.py --port 8765

Smoke-test:
  curl -s http://127.0.0.1:8765/health
  curl -s -X POST http://127.0.0.1:8765/load -H 'Content-Type: application/json' --data-binary @platos_cave_minimal_labeled.json | jq
  curl -s http://127.0.0.1:8765/graph | jq '.nodes|length'
  curl -s http://127.0.0.1:8765/state | jq
  curl -s -X POST http://127.0.0.1:8765/step -H 'Content-Type: application/json' -d '{"op":"next"}' | jq
  curl -s -X POST http://127.0.0.1:8765/score -H 'Content-Type: application/json' -d '{"id":"0","credibility":0.9}' | jq
"""

from __future__ import annotations
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import argparse

# --- Reuse your repo logic (DAG repair + root + levels + BFS, minimal->PaperGraph) ---
from app.graph_build import minimal_to_papergraph, build_graph_and_bfs  # type: ignore

# ---------------- Controller (no agent loop) ----------------

class BFSController:
    METRIC_KEYS = [
        "credibility", "relevance", "evidence_strength",
        "method_rigor", "reproducibility", "citation_support",
    ]

    def __init__(self) -> None:
        self.G = None                       # networkx.DiGraph
        self.root: Optional[str] = None
        self.order: List[str] = []
        self.removed: List[Tuple[str, str]] = []
        self.index: int = 0
        self.paper_id: str = "unknown"
        self.title: str = "Untitled"

    # ---- Load / Build ----
    def load(self, raw: Dict[str, Any], *, paper_id: str = "unknown",
             title: str = "Untitled", default_role: str = "other",
             default_relation: str = "supports") -> Dict[str, Any]:
        nodes = raw.get("nodes", [])
        edges = raw.get("edges", [])
        has_role = any(isinstance(n, dict) and ("role" in n) for n in nodes)
        has_rel  = any(isinstance(e, dict) and ("relation" in e) for e in edges)

        if not has_role or not has_rel:
            pg = minimal_to_papergraph(
                raw,
                paper_id=paper_id,
                title=title,
                default_role=default_role,
                default_relation=default_relation,
            )
            pg_dict = pg.model_dump()
        else:
            pg_dict = {**raw}
            pg_dict.setdefault("paper_id", paper_id)
            pg_dict.setdefault("title", title)

        G, _idmap, root, order, removed = build_graph_and_bfs(pg_dict)
        self.G = G
        self.root = root
        self.order = order
        self.removed = removed
        self.index = 0
        self.paper_id = pg_dict.get("paper_id", paper_id)
        self.title = pg_dict.get("title", title)
        return self.state()

    # ---- Accessors ----
    def current_id(self) -> str:
        if not self.order:
            return ""
        return self.order[max(0, min(self.index, len(self.order) - 1))]

    def successors(self, nid: str) -> List[str]:
        if self.G is None or nid not in self.G.nodes:
            return []
        return sorted(self.G.successors(nid), key=str)

    def node_json(self, nid: str) -> Dict[str, Any]:
        if self.G is None or nid not in self.G.nodes:
            return {}
        d = self.G.nodes[nid]
        return {
            "id": str(nid),
            "text": d.get("text", ""),
            "role": d.get("role", "other"),
            "level": int(d.get("level", 0)),
            "score": d.get("score"),
        }

    def graph(self) -> Dict[str, Any]:
        if self.G is None:
            return {"paper_id": self.paper_id, "title": self.title,
                    "root": self.root, "nodes": [], "edges": []}
        nodes = [self.node_json(nid) for nid in self.G.nodes]
        edges = [{"source": u, "target": v, "relation": self.G.edges[u, v].get("relation", "supports")}
                 for (u, v) in self.G.edges]
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "root": self.root,
            "nodes": nodes,
            "edges": edges,
        }

    def state(self) -> Dict[str, Any]:
        cid = self.current_id()
        visited = self.order[: self.index + 1] if self.order else []
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "root": self.root,
            "order": self.order,
            "index": self.index,
            "current": self.node_json(cid),
            "successors": self.successors(cid),
            "visited": visited,
            "removed_edges": [[u, v] for (u, v) in self.removed],
            "metric_keys": self.METRIC_KEYS,
        }

    # ---- Step / Score ----
    def step(self, op: str = "next", target: Optional[str] = None) -> Dict[str, Any]:
        if not self.order:
            return self.state()
        op = (op or "next").lower()
        if op == "next":
            self.index = min(self.index + 1, len(self.order) - 1)
        elif op == "prev":
            self.index = max(self.index - 1, 0)
        elif op == "goto" and target in self.order:
            self.index = self.order.index(target)
        return self.state()

    def set_score(self, node_id: str, score: Dict[str, Any]) -> Dict[str, Any]:
        if self.G is None or node_id not in self.G.nodes:
            return {"ok": False, "error": "unknown node"}
        s = {}
        for k in self.METRIC_KEYS:
            try:
                s[k] = max(0.0, min(1.0, float(score.get(k, 0) or 0.0)))
            except Exception:
                s[k] = 0.0
        s["composite"] = round(sum(s[k] for k in self.METRIC_KEYS) / len(self.METRIC_KEYS), 4)
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
        if path == "/health":
            self._send(200, {"ok": True}); return
        if path == "/graph":
            self._send(200, _CTRL.graph()); return
        if path == "/state":
            self._send(200, _CTRL.state()); return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        body = self._read_json()

        if path == "/load":
            # Accept either {"graph": {...}, "paper_id": "...", "title": "..."} or raw graph
            graph_obj = body.get("graph") if isinstance(body.get("graph"), dict) else body
            paper_id = body.get("paper_id", "unknown")
            title = body.get("title", "Untitled")
            default_role = body.get("default_role", "other")
            default_relation = body.get("default_relation", "supports")
            if not isinstance(graph_obj, dict) or not graph_obj.get("nodes"):
                self._send(400, {"ok": False, "error": "missing/invalid graph"}); return
            try:
                st = _CTRL.load(
                    graph_obj,
                    paper_id=paper_id,
                    title=title,
                    default_role=default_role,
                    default_relation=default_relation,
                )
                self._send(200, {"ok": True, "state": st})
            except Exception as e:
                self._send(400, {"ok": False, "error": f"load failed: {e!r}"})
            return

        if path == "/step":
            op = body.get("op", "next")
            target = body.get("target")
            st = _CTRL.step(op=op, target=target)
            self._send(200, st); return

        if path == "/score":
            nid = body.get("id")
            if not nid:
                self._send(400, {"ok": False, "error": "missing id"}); return
            res = _CTRL.set_score(str(nid), body)
            self._send(200, res); return

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
