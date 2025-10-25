"""
Frontend bridge: minimal, importable controller + optional HTTP endpoints.

Provides:
  - init_state(input_json_or_path, **kwargs) -> dict    # build graph & BFS order
  - get_state() -> dict                                 # current step payload
  - update_state(op='next'|'prev'|'goto', target=None)  # advance/seek BFS
  - (optional) `--serve` flag to run a tiny HTTP JSON API

Payload shape returned to frontend (example):
{
  "paper_id": "...",
  "title": "...",
  "root": "0",
  "order": ["0","1","2",...],           # full BFS order
  "index": 3,                            # current position in order
  "current": { "id":"...", "text":"...", "role":"...", "level":0 },
  "successors": ["...","..."],           # direct children to highlight
  "visited": ["0","1","2","3"],          # ids up to and including index
  "removed_edges": [["u","v"], ...]      # edges dropped by DAG repair
}
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json

from .graph_build import (
    minimal_to_papergraph,
    build_graph_and_bfs,
)  # uses your existing pipeline  :contentReference[oaicite:3]{index=3}

# ---------------------- Controller ----------------------

class BFSController:
    def __init__(self):
        self.G = None
        self.id_map = None
        self.root = None
        self.order: List[str] = []
        self.removed: List[Tuple[str, str]] = []
        self.index: int = 0
        self.paper_id: str = "unknown"
        self.title: str = "Untitled"

    def load(self, pg: Dict[str, Any]):
        # Build graph, enforce DAG, root, levels, BFS
        G, id_map, root, order, removed = build_graph_and_bfs(pg)  # :contentReference[oaicite:4]{index=4}
        self.G, self.id_map, self.root, self.order, self.removed = G, id_map, root, order, removed
        self.paper_id = getattr(pg, "paper_id", None) or pg.get("paper_id", "unknown")
        self.title = getattr(pg, "title", None) or pg.get("title", "Untitled")
        self.index = 0

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
        }

    def payload(self) -> Dict[str, Any]:
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
        }

    def update(self, op: str = "next", target: Optional[str] = None) -> Dict[str, Any]:
        if not self.order:
            return self.payload()
        if op == "next":
            self.index = min(self.index + 1, len(self.order) - 1)
        elif op == "prev":
            self.index = max(self.index - 1, 0)
        elif op == "goto":
            if target in self.order:
                self.index = self.order.index(target)
        # else ignore unknown ops
        return self.payload()


# Singleton-style controller for simple imports
_CTRL = BFSController()

# ---------------------- Public API ----------------------

def _load_input(input_json_or_path: Union[str, Dict[str, Any]],
                *,
                paper_id: str = "unknown",
                title: str = "Untitled",
                default_role: str = "other",
                default_relation: str = "supports") -> Dict[str, Any]:
    """
    Detect minimal schema (id/text + source/target) vs full PaperGraph and coerce.
    """
    if isinstance(input_json_or_path, str) and Path(input_json_or_path).exists():
        raw = json.loads(Path(input_json_or_path).read_text(encoding="utf-8"))
    elif isinstance(input_json_or_path, str):
        # Treat as already-JSON string
        raw = json.loads(input_json_or_path)
    else:
        raw = dict(input_json_or_path)

    # Minimal schema detection: nodes missing 'role' or edges missing 'relation'
    nodes = raw.get("nodes", [])
    edges = raw.get("edges", [])
    has_any_role = any(isinstance(n, dict) and ("role" in n) for n in nodes)
    has_any_relation = any(isinstance(e, dict) and ("relation" in e) for e in edges)
    if (not has_any_role) or (not has_any_relation):
        pg = minimal_to_papergraph(
            raw,
            paper_id=paper_id,
            title=title,
            default_role=default_role,
            default_relation=default_relation,
        )  # returns PaperGraph (pydantic)  :contentReference[oaicite:5]{index=5}
        return pg.model_dump()  # dict for downstream
    return raw  # assumed full PaperGraph-shaped dict


def init_state(input_json_or_path: Union[str, Dict[str, Any]],
               *,
               paper_id: str = "unknown",
               title: str = "Untitled",
               default_role: str = "other",
               default_relation: str = "supports") -> Dict[str, Any]:
    """
    Initialize controller from minimal JSON or full PaperGraph-shaped JSON.
    Returns the initial payload (index=0).
    """
    pg = _load_input(input_json_or_path,
                     paper_id=paper_id,
                     title=title,
                     default_role=default_role,
                     default_relation=default_relation)
    _CTRL.load(pg)
    return _CTRL.payload()


def get_state() -> Dict[str, Any]:
    """Return current state payload (without changing index)."""
    return _CTRL.payload()


def update_state(op: str = "next", target: Optional[str] = None) -> Dict[str, Any]:
    """
    Advance or seek in BFS:
        op='next' | 'prev' | 'goto' (use target=<node_id> for goto)
    Returns the updated state payload.
    """
    return _CTRL.update(op=op, target=target)

# ---------------------- CLI / Demo ----------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Frontend bridge demo for BFS state stepping.")
    parser.add_argument("input", help="Path to JSON (minimal or full). Use '-' to read from stdin.")
    parser.add_argument("--paper-id", default="demo")
    parser.add_argument("--title", default="Demo")
    parser.add_argument("--serve", action="store_true",
                        help="Run a tiny HTTP API at http://127.0.0.1:8765")
    args = parser.parse_args()

    # Load
    raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
    state = init_state(raw, paper_id=args.paper_id, title=args.title)
    print("[init]", json.dumps(state, indent=2))

    if not args.serve:
        # Step a couple times for demo
        print("[next]", json.dumps(update_state("next"), indent=2))
        print("[goto id=0]", json.dumps(update_state("goto", target="0"), indent=2))
        sys.exit(0)

    # Optional ultra-light HTTP JSON API using stdlib (no dependencies)
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import urllib.parse

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: Dict[str, Any]):
            b = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/state":
                self._send(200, get_state()); return
            elif parsed.path == "/update_state":
                qs = urllib.parse.parse_qs(parsed.query)
                op = (qs.get("op", ["next"])[0] or "next").lower()
                target = qs.get("target", [None])[0]
                self._send(200, update_state(op, target)); return
            self._send(404, {"error": "not found"})

        def log_message(self, *a, **k):
            return  # quiet

    srv = HTTPServer(("127.0.0.1", 8765), Handler)
    print("Serving JSON API at http://127.0.0.1:8765  (GET /state, GET /update_state?op=next|prev|goto&target=id)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
