# app/graph_build.py
"""
Graph builder utilities for PaperGraph → networkx.DiGraph,
DAG repair, stable BFS traversal, and PyVis HTML export.

Public API
----------
json_to_nx(pg: PaperGraph|dict) -> (nx.DiGraph, dict[str, KGNode])
ensure_dag(G: nx.DiGraph) -> list[tuple[str,str]]
pick_root(G: nx.DiGraph, id_to_node: dict[str, KGNode]|None=None) -> str
relevel_from_root(G: nx.DiGraph, root: str, id_to_node: dict[str, KGNode]|None=None) -> None
bfs_order(G: nx.DiGraph, root: str) -> list[str]
export_pyvis(G: nx.DiGraph, path: str|Path, root: str|None=None) -> str

Acceptance notes
----------------
• Cycles auto-repaired deterministically (see RELATION_PRIORITY below).
• BFS order is **stable across runs** (neighbors are sorted by (level, role priority, id)).
• PyVis HTML renders with role-based colors and hierarchical layout.

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Union, Any

from .schemas import PaperGraph, KGNode, KGEdge  # pydantic models
import networkx as nx

# Optional dependency; only needed for export_pyvis
try:
    from pyvis.network import Network
    _HAS_PYVIS = True
except Exception:
    _HAS_PYVIS = False

# ---- Styling / priorities ----------------------------------------------------

ROLE_COLORS: Dict[str, str] = {
    "hypothesis": "#1E90FF",
    "claim": "#00B894",
    "method": "#6C5CE7",
    "evidence": "#0984E3",
    "result": "#F39C12",
    "conclusion": "#E74C3C",
    "limitation": "#7F8C8D",
    "other": "#636E72",
}

# Lower number = more essential; higher = easier to drop when repairing cycles
RELATION_PRIORITY: Dict[str, int] = {
    "supports": 0,
    "based_on": 1,
    "leads_to": 2,
    "qualifies": 3,
    "contradicts": 4,   # easiest to drop during cycle repair
}

ROLE_PRIORITY: Dict[str, int] = {
    "hypothesis": 0,
    "method": 1,
    "evidence": 2,
    "claim": 3,
    "result": 4,
    "conclusion": 5,
    "limitation": 6,
    "other": 7,
}

# ---- Builders ----------------------------------------------------------------

def _coerce_pg(pg: Union[PaperGraph, dict]) -> PaperGraph:
    if isinstance(pg, PaperGraph):
        return pg
    if isinstance(pg, dict):
        return PaperGraph(**pg)
    raise TypeError("Expected PaperGraph or dict")

def json_to_nx(pg: Union[PaperGraph, dict]) -> Tuple[nx.DiGraph, Dict[str, KGNode]]:
    """
    Convert a PaperGraph into a networkx.DiGraph with node/edge attributes.

    Returns:
        G: nx.DiGraph whose nodes are node.id (str).
        id_to_node: dict mapping node.id -> KGNode (original objects).
    """
    pg = _coerce_pg(pg)
    G = nx.DiGraph()
    id_to_node: Dict[str, KGNode] = {n.id: n for n in pg.nodes}

    # Add nodes
    for n in pg.nodes:
        G.add_node(
            n.id,
            role=n.role,
            level=int(n.level),
            text=n.text,
            span_hint=n.span_hint or "",
            paper_id=pg.paper_id,
            title=pg.title,
        )

    # Add edges (skip self-loops and missing ids)
    for e in pg.edges:
        if e.source == e.target:
            continue
        if e.source not in id_to_node or e.target not in id_to_node:
            # silently skip invalid references
            continue
        rel = e.relation
        prio = RELATION_PRIORITY.get(rel, 99)
        G.add_edge(e.source, e.target, relation=rel, priority=prio)

    return G, id_to_node

def minimal_to_papergraph(
    data: Dict[str, Any],
    *,
    paper_id: str = "unknown",
    title: str = "Untitled",
    default_role: str = "other",
    default_relation: str = "supports"
) -> PaperGraph:
    nodes = []
    for n in data.get("nodes", []):
        nodes.append({
            "id": str(n["id"]),                 # cast ints to str
            "role": default_role,               # required by your schema
            "level": 0,                         # will be normalized later
            "text": n.get("text", ""),
        })

    edges = []
    for e in data.get("edges", []):
        edges.append({
            "source": str(e["source"]),
            "target": str(e["target"]),
            "relation": default_relation,       # required by your schema
        })

    pg_dict = {
        "paper_id": paper_id,
        "title": title,
        "nodes": nodes,
        "edges": edges,
        "meta": {}
    }
    return PaperGraph(**pg_dict)

# ---- DAG repair --------------------------------------------------------------

def _find_cycle_edges(G: nx.DiGraph) -> Optional[List[Tuple[str, str]]]:
    try:
        cyc = list(nx.find_cycle(G, orientation="original"))
        # nx returns tuples like (u, v, 'forward'); take u, v
        return [(u, v) for (u, v, _dir) in cyc]
    except nx.NetworkXNoCycle:
        return None

def _choose_edge_to_break(G: nx.DiGraph, cycle_edges: Iterable[Tuple[str, str]]) -> Tuple[str, str]:
    """
    Deterministic choice: remove the least-essential edge in the cycle.

    Tie-breaks by:
        (edge priority, sink_level, source_level, sink_roleprio, source_roleprio, source_id, target_id)
    """
    def key(edge: Tuple[str, str]):
        u, v = edge
        data = G.get_edge_data(u, v, default={})
        prio = int(data.get("priority", 99))
        ulev = int(G.nodes[u].get("level", 10**6))
        vlev = int(G.nodes[v].get("level", 10**6))
        urole = G.nodes[u].get("role", "other")
        vrole = G.nodes[v].get("role", "other")
        urole_p = ROLE_PRIORITY.get(urole, 99)
        vrole_p = ROLE_PRIORITY.get(vrole, 99)
        return (prio, vlev, ulev, vrole_p, urole_p, str(u), str(v))
    # choose the MAX by our "worst" ordering? We want the EASIEST to drop (highest prio).
    # Using min with tuple where first item is priority (lower is more essential),
    # but we want the least-essential => max over negative priority is messy.
    # Simpler: pick the edge with the largest key lexicographically? No:
    # We constructed key where LOWER priority is "better". To drop least-essential,
    # we want the LARGEST `prio`. So sort descending by prio; to do deterministic overall,
    # we invert priority sign and keep the rest as inverse? Simpler: custom tuple with (-prio).
    # Let's build a separate tuple with (-prio) so max(...) drops largest priority; hmm.
    # We'll just sort by (-prio, -vlev, -ulev, -vrole_p, -urole_p, u, v) and take first.
    def drop_key(edge: Tuple[str, str]):
        u, v = edge
        data = G.get_edge_data(u, v, default={})
        prio = int(data.get("priority", 99))
        ulev = int(G.nodes[u].get("level", 10**6))
        vlev = int(G.nodes[v].get("level", 10**6))
        urole = G.nodes[u].get("role", "other")
        vrole = G.nodes[v].get("role", "other")
        urole_p = ROLE_PRIORITY.get(urole, 99)
        vrole_p = ROLE_PRIORITY.get(vrole, 99)
        return (prio, vlev, ulev, vrole_p, urole_p, str(u), str(v))

    # We want to drop the edge with the HIGHEST (prio, vlev, ...) tuple.
    to_drop = sorted(cycle_edges, key=drop_key, reverse=True)[0]
    return to_drop

def ensure_dag(G: nx.DiGraph) -> List[Tuple[str, str]]:
    """
    Remove edges until G is a DAG. Returns the list of removed edges (u, v) in order.

    Deterministic: for each cycle, remove the least-essential edge based on RELATION_PRIORITY;
    ties broken by deeper sink, then by roles, then by ids.
    """
    removed: List[Tuple[str, str]] = []
    while True:
        cyc = _find_cycle_edges(G)
        if not cyc:
            break
        drop = _choose_edge_to_break(G, cyc)
        if G.has_edge(*drop):
            G.remove_edge(*drop)
            removed.append(drop)
        else:
            # Should not happen; break to avoid infinite loop
            break
    return removed

# ---- Root picking & leveling -------------------------------------------------

def pick_root(G: nx.DiGraph, id_to_node: Optional[Dict[str, KGNode]] = None) -> str:
    """
    Pick a single root deterministically.

    Priority:
      1) node with role='hypothesis' and in-degree 0
      2) node with role='hypothesis'
      3) any node with in-degree 0
      4) lexicographically smallest node id

    If multiple candidates in a tier, prefer the lowest 'level' attribute, then smallest id.
    """
    def sort_key(nid: str):
        level = int(G.nodes[nid].get("level", 10**6))
        return (level, str(nid))

    # 1) hypothesis & indeg 0
    hyp0 = [n for n in G.nodes if G.nodes[n].get("role") == "hypothesis" and G.in_degree(n) == 0]
    if hyp0:
        return sorted(hyp0, key=sort_key)[0]

    # 2) any hypothesis
    hyp = [n for n in G.nodes if G.nodes[n].get("role") == "hypothesis"]
    if hyp:
        return sorted(hyp, key=sort_key)[0]

    # 3) indeg 0
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if roots:
        return sorted(roots, key=sort_key)[0]

    # 4) fallback
    return sorted(G.nodes, key=lambda x: (int(G.nodes[x].get("level", 10**6)), str(x)))[0]

def relevel_from_root(
    G: nx.DiGraph,
    root: str,
    id_to_node: Optional[Dict[str, KGNode]] = None,
) -> None:
    """
    Set 'level' to shortest-path distance (BFS depth) from root.
    Disconnected nodes get level = max_reachable_level + 1 (small, finite).
    """
    levels = {root: 0}
    queue: List[str] = [root]
    seen = {root}
    while queue:
        u = queue.pop(0)
        for v in sorted(G.successors(u), key=lambda x: str(x)):
            if v not in seen:
                levels[v] = levels[u] + 1
                seen.add(v)
                queue.append(v)

    # Compact, contiguous levels: put unreachable nodes just one layer below the deepest reachable.
    max_reach = max(levels.values()) if levels else 0
    fallback_level = max_reach + 1

    for n in G.nodes:
        lvl = int(levels.get(n, fallback_level))
        G.nodes[n]["level"] = lvl
        if id_to_node and n in id_to_node:
            try:
                id_to_node[n].level = lvl
            except Exception:
                pass

# ---- Stable BFS --------------------------------------------------------------

def _neighbor_sort_key(G: nx.DiGraph, nid: str):
    lvl = int(G.nodes[nid].get("level", 10**6))
    role = G.nodes[nid].get("role", "other")
    role_p = ROLE_PRIORITY.get(role, 99)
    return (lvl, role_p, str(nid))

def bfs_order(G: nx.DiGraph, root: str) -> List[str]:
    """
    Deterministic BFS order:
      - Queue-based BFS
      - For each pop, push neighbors sorted by (level, role priority, id)
    """
    order: List[str] = []
    seen = set([root])
    q: List[str] = [root]
    while q:
        u = q.pop(0)
        order.append(u)
        nbrs = sorted(G.successors(u), key=lambda v: _neighbor_sort_key(G, v))
        for v in nbrs:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return order

# ---- PyVis export ------------------------------------------------------------

def export_pyvis(
    G: nx.DiGraph,
    path: Union[str, Path],
    root: Optional[str] = None,
    *,
    hierarchical: bool = True,
    physics: bool = True,
    show_buttons: bool = True,
) -> str:
    """
    Export an interactive HTML graph using pyvis/vis.js.
    - hierarchical: layered UD layout (root at top if provided)
    - physics: enable force-based physics (gravity-like)
    - show_buttons: show vis.js 'configure' panel (no pyvis.show_buttons())
    """
    if not _HAS_PYVIS:
        raise RuntimeError("pyvis is not installed. `pip install pyvis`")

    path = str(Path(path).absolute())
    net = Network(height="820px", width="100%", directed=True, notebook=False)

    # Good default physics (fallback to barnes_hut on older pyvis)
    try:
        net.force_atlas_2based(
            gravity=-35,
            central_gravity=0.015,
            spring_length=180,
            spring_strength=0.08,
            damping=0.45,
        )
    except TypeError:
        net.barnes_hut(gravity=-20000, central_gravity=0.2, spring_length=180, spring_strength=0.03)

    # Nodes
    for nid, data in G.nodes(data=True):
        role = data.get("role", "other")
        color = ROLE_COLORS.get(role, "#95A5A6")
        lvl = data.get("level", 0)
        text = data.get("text", "")
        label = _truncate(text, 36) or f"{role}:{nid}"
        title = f"<b>{role}</b> (level {lvl})<br><i>{_escape_html(text)}</i>"
        net.add_node(
            nid,
            label=label,
            title=title,
            color=color,
            shape="dot",
            size=_node_size(role),
            level=int(lvl),
        )

    # Edges
    for u, v, ed in G.edges(data=True):
        rel = ed.get("relation", "")
        dashes = (rel == "contradicts")
        net.add_edge(u, v, label=rel, arrows="to", smooth=False, physics=True, dashes=dashes)

    # Build a shared interaction block
    configure_block = f'''"configure": {{
        "enabled": {str(show_buttons).lower()},
        "filter": ["layout", "physics", "interaction"]
    }}'''
    interaction_block = f'''"interaction": {{
        "hover": true,
        "navigationButtons": true,
        "keyboard": {{
          "enabled": true,
          "speed": {{ "x": 10, "y": 10, "zoom": 0.02 }}
        }}
    }}'''

    if hierarchical and root is not None and root in G.nodes:
        net.set_options(f"""
        {{
          "layout": {{
            "hierarchical": {{
              "enabled": true,
              "direction": "UD",
              "sortMethod": "hubsize",
              "shakeTowards": "roots",
              "nodeSpacing": 180,
              "levelSeparation": 120
            }}
          }},
          "physics": {{
            "enabled": {str(physics).lower()},
            "solver": "hierarchicalRepulsion",
            "hierarchicalRepulsion": {{
              "centralGravity": 0.0,
              "springLength": 140,
              "springConstant": 0.01,
              "nodeDistance": 180,
              "damping": 0.35
            }},
            "stabilization": {{ "enabled": true, "iterations": 200 }},
            "minVelocity": 0.1
          }},
          {interaction_block},
          {configure_block}
        }}
        """)
    else:
        net.set_options(f"""
        {{
          "layout": {{ "improvedLayout": true }},
          "physics": {{
            "enabled": {str(physics).lower()},
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {{
              "gravitationalConstant": -35,
              "centralGravity": 0.015,
              "springLength": 180,
              "springConstant": 0.08,
              "damping": 0.45,
              "avoidOverlap": 0.6
            }},
            "stabilization": {{ "enabled": true, "iterations": 250 }},
            "minVelocity": 0.1
          }},
          {interaction_block},
          {configure_block}
        }}
        """)

    net.write_html(path, open_browser=False, notebook=False)
    return path

def _node_size(role: str) -> int:
    # slightly larger tokens for top-level narrative nodes
    base = {
        "hypothesis": 28,
        "conclusion": 26,
        "claim": 24,
        "result": 22,
        "method": 20,
        "evidence": 20,
        "limitation": 18,
        "other": 18,
    }
    return base.get(role, 18)

def _truncate(s: str, n: int) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: max(0, n - 1)].rstrip() + "…"

def _escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )

def export_graphml(G: nx.DiGraph, path: Union[str, Path], root: Optional[str] = None) -> str:
    """
    Export the graph to GraphML (.graphml), an XML format.
    If root is provided, annotates nodes with a 'bfs_order' index.
    """
    P = str(Path(path).absolute())
    H = G.copy()
    if root is not None and root in H.nodes:
        order = bfs_order(H, root)
        nx.set_node_attributes(H, {n: i for i, n in enumerate(order)}, "bfs_order")
    # Ensure attributes are GraphML-friendly (no None)
    for n, d in H.nodes(data=True):
        for k, v in list(d.items()):
            if v is None:
                d[k] = ""
    for u, v, d in H.edges(data=True):
        for k, val in list(d.items()):
            if val is None:
                d[k] = ""
    nx.write_graphml(H, P)
    return P

def export_gexf(G: nx.DiGraph, path: Union[str, Path], root: Optional[str] = None) -> str:
    """
    Export the graph to GEXF (.gexf), an XML format.
    If root is provided, annotates nodes with a 'bfs_order' index.
    """
    P = str(Path(path).absolute())
    H = G.copy()
    if root is not None and root in H.nodes:
        order = bfs_order(H, root)
        nx.set_node_attributes(H, {n: i for i, n in enumerate(order)}, "bfs_order")
    for n, d in H.nodes(data=True):
        for k, v in list(d.items()):
            if v is None:
                d[k] = ""
    for u, v, d in H.edges(data=True):
        for k, val in list(d.items()):
            if val is None:
                d[k] = ""
    nx.write_gexf(H, P)
    return P

def export_xml(G: nx.DiGraph, path: Union[str, Path], root: Optional[str] = None) -> str:
    """
    Convenience: chooses GraphML or GEXF based on file extension.
    Use '.graphml' for GraphML or '.gexf' for GEXF.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".gexf":
        return export_gexf(G, path, root=root)
    # default to GraphML
    return export_graphml(G, path, root=root)

def write_pc_payload(pg_or_dict, out_json: str | Path) -> str:
    """
    Build graph, repair DAG, pick root, relevel, BFS, then write a frontend-friendly payload:
      {
        "paper_id": "...",
        "title": "...",
        "root": "<root-id>",
        "bfs_order": ["<id0>", "<id1>", ...],
        "nodes": [{"id": "...", "text": "...", "role": "...", "level": 0}, ...],
        "edges": [{"source": "...", "target": "...", "relation": "supports"}, ...]
      }
    """
    import json
    from pathlib import Path

    # Accept dict or PaperGraph
    if isinstance(pg_or_dict, dict):
        # If it's the minimal schema (no roles/relations), adapt it:
        nodes = pg_or_dict.get("nodes", [])
        edges = pg_or_dict.get("edges", [])
        has_role = any(isinstance(n, dict) and ("role" in n) for n in nodes)
        has_rel = any(isinstance(e, dict) and ("relation" in e) for e in edges)
        if not has_role or not has_rel:
            pg = minimal_to_papergraph(pg_or_dict)
        else:
            pg = pg_or_dict
    else:
        pg = pg_or_dict

    G, id_map, root, order, removed = build_graph_and_bfs(pg)  # BFS, root, etc. already defined

    payload = {
        "paper_id": getattr(pg, "paper_id", getattr(pg, "get", lambda k, d=None: d)("paper_id", "unknown")),
        "title": getattr(pg, "title", getattr(pg, "get", lambda k, d=None: d)("title", "Untitled")),
        "root": root,
        "bfs_order": order,
        "nodes": [
            {
                "id": nid,
                "text": G.nodes[nid].get("text", ""),
                "role": G.nodes[nid].get("role", "other"),
                "level": int(G.nodes[nid].get("level", 0)),
            }
            for nid in G.nodes
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                "relation": G.edges[u, v].get("relation", "supports")
            }
            for (u, v) in G.edges
        ],
        "removed_edges_for_dag": removed,
    }

    out_json = str(Path(out_json))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_json

# ---- Convenience: end-to-end builder ----------------------------------------

def build_graph_and_bfs(
    pg: Union[PaperGraph, dict],
) -> Tuple[nx.DiGraph, Dict[str, KGNode], str, List[str], List[Tuple[str, str]]]:
    """
    One-call convenience for UI/Integrator:

    Steps:
      1) JSON → G
      2) ensure_dag
      3) pick_root
      4) relevel_from_root
      5) stable bfs_order

    Returns:
      G, id_to_node, root, bfs_ids, removed_edges
    """
    G, id_to_node = json_to_nx(pg)
    removed = ensure_dag(G)
    root = pick_root(G, id_to_node)
    relevel_from_root(G, root, id_to_node)
    order = bfs_order(G, root)
    return G, id_to_node, root, order, removed

# ---- Example usage (manual smoke test) --------------------------------------

if __name__ == "__main__":
    """
    CLI driver that accepts either:
      • Minimal JSON: {"nodes":[{"id":0,"text":"..."},...], "edges":[{"source":0,"target":1},...]}
      • Full PaperGraph-shaped JSON (paper_id/title/nodes/edges with roles/relations)

    Usage examples:
      python -m app.graph_build minimal.json --out-dir demo_out --paper-id platos-cave-001 --title "Plato's Cave"
      python -m app.graph_build papergraph.json --out-dir demo_out --out-xml graph.gexf
      cat minimal.json | python -m app.graph_build - --no-html

    Notes:
      • XML export format is chosen by extension via export_xml: .graphml (default) or .gexf
      • HTML export requires `pyvis`; pass --no-html to skip if not installed.
    """
    import argparse, json, sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Build graph, repair DAG, relevel, BFS, and export.")
    parser.add_argument("input", help="Path to JSON file, or '-' to read JSON from stdin.")
    parser.add_argument("--out-dir", default="demo_out", help="Output directory (will be created).")
    parser.add_argument("--out-xml", default="graph.graphml",
                        help="XML filename (.graphml or .gexf) inside out-dir.")
    parser.add_argument("--out-html", default="graph.html",
                        help="PyVis HTML filename inside out-dir (requires pyvis).")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML export.")
    # Defaults used only when we detect the minimal schema:
    parser.add_argument("--paper-id", default="unknown", help="Paper id (for minimal schema).")
    parser.add_argument("--title", default="Untitled", help="Title (for minimal schema).")
    parser.add_argument("--default-role", default="other",
                        help="Role to assign to minimal nodes (schema requires a role).")
    parser.add_argument("--default-relation", default="supports",
                        choices=["supports","based_on","leads_to","contradicts","qualifies"],
                        help="Relation to assign to minimal edges (schema requires a relation).")
    args = parser.parse_args()

    # ---- Load input JSON
    try:
        raw_text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
        raw = json.loads(raw_text)
    except Exception as e:
        print(f"[error] Failed to read/parse JSON: {e}", file=sys.stderr)
        sys.exit(2)

    # ---- Detect schema (minimal vs full)
    def _is_minimal_schema(d: dict) -> bool:
        if not isinstance(d, dict):
            return False
        nodes = d.get("nodes")
        edges = d.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            return False
        # If nodes lack 'role' or edges lack 'relation', treat as minimal.
        has_any_role = any(isinstance(n, dict) and ("role" in n) for n in nodes)
        has_any_relation = any(isinstance(e, dict) and ("relation" in e) for e in edges)
        return (not has_any_role) or (not has_any_relation)

    if _is_minimal_schema(raw):
        # Adapt minimal → PaperGraph using helper already in this module
        pg = minimal_to_papergraph(
            raw,
            paper_id=args.paper_id,
            title=args.title,
            default_role=args.default_role,
            default_relation=args.default_relation,
        )
    else:
        # Assume it's already PaperGraph-shaped; your json_to_nx will validate via pydantic
        pg = raw

    # ---- Pipeline: build, repair, root, relevel, BFS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    G, id_map, root, order, removed = build_graph_and_bfs(pg)

    # ---- Exports
    xml_path = out_dir / args.out_xml
    export_xml(G, xml_path, root=root)

    if not args.no_html:
        try:
            html_path = out_dir / args.out_html
            export_pyvis(G, html_path, root=root, hierarchical=True, physics=True)
        except RuntimeError:
            print("[warn] pyvis not installed; skipping HTML export.", file=sys.stderr)

    # ---- Console summary
    print(f"Root: {root}")
    print(f"Nodes: {G.number_of_nodes()} | Edges (after repair): {G.number_of_edges()}")
    if removed:
        print("Removed edges for DAG repair:", removed)
    print("BFS order:", order)
    print(f"Wrote XML to: {xml_path}")
    if not args.no_html:
        print(f"Wrote HTML to: {out_dir / args.out_html}")