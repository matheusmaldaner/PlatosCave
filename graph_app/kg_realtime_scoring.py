# kg_realtime_scoring.py
"""
Real-time DAG edge scoring and graph scoring for knowledge graphs.

Nodes have:
  - id: str
  - role: str   (one of: Hypothesis, Conclusion, Claim, Evidence, Method, Result,
                 Assumption, Counterevidence, Limitation, Context/Contex)
  - text: str
  - parents: list[str]
  - agent metrics (when available): credibility, relevance, evidence_strength,
                                     method_rigor, reproducibility, citation_support  ∈ [0,1]

Edges (u -> v) are scored ("confidence") when:
  - v has all six agent metrics, AND
  - every parent of v has all six agent metrics.

Edge confidence uses:
  - Role transition prior (from_role -> to_role)
  - Parent node quality (weighted mix of the 6 metrics; role-specific weights)
  - Child node quality (weighted mix; role-specific weights)
  - Text alignment (Jaccard overlap of content words)
  - Pairwise synergy (role-aware combination of specific metrics from parent+child)

Graph score (optional) is also configurable.

Author: you :)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Callable, Iterable, List, Optional, Set, Any
import re
import math
import networkx as nx
import numpy as np

# --------- Roles & helpers ---------

_CANON_ROLE = {
    "hypothesis": "Hypothesis", "hypothesisroot": "Hypothesis", "h": "Hypothesis",
    "conclusion": "Conclusion", "conclusionroot": "Conclusion", "c": "Conclusion",
    "claim": "Claim", "evidence": "Evidence", "method": "Method", "result": "Result",
    "assumption": "Assumption", "counterevidence": "Counterevidence",
    "limitation": "Limitation", "context": "Context", "contex": "Context",
}

# Make role list independent of class definition order
_CANON_ROLES = ["Assumption","Claim","Conclusion","Context","Counterevidence",
                "Evidence","Hypothesis","Limitation","Method","Result"]

_ROLE_LIST = _CANON_ROLES

_ROLE_TO_IDX = {r: i for i, r in enumerate(_ROLE_LIST)}

METRIC_KEYS = (
    "credibility",
    "relevance",
    "evidence_strength",
    "method_rigor",
    "reproducibility",
    "citation_support",
)

def _role_one_hot(role: str):
    vec = [0.0] * len(_ROLE_LIST)
    i = _ROLE_TO_IDX.get(role)
    if i is not None:
        vec[i] = 1.0
    return vec

def canon_role(role: str) -> str:
    r = _CANON_ROLE.get(role.strip().lower())
    if not r:
        raise ValueError(f"Unknown role '{role}'. Allowed: {sorted(set(_CANON_ROLE.values()))}")
    return r

_STOP = {
    "a","an","the","and","or","but","if","then","thus","therefore","so","of","to",
    "in","on","for","with","by","as","at","from","is","are","was","were","be","been",
    "this","that","these","those","it","its","we","our","you","your","they","their"
}

_TOK = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str) -> Set[str]:
    return {t.lower() for t in _TOK.findall(text) if t.lower() not in _STOP}

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b: return 0.0
    u = len(a | b); i = len(a & b)
    return i / u if u else 0.0

def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

# --------- Config dataclasses ---------
# --------- Config dataclasses ---------
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Set, Any, Optional, Callable

@dataclass
class GlobalMetricWeights:
    """
    Global per-metric weights applied to raw agent metrics before any mixing.
    Keys must be from:
      {'credibility','relevance','evidence_strength',
       'method_rigor','reproducibility','citation_support'}
    """
    weights: Dict[str, float] = field(default_factory=lambda: {
        "credibility":        1.0,
        "relevance":          1.0,
        "evidence_strength":  1.0,
        "method_rigor":       1.0,
        "reproducibility":    1.0,
        "citation_support":   1.0,
    })

@dataclass
class EdgeCombineWeights:
    """How to combine edge features into a single confidence score."""
    role_prior: float = 0.30
    parent_quality: float = 0.20
    child_quality: float = 0.20
    alignment: float = 0.10
    synergy: float = 0.20
    # Must sum to <= 1.0 (extra headroom is fine if you expect penalties later)

@dataclass
class NodeQualityWeights:
    """Role-specific weights over the six agent metrics."""
    # keys are role names; values are dict(metric->weight). They will be normalized.
    per_role: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Hypothesis":       {"credibility": 0.5, "relevance": 0.5},
        "Conclusion":       {"credibility": 0.6, "relevance": 0.4},
        "Claim":            {"credibility": 0.45, "relevance": 0.35, "evidence_strength": 0.20},
        "Evidence":         {"evidence_strength": 0.5, "citation_support": 0.3,
                             "credibility": 0.2},
        "Method":           {"method_rigor": 0.6, "reproducibility": 0.3, "credibility": 0.1},
        "Result":           {"credibility": 0.4, "relevance": 0.3, "evidence_strength": 0.3},
        "Assumption":       {"credibility": 0.6, "relevance": 0.4},
        "Counterevidence":  {"evidence_strength": 0.5, "citation_support": 0.3,
                             "credibility": 0.2},
        "Limitation":       {"credibility": 0.5, "relevance": 0.5},
        "Context":          {"credibility": 0.4, "relevance": 0.6},
    })

@dataclass
class PropagationPenalty:
    enabled: bool = True
    agg: str = "min"      # "min" | "mean" | "softmin" | "dampmin"
    alpha: float = 1.0
    eta: float = 2**(-1/8)
    default_raw_conf: float = 0.5

    # NEW (reasonable defaults)
    softmin_beta: float = 6.0     # higher => closer to min
    dampmin_lambda: float = 0.35  # higher => closer to mean

@dataclass
class RoleTransitionPrior:
    """Coarse prior on how sensible an edge (role_u -> role_v) is, in [0,1]."""
    table: Dict[Tuple[str, str], float] = field(default_factory=lambda: {
        ("Hypothesis","Claim"): 0.75, ("Hypothesis","Evidence"): 0.75,
        ("Hypothesis","Method"): 0.50, ("Hypothesis","Result"): 0.25,
        ("Hypothesis","Conclusion"): 0.25,
        ("Evidence","Result"): 1.00, ("Evidence","Claim"): 0.50,
        ("Evidence","Conclusion"): 0.75,
        ("Method","Result"): 0.75, ("Method","Evidence"): 0.50,
        ("Result","Conclusion"): 0.75, ("Claim","Conclusion"): 0.50,
        ("Claim","Evidence"): 0.50, ("Context","Claim"): 0.50,
        ("Assumption","Claim"): 0.50, ("Assumption","Method"): 0.50,
        ("Counterevidence","Claim"): 0.75, ("Counterevidence","Conclusion"): 0.75,
        # Default if missing: 0.5 (weakly sensible)
    })
    default_value: float = 0.5

    def get(self, r_from: str, r_to: str) -> float:
        return self.table.get((r_from, r_to), self.default_value)

@dataclass
class PairSynergyWeights:
    """
    Role-pair-specific metric mixing for synergy ∈ [0,1].
    Each mapping provides weights over PARENT and CHILD metrics; normalized internally.
    """
    per_pair: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = field(
        default_factory=lambda: {
            # Evidence -> Claim/Result/Conclusion: lean on evidence + child credibility
            ("Evidence","Claim"): {
                "parent": {"evidence_strength": 0.5, "citation_support": 0.3, "credibility": 0.2},
                "child":  {"credibility": 0.6, "relevance": 0.4}
            },
            ("Evidence","Result"): {
                "parent": {"evidence_strength": 0.5, "citation_support": 0.3, "credibility": 0.2},
                "child":  {"credibility": 0.5, "relevance": 0.5}
            },
            ("Evidence","Conclusion"): {
                "parent": {"evidence_strength": 0.5, "citation_support": 0.4, "credibility": 0.1},
                "child":  {"credibility": 0.7, "relevance": 0.3}
            },
            # Method -> Result: lean on rigor & reproducibility + child credibility
            ("Method","Result"): {
                "parent": {"method_rigor": 0.6, "reproducibility": 0.3, "credibility": 0.1},
                "child":  {"credibility": 0.6, "relevance": 0.4}
            },
            # Hypothesis -> anything: the child's credibility + relevance matter most
            ("Hypothesis","Claim"): {
                "parent": {"credibility": 0.3, "relevance": 0.7},
                "child":  {"credibility": 0.6, "relevance": 0.4}
            },
            ("Hypothesis","Evidence"): {
                "parent": {"credibility": 0.4, "relevance": 0.6},
                "child":  {"credibility": 0.5, "relevance": 0.5}
            },
            ("Claim","Conclusion"): {
                "parent": {"credibility": 0.6, "relevance": 0.4},
                "child":  {"credibility": 0.7, "relevance": 0.3}
            },
        }
    )
    # Fallback if a pair isn't listed: 50/50 parent/child using all available metrics.

@dataclass
class GraphScoreWeights:
    """Top-level weights for graph score, all in [0,1], sum doesn't need to be 1."""
    bridge_coverage: float = 0.25
    best_path: float = 0.25
    redundancy: float = 0.15
    fragility: float = -0.15     # negative weight (penalty)
    coherence: float = 0.10
    coverage: float = 0.10

@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    edge_mismatches: List[Tuple[str, str, str]] = field(default_factory=list)  # (src, dst, reason)
    reconciled_edges: Set[Tuple[str, str]] = field(default_factory=set)

    def ok(self) -> bool:
        return len(self.errors) == 0

# --- Helper to canonicalize/guard JSON fields ---
def _as_str_id(x: Any) -> str:
    if isinstance(x, (int, float)):  # keep ints like 0,1 as "0","1"
        if isinstance(x, float) and not x.is_integer():
            raise ValueError(f"Non-integer numeric id '{x}' not allowed.")
        return str(int(x))
    if isinstance(x, str):
        sx = x.strip()
        if not sx:
            raise ValueError("Blank id not allowed.")
        return sx
    raise ValueError(f"Unsupported id type: {type(x)}")

def _list_of_ids(seq: Any) -> List[str]:
    if seq is None: return []
    if not isinstance(seq, (list, tuple)): 
        raise ValueError("parents/children must be a list")
    return [_as_str_id(z) for z in seq]

def _role_ok(role: str) -> bool:
    try:
        canon_role(role); return True
    except Exception:
        return False

# --- Public API: validate JSON & build KGScorer ---
class DAGValidation:
    @staticmethod
    def validate_and_build_from_json(
        data: Dict[str, Any],
        *,
        reconcile: str = "prefer_parents",  # "strict"|"prefer_parents"|"prefer_children"|"union"|"intersection"
        strict_roles: bool = True,          # if True, unknown roles => error
        expect_roots: bool = True,          # if True, warn if missing Hypothesis/Conclusion roots
        forbid_self_loops: bool = True
    ) -> Tuple["KGScorer", ValidationReport]:
        """
        Validates a JSON knowledge graph and returns a KGScorer plus a ValidationReport.
        JSON schema:
        {
          "nodes": [
            {"id": Num|Str, "text": str, "role": str, "parents": [ids], "children": [ids]},
            ...
          ]
        }
        """
        rep = ValidationReport()
        if not isinstance(data, dict) or "nodes" not in data or not isinstance(data["nodes"], list):
            raise ValueError("Input must be a dict with key 'nodes' as a list.")

        # 1) Canonicalize nodes
        raw_nodes = data["nodes"]
        seen: Set[str] = set()
        nodes_norm: Dict[str, Dict[str, Any]] = {}
        for i, n in enumerate(raw_nodes):
            if not isinstance(n, dict):
                rep.errors.append(f"nodes[{i}] is not an object")
                continue
            try:
                nid = _as_str_id(n.get("id", None))
            except Exception as e:
                rep.errors.append(f"nodes[{i}] id error: {e}")
                continue

            if nid in seen:
                rep.errors.append(f"Duplicate node id '{nid}'")
                continue
            seen.add(nid)

            text = n.get("text", "")
            role = n.get("role", "")
            if not isinstance(text, str):
                rep.errors.append(f"Node {nid}: text must be string")
                text = str(text)
            if not isinstance(role, str):
                rep.errors.append(f"Node {nid}: role must be string")
                role = str(role)

            if strict_roles and not _role_ok(role):
                rep.errors.append(f"Node {nid}: unknown role '{role}'")
            role = canon_role(role) if _role_ok(role) else role  # normalize if possible

            try:
                parents = _list_of_ids(n.get("parents", []))
            except Exception as e:
                rep.errors.append(f"Node {nid}: parents list error: {e}")
                parents = []
            try:
                children = _list_of_ids(n.get("children", []))
            except Exception as e:
                rep.errors.append(f"Node {nid}: children list error: {e}")
                children = []

            nodes_norm[nid] = {"id": nid, "text": text, "role": role,
                               "parents": parents, "children": children}

        # Early abort if fundamental issues
        if rep.errors:
            # Build empty scorer so caller can still inspect report
            return KGScorer(), rep

        # 2) Detect dangling refs and collect edges from both views
        all_ids = set(nodes_norm.keys())
        edges_from_parents: Set[Tuple[str, str]] = set()
        edges_from_children: Set[Tuple[str, str]] = set()
        for nid, nd in nodes_norm.items():
            # parents ⇒ parent -> nid
            for p in nd["parents"]:
                if p not in all_ids:
                    rep.errors.append(f"Node {nid}: parent '{p}' does not exist")
                else:
                    edges_from_parents.add((p, nid))
            # children ⇒ nid -> child
            for c in nd["children"]:
                if c not in all_ids:
                    rep.errors.append(f"Node {nid}: child '{c}' does not exist")
                else:
                    edges_from_children.add((nid, c))

        if rep.errors:
            return KGScorer(), rep

        # 3) Self-loops check
        if forbid_self_loops:
            for (u, v) in list(edges_from_parents | edges_from_children):
                if u == v:
                    rep.errors.append(f"Self-loop edge '{u} -> {v}' not allowed")
        if rep.errors:
            return KGScorer(), rep

        # 4) Reconcile edge sets
        if reconcile not in {"strict", "prefer_parents", "prefer_children", "union", "intersection"}:
            raise ValueError("Invalid reconcile mode.")
        if reconcile == "strict":
            # Both must match exactly
            if edges_from_parents != edges_from_children:
                # Log mismatches
                only_pp = edges_from_parents - edges_from_children
                only_cc = edges_from_children - edges_from_parents
                for e in sorted(only_pp):
                    rep.edge_mismatches.append((e[0], e[1], "present_in_parents_only"))
                for e in sorted(only_cc):
                    rep.edge_mismatches.append((e[0], e[1], "present_in_children_only"))
                rep.errors.append("Edges from parents/children do not match in strict mode.")
                return KGScorer(), rep
            reconciled = set(edges_from_parents)
        elif reconcile == "prefer_parents":
            reconciled = set(edges_from_parents)
            for e in sorted(edges_from_children - edges_from_parents):
                rep.warnings.append(f"Child-only edge {e[0]}->{e[1]} ignored (prefer_parents).")
                rep.edge_mismatches.append((e[0], e[1], "ignored_child_only"))
        elif reconcile == "prefer_children":
            reconciled = set(edges_from_children)
            for e in sorted(edges_from_parents - edges_from_children):
                rep.warnings.append(f"Parent-only edge {e[0]}->{e[1]} ignored (prefer_children).")
                rep.edge_mismatches.append((e[0], e[1], "ignored_parent_only"))
        elif reconcile == "union":
            reconciled = set(edges_from_parents | edges_from_children)
            for e in sorted(edges_from_parents ^ edges_from_children):
                rep.warnings.append(f"Edge {e[0]}->{e[1]} added by union reconciliation.")
                rep.edge_mismatches.append((e[0], e[1], "union_added"))
        else:  # intersection
            reconciled = set(edges_from_parents & edges_from_children)
            dropped = (edges_from_parents | edges_from_children) - reconciled
            for e in sorted(dropped):
                rep.warnings.append(f"Edge {e[0]}->{e[1]} dropped by intersection reconciliation.")
                rep.edge_mismatches.append((e[0], e[1], "intersection_dropped"))

        # 5) Build a temporary graph to check DAG & cycles
        Gtmp = nx.DiGraph()
        Gtmp.add_nodes_from(all_ids)
        Gtmp.add_edges_from(reconciled)
        if not nx.is_directed_acyclic_graph(Gtmp):
            try:
                cyc = nx.find_cycle(Gtmp, orientation="original")
                cyc_path = " -> ".join([a for (a, b, _) in cyc] + [cyc[0][0]])
                rep.errors.append(f"Cycle detected: {cyc_path}")
            except Exception:
                rep.errors.append("Graph is not a DAG (cycle present).")
            return KGScorer(), rep

        # 6) Optional sanity: roots and leaves by role
        if expect_roots:
            hyp_roots = [n for n in all_ids if canon_role(nodes_norm[n]["role"]) == "Hypothesis" and Gtmp.in_degree(n) == 0]
            con_leaves = [n for n in all_ids if canon_role(nodes_norm[n]["role"]) == "Conclusion" and Gtmp.out_degree(n) == 0]
            if not hyp_roots:
                rep.warnings.append("No Hypothesis root with in-degree 0 found.")
            if not con_leaves:
                rep.warnings.append("No Conclusion leaf with out-degree 0 found.")
            if len(hyp_roots) > 1:
                rep.warnings.append(f"Multiple Hypothesis roots found: {hyp_roots}")
            if len(con_leaves) > 1:
                rep.warnings.append(f"Multiple Conclusion leaves found: {con_leaves}")

        rep.stats = {
            "num_nodes": len(all_ids),
            "num_edges": len(reconciled),
            "num_roles": len(set(canon_role(nodes_norm[n]["role"]) for n in all_ids)),
        }
        rep.reconciled_edges = reconciled

        # 7) Build the KGScorer with reconciled edges; reconcile parents lists accordingly
        kg = KGScorer()
        # add nodes (parents will be set after edges are added)
        for nid, nd in nodes_norm.items():
            kg.add_node(Node(
                id=nid, role=nd["role"], text=nd["text"], parents=[]
            ))
        # add edges (authoritative set)
        for (u, v) in reconciled:
            kg.add_edge(u, v)

        # Optionally: warn if any Hypothesis has parents or any Conclusion has children
        for nid, nd in nodes_norm.items():
            role = canon_role(nd["role"])
            if role == "Hypothesis" and kg.G.in_degree(nid) > 0:
                rep.warnings.append(f"Hypothesis node '{nid}' has parents in the reconciled DAG.")
            if role == "Conclusion" and kg.G.out_degree(nid) > 0:
                rep.warnings.append(f"Conclusion node '{nid}' has children in the reconciled DAG.")

        return kg, rep

# --------- Core classes ---------

@dataclass
class Node:
    id: str
    role: str
    text: str
    parents: List[str] = field(default_factory=list)
    # agent metrics (None = unknown)
    credibility: Optional[float] = None
    relevance: Optional[float] = None
    evidence_strength: Optional[float] = None
    method_rigor: Optional[float] = None
    reproducibility: Optional[float] = None
    citation_support: Optional[float] = None

    def has_all_metrics(self) -> bool:
        return all(
            getattr(self, k) is not None
            for k in ("credibility","relevance","evidence_strength",
                      "method_rigor","reproducibility","citation_support")
        )

    def metric_dict(self) -> Dict[str, float]:
        return {
            "credibility": self.credibility or 0.0,
            "relevance": self.relevance or 0.0,
            "evidence_strength": self.evidence_strength or 0.0,
            "method_rigor": self.method_rigor or 0.0,
            "reproducibility": self.reproducibility or 0.0,
            "citation_support": self.citation_support or 0.0,
        }

class KGScorer:
    """
    Holds a DAG, keeps node/edge data, recomputes edge confidences when ready,
    and can compute a graph-level score.
    """
    def __init__(self,
                 edge_weights: EdgeCombineWeights | None = None,
                 node_quality: NodeQualityWeights | None = None,
                 role_prior: RoleTransitionPrior | None = None,
                 pair_synergy: PairSynergyWeights | None = None,
                 graph_score_weights: GraphScoreWeights | None = None,
                 global_metric_weights: GlobalMetricWeights | None = None):
        self.G = nx.DiGraph()
        self.nodes: Dict[str, Node] = {}
        self.tokens: Dict[str, Set[str]] = {}
        self.edge_update_callbacks: List[Callable[[str, str, float, Dict[str, float]], None]] = []

        self.edge_w = edge_weights or EdgeCombineWeights()
        self.node_q = node_quality or NodeQualityWeights()
        self.role_prior = role_prior or RoleTransitionPrior()
        self.pair_syn = pair_synergy or PairSynergyWeights()
        self.graph_w = graph_score_weights or GraphScoreWeights()
        self.metric_w = global_metric_weights or GlobalMetricWeights()
        self.default_edge_confidence = 0.5  # set to None to forbid fallback
        self.penalty = PropagationPenalty()
        self.trust: Dict[str, float] = {}

        # ---------- NEW: topology + adjacency caches ----------
        self._topo_dirty: bool = True
        self._topo_order: List[str] = []
        self._topo_index: Dict[str, int] = {}

        self._pred_cache: Dict[str, Tuple[str, ...]] = {}
        self._succ_cache: Dict[str, Tuple[str, ...]] = {}

        # ---------- NEW: per-node metric/quality caches ----------
        self._node_metric_version: Dict[str, int] = {}
        self._global_metric_weight_version: int = 0
        # node_id -> (node_ver, global_ver, cached_value)
        self._weighted_metrics_cache: Dict[str, Tuple[int, int, Dict[str, float]]] = {}
        self._node_quality_cache: Dict[str, Tuple[int, int, float]] = {}
        self._score_state_version: int = 0
        self._validated_score_state_version: int = -1
        self._validated_score_state_report: Optional[Dict[str, Any]] = None
        self._alignment_cache: Dict[Tuple[str, str], float] = {}

    # --- Build / update API ---

    def add_node(self, node: Node):
        node.role = canon_role(node.role)
        self.nodes[node.id] = node
        self.G.add_node(node.id, role=node.role)
        self.tokens[node.id] = tokenize(node.text)

        # NEW: cache invalidation
        self._node_metric_version.setdefault(node.id, 0)
        self._pred_cache.pop(node.id, None)
        self._succ_cache.pop(node.id, None)
        self._weighted_metrics_cache.pop(node.id, None)
        self._node_quality_cache.pop(node.id, None)
        self.trust.pop(node.id, None)
        self._mark_topology_dirty()

    def add_edge(self, u: str, v: str):
        if u not in self.nodes or v not in self.nodes:
            raise KeyError("add_edge: both nodes must be added first")
        self.G.add_edge(u, v)
        if u not in self.nodes[v].parents:
            self.nodes[v].parents.append(u)

        # NEW: adjacency/topology cache invalidation (localized)
        self._pred_cache.pop(v, None)
        self._succ_cache.pop(u, None)
        self._mark_topology_dirty()
        self._alignment_cache[(u, v)] = jaccard(self.tokens[u], self.tokens[v])

    def register_edge_update_callback(self, fn: Callable[[str, str, float, Dict[str,float]], None]):
        """fn(u, v, weight, feature_dict)"""
        self.edge_update_callbacks.append(fn)

    def _invalidate_trust_from(self, nid: str):
        if not self.G.has_node(nid):
            self.trust.pop(nid, None)
            return
        impacted = {nid} | self._descendants_set(nid)
        for x in impacted:
            self.trust.pop(x, None)

    def update_node_metrics(self, node_id: str, **metrics: float):
        """
        Update any subset of the six agent metrics. Recompute only impacted nodes/edges.
        """
        n = self.nodes[node_id]
        changed_any = False

        for k, v in metrics.items():
            if k not in n.__dict__:
                raise KeyError(f"Unknown metric '{k}'")
            if v is not None and not (0.0 <= v <= 1.0):
                raise ValueError(f"Metric '{k}' must be in [0,1]")
            if getattr(n, k) != v:
                setattr(n, k, v)
                changed_any = True

        if not changed_any:
            return

        # NEW: bump per-node version so metric/quality caches naturally invalidate
        self._node_metric_version[node_id] = self._node_metric_version.get(node_id, 0) + 1
        self._weighted_metrics_cache.pop(node_id, None)
        self._node_quality_cache.pop(node_id, None)

        # 0) Invalidate trust for this node and all descendants (trust recursion depends on it)
        self._invalidate_trust_from(node_id)

        # 1) Only recompute the impacted subgraph, in topological order
        impacted = {node_id} | self._descendants_set(node_id)
        for v in self._impacted_nodes_in_topo_order(impacted):
            self._try_update_incoming_edges(v)

    # ---------- Cache helpers ----------

    def _mark_topology_dirty(self) -> None:
        self._topo_dirty = True
        # adjacency caches are safe to keep, but they must be invalidated per-node on changes

    def _ensure_topology_cache(self) -> None:
        if not self._topo_dirty:
            return
        self._topo_order = list(nx.topological_sort(self.G))
        self._topo_index = {nid: i for i, nid in enumerate(self._topo_order)}
        self._topo_dirty = False

    def _preds(self, v: str) -> Tuple[str, ...]:
        cached = self._pred_cache.get(v)
        if cached is not None:
            return cached
        preds = tuple(self.G.predecessors(v))
        self._pred_cache[v] = preds
        return preds

    def _succs(self, u: str) -> Tuple[str, ...]:
        cached = self._succ_cache.get(u)
        if cached is not None:
            return cached
        succs = tuple(self.G.successors(u))
        self._succ_cache[u] = succs
        return succs

    def _descendants_set(self, start: str) -> Set[str]:
        """
        Faster descendant collection than nx.descendants for repeated calls.
        Uses cached successor lists (which are invalidated on add_edge).
        """
        if not self.G.has_node(start):
            return set()
        seen: Set[str] = set()
        stack = list(self._succs(start))
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(self._succs(x))
        return seen

    def _impacted_nodes_in_topo_order(self, impacted: Set[str]) -> List[str]:
        """
        Returns impacted nodes ordered topologically with a cost proportional to |impacted|
        for small impacted sets, falling back to a scan for very large impacted sets.
        """
        self._ensure_topology_cache()
        n_total = len(self._topo_order) or 1
        k = len(impacted)

        # If most nodes are impacted, scanning once is cheaper than sorting k nodes
        if k >= int(0.6 * n_total):
            imp = impacted  # local ref
            return [nid for nid in self._topo_order if nid in imp]

        idx = self._topo_index
        return sorted(impacted, key=lambda nid: idx.get(nid, 1 << 30))

    # --- Internal scoring ---

    def _try_update_incoming_edges(self, v: str):
        """If v + all parents have metrics, update each edge (u->v)."""
        node_v = self.nodes[v]
        if not node_v.has_all_metrics():
            return

        parents = self._preds(v)
        if not parents:
            return

        # Fast fail: if any parent missing metrics, nothing to do
        for u in parents:
            if not self.nodes[u].has_all_metrics():
                return

        # 1) Compute RAW edge confidences (no propagation penalty)
        raw_by_u: Dict[str, float] = {}
        feats_by_u: Dict[str, Dict[str, float]] = {}

        for u in parents:
            w_raw, feats = self._compute_edge_confidence(u, v)
            self.G[u][v]["confidence_raw"] = w_raw
            raw_by_u[u] = w_raw
            feats_by_u[u] = feats

        # 2) Compute TRUST for v (fills self.trust recursively as needed)
        tv = self._compute_node_trust(v)

        # 3) Set FINAL edge confidences with gating by parent trust
        eta = self.penalty.eta
        if eta < 0.0:
            eta = 0.0
        elif eta > 1.0:
            eta = 1.0

        alpha = self.penalty.alpha
        if alpha < 0.0:
            alpha = 0.0

        for u in parents:
            tu = self.trust.get(u)
            if tu is None:
                tu = self._node_quality(u)

            w_raw = raw_by_u[u]
            w_final = clip01(w_raw * (eta + (1.0 - eta) * (tu ** alpha)))
            self.G[u][v]["confidence"] = w_final

            feats = feats_by_u[u]  # NO copy
            feats["confidence_raw"] = w_raw
            feats["trust_parent"] = tu
            feats["trust_child"] = tv

            # keep compatibility: preserve the old raw “confidence” as confidence_combined
            if "confidence" in feats and feats["confidence"] != w_final:
                feats["confidence_combined"] = feats["confidence"]
            feats["confidence"] = w_final

            self.G[u][v]["features"] = feats

            for cb in self.edge_update_callbacks:
                cb(u, v, w_final, feats)

        # NEW: bump a version counter so graph_score() can reuse a cached validate_scoring_state()
        self._score_state_version += 1
        self._validated_score_state_version = -1
        self._validated_score_state_report = None

    def _node_quality(self, node_id: str) -> float:
        node_ver = self._node_metric_version.get(node_id, 0)
        glob_ver = self._global_metric_weight_version

        cached = self._node_quality_cache.get(node_id)
        if cached is not None and cached[0] == node_ver and cached[1] == glob_ver:
            return cached[2]

        node = self.nodes[node_id]
        role = node.role
        weights = self.node_q.per_role.get(role, {})

        md = self._weighted_metrics(node_id)

        if not weights:
            q = sum(md.values()) / 6.0
            q = clip01(q)
            self._node_quality_cache[node_id] = (node_ver, glob_ver, q)
            return q

        total = sum(abs(v) for v in weights.values()) or 1.0
        q = 0.0
        for m, w in weights.items():
            q += w * md.get(m, 0.0)
        q = clip01(q / total)

        self._node_quality_cache[node_id] = (node_ver, glob_ver, q)
        return q

    def _pair_synergy(self, u: str, v: str) -> float:
        ru, rv = self.nodes[u].role, self.nodes[v].role
        spec = self.pair_syn.per_pair.get((ru, rv))
        parent_md = self._weighted_metrics(u)
        child_md = self._weighted_metrics(v)

        if not spec:
            # default: average parent+child with equal weights over all metrics
            p = sum(parent_md.values()) / 6.0
            c = sum(child_md.values()) / 6.0
            return clip01(0.5*p + 0.5*c)

        def mix(md: Dict[str,float], w: Dict[str,float]) -> float:
            if not w: return 0.0
            tot = sum(abs(x) for x in w.values()) or 1.0
            return sum(w[k]*md.get(k,0.0) for k in w) / tot

        p_score = mix(parent_md, spec.get("parent", {}))
        c_score = mix(child_md,  spec.get("child", {}))
        return clip01(0.5*p_score + 0.5*c_score)

    def _alignment(self, u: str, v: str) -> float:
        key = (u, v)
        cached = self._alignment_cache.get(key)
        if cached is not None:
            return cached
        al = jaccard(self.tokens[u], self.tokens[v])
        self._alignment_cache[key] = al
        return al

    def _role_prior(self, u: str, v: str) -> float:
        return self.role_prior.get(self.nodes[u].role, self.nodes[v].role)

    def _compute_edge_confidence(self, u: str, v: str) -> Tuple[float, Dict[str,float]]:
        rp = self._role_prior(u, v)
        q_u = self._node_quality(u)
        q_v = self._node_quality(v)
        al = self._alignment(u, v)
        syn = self._pair_synergy(u, v)

        wcfg = self.edge_w
        # weighted sum; you can add penalties here if needed
        raw = (wcfg.role_prior   * rp +
               wcfg.parent_quality * q_u +
               wcfg.child_quality  * q_v +
               wcfg.alignment      * al +
               wcfg.synergy        * syn)

        w = clip01(raw)
        feats = {
            "role_prior": rp,
            "parent_quality": q_u,
            "child_quality": q_v,
            "alignment": al,
            "synergy": syn,
            "confidence": w
        }
        return w, feats

    def _get_raw_conf(self, u: str, v: str) -> float:
        return float(self.G[u][v].get("confidence_raw", self.penalty.default_raw_conf))
    
    def _agg_parent_vals(self, vals: list[float]) -> float:
        if not vals:
            return 0.0

        mode = self.penalty.agg
        if mode == "min":
            return min(vals)
        if mode == "mean":
            return sum(vals) / len(vals)

        if mode == "softmin":
            beta = max(0.0, float(getattr(self.penalty, "softmin_beta", 6.0)))
            vmin = min(vals)
            vmax = max(vals)
            rng = vmax - vmin
            if rng <= 1e-12 or beta == 0.0:
                return sum(vals) / len(vals)

            inv_rng = 1.0 / rng
            wsum = 0.0
            vsum = 0.0
            for v in vals:
                x = (v - vmin) * inv_rng
                w = math.exp(-beta * x)
                wsum += w
                vsum += w * v
            return vsum / (wsum or 1.0)

        if mode == "dampmin":
            lam = float(getattr(self.penalty, "dampmin_lambda", 0.35))
            lam = 0.0 if lam < 0.0 else 1.0 if lam > 1.0 else lam
            vmin = min(vals)
            vmean = sum(vals) / len(vals)
            return (1.0 - lam) * vmin + lam * vmean

        raise ValueError(f"Unknown penalty.agg='{mode}'")

    def _compute_node_trust(self, v: str, _memo: Optional[Dict[str, float]] = None) -> float:
        if not self.penalty.enabled:
            return clip01(self._node_quality(v))

        if _memo is None:
            _memo = {}
        else:
            tv = _memo.get(v)
            if tv is not None:
                return tv

        tv_cached = self.trust.get(v)
        if tv_cached is not None:
            _memo[v] = tv_cached
            return tv_cached

        qv = clip01(self._node_quality(v))
        parents = self._preds(v)
        if not parents:
            tv = qv
        else:
            alpha = self.penalty.alpha
            if alpha < 0.0:
                alpha = 0.0

            vals: List[float] = []
            for u in parents:
                tu = _memo.get(u)
                if tu is None:
                    tu = self.trust.get(u)
                if tu is None:
                    tu = self._compute_node_trust(u, _memo)

                c_raw = self._get_raw_conf(u, v)
                vals.append((max(1e-6, tu) ** alpha) * c_raw)

            agg_val = self._agg_parent_vals(vals)
            tv = clip01(qv * agg_val)

        _memo[v] = tv
        self.trust[v] = tv
        return tv

    def _weighted_metrics(self, node_id: str) -> Dict[str, float]:
        """
        Return this node's six metrics after applying the global weights.
        Cached per node_id across calls until that node's metrics or global weights change.
        """
        node_ver = self._node_metric_version.get(node_id, 0)
        glob_ver = self._global_metric_weight_version

        cached = self._weighted_metrics_cache.get(node_id)
        if cached is not None and cached[0] == node_ver and cached[1] == glob_ver:
            return cached[2]

        md = self.nodes[node_id].metric_dict()  # already returns a fresh dict
        w = self.metric_w.weights or {}

        if w:
            for k in METRIC_KEYS:
                wk = float(w.get(k, 1.0))
                v = md.get(k, 0.0) * wk
                md[k] = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

        self._weighted_metrics_cache[node_id] = (node_ver, glob_ver, md)
        return md

    def recompute_all_confidences(self):
        """
        Recompute all edge confidences (and trusts) under current weights.
        Uses cached topological order.
        """
        if not self.G.nodes:
            return
        self.trust.clear()
        self._ensure_topology_cache()
        for v in self._topo_order:
            self._try_update_incoming_edges(v)

    def validate_scoring_state(self, *, strict: bool = False) -> Dict[str, Any]:
        """
        Validate that the scorer has enough information to produce stable scores.

        Definitions:
        - A node is "ready" if it has all six metrics (Node.has_all_metrics()).
        - An edge (u -> v) is "eligible" for derived confidences if v is ready and ALL
            of its parents are ready. Eligible edges should have both:
            * G[u][v]['confidence_raw']
            * G[u][v]['confidence']

        This flags the most common root cause of unexpectedly low scores: scoring paths
        falling back to defaults because eligible edges never got their derived fields.
        """
        rep: Dict[str, Any] = {
            "errors": [],
            "warnings": [],
            "nodes_missing_metrics": [],
            "eligible_edges": 0,
            "eligible_edges_with_confidence": 0,
            "eligible_edges_with_confidence_raw": 0,
            "eligible_edges_missing_confidence": [],
            "eligible_edges_missing_confidence_raw": [],
            "eligible_edges_missing_features": [],
        }

        # 1) Node readiness
        for nid, n in self.nodes.items():
            if not n.has_all_metrics():
                rep["nodes_missing_metrics"].append(nid)

        # 2) Eligible edges that should have derived confidence fields
        for v in self.G.nodes():
            parents = self._preds(v)
            if not parents:
                continue
            if not self.nodes[v].has_all_metrics():
                continue
            if not all(self.nodes[u].has_all_metrics() for u in parents):
                continue

            for u in parents:
                rep["eligible_edges"] += 1
                d = self.G[u][v]

                if "confidence" in d:
                    rep["eligible_edges_with_confidence"] += 1
                else:
                    rep["eligible_edges_missing_confidence"].append((u, v))

                if "confidence_raw" in d:
                    rep["eligible_edges_with_confidence_raw"] += 1
                else:
                    rep["eligible_edges_missing_confidence_raw"].append((u, v))

                if "features" not in d:
                    rep["eligible_edges_missing_features"].append((u, v))

        # 3) Summarize as warnings/errors
        if rep["eligible_edges_missing_confidence"]:
            msg = (
                f"{len(rep['eligible_edges_missing_confidence'])} eligible edges are missing 'confidence'. "
                "Run recompute_all_confidences() (or ensure update_node_metrics() reaches all impacted nodes) "
                "before scoring."
            )
            if self.default_edge_confidence is None:
                rep["errors"].append(msg)
            else:
                rep["warnings"].append(msg)

        if self.penalty.enabled and rep["eligible_edges_missing_confidence_raw"]:
            msg = (
                f"{len(rep['eligible_edges_missing_confidence_raw'])} eligible edges are missing 'confidence_raw'. "
                "Propagation trust uses confidence_raw; missing values typically indicate the edge-update path did not run."
            )
            if self.penalty.default_raw_conf is None:
                rep["errors"].append(msg)
            else:
                rep["warnings"].append(msg)

        if rep["eligible_edges_missing_features"]:
            rep["warnings"].append(
                f"{len(rep['eligible_edges_missing_features'])} eligible edges are missing 'features' dicts."
            )

        if strict and rep["errors"]:
            raise ValueError("; ".join(rep["errors"]))

        return rep

    def set_metric_weights(self, weights: Dict[str, float], *, normalize: bool = False):
        """
        Update the global per-metric weights (single vector for all nodes).
        Example:
            set_metric_weights({
                "credibility": 1.0,
                "relevance": 0.8,
                "evidence_strength": 1.2,
                "method_rigor": 1.1,
                "reproducibility": 0.9,
                "citation_support": 1.0
            }, normalize=False)
        Notes:
        • Values are nonnegative; we clip their effect on metrics to keep [0,1].
        • If normalize=True, we scale so the mean weight is 1.0 (keeps overall
            scale roughly stable across mixes).
        """
        valid = {"credibility","relevance","evidence_strength",
                "method_rigor","reproducibility","citation_support"}
        # Merge+validate
        curr = dict(self.metric_w.weights)
        for k, v in weights.items():
            if k not in valid:
                raise KeyError(f"Unknown metric '{k}'")
            fv = float(v)
            if fv < 0:
                raise ValueError(f"Metric weight '{k}' must be >= 0")
            curr[k] = fv

        if normalize and curr:
            avg = sum(curr.values()) / len(curr.values())
            avg = avg if avg > 0 else 1.0
            curr = {k: (x / avg) for k, x in curr.items()}

        self.metric_w.weights = curr

        # NEW: bump global version to invalidate all cached metric mixes / qualities
        self._global_metric_weight_version += 1
        self._weighted_metrics_cache.clear()
        self._node_quality_cache.clear()

        self.recompute_all_confidences()

    # --------- Graph score (optional; configurable) ---------

    def graph_score(self, *, validate: bool = True) -> Tuple[float, Dict[str, float]]:
        """
        Optimized graph_score:
        - Uses cached topo order (self._ensure_topology_cache)
        - Uses boolean reachability (no big-int path counting)
        - Best-path DP in log-space for numeric stability (still “max product” semantics)
        - Restricts coherence/flow/cut work to the bridge subgraph
        - Optionally skips validate_scoring_state() if state unchanged
        - Skips maxflow/mincut if their weights are 0
        """
        # 0) Get topo once (and validate DAG implicitly)
        try:
            self._ensure_topology_cache()
        except nx.NetworkXUnfeasible as e:
            raise ValueError("Graph score expects a DAG") from e

        G = self.G
        topo = self._topo_order
        if not topo:
            details = {
                "bridge_coverage": 0.0,
                "best_path": 0.0,
                "redundancy": 0.0,
                "fragility": 0.0,
                "coherence": 0.0,
                "coverage": 0.0,
                "score_raw": 0.0,
            }
            # keep mapping consistent with your existing normalization logic
            return 0.5, details

        # Identify roots
        H = [n for n in G.nodes() if self.nodes[n].role == "Hypothesis"]
        C = [n for n in G.nodes() if self.nodes[n].role == "Conclusion"]

        # 1) Optional validation (cached by a version counter; see section #2)
        if validate:
            rep = None
            if (
                getattr(self, "_validated_score_state_version", -1) == getattr(self, "_score_state_version", 0)
                and getattr(self, "_validated_score_state_report", None) is not None
            ):
                rep = self._validated_score_state_report
            else:
                rep = self.validate_scoring_state(strict=False)
                self._validated_score_state_version = self._score_state_version
                self._validated_score_state_report = rep

            if rep["eligible_edges"] and rep["eligible_edges_with_confidence"] < rep["eligible_edges"]:
                if self.default_edge_confidence is None:
                    missing = rep["eligible_edges"] - rep["eligible_edges_with_confidence"]
                    raise ValueError(f"Scoring not ready: {missing} eligible edges are missing 'confidence'.")

        # Keep your “all edges must have confidence if no default” behavior
        total_e = G.number_of_edges() or 1
        weighted_e = sum(1 for u, v in G.edges() if "confidence" in G[u][v])
        if self.default_edge_confidence is None and weighted_e < total_e:
            raise ValueError("Not enough edges have confidence; defer scoring.")

        preds = self._preds
        succs = self._succs

        # 2) Boolean reachability from H (single topo pass)
        reach_from: Set[str] = set(H)
        for v in topo:
            if v in reach_from:
                continue
            for u in preds(v):
                if u in reach_from:
                    reach_from.add(v)
                    break

        # 3) Boolean reachability to C (single reverse-topo pass)
        reach_to: Set[str] = set(C)
        for u in reversed(topo):
            if u in reach_to:
                continue
            for v in succs(u):
                if v in reach_to:
                    reach_to.add(u)
                    break

        bridge_nodes = reach_from & reach_to
        bridge_cov = len(bridge_nodes) / max(1, G.number_of_nodes())

        # 4) Best-path “max product” DP, but in log-space (faster + avoids underflow)
        idx = self._topo_index
        NEG_INF = -1e300
        best_log = [NEG_INF] * len(topo)
        best_len = [0] * len(topo)

        for h in H:
            # Only H that can reach some Conclusion matters
            if h in bridge_nodes:
                hi = idx[h]
                best_log[hi] = 0.0
                best_len[hi] = 0

        default_w = self.default_edge_confidence
        for v in topo:
            if v not in bridge_nodes:
                continue
            vi = idx[v]
            for u in preds(v):
                ui = idx[u]
                ul = best_log[ui]
                if ul <= NEG_INF / 2:
                    continue
                w = G[u][v].get("confidence", default_w)
                if w is None or w <= 0.0:
                    continue
                cand_log = ul + math.log(w)
                if cand_log > best_log[vi]:
                    best_log[vi] = cand_log
                    best_len[vi] = best_len[ui] + 1

        best_path = 0.0
        for c in C:
            if c not in bridge_nodes:
                continue
            ci = idx[c]
            bl = best_log[ci]
            if bl <= NEG_INF / 2:
                continue
            L = max(1, best_len[ci])
            gm = math.exp(bl / L)  # geometric mean of product-maximizing path
            if gm > best_path:
                best_path = gm

        # 5) Coherence & coverage computed only on bridge edges/nodes
        bridge_edges: List[Tuple[str, str]] = []
        coh_hits = 0.0
        role_prior = self._role_prior
        for u, v in G.edges():
            if u in bridge_nodes and v in bridge_nodes:
                bridge_edges.append((u, v))
                if role_prior(u, v) >= 0.5:
                    coh_hits += 1.0
        coh = (coh_hits / len(bridge_edges)) if bridge_edges else 0.0

        target_roles = {"Method", "Evidence", "Result"}
        roles_present = {self.nodes[n].role for n in bridge_nodes}
        cov = len(roles_present & target_roles) / len(target_roles) if target_roles else 1.0

        # 6) Redundancy / fragility: skip entirely if their weights are 0
        W = self.graph_w
        redundancy = 0.0
        fragility = 0.0

        # Filter roots to those actually on the bridge
        H2 = [h for h in H if h in bridge_nodes]
        C2 = [c for c in C if c in bridge_nodes]

        # Prefer a faster flow implementation for typical sparse DAGs
        flow_func = nx.algorithms.flow.shortest_augmenting_path

        if W.redundancy != 0.0 and bridge_edges and H2 and C2:
            Gf = nx.DiGraph()
            S, T = "_S_", "_T_"
            for u, v in bridge_edges:
                Gf.add_edge(u, v, capacity=1.0)
            for h in H2:
                Gf.add_edge(S, h, capacity=float("inf"))
            for c in C2:
                Gf.add_edge(c, T, capacity=float("inf"))
            try:
                flow_val, _ = nx.maximum_flow(Gf, S, T, flow_func=flow_func)
                redundancy = min(flow_val / 3.0, 1.0)
            except Exception:
                redundancy = 0.0

        if W.fragility != 0.0 and bridge_edges and H2 and C2:
            Gc = nx.DiGraph()
            S, T = "_S_", "_T_"
            for u, v in bridge_edges:
                w = G[u][v].get("confidence", default_w)
                if w is None:
                    # if default_w is None, earlier checks would have raised; keep safe fallback
                    w = 0.0
                cap = max(1e-6, 1.0 - w)
                Gc.add_edge(u, v, capacity=cap)
            for h in H2:
                Gc.add_edge(S, h, capacity=float("inf"))
            for c in C2:
                Gc.add_edge(c, T, capacity=float("inf"))
            try:
                cut_val, _ = nx.minimum_cut(Gc, S, T, flow_func=flow_func)
                denom = max(1, len(bridge_edges))
                fragility = clip01(cut_val / denom)
            except Exception:
                fragility = 1.0

        # 7) Final score (unchanged normalization logic)
        score_raw = (
            W.bridge_coverage * bridge_cov +
            W.best_path       * best_path +
            W.redundancy      * redundancy +
            W.fragility       * fragility +
            W.coherence       * coh +
            W.coverage        * cov
        )

        wvec = [W.bridge_coverage, W.best_path, W.redundancy, W.fragility, W.coherence, W.coverage]
        pos_sum = sum(x for x in wvec if x > 0)
        neg_sum = sum(x for x in wvec if x < 0)

        denom = max(1e-9, (pos_sum - neg_sum))
        score_01 = (score_raw - neg_sum) / denom
        score_01 = clip01(score_01)

        eps = 1e-9
        score = eps + (1.0 - 2.0 * eps) * score_01

        details = {
            "bridge_coverage": bridge_cov,
            "best_path": best_path,
            "redundancy": redundancy,
            "fragility": fragility,
            "coherence": coh,
            "coverage": cov,
            "score_raw": score_raw,
        }
        return score, details

    # ---------- Embedding exports ----------

    def export_edge_features(self) -> list[dict]:
        """
        Return a list of edge feature dicts:
          {'u','v','confidence', 'role_prior','parent_quality','child_quality','alignment','synergy'}
        Uses cached features when present; otherwise computes with defaults.
        """
        feats = []
        for u, v in self.G.edges():
            d = self.G[u][v]
            conf = d.get("confidence", self.default_edge_confidence)
            f = d.get("features")
            if not f:
                w, f = self._compute_edge_confidence(u, v)
                conf = w
            conf_raw = d.get("confidence_raw", self.default_edge_confidence)
            tp = d.get("features", {}).get("trust_parent")
            tc = d.get("features", {}).get("trust_child")
            feats.append({
            "u": u, "v": v,
            "confidence_raw": float(conf_raw),
            "confidence": float(conf),
            "trust_parent": None if tp is None else float(tp),
            "trust_child": None if tc is None else float(tc),
                "role_prior": float(f["role_prior"]),
                "parent_quality": float(f["parent_quality"]),
                "child_quality": float(f["child_quality"]),
                "alignment": float(f["alignment"]),
                "synergy": float(f["synergy"]),
            })
        return feats

    def export_node_feature_matrix(
        self,
        text_embeddings: dict[str, "np.ndarray"] | None = None,
        include_role_onehot: bool = True,
        include_metrics: bool = True
    ):
        """
        Build per-node feature vectors in consistent id order.
        Returns (X, ids, feature_names)
        - X: np.ndarray [N, D] if numpy is available, else List[List[float]]
        - feature_names: names matching columns in X
        """
        ids = sorted(self.nodes.keys(), key=lambda x: str(x))
        feat_names = []
        X = []

        base_role_names = [f"role::{r}" for r in _ROLE_LIST] if include_role_onehot else []
        base_metric_names = ["credibility","relevance","evidence_strength",
                             "method_rigor","reproducibility","citation_support"] if include_metrics else []
        text_dim = None

        # If text embeddings provided, infer dimension and column names
        if text_embeddings:
            # Grab first available vector to infer dim
            for nid in ids:
                vec = text_embeddings.get(nid)
                if vec is not None:
                    text_dim = len(vec.tolist() if hasattr(vec, "tolist") else vec)
                    break

        feat_names = []
        feat_names += base_role_names
        feat_names += base_metric_names
        if text_dim:
            feat_names += [f"text_emb_{i}" for i in range(text_dim)]

        for nid in ids:
            node = self.nodes[nid]
            row = []
            if include_role_onehot:
                row.extend(_role_one_hot(node.role))
            if include_metrics:
                md = node.metric_dict()
                row.extend([md[k] for k in base_metric_names])
            if text_dim:
                vec = text_embeddings.get(nid)
                if vec is None:
                    row.extend([0.0]*text_dim)
                else:
                    row.extend(vec.tolist() if hasattr(vec, "tolist") else list(vec))
            X.append(row)

        if np is not None:
            X = np.asarray(X, dtype=float)
        return X, ids, feat_names

    def random_walk_corpus(
        self,
        num_walks: int = 10,
        walk_length: int = 8,
        bias_p: float = 1.0,
        bias_q: float = 1.0,
        min_conf: float = 0.0
    ) -> list[list[str]]:
        """
        Weighted, second-order random-walk corpus (node2vec flavor) over the DAG.
        Uses 'confidence' as edge weights (falls back to default_edge_confidence).
        Returns a list of walks (each a list of node ids as strings).
        """
        import random
        G = self.G
        # Precompute neighbor lists with weights
        nbrs = {}
        for u in G.nodes():
            outs = []
            for v in G.successors(u):
                w = G[u][v].get("confidence", self.default_edge_confidence)
                if w is None: 
                    continue
                if w >= min_conf:
                    outs.append((v, float(w)))
            nbrs[u] = outs

        def weighted_choice(options):
            if not options: return None
            vs, ws = zip(*options)
            s = sum(ws)
            if s <= 0: return random.choice(vs)
            r = random.random() * s
            acc = 0.0
            for v, w in options:
                acc += w
                if acc >= r:
                    return v
            return vs[-1]

        walks = []
        nodes = list(G.nodes())
        for _ in range(num_walks):
            random.shuffle(nodes)
            for start in nodes:
                walk = [start]
                if not nbrs[start]:
                    walks.append(walk); continue
                # first step: simple weighted pick
                v1 = weighted_choice(nbrs[start])
                if v1 is None:
                    walks.append(walk); continue
                walk.append(v1)
                # subsequent steps: biased by p,q around previous hop
                while len(walk) < walk_length:
                    t, v = walk[-2], walk[-1]
                    cand = []
                    for x, w in nbrs.get(v, []):
                        # node2vec bias:
                        # backtrack t -> v -> x gets weight /p
                        # if x connected to t (here, use DAG structure: treat equality as "backtrack")
                        alpha = (1.0/bias_p) if (x == t) else (1.0 if (t in G.predecessors(x) or x in G.successors(t)) else 1.0/bias_q)
                        cand.append((x, w * alpha))
                    nxt = weighted_choice(cand)
                    if nxt is None:
                        break
                    walk.append(nxt)
                walks.append([str(z) for z in walk])
        return walks

    def paper_fingerprint(self) -> tuple[list[float], list[str]]:
        """
        Deterministic 'graph embedding' for a whole paper using:
          - graph_score details (bridge_coverage, best_path, redundancy, fragility, coherence, coverage)
          - role counts (normalized)
          - role-pair edge histogram weighted by confidence (normalized)
          - edge confidence histogram (5 bins)
        Returns (vector, names)
        """
        # 1) graph_score details
        _, det = self.graph_score()
        keys = ["bridge_coverage","best_path","redundancy","fragility","coherence","coverage"]
        vec = [float(det[k]) for k in keys]
        names = [f"score::{k}" for k in keys]

        # 2) role counts
        total_nodes = max(1, self.G.number_of_nodes())
        role_counts = {r: 0 for r in _ROLE_LIST}
        for n in self.G.nodes():
            role_counts[self.nodes[n].role] += 1
        for r in _ROLE_LIST:
            names.append(f"role_frac::{r}")
            vec.append(role_counts[r] / total_nodes)

        # 3) role-pair histogram (weighted by confidence)
        pair_sum = 0.0
        pair_counts = {(ru, rv): 0.0 for ru in _ROLE_LIST for rv in _ROLE_LIST}
        for u, v in self.G.edges():
            ru, rv = self.nodes[u].role, self.nodes[v].role
            w = self.G[u][v].get("confidence", 0.5)
            pair_counts[(ru, rv)] += float(w)
            pair_sum += float(w)
        denom = max(pair_sum, 1e-6)
        for ru in _ROLE_LIST:
            for rv in _ROLE_LIST:
                names.append(f"pairW::{ru}->{rv}")
                vec.append(pair_counts[(ru, rv)] / denom)

        # 4) confidence histogram (coarse)
        bins = [0, 0, 0, 0, 0]
        for u, v in self.G.edges():
            w = self.G[u][v].get("confidence", 0.5)
            if   w < 0.2: bins[0]+=1
            elif w < 0.4: bins[1]+=1
            elif w < 0.6: bins[2]+=1
            elif w < 0.8: bins[3]+=1
            else:         bins[4]+=1
        total_e = max(1, self.G.number_of_edges())
        for i, c in enumerate(bins):
            names.append(f"conf_bin_{i}")
            vec.append(c/total_e)

        return vec, names

# --------- Example usage (remove or keep for a quick smoke test) ---------
if __name__ == "__main__":
    kg = KGScorer()

    # Nodes
    kg.add_node(Node(id="H", role="Hypothesis", text="Drug X reduces blood pressure compared to placebo."))
    kg.add_node(Node(id="M1", role="Method", text="Randomized double-blind RCT with 200 participants."))
    kg.add_node(Node(id="E1", role="Evidence", text="SBP decreased by 8mmHg (p<0.01). Smith 2024. [1]"))
    kg.add_node(Node(id="R1", role="Result", text="Observed mean SBP reduction vs baseline."))
    kg.add_node(Node(id="C", role="Conclusion", text="Drug X is effective for lowering blood pressure."))

    # Edges (DAG)
    kg.add_edge("H", "M1")
    kg.add_edge("M1", "R1")
    kg.add_edge("E1", "R1")
    kg.add_edge("R1", "C")
    kg.add_edge("H", "E1")
    kg.add_edge("H", "R1")  # maybe weaker path

    # Listen for real-time updates
    def on_edge(u, v, w, feats):
        print(f"EDGE UPDATED {u}->{v}: {w:.3f}  {feats}")
    kg.register_edge_update_callback(on_edge)

    # Simulate agent filling metrics out of order
    kg.update_node_metrics("H", credibility=0.8, relevance=0.9, evidence_strength=0.3,
                           method_rigor=0.2, reproducibility=0.2, citation_support=0.1)
    kg.update_node_metrics("M1", credibility=0.7, relevance=0.6, evidence_strength=0.3,
                           method_rigor=0.9, reproducibility=0.8, citation_support=0.2)
    kg.update_node_metrics("E1", credibility=0.8, relevance=0.7, evidence_strength=0.9,
                           method_rigor=0.4, reproducibility=0.6, citation_support=0.9)
    kg.update_node_metrics("R1", credibility=0.75, relevance=0.8, evidence_strength=0.8,
                           method_rigor=0.6, reproducibility=0.6, citation_support=0.8)
    kg.update_node_metrics("C", credibility=0.7, relevance=0.8, evidence_strength=0.7,
                           method_rigor=0.5, reproducibility=0.6, citation_support=0.5)

    # Graph score
    score, details = kg.graph_score()
    print("GRAPH SCORE:", score, details)
