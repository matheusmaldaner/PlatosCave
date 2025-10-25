# app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict

Role = Literal[
    "hypothesis","claim","method","evidence",
    "result","conclusion","limitation","other"
]

class NodeScore(BaseModel):
    credibility: float = Field(ge=0, le=1, default=0.0)
    relevance: float = Field(ge=0, le=1, default=0.0)
    evidence_strength: float = Field(ge=0, le=1, default=0.0)
    method_rigor: float = Field(ge=0, le=1, default=0.0)
    reproducibility: float = Field(ge=0, le=1, default=0.0)
    citation_support: float = Field(ge=0, le=1, default=0.0)
    composite: float = Field(ge=0, le=1, default=0.0)
    mode: Literal["web","opinion"] = "opinion"
    citations: List[str] = []
    notes: str = ""

class KGNode(BaseModel):
    id: str
    role: Role
    level: int
    text: str
    span_hint: Optional[str] = None
    children: List[str] = []
    score: Optional[NodeScore] = None

class KGEdge(BaseModel):
    source: str
    target: str
    relation: Literal["supports","based_on","leads_to","contradicts","qualifies"]

class PaperGraph(BaseModel):
    paper_id: str
    title: str
    nodes: List[KGNode]
    edges: List[KGEdge]
    meta: Dict[str, str] = {}
