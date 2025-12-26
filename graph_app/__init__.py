"""
Realtime knowledge-graph scoring service.

Exports:
- KGScorer, DAGValidation: core scoring and validation utilities
- KGSession: thin session adapter intended for API/front-end integration
"""

from .kg_realtime_scoring import KGScorer, DAGValidation
from .service_adapter import KGSession

__all__ = ["KGScorer", "DAGValidation", "KGSession"]

