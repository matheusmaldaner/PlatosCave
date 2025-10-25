export interface NodeScore {
  credibility: number;
  relevance: number;
  evidence_strength: number;
  method_rigor: number;
  reproducibility: number;
  citation_support: number;
  role: string;  // e.g. "hypothesis", "result", etc.
  level: number; // depth in the knowledge graph
}

 // Computes a simple average composite score for one node
export function computeComposite(node: NodeScore): number {
  const metrics = [
    node.credibility,
    node.relevance,
    node.evidence_strength,
    node.method_rigor,
    node.reproducibility,
    node.citation_support,
  ];
  return metrics.reduce((a, b) => a + b, 0) / metrics.length;
}

/**
 * Computes an overall weighted score for a paper
 * @param nodes  List of nodes in the knowledge graph
 * @param alpha  Depth decay factor (higher alpha → more penalty for deeper nodes)
 * @param roleWeights Optional weighting per role type
 */
export function computePaperScore(
  nodes: NodeScore[],
  alpha = 0.15,
  roleWeights?: Record<string, number>
): number {
  const defaultWeights: Record<string, number> = {
    hypothesis: 1.2,
    claim: 1.1,
    method: 1.0,
    evidence: 1.0,
    result: 1.1,
    conclusion: 1.2,
    limitation: 0.9,
    other: 0.8,
  };

  const weights = roleWeights || defaultWeights;

  let numerator = 0;
  let denominator = 0;

  for (const node of nodes) {
    const w = (weights[node.role] || 1) * Math.exp(-alpha * node.level);
    const comp = computeComposite(node);
    numerator += w * comp;
    denominator += w;
  }

  return denominator > 0 ? numerator / denominator : 0;
}

