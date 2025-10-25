export interface FactDagNode {
  id: number; // 0..N-1
  text: string; // concise fact statement
}

export interface FactDagEdge {
  source: number; // node id
  target: number; // node id (must be > source to ensure acyclicity)
}

export interface FactDAG {
  nodes: FactDagNode[];
  edges: FactDagEdge[];
}

export interface LLMClient {
  generate(prompt: string): Promise<string>;
}

export function buildFactDagPrompt(rawText: string): string {
  return [
    "Extract factual statements and connect related facts as a single acyclic directed graph (DAG).",
    "Rules:",
    "- Output STRICT JSON ONLY with keys: nodes, edges.",
    "- nodes: array of { id: number (0..N-1), text: string } (concise, self-contained facts).",
    "- edges: array of { source: number, target: number } with target > source to enforce DAG.",
    "- Connect a fact to all directly related subsequent facts; omit unrelated links.",
    "- No commentary, no extra keys.",
    "Example:\n{\n  \"nodes\": [ { \"id\": 0, \"text\": \"...\" }, { \"id\": 1, \"text\": \"...\" } ],\n  \"edges\": [ { \"source\": 0, \"target\": 1 } ]\n}",
    "TEXT:",
    rawText.trim(),
  ].join("\n\n");
}

export async function extractFactDAGWithLLM(
  rawText: string,
  llm: LLMClient
): Promise<FactDAG> {
  const prompt = buildFactDagPrompt(rawText);
  const completion = await llm.generate(prompt);
  const parsed = parseDagJson(completion);
  if (parsed) return parsed;
  return heuristicDag(rawText);
}

function parseDagJson(s: string): FactDAG | null {
  try {
    const jsonStart = s.indexOf("{");
    const jsonEnd = s.lastIndexOf("}");
    const trimmed = jsonStart >= 0 && jsonEnd >= 0 ? s.slice(jsonStart, jsonEnd + 1) : s;
    const obj = JSON.parse(trimmed) as Partial<FactDAG>;
    if (!obj || !Array.isArray(obj.nodes) || !Array.isArray(obj.edges)) return null;
    // Basic validation
    for (const n of obj.nodes) {
      if (typeof (n as any)?.id !== "number" || typeof (n as any)?.text !== "string") return null;
    }
    for (const e of obj.edges) {
      const ee = e as any;
      if (typeof ee?.source !== "number" || typeof ee?.target !== "number") return null;
      if (!(ee.target > ee.source)) return null; // enforce DAG direction
    }
    return obj as FactDAG;
  } catch {
    return null;
  }
}

function heuristicDag(rawText: string): FactDAG {
  const normalized = rawText
    .replace(/\r/g, "")
    .replace(/[\t ]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  const paragraphs = normalized
    .split(/\n\s*\n/)
    .map(p => p.trim())
    .filter(Boolean);

  const sentenceSplit = (text: string): string[] => {
    const parts = text
      .split(/(?<=[\.!?])\s+(?=[A-Z(\[])|(?<=[\.!?])\s+(?=\d)/)
      .map(s => s.trim())
      .filter(Boolean);
    if (parts.length <= 1) {
      const lines = text.split(/\n+/).map(l => l.trim()).filter(Boolean);
      return lines.length > 1 ? lines : [text];
    }
    return parts;
  };

  const sentences: string[] = [];
  for (const para of paragraphs) {
    sentences.push(...sentenceSplit(para));
  }
  const nodes: FactDagNode[] = sentences.map((t, i) => ({ id: i, text: t }));
  const edges: FactDagEdge[] = [];
  // Simple acyclic structure: connect each sentence to the next within a small window
  const windowSize = 1; // can be tuned or made semantic later
  for (let i = 0; i < nodes.length; i++) {
    for (let w = 1; w <= windowSize; w++) {
      const j = i + w;
      if (j < nodes.length) edges.push({ source: i, target: j });
    }
  }
  return { nodes, edges };
}

export class MockDagLLM implements LLMClient {
  async generate(prompt: string): Promise<string> {
    // Derive a basic DAG using heuristic and return as strict JSON
    const text = prompt.slice(prompt.lastIndexOf("TEXT:") + 5).trim();
    const dag = heuristicDag(text);
    return JSON.stringify(dag);
  }
}


