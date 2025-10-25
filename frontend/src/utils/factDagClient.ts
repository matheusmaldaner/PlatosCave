import type { FactDAG, LLMClient } from "./factDagLLM";

export class ApiDagLLM implements LLMClient {
  async generate(prompt: string): Promise<string> {
    const startedAt = Date.now();
    const res = await fetch("/api/fact-dag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    if (!res.ok) {
      const text = await res.text();
      // Surface vendor/model/reqId if present
      try {
        const j = JSON.parse(text);
        console.error("/api/fact-dag error", { status: res.status, ...j, ms: Date.now() - startedAt });
      } catch {
        console.error("/api/fact-dag error", { status: res.status, body: text, ms: Date.now() - startedAt });
      }
      throw new Error(`API error ${res.status}: ${text}`);
    }
    const data = await res.json();
    console.log("/api/fact-dag success", { ms: Date.now() - startedAt });
    return data.completion as string;
  }
}


