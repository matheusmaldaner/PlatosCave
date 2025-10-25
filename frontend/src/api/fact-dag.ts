import { buildFactDagPrompt } from "../utils/factDagLLM";
import fs from "fs";
import path from "path";

// Gatsby Functions: default export handler(req, res)
export default async function handler(req: any, res: any) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method Not Allowed" });
    return;
  }

  try {
    const reqId = Math.random().toString(36).slice(2);
    console.log(`[fact-dag][${reqId}] invoked`);
    // Attempt to load key/provider/model from config.yaml at repo root; fallback to env var
    const cfg = await loadConfigFromYaml();
    const apiKey = cfg?.apiKey || process.env.OPENAI_API_KEY;
    if (!apiKey) {
      console.error(`[fact-dag][${reqId}] missing API key`);
      res.status(500).json({ error: "Missing API key (config.yaml or OPENAI_API_KEY)", reqId });
      return;
    }

    const { text, prompt } = typeof req.body === "string" ? JSON.parse(req.body) : req.body || {};
    if (!text && !prompt) {
      console.error(`[fact-dag][${reqId}] missing text/prompt`);
      res.status(400).json({ error: "Provide 'text' or 'prompt' in JSON body", reqId });
      return;
    }

    const finalPrompt = prompt || buildFactDagPrompt(String(text));

    if (cfg?.vendor === "anthropic" || /^sk-ant-/.test(apiKey)) {
      // Anthropic Messages API
      const attempts = buildAnthropicAttempts(cfg?.model);
      let lastStatus = 0;
      let lastBody = "";
      for (const model of attempts) {
        console.log(`[fact-dag][${reqId}] vendor=anthropic try model=${model}`);
        const response = await fetch("https://api.anthropic.com/v1/messages", {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-api-key": apiKey,
            "anthropic-version": "2023-06-01",
          },
          body: JSON.stringify({
            model,
            max_tokens: 8000,
            temperature: 0.2,
            messages: [
              { role: "user", content: `Return STRICT JSON only.\n\n${finalPrompt}` },
            ],
          }),
        });
        if (response.ok) {
          const data = await response.json();
          const content = Array.isArray(data?.content) && data.content[0]?.text ? data.content[0].text : "";
          console.log(`[fact-dag][${reqId}] anthropic success model=${model}`);
          res.status(200).json({ completion: content, model });
          return;
        } else {
          lastStatus = response.status;
          lastBody = await response.text();
          console.error(`[fact-dag][${reqId}] anthropic error status=${lastStatus} body=${lastBody}`);
          // Only continue to next attempt on 404 model not found; otherwise stop early
          if (!(lastStatus === 404 && /not_found_error/.test(lastBody) && /model/i.test(lastBody))) {
            res.status(lastStatus).json({ error: lastBody, vendor: "anthropic", model, reqId });
            return;
          }
        }
      }
      res.status(lastStatus || 404).json({ error: lastBody || "model not found", vendor: "anthropic", attempts, reqId });
      return;
    }

    // Default: OpenAI Chat Completions API
    {
      const model = cfg?.model || "gpt-4o-mini";
      console.log(`[fact-dag][${reqId}] vendor=openai model=${model}`);
      const response = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages: [
            { role: "system", content: "You return STRICT JSON only, no commentary." },
            { role: "user", content: finalPrompt },
          ],
          temperature: 0.2,
          response_format: { type: "json_object" },
          max_tokens: 8000,
        }),
      });
      if (!response.ok) {
        const errText = await response.text();
        console.error(`[fact-dag][${reqId}] openai error status=${response.status} body=${errText}`);
        res.status(response.status).json({ error: errText, vendor: "openai", model, reqId });
        return;
      }
      const data = await response.json();
      const content = data?.choices?.[0]?.message?.content ?? "";
      console.log(`[fact-dag][${reqId}] openai success`);
      res.status(200).json({ completion: content });
    }
  } catch (err: any) {
    console.error(`[fact-dag] unhandled error`, err);
    res.status(500).json({ error: err?.message || "Unknown error" });
  }
}

async function loadConfigFromYaml(): Promise<{ apiKey: string; vendor?: string; model?: string } | null> {
  try {
    const cfgPath = path.resolve(process.cwd(), "../config.yaml");
    if (!fs.existsSync(cfgPath)) return null;
    const content = await fs.promises.readFile(cfgPath, "utf8");
    const apiKeyMatch = content.match(/\bapi_key\s*:\s*([^\n#]+)/);
    const modelMatch = content.match(/\bmodel\s*:\s*([^\n#]+)/);
    const vendorMatch = content.match(/\bvendor\s*:\s*([^\n#]+)/);
    const apiKey = apiKeyMatch ? apiKeyMatch[1].trim() : "";
    const model = modelMatch ? modelMatch[1].trim() : undefined;
    const vendor = vendorMatch ? vendorMatch[1].trim() : undefined;
    if (!apiKey) return null;
    return { apiKey, vendor, model };
  } catch {
    return null;
  }
}

function normalizeAnthropicModel(input?: string): string {
  if (!input) return "claude-3-5-sonnet-latest";
  const s = input.trim();
  if (/^claude-3-5-sonnet$/i.test(s)) return "claude-3-5-sonnet-latest";
  if (/^claude-3-5-haiku$/i.test(s)) return "claude-3-5-haiku-latest";
  if (/^claude-3-5-opus$/i.test(s)) return "claude-3-5-opus-latest";
  return s;
}

function ensureLatestSuffix(model: string): string {
  return /-latest$/i.test(model) ? model : `${model}-latest`;
}

function buildAnthropicAttempts(preferred?: string): string[] {
  const base = normalizeAnthropicModel(preferred);
  const candidates = [
    base,
    ensureLatestSuffix(base),
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-5-opus-20240229",
    "claude-3-opus-20240229",
  ];
  // Deduplicate while preserving order
  return Array.from(new Set(candidates.filter(Boolean)));
}


