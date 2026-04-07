import React, { useMemo } from "react";

type NodeDetail = {
  nodeId: string;
  role: string;
  text: string;
  integrityScorePct: number | null;
};

const normalizeRole = (role: string) => role.trim().toLowerCase();

const ROLE_BADGE: Record<string, { bg: string; text: string }> = {
  evidence: { bg: "#efe8fb", text: "#3f3f46" },
  claim: { bg: "#eef6f7", text: "#3f3f46" },
  result: { bg: "#f3f3f2", text: "#3f3f46" },
  methodology: { bg: "#f3f3f2", text: "#3f3f46" },
  method: { bg: "#f3f3f2", text: "#3f3f46" },
  hypothesis: { bg: "#f3f3f2", text: "#3f3f46" },
  context: { bg: "#f3f3f2", text: "#3f3f46" },
};

const splitBodyAndMeta = (text: string) => {
  const raw = (text || "").trim();
  if (!raw) return { body: "", meta: "" };
  const lines = raw.split("\n").map((l) => l.trim()).filter(Boolean);
  if (lines.length <= 1) return { body: raw, meta: "" };

  const last = lines[lines.length - 1];
  const isMeta =
    /\b(19|20)\d{2}\b/.test(last) ||
    /\bet\s+al\.\b/i.test(last) ||
    /vol\.|press/i.test(last) ||
    /^fig\.?\s*\d+/i.test(last) ||
    /inferred from/i.test(last);

  if (!isMeta) return { body: raw, meta: "" };
  return { body: lines.slice(0, -1).join("\n"), meta: last };
};

const makeTitle = (body: string) => {
  const t = (body || "").trim();
  if (!t) return "Node";
  // Prefer a "title-ish" first line/sentence, then fall back to truncation.
  const firstLine = t.split("\n")[0].trim();
  const firstSentence = firstLine.split(/(?<=[.!?])\s+/)[0].trim();
  const base = firstSentence || firstLine;
  return base.length > 64 ? `${base.slice(0, 64).trim()}…` : base;
};

const clampPct = (n: number) => Math.max(0, Math.min(100, Math.round(n)));

const NodeDetailPanel: React.FC<{
  isOpen: boolean;
  node: NodeDetail | null;
  onClose: () => void;
  onViewSource: () => void;
}> = ({ isOpen, node, onClose, onViewSource }) => {
  const roleKey = useMemo(() => normalizeRole(node?.role || "context"), [node?.role]);
  const badge = ROLE_BADGE[roleKey] || ROLE_BADGE.context;
  const { body } = useMemo(() => splitBodyAndMeta(node?.text || ""), [node?.text]);
  const title = useMemo(() => makeTitle(body), [body]);
  const score = node?.integrityScorePct ?? null;
  const scorePct = score == null ? null : clampPct(score);

  if (!isOpen || !node) return null;

  return (
    <aside
      className="absolute right-0 top-0 h-full border-l border-gray-200 bg-[#f8f7f4]"
      style={{
        fontFamily: '"DM Sans", sans-serif',
        width: "var(--detail-panel-w, 360px)",
        boxSizing: "border-box",
      }}
    >
      <div className="flex h-full flex-col">
        {/* Top bar */}
        <div className="flex items-center justify-between px-6 pt-5">
          <span
            className="inline-flex items-center rounded px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.16em]"
            style={{ background: badge.bg, color: badge.text }}
          >
            {node.role}
          </span>
          <button
            onClick={onClose}
            className="rounded p-2 text-gray-400 hover:bg-white/70 hover:text-gray-700 transition-colors"
            aria-label="Close"
            title="Close"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto px-6 pb-6">
          <h2
            className="mt-3 text-[22px] leading-tight text-gray-900"
            style={{ fontFamily: '"EB Garamond", Georgia, serif', fontWeight: 600 }}
          >
            {title}
          </h2>

          {/* Integrity score */}
          <div className="mt-7">
            <div className="flex items-center justify-between text-[11px] tracking-wide text-gray-600">
              <span>Integrity Score</span>
              <span className="font-medium text-gray-800">{scorePct == null ? "—" : `${scorePct}%`}</span>
            </div>
            <div className="mt-2 h-[6px] w-full rounded-full bg-gray-200">
              <div
                className="h-[6px] rounded-full bg-[#1f7898]"
                style={{ width: `${scorePct == null ? 0 : scorePct}%` }}
              />
            </div>
          </div>

          {/* Extracted source */}
          <div className="mt-7">
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-gray-400">
              Extracted Source
            </div>
            <div className="mt-3 border-l-2 border-[#1f7898] pl-4">
              <p
                className="text-[13px] leading-relaxed text-gray-700 italic"
                style={{ fontFamily: '"EB Garamond", Georgia, serif' }}
              >
                “{body || "—"}”
              </p>
            </div>
          </div>

          {/* Source metadata (placeholder until backend provides fields) */}
          <div className="mt-8">
            <div className="flex items-center gap-3">
              <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-gray-400">
                Source Metadata
              </div>
              <div className="h-px flex-1 bg-gray-200" />
            </div>

            <div className="mt-4 grid grid-cols-2 gap-x-8 gap-y-5">
              <div>
                <div className="text-[9px] font-semibold uppercase tracking-[0.18em] text-gray-400">Author</div>
                <div className="mt-1 text-[12px] text-gray-900">—</div>
              </div>
              <div>
                <div className="text-[9px] font-semibold uppercase tracking-[0.18em] text-gray-400">Published</div>
                <div className="mt-1 text-[12px] text-gray-900">—</div>
              </div>
              <div>
                <div className="text-[9px] font-semibold uppercase tracking-[0.18em] text-gray-400">Publisher</div>
                <div className="mt-1 text-[12px] text-gray-900">—</div>
              </div>
              <div>
                <div className="text-[9px] font-semibold uppercase tracking-[0.18em] text-gray-400">Bias Rating</div>
                <div className="mt-1 text-[12px] text-gray-900">—</div>
              </div>
            </div>

            <p className="mt-8 text-[11px] leading-relaxed text-gray-400">
              Node ID: <span className="font-mono text-gray-500">{node.nodeId}</span>
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 bg-[#f8f7f4] p-5">
          <button
            onClick={onViewSource}
            className="w-full rounded-md bg-[#1f7898] px-4 py-3 text-xs font-semibold tracking-[0.18em] text-white shadow-sm hover:bg-[#1a6a87] transition-colors"
          >
            <span className="inline-flex items-center justify-center gap-2">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M14 3h7v7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                <path d="M21 3l-9 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                <path d="M10 7H7a4 4 0 00-4 4v6a4 4 0 004 4h6a4 4 0 004-4v-3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              VIEW ORIGINAL SOURCE
            </span>
          </button>
        </div>
      </div>
    </aside>
  );
};

export default NodeDetailPanel;

