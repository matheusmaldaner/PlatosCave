import React, { useState, useEffect, useCallback, useMemo, useRef } from "react";
import ReactFlow, {
  Node, Edge, MarkerType, NodeProps, Handle, Position,
  useReactFlow, NodeToolbar, useNodesState, useEdgesState
} from "reactflow";
import dagre from "dagre";
import "reactflow/dist/style.css";
import { Plus, Minus, Maximize2 } from "lucide-react";

interface XmlGraphViewerProps {
  graphmlData: string | null;
  isDrawerOpen?: boolean;
  activeNodeId?: string | null;
  edgeUpdates?: Record<string, number>;
  verifiedNodeIds?: Set<string>;
  onBrowserClick?: () => void;
  onNodeSelect?: (payload: {
    nodeId: string;
    role: string;
    text: string;
    integrityScorePct: number | null;
  }) => void;
}

type GraphNodeData = {
  role: string;
  text: string;
  cardWidth: number;
  isActive?: boolean;
  isVerified?: boolean;
  onBrowserClick?: () => void;
};

type GraphNode = Node<GraphNodeData>;
type GraphEdgeData = { value?: number };
type GraphEdge = Edge<GraphEdgeData>;

/* ── academic palette ── */

const ROLE_THEME: Record<
  string,
  { border: string; shadow: string; surface?: string; label: string; meta: string }
> = {
  claim: {
    border: "#2f7f86",
    shadow: "0 10px 22px rgba(0,0,0,0.06)",
    label: "#a3a3a3",
    meta: "#a3a3a3",
  },
  evidence: {
    border: "#e7dcf6",
    shadow: "0 10px 22px rgba(0,0,0,0.06)",
    surface: "#fbf9ff",
    label: "#a3a3a3",
    meta: "#a3a3a3",
  },
  hypothesis: {
    border: "#e7e5e4",
    shadow: "0 10px 22px rgba(0,0,0,0.06)",
    label: "#a3a3a3",
    meta: "#a3a3a3",
  },
  result: {
    border: "#e7e5e4",
    shadow: "0 10px 22px rgba(0,0,0,0.06)",
    label: "#a3a3a3",
    meta: "#a3a3a3",
  },
  method: {
    border: "#e7e5e4",
    shadow: "0 10px 22px rgba(0,0,0,0.06)",
    label: "#a3a3a3",
    meta: "#a3a3a3",
  },
  methodology: {
    border: "#e7e5e4",
    shadow: "0 10px 22px rgba(0,0,0,0.06)",
    label: "#a3a3a3",
    meta: "#a3a3a3",
  },
  context: {
    border: "#e7e5e4",
    shadow: "0 10px 22px rgba(0,0,0,0.06)",
    label: "#a3a3a3",
    meta: "#a3a3a3",
  },
};

const edgeColor = "#2b2b2b";
const edgeLabelColor = "#2b2b2b";

const normalizeRole = (role: string) => role.trim().toLowerCase();

const roleKeyForTheme = (role: string) => {
  // Keep roles in sync with the old mapping (no new role names).
  const r = normalizeRole(role);
  if (r === "methods") return "method";
  if (r === "methodology") return "methodology";
  if (r === "method") return "method";
  if (r === "claim") return "claim";
  if (r === "evidence") return "evidence";
  if (r === "hypothesis") return "hypothesis";
  if (r === "result") return "result";
  if (r === "context") return "context";
  return "context";
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
    /vol\.|journal|conference|press/i.test(last) ||
    /^fig\.?\s*\d+/i.test(last) ||
    /inferred from/i.test(last);

  if (!isMeta) return { body: raw, meta: "" };
  return { body: lines.slice(0, -1).join("\n"), meta: last };
};

/* ── node card ── */

const GraphNodeCard: React.FC<NodeProps<GraphNodeData>> = ({ data }) => {
  const roleKey = roleKeyForTheme(data.role);
  const theme = ROLE_THEME[roleKey] || ROLE_THEME.context;
  const isEvidence = roleKey === "evidence";
  const { body, meta } = splitBodyAndMeta(data.text);

  return (
    <>
      <NodeToolbar isVisible={data.isActive} position={Position.Top} style={{ pointerEvents: "auto" }}>
        <button
          onClick={(e) => { e.stopPropagation(); e.preventDefault(); data.onBrowserClick?.(); }}
          className="flex items-center gap-2 px-3 py-2 bg-white border border-gray-200 rounded-md shadow-sm hover:shadow transition-all duration-150 cursor-pointer"
          style={{ pointerEvents: "auto", fontFamily: '"DM Sans", sans-serif' }}
        >
          <div className="w-1.5 h-1.5 rounded-full bg-teal-600 animate-pulse" />
          <span className="font-medium text-gray-600 text-xs tracking-wide">Verifying</span>
          <svg className="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
          </svg>
        </button>
      </NodeToolbar>

      <div
        style={{
          width: `${data.cardWidth}px`,
          fontFamily: '"DM Sans", sans-serif',
          borderColor: data.isActive ? theme.border : "#efefef",
          background: theme.surface || "#ffffff",
          boxShadow: data.isActive ? theme.shadow : "0 8px 18px rgba(0,0,0,0.05)",
        }}
        className={`relative rounded-lg transition-all duration-300 ${
          data.isActive
            ? "border-2 shadow-md"
            : data.isVerified
            ? "border shadow-sm"
            : "border shadow-sm"
        }`}
      >
        {/* Left role color slice (full height) */}
        <div
          className="absolute left-0 top-0 h-full w-[6px] rounded-l-lg"
          style={{ background: theme.border }}
          aria-hidden="true"
        />
        <Handle
          type="target"
          position={Position.Top}
          style={{
            width: 6, height: 6,
            background: "#d1d5db",
            border: "none",
            top: -3,
          }}
        />

        {/* Card body */}
        <div className="p-4 pl-5">
          {/* Role label + verified badge row */}
          <div className="flex items-center justify-between mb-2.5">
            <span
              className="text-[10px] font-semibold uppercase tracking-[0.12em]"
              style={{ color: theme.label }}
            >
              {data.role}
            </span>
            {data.isActive && (
              <button
                aria-label="Node options"
                className="text-gray-400 hover:text-gray-600 transition-colors"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                }}
                style={{ lineHeight: 1 }}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                  <circle cx="5" cy="12" r="1.6" />
                  <circle cx="12" cy="12" r="1.6" />
                  <circle cx="19" cy="12" r="1.6" />
                </svg>
              </button>
            )}
            {data.isVerified && (
              <span className="inline-flex items-center gap-1 text-[10px] font-medium text-green-700 animate-fade-in">
                <svg className="w-3 h-3 text-green-600" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                Verified
              </span>
            )}
          </div>

          {/* Node text */}
          {isEvidence ? (
            <p
              className="text-[16px] leading-relaxed text-gray-600 italic"
              style={{ fontFamily: '"EB Garamond", Georgia, serif' }}
            >
              &ldquo;{body}&rdquo;
            </p>
          ) : (
            <p
              className="text-[16px] leading-relaxed text-gray-900"
              style={{ fontFamily: '"EB Garamond", Georgia, serif' }}
            >
              {body}
            </p>
          )}

          {meta && (
            <div className="mt-3 flex items-center gap-2 text-[10.5px]" style={{ color: theme.meta }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M6 4h9a3 3 0 013 3v13H9a3 3 0 00-3 3V4z" opacity="0.25" />
                <path d="M6 4h10a2 2 0 012 2v14H9a3 3 0 00-3 3V4zm3 6h7v2H9v-2zm0 4h7v2H9v-2z" />
              </svg>
              <span className="truncate">{meta}</span>
            </div>
          )}
        </div>

        <Handle
          type="source"
          position={Position.Bottom}
          style={{
            width: 6, height: 6,
            background: "#d1d5db",
            border: "none",
            bottom: -3,
          }}
        />
      </div>
    </>
  );
};

/* ── layout ── */

type LayoutConfig = {
  id: "default" | "compact";
  nodeWidth: number;
  nodeHeight: number;
  nodesep: number;
  ranksep: number;
  margin: number;
  markerSize: number;
  strokeWidth: number;
};

const LAYOUT_PRESETS: Record<"default" | "compact", LayoutConfig> = {
  default: { id: "default", nodeWidth: 280, nodeHeight: 150, nodesep: 80, ranksep: 110, margin: 50, markerSize: 8, strokeWidth: 1 },
  compact: { id: "compact", nodeWidth: 220, nodeHeight: 130, nodesep: 50, ranksep: 80, margin: 30, markerSize: 7, strokeWidth: 1 },
};

const formatRole = (role: string) => role.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

const getLayoutedElements = (nodes: GraphNode[], edges: GraphEdge[], config: LayoutConfig) => {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB", nodesep: config.nodesep, ranksep: config.ranksep, marginx: config.margin, marginy: config.margin });
  nodes.forEach((node) => g.setNode(node.id, { width: config.nodeWidth, height: config.nodeHeight }));
  edges.forEach((edge) => g.setEdge(edge.source, edge.target));
  dagre.layout(g);
  nodes.forEach((node) => {
    const pos = g.node(node.id);
    node.position = { x: pos.x - config.nodeWidth / 2, y: pos.y - config.nodeHeight / 2 };
  });
  return { nodes, edges };
};

/** Build a styled edge. confidence === undefined means "pending". */
const makeEdge = (
  id: string, source: string, target: string,
  confidence: number | undefined, config: LayoutConfig
): GraphEdge => {
  const hasCon = confidence !== undefined;
  const pct = hasCon ? `${Math.round(confidence * 100)}%` : "";
  return {
    id,
    source,
    target,
    type: "smoothstep",
    data: { value: confidence },
    label: pct,
    labelStyle: {
      fill: edgeLabelColor,
      fontSize: 11,
      fontWeight: 500,
      fontFamily: '"DM Sans", sans-serif',
    },
    labelBgStyle: { fill: "#ffffff", fillOpacity: 0.92 },
    labelBgPadding: [6, 3] as [number, number],
    labelBgBorderRadius: 999,
    labelShowBg: hasCon,
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: edgeColor,
      width: config.markerSize,
      height: config.markerSize,
    },
    style: {
      stroke: edgeColor,
      strokeWidth: 1.1,
      opacity: 0.55,
    },
  };
};

const parseGraphML = (xmlString: string, config: LayoutConfig): { nodes: GraphNode[]; edges: GraphEdge[] } => {
  const parser = new DOMParser();
  const xmlDoc = parser.parseFromString(xmlString, "application/xml");
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];

  Array.from(xmlDoc.getElementsByTagName("node")).forEach((nodeEl) => {
    const id = nodeEl.getAttribute("id")!;
    const dataEls = nodeEl.getElementsByTagName("data");
    const role = Array.from(dataEls).find((d) => d.getAttribute("key") === "d0")?.textContent || "node";
    const text = Array.from(dataEls).find((d) => d.getAttribute("key") === "d2")?.textContent || id;
    nodes.push({
      id,
      type: "graphNode",
      data: { role: formatRole(role.trim()), text: text.trim(), cardWidth: config.nodeWidth },
      position: { x: 0, y: 0 },
      style: { width: config.nodeWidth },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
    });
  });

  Array.from(xmlDoc.getElementsByTagName("edge")).forEach((edgeEl, index) => {
    const source = edgeEl.getAttribute("source")!;
    const target = edgeEl.getAttribute("target")!;
    const dataEls = edgeEl.getElementsByTagName("data");
    const valueText = Array.from(dataEls).find((d) => d.getAttribute("key") === "weight")?.textContent;
    const value = valueText != null ? parseFloat(valueText) : undefined;
    edges.push(makeEdge(
      edgeEl.getAttribute("id") || `edge-${source}-${target}-${index}`,
      source, target, value, config
    ));
  });

  return getLayoutedElements(nodes, edges, config);
};

/* ── controls ── */

const GraphControls: React.FC<{ isDrawerOpen: boolean }> = ({ isDrawerOpen }) => {
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  return (
    <div
      className="absolute z-10 flex flex-col transition-all duration-300"
      style={{
        right: "1rem",
        bottom: "1rem",
        fontFamily: '"DM Sans", sans-serif',
      }}
    >
      <button
        onClick={() => zoomIn()}
        className="w-9 h-9 flex items-center justify-center bg-white border border-gray-200 rounded-t-md hover:bg-gray-50 text-gray-500 hover:text-gray-700 transition-colors shadow-sm"
        title="Zoom In"
      >
        <Plus size={16} strokeWidth={1.5} />
      </button>
      <button
        onClick={() => zoomOut()}
        className="w-9 h-9 flex items-center justify-center bg-white border-x border-gray-200 hover:bg-gray-50 text-gray-500 hover:text-gray-700 transition-colors shadow-sm"
        title="Zoom Out"
      >
        <Minus size={16} strokeWidth={1.5} />
      </button>
      <button
        onClick={() => fitView()}
        className="w-9 h-9 flex items-center justify-center bg-white border border-gray-200 rounded-b-md hover:bg-gray-50 text-gray-500 hover:text-gray-700 transition-colors shadow-sm"
        title="Fit View"
      >
        <Maximize2 size={14} strokeWidth={1.5} />
      </button>
    </div>
  );
};

/* ── main component ── */

const XmlGraphViewer: React.FC<XmlGraphViewerProps> = ({
  graphmlData,
  isDrawerOpen = false,
  activeNodeId = null,
  edgeUpdates = {},
  verifiedNodeIds = new Set(),
  onBrowserClick,
  onNodeSelect,
}) => {
  const [nodes, setNodes, onNodesChange] = useNodesState<GraphNodeData>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [layoutConfig, setLayoutConfig] = useState<LayoutConfig>(LAYOUT_PRESETS.default);
  const nodeTypes = useMemo(() => ({ graphNode: GraphNodeCard }), []);
  const isCompact = layoutConfig.id === "compact";

  // Keep a ref to the latest onBrowserClick so node data doesn't go stale
  const browserClickRef = useRef(onBrowserClick);
  browserClickRef.current = onBrowserClick;
  const stableBrowserClick = useCallback(() => browserClickRef.current?.(), []);

  // Responsive layout
  useEffect(() => {
    if (typeof window === "undefined") return;
    const update = () => setLayoutConfig(window.innerWidth < 768 ? LAYOUT_PRESETS.compact : LAYOUT_PRESETS.default);
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  // ─── Parse GraphML only when graphmlData or layoutConfig change ───
  useEffect(() => {
    if (!graphmlData) { setNodes([]); setEdges([]); return; }
    try {
      const { nodes: ln, edges: le } = parseGraphML(graphmlData, layoutConfig);
      setNodes(ln);
      setEdges(le);
    } catch (e) {
      console.error("Failed to parse GraphML data:", e);
    }
  }, [graphmlData, layoutConfig]);

  // ─── Update node active / verified / callback state ───
  useEffect(() => {
    setNodes((prev) =>
      prev.map((node) => {
        const numId = node.id.replace("n", "");
        const isActive = node.id === `n${activeNodeId}`;
        const isVerified = verifiedNodeIds.has(numId);
        if (
          node.data.isActive === isActive &&
          node.data.isVerified === isVerified &&
          node.data.onBrowserClick === stableBrowserClick
        ) return node;
        return { ...node, data: { ...node.data, isActive, isVerified, onBrowserClick: stableBrowserClick } };
      })
    );
  }, [activeNodeId, verifiedNodeIds, stableBrowserClick, setNodes]);

  // ─── Apply edge confidence updates in real-time ───
  useEffect(() => {
    const keys = Object.keys(edgeUpdates);
    if (keys.length === 0) return;

    setEdges((prev) =>
      prev.map((edge) => {
        const srcNum = edge.source.replace("n", "");
        const tgtNum = edge.target.replace("n", "");
        const edgeKey = `${srcNum}->${tgtNum}`;
        const confidence = edgeUpdates[edgeKey];
        if (confidence === undefined) return edge;
        if (edge.data?.value === confidence) return edge;

        const pct = `${Math.round(confidence * 100)}%`;
        return {
          ...edge,
          data: { ...edge.data, value: confidence },
          label: pct,
          labelStyle: {
            fill: edgeLabelColor,
            fontSize: 11,
            fontWeight: 500,
            fontFamily: '"DM Sans", sans-serif',
          },
          labelShowBg: true,
          style: {
            ...edge.style,
            strokeWidth: 1.1,
            opacity: 0.55,
          },
        };
      })
    );
  }, [edgeUpdates, setEdges]);

  const computeIntegrityScorePct = useCallback(
    (nodeId: string) => {
      const vals = edges
        .filter((e: any) => e?.source === nodeId || e?.target === nodeId)
        .map((e: any) => e?.data?.value)
        .filter((v: any) => typeof v === "number" && Number.isFinite(v)) as number[];
      if (vals.length === 0) return null;
      const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
      return Math.round(avg * 100);
    },
    [edges]
  );

  if (!graphmlData)
    return (
      <div
        className="flex h-full w-full items-center justify-center border border-gray-200 bg-[#f8f7f4] p-6 text-center"
        style={{ fontFamily: '"DM Sans", sans-serif' }}
      >
        <div className="max-w-md rounded-lg border border-gray-200 bg-white px-6 py-5 shadow-sm">
          <div className="mx-auto mb-3 h-6 w-6 rounded-full border-2 border-gray-200 border-t-gray-500 animate-spin" />
          <p className="text-sm text-gray-600 tracking-wide">
            Waiting for the knowledge graph to be generated
          </p>
          <p className="mt-1 text-xs text-gray-400">
            The canvas will populate automatically as soon as graph data arrives.
          </p>
        </div>
      </div>
    );

  return (
    <div
      className="relative flex h-full w-full flex-col overflow-hidden bg-[#f8f7f4] border border-gray-200"
      style={{ fontFamily: '"DM Sans", sans-serif' }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        onNodeClick={(_, node) => {
          if (!onNodeSelect) return;
          const integrityScorePct = computeIntegrityScorePct(node.id);
          onNodeSelect({
            nodeId: node.id,
            role: node.data?.role ?? "Context",
            text: node.data?.text ?? "",
            integrityScorePct,
          });
        }}
        fitView
        className="flex-1"
        fitViewOptions={{ padding: isCompact ? 0.3 : 0.25 }}
        minZoom={0.3}
        maxZoom={1.8}
        panOnDrag
        zoomOnPinch
        nodesDraggable={false}
        nodesConnectable={false}
        proOptions={{ hideAttribution: true }}
      >
        <GraphControls isDrawerOpen={isDrawerOpen} />
      </ReactFlow>
    </div>
  );
};

export default XmlGraphViewer;
