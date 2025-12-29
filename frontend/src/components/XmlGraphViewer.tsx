import React, { useState, useEffect, useMemo } from "react";
import ReactFlow, { Node, Edge, MarkerType, NodeProps, Handle, Position, useReactFlow, NodeToolbar } from "reactflow";
import dagre from "dagre";
import "reactflow/dist/style.css";
import { ZoomIn, ZoomOut, Maximize2 } from "lucide-react";

interface XmlGraphViewerProps {
  graphmlData: string | null;
  isDrawerOpen?: boolean;
  activeNodeId?: string | null;
  edgeUpdates?: Record<string, number>;  // Map of "source->target" to confidence value
  onBrowserClick?: () => void;
}

type GraphNodeData = {
  role: string;
  text: string;
  cardWidth: number;
  isActive?: boolean;
  onBrowserClick?: () => void;
};

type GraphNode = Node<GraphNodeData>;
type GraphEdgeData = {
  value?: number;
};
type GraphEdge = Edge<GraphEdgeData>;

const roleStyles: Record<string, { badgeBg: string; badgeText: string; bodyBg: string; border: string }> = {
  // Base roles
  context: { badgeBg: "bg-emerald-50", badgeText: "text-emerald-700", bodyBg: "bg-white", border: "border-emerald-200" },
  claim: { badgeBg: "bg-green-50", badgeText: "text-green-700", bodyBg: "bg-white", border: "border-green-200" },
  evidence: { badgeBg: "bg-sky-50", badgeText: "text-sky-700", bodyBg: "bg-white", border: "border-sky-200" },
  hypothesis: { badgeBg: "bg-teal-50", badgeText: "text-teal-700", bodyBg: "bg-white", border: "border-teal-200" },
  result: { badgeBg: "bg-lime-50", badgeText: "text-lime-700", bodyBg: "bg-white", border: "border-lime-200" },
  method: { badgeBg: "bg-cyan-50", badgeText: "text-cyan-700", bodyBg: "bg-white", border: "border-cyan-200" },
  conclusion: { badgeBg: "bg-indigo-50", badgeText: "text-indigo-700", bodyBg: "bg-white", border: "border-indigo-200" },
  assumption: { badgeBg: "bg-amber-50", badgeText: "text-amber-700", bodyBg: "bg-white", border: "border-amber-200" },
  counterevidence: { badgeBg: "bg-rose-50", badgeText: "text-rose-700", bodyBg: "bg-white", border: "border-rose-200" },
  limitation: { badgeBg: "bg-orange-50", badgeText: "text-orange-700", bodyBg: "bg-white", border: "border-orange-200" },
  // Journal mode roles
  peerreviewstatus: { badgeBg: "bg-violet-50", badgeText: "text-violet-700", bodyBg: "bg-white", border: "border-violet-200" },
  statisticalmethod: { badgeBg: "bg-fuchsia-50", badgeText: "text-fuchsia-700", bodyBg: "bg-white", border: "border-fuchsia-200" },
  replicationstatus: { badgeBg: "bg-purple-50", badgeText: "text-purple-700", bodyBg: "bg-white", border: "border-purple-200" },
  conflictofinterest: { badgeBg: "bg-red-50", badgeText: "text-red-700", bodyBg: "bg-white", border: "border-red-200" },
  samplesize: { badgeBg: "bg-pink-50", badgeText: "text-pink-700", bodyBg: "bg-white", border: "border-pink-200" },
  // Finance mode roles
  financialmetric: { badgeBg: "bg-emerald-50", badgeText: "text-emerald-700", bodyBg: "bg-white", border: "border-emerald-200" },
  forwardguidance: { badgeBg: "bg-blue-50", badgeText: "text-blue-700", bodyBg: "bg-white", border: "border-blue-200" },
  riskfactor: { badgeBg: "bg-red-50", badgeText: "text-red-700", bodyBg: "bg-white", border: "border-red-200" },
  regulatorycompliance: { badgeBg: "bg-slate-50", badgeText: "text-slate-700", bodyBg: "bg-white", border: "border-slate-200" },
  comparableanalysis: { badgeBg: "bg-zinc-50", badgeText: "text-zinc-700", bodyBg: "bg-white", border: "border-zinc-200" },
  managementdiscussion: { badgeBg: "bg-stone-50", badgeText: "text-stone-700", bodyBg: "bg-white", border: "border-stone-200" },
};

const GraphNodeCard: React.FC<NodeProps<GraphNodeData>> = ({ data }) => {
  const roleKey = data.role.toLowerCase();
  const palette = roleStyles[roleKey] || { badgeBg: "bg-gray-50", badgeText: "text-gray-700", bodyBg: "bg-white", border: "border-gray-200" };

  return (
    <>
      {/* Browser Active Toolbar - appears above node when active */}
      <NodeToolbar isVisible={data.isActive} position={Position.Top} style={{ pointerEvents: 'auto' }}>
        <button
          onClick={(e) => {
            e.stopPropagation();
            e.preventDefault();
            console.log('[DEBUG] Browser Active clicked, onBrowserClick:', data.onBrowserClick);
            if (data.onBrowserClick) {
              data.onBrowserClick();
            } else {
              console.error('[DEBUG] onBrowserClick is undefined!');
            }
          }}
          className="flex items-center gap-2 px-3 py-2 bg-white border border-gray-200 rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105 cursor-pointer"
          style={{ pointerEvents: 'auto' }}
        >
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="font-medium text-gray-700 text-sm">Browser Active</span>
          <svg
            className="w-4 h-4 text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
        </button>
      </NodeToolbar>

      {/* Node card with active state styling */}
      <div
        style={{ width: `${data.cardWidth}px` }}
        className={`relative rounded-xl border ${data.isActive ? 'border-emerald-400 ring-2 ring-emerald-200' : palette.border} ${palette.bodyBg} p-3 shadow-sm transition-all duration-300`}
      >
        <Handle type="target" position={Position.Top} className="h-2 w-2 rounded-full bg-brand-green" />
        <p className={`inline-flex items-center rounded-full px-2 py-1 text-[10px] font-semibold uppercase tracking-wide ${palette.badgeBg} ${palette.badgeText}`}>
          {data.role}
        </p>
        <p className="mt-2 text-sm leading-relaxed text-gray-700 whitespace-pre-line">{data.text}</p>
        <Handle type="source" position={Position.Bottom} className="h-2 w-2 rounded-full bg-brand-green" />
      </div>
    </>
  );
};

const dagreGraph = new dagre.graphlib.Graph();
dagreGraph.setDefaultEdgeLabel(() => ({}));

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
  default: {
    id: "default",
    nodeWidth: 240,
    nodeHeight: 140,
    nodesep: 70,
    ranksep: 100,
    margin: 40,
    markerSize: 10,
    strokeWidth: 1.8,
  },
  compact: {
    id: "compact",
    nodeWidth: 200,
    nodeHeight: 120,
    nodesep: 48,
    ranksep: 70,
    margin: 28,
    markerSize: 9,
    strokeWidth: 1.6,
  },
};

const brandGreen = "#2F855A";
// Format role: convert camelCase/PascalCase to "Title Case With Spaces"
// e.g., "PeerReviewStatus" → "Peer Review Status", "conflictOfInterest" → "Conflict Of Interest"
const formatRole = (role: string) => 
  role
    .replace(/_/g, " ")                           // Replace underscores with spaces
    .replace(/([a-z])([A-Z])/g, "$1 $2")         // Add space before uppercase letters (camelCase)
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")   // Handle acronyms like "XMLParser" → "XML Parser"
    .replace(/\b\w/g, (c) => c.toUpperCase());    // Capitalize first letter of each word

const getLayoutedElements = (nodes: GraphNode[], edges: GraphEdge[], config: LayoutConfig) => {
  dagreGraph.setGraph({ rankdir: "TB", nodesep: config.nodesep, ranksep: config.ranksep, marginx: config.margin, marginy: config.margin });
  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: config.nodeWidth, height: config.nodeHeight });
  });
  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });
  dagre.layout(dagreGraph);
  nodes.forEach((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    node.position = {
      x: nodeWithPosition.x - config.nodeWidth / 2,
      y: nodeWithPosition.y - config.nodeHeight / 2,
    };
  });
  return { nodes, edges };
};

const parseGraphML = (xmlString: string, config: LayoutConfig): { nodes: GraphNode[]; edges: GraphEdge[] } => {
  const parser = new DOMParser();
  const xmlDoc = parser.parseFromString(xmlString, "application/xml");
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const nodeElements = xmlDoc.getElementsByTagName("node");
  const edgeElements = xmlDoc.getElementsByTagName("edge");

  Array.from(nodeElements).forEach((nodeEl) => {
    const id = nodeEl.getAttribute("id")!;
    const dataElements = nodeEl.getElementsByTagName("data");
    const role = Array.from(dataElements).find((d) => d.getAttribute("key") === "d0")?.textContent || "node";
    const text = Array.from(dataElements).find((d) => d.getAttribute("key") === "d2")?.textContent || id;

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

  Array.from(edgeElements).forEach((edgeEl, index) => {
    const source = edgeEl.getAttribute("source")!;
    const target = edgeEl.getAttribute("target")!;
    const dataElements = edgeEl.getElementsByTagName("data");
    const valueText = Array.from(dataElements).find((d) => d.getAttribute("key") === "weight")?.textContent;
    const hasWeight = valueText !== null && valueText !== undefined;
    const value = hasWeight ? parseFloat(valueText) : undefined;

    edges.push({
      id: edgeEl.getAttribute("id") || `edge-${source}-${target}-${index}`,
      source,
      target,
      type: "smoothstep",
      data: { value },
      // Show actual value if present, otherwise show "—" to indicate pending
      label: hasWeight ? value!.toFixed(2) : "—",
      labelStyle: { fill: hasWeight ? brandGreen : "#9CA3AF", fontSize: 11, fontWeight: 600 },
      labelBgStyle: { fill: "white", fillOpacity: 0.7 },
      labelShowBg: true,
      markerEnd: { type: MarkerType.ArrowClosed, color: brandGreen, width: config.markerSize, height: config.markerSize },
      style: { stroke: brandGreen, strokeWidth: hasWeight ? Math.max(1.2, value! * 2) : 1.2, opacity: hasWeight ? 0.85 : 0.5 },
    });
  });

  return getLayoutedElements(nodes, edges, config);
};

const GraphControls: React.FC<{ isDrawerOpen: boolean }> = ({ isDrawerOpen }) => {
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  return (
    <div
      className="absolute z-10 flex flex-col gap-2 bg-white/80 p-2 rounded-lg shadow-md transition-all duration-300"
      style={{
        left: isDrawerOpen ? "21rem" : "1rem",
        bottom: isDrawerOpen ? "4rem" : "1rem", // ✅ slides upward when drawer opens
      }}
    >
      <button onClick={() => zoomIn()} className="p-2 bg-gray-100 rounded hover:bg-gray-200" title="Zoom In">
        <ZoomIn size={20} />
      </button>
      <button onClick={() => zoomOut()} className="p-2 bg-gray-100 rounded hover:bg-gray-200" title="Zoom Out">
        <ZoomOut size={20} />
      </button>
      <button onClick={() => fitView()} className="p-2 bg-gray-100 rounded hover:bg-gray-200" title="Fit View">
        <Maximize2 size={20} />
      </button>
    </div>
  );
};


const XmlGraphViewer: React.FC<XmlGraphViewerProps> = ({
  graphmlData,
  isDrawerOpen = false,
  activeNodeId = null,
  edgeUpdates = {},
  onBrowserClick
}) => {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [layoutConfig, setLayoutConfig] = useState<LayoutConfig>(LAYOUT_PRESETS.default);
  const nodeTypes = useMemo(() => ({ graphNode: GraphNodeCard }), []);
  const isCompact = layoutConfig.id === "compact";

  useEffect(() => {
    if (typeof window === "undefined") return;
    const updateLayout = () => {
      const width = window.innerWidth;
      setLayoutConfig(width < 768 ? LAYOUT_PRESETS.compact : LAYOUT_PRESETS.default);
    };
    updateLayout();
    window.addEventListener("resize", updateLayout);
    return () => window.removeEventListener("resize", updateLayout);
  }, []);

  useEffect(() => {
    if (graphmlData) {
      try {
        const { nodes: layoutedNodes, edges: layoutedEdges } = parseGraphML(graphmlData, layoutConfig);

        console.log('[DEBUG] XmlGraphViewer useEffect - activeNodeId:', activeNodeId, 'onBrowserClick defined:', !!onBrowserClick);

        // Mark active node and pass browser click callback
        const nodesWithActiveState = layoutedNodes.map(node => ({
          ...node,
          data: {
            ...node.data,
            isActive: node.id === `n${activeNodeId}`,
            onBrowserClick
          }
        }));

        setNodes(nodesWithActiveState);
        setEdges(layoutedEdges);
      } catch (e) {
        console.error("Failed to parse GraphML data:", e);
      }
    } else {
      setNodes([]);
      setEdges([]);
    }
  }, [graphmlData, layoutConfig, activeNodeId, onBrowserClick]);

  // Apply real-time edge updates as verification progresses
  useEffect(() => {
    if (Object.keys(edgeUpdates).length === 0) return;

    setEdges(prevEdges => prevEdges.map(edge => {
      // Convert node IDs to match the edgeUpdates format
      // Edge source/target are like "n0", "n1" - need to extract just the number
      const sourceNum = edge.source.replace('n', '');
      const targetNum = edge.target.replace('n', '');
      const edgeKey = `${sourceNum}->${targetNum}`;

      if (edgeUpdates[edgeKey] !== undefined) {
        const confidence = edgeUpdates[edgeKey];
        return {
          ...edge,
          data: { ...edge.data, value: confidence },
          label: confidence.toFixed(2),
          labelStyle: { fill: brandGreen, fontSize: 11, fontWeight: 600 },
          style: {
            ...edge.style,
            strokeWidth: Math.max(1.2, confidence * 2),
            opacity: 0.85,
          },
        };
      }
      return edge;
    }));
  }, [edgeUpdates]);

  const handleSaveGraph = () => {
    if (!graphmlData) return;
    const blob = new Blob([graphmlData], { type: "application/xml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "logic_graph.graphml";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (!graphmlData)
    return (
      <div className="flex h-full w-full items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-white/60 p-6 text-center text-sm text-gray-500">
        Waiting for the knowledge graph to be generated...
      </div>
    );

  return (
    <div className="relative flex h-full w-full flex-col overflow-hidden rounded-2xl border border-gray-100 bg-white shadow-sm">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        className="flex-1"
        fitViewOptions={{ padding: isCompact ? 0.3 : 0.2 }}
        minZoom={0.35}
        maxZoom={1.6}
        panOnDrag
        zoomOnPinch
        nodesDraggable={false}
        nodesConnectable={false}
      >
        <GraphControls isDrawerOpen={isDrawerOpen} />
      </ReactFlow>

      
    </div>
  );
};

export default XmlGraphViewer;
