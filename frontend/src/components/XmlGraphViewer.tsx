import React, { useState, useEffect } from "react";
import ReactFlow, { Background, Node, Edge, MarkerType, useReactFlow } from "reactflow";
import "reactflow/dist/style.css";
import dagre from "dagre";
import { Settings } from "./SettingsModal";
import { ZoomIn, ZoomOut, Maximize2 } from "lucide-react";

interface XmlGraphViewerProps {
  graphmlData: string | null;
  isDrawerOpen?: boolean;
  settings?: Settings;
  showTypeColors?: boolean;
}

const dagreGraph = new dagre.graphlib.Graph();
dagreGraph.setDefaultEdgeLabel(() => ({}));
const nodeWidth = 200;
const nodeHeight = 50;

const getLayoutedElements = (nodes: Node[], edges: Edge[]) => {
  dagreGraph.setGraph({ rankdir: "TB" });
  nodes.forEach((n) => dagreGraph.setNode(n.id, { width: nodeWidth, height: nodeHeight }));
  edges.forEach((e) => dagreGraph.setEdge(e.source, e.target));
  dagre.layout(dagreGraph);

  nodes.forEach((n) => {
    const pos = dagreGraph.node(n.id);
    n.position = { x: pos.x - nodeWidth / 2, y: pos.y - nodeHeight / 2 };
  });
  return { nodes, edges };
};

const getRoleColor = (role: string) => {
  switch (role.toLowerCase()) {
    case "hypothesis": return "#E53935"; // red
    case "claim": return "#FB8C00";      // orange
    case "method": return "#FDD835";     // yellow
    case "evidence": return "#43A047";   // green
    case "result": return "#1E88E5";     // blue
    case "conclusion": return "#5E35B1"; // indigo
    case "limitation": return "#8E24AA"; // violet
    default: return "#9E9E9E";
  }
};

const parseGraphML = (
  xml: string,
  settings?: Settings,
  showTypeColors?: boolean
): { nodes: Node[]; edges: Edge[] } => {
  const parser = new DOMParser();
  const doc = parser.parseFromString(xml, "application/xml");
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  Array.from(doc.getElementsByTagName("node")).forEach((el) => {
    const id = el.getAttribute("id")!;
    const dataEls = el.getElementsByTagName("data");
    const role =
      Array.from(dataEls).find((d) => d.getAttribute("key") === "d0")?.textContent || "node";
    const text =
      Array.from(dataEls).find((d) => d.getAttribute("key") === "d2")?.textContent || id;

    const borderColor = showTypeColors ? getRoleColor(role) : "#9E9E9E";

    nodes.push({
      id,
      data: { label: `${role.toUpperCase()}: ${text}` },
      position: { x: 0, y: 0 },
      style: {
        width: nodeWidth,
        height: nodeHeight,
        textAlign: "center",
        fontWeight: 600,
        background: "#fff",
        border: `3px solid ${borderColor}`,
        borderRadius: "8px",
        boxShadow: showTypeColors ? `0 0 8px ${borderColor}40` : "none",
        opacity: settings?.credibility ?? 1,
        transition: "all 0.3s ease",
      },
    });
  });

  Array.from(doc.getElementsByTagName("edge")).forEach((el) => {
    const source = el.getAttribute("source")!;
    const target = el.getAttribute("target")!;
    edges.push({
      id: `e-${source}-${target}-${Math.random()}`,
      source,
      target,
      markerEnd: { type: MarkerType.ArrowClosed },
      style: {
        strokeWidth: 1 + (settings?.evidenceStrength ?? 1) * 2,
        opacity: settings?.relevance ?? 1,
        stroke: "#999",
      },
    });
  });

  return getLayoutedElements(nodes, edges);
};

const GraphControls: React.FC<{ isDrawerOpen: boolean }> = ({ isDrawerOpen }) => {
  const { zoomIn, zoomOut, fitView } = useReactFlow();

  return (
    <div
      className="absolute bottom-4 z-10 flex flex-col gap-2 bg-white/80 p-2 rounded-lg shadow-md transition-all duration-300"
      style={{
        left: isDrawerOpen ? "21rem" : "1rem",
      }}
    >
      <button
        onClick={zoomIn}
        className="p-2 bg-gray-100 rounded hover:bg-gray-200 transition"
        title="Zoom In"
      >
        <ZoomIn size={20} />
      </button>
      <button
        onClick={zoomOut}
        className="p-2 bg-gray-100 rounded hover:bg-gray-200 transition"
        title="Zoom Out"
      >
        <ZoomOut size={20} />
      </button>
      <button
        onClick={fitView}
        className="p-2 bg-gray-100 rounded hover:bg-gray-200 transition"
        title="Fit View"
      >
        <Maximize2 size={20} />
      </button>
    </div>
  );
};

const XmlGraphViewer: React.FC<XmlGraphViewerProps> = ({
  graphmlData,
  isDrawerOpen = false,
  settings,
  showTypeColors = false,
}) => {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  useEffect(() => {
    if (graphmlData) {
      try {
        const { nodes: n, edges: e } = parseGraphML(graphmlData, settings, showTypeColors);
        setNodes(n);
        setEdges(e);
      } catch (err) {
        console.error("Failed to parse GraphML:", err);
      }
    } else {
      setNodes([]);
      setEdges([]);
    }
  }, [graphmlData, settings, showTypeColors]);

  if (!graphmlData)
    return <div className="p-4 text-center">Waiting for logical graph to be generated...</div>;

  return (
    <div className="w-full h-full rounded-md relative bg-gray-50">
      <ReactFlow nodes={nodes} edges={edges} fitView>
        <Background />
        <GraphControls isDrawerOpen={isDrawerOpen} />
      </ReactFlow>
    </div>
  );
};

export default XmlGraphViewer;
