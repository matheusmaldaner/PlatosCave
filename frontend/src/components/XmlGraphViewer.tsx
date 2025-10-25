// frontend/src/components/XmlGraphViewer.tsx
import React, { useState, useEffect } from "react";
import ReactFlow, { Background, Node, Edge, MarkerType, useReactFlow } from "reactflow";
import "reactflow/dist/style.css";
import dagre from "dagre";

interface XmlGraphViewerProps {
  graphmlData: string | null;
  isDrawerOpen?: boolean;
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

const parseGraphML = (xml: string): { nodes: Node[]; edges: Edge[] } => {
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

    nodes.push({
      id,
      data: { label: `${role.toUpperCase()}: ${text}` },
      position: { x: 0, y: 0 },
      style: {
        width: nodeWidth,
        textAlign: "center",
        background: "#f0f0f0",
        border: "1px solid #333",
        borderRadius: "4px",
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
    });
  });

  return getLayoutedElements(nodes, edges);
};

const VerticalControls: React.FC<{ isDrawerOpen: boolean }> = ({ isDrawerOpen }) => {
  const { zoomIn, zoomOut, fitView } = useReactFlow();

  return (
    <div
      className={`absolute bottom-4 z-10 flex flex-col gap-2 bg-white/80 p-2 rounded-lg shadow-md transition-all duration-300`}
      style={{
        left: isDrawerOpen ? "21rem" : "1rem",
      }}
    >
      <button
        onClick={() => zoomIn()}
        className="px-2 py-1 bg-gray-100 rounded hover:bg-gray-200 transition"
        title="Zoom In"
      >
        🔍➕
      </button>
      <button
        onClick={() => zoomOut()}
        className="px-2 py-1 bg-gray-100 rounded hover:bg-gray-200 transition"
        title="Zoom Out"
      >
        🔍➖
      </button>
      <button
        onClick={() => fitView()}
        className="px-2 py-1 bg-gray-100 rounded hover:bg-gray-200 transition"
        title="Fit View"
      >
        ⛶
      </button>
    </div>
  );
};

const XmlGraphViewer: React.FC<XmlGraphViewerProps> = ({ graphmlData, isDrawerOpen = false }) => {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  useEffect(() => {
    if (graphmlData) {
      try {
        const { nodes: n, edges: e } = parseGraphML(graphmlData);
        setNodes(n);
        setEdges(e);
      } catch (err) {
        console.error("Failed to parse GraphML:", err);
      }
    } else {
      setNodes([]);
      setEdges([]);
    }
  }, [graphmlData]);

  const handleSaveGraph = () => {
    if (!graphmlData) return;
    const blob = new Blob([graphmlData], { type: "application/xml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "logic_graph.graphml";
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!graphmlData)
    return <div className="p-4 text-center">Waiting for logical graph to be generated...</div>;

  return (
    <div className="w-full h-full rounded-md relative bg-gray-50">
      <ReactFlow nodes={nodes} edges={edges} fitView>
        <Background />
        <VerticalControls isDrawerOpen={isDrawerOpen} />
      </ReactFlow>
    </div>
  );
};

export default XmlGraphViewer;
