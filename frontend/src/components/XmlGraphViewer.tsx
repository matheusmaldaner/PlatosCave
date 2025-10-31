// frontend/src/components/XmlGraphViewer.tsx
import React, { useState, useEffect, useMemo } from 'react';
import ReactFlow, { Controls, Node, Edge, MarkerType, NodeProps, Handle, Position } from 'reactflow';
import dagre from 'dagre';
import 'reactflow/dist/style.css';

interface XmlGraphViewerProps {
    graphmlData: string | null;
}

type GraphNodeData = {
    role: string;
    text: string;
    cardWidth: number;
};

type GraphNode = Node<GraphNodeData>;
type GraphEdgeData = {
    value?: number;
};
type GraphEdge = Edge<GraphEdgeData>;

const roleStyles: Record<string, { badgeBg: string; badgeText: string; bodyBg: string; border: string }> = {
    context: { badgeBg: 'bg-emerald-50', badgeText: 'text-emerald-700', bodyBg: 'bg-white', border: 'border-emerald-200' },
    claim: { badgeBg: 'bg-green-50', badgeText: 'text-green-700', bodyBg: 'bg-white', border: 'border-green-200' },
    evidence: { badgeBg: 'bg-sky-50', badgeText: 'text-sky-700', bodyBg: 'bg-white', border: 'border-sky-200' },
    hypothesis: { badgeBg: 'bg-teal-50', badgeText: 'text-teal-700', bodyBg: 'bg-white', border: 'border-teal-200' },
    result: { badgeBg: 'bg-lime-50', badgeText: 'text-lime-700', bodyBg: 'bg-white', border: 'border-lime-200' },
    method: { badgeBg: 'bg-cyan-50', badgeText: 'text-cyan-700', bodyBg: 'bg-white', border: 'border-cyan-200' },
};

const GraphNodeCard: React.FC<NodeProps<GraphNodeData>> = ({ data }) => {
    const roleKey = data.role.toLowerCase();
    const palette = roleStyles[roleKey] || { badgeBg: 'bg-gray-50', badgeText: 'text-gray-700', bodyBg: 'bg-white', border: 'border-gray-200' };

    return (
        <div
            style={{ width: `${data.cardWidth}px` }}
            className={`relative rounded-xl border ${palette.border} ${palette.bodyBg} p-3 shadow-sm`}
        >
            <Handle type="target" position={Position.Top} className="h-2 w-2 rounded-full bg-brand-green" />
            <p className={`inline-flex items-center rounded-full px-2 py-1 text-[10px] font-semibold uppercase tracking-wide ${palette.badgeBg} ${palette.badgeText}`}>
                {data.role}
            </p>
            <p className="mt-2 text-sm leading-relaxed text-gray-700 whitespace-pre-line">
                {data.text}
            </p>
            <Handle type="source" position={Position.Bottom} className="h-2 w-2 rounded-full bg-brand-green" />
        </div>
    );
};

const dagreGraph = new dagre.graphlib.Graph();
dagreGraph.setDefaultEdgeLabel(() => ({}));

type LayoutConfig = {
    id: 'default' | 'compact';
    nodeWidth: number;
    nodeHeight: number;
    nodesep: number;
    ranksep: number;
    margin: number;
    markerSize: number;
    strokeWidth: number;
};

const LAYOUT_PRESETS: Record<'default' | 'compact', LayoutConfig> = {
    default: {
        id: 'default',
        nodeWidth: 240,
        nodeHeight: 140,
        nodesep: 70,
        ranksep: 100,
        margin: 40,
        markerSize: 10,
        strokeWidth: 1.8,
    },
    compact: {
        id: 'compact',
        nodeWidth: 200,
        nodeHeight: 120,
        nodesep: 48,
        ranksep: 70,
        margin: 28,
        markerSize: 9,
        strokeWidth: 1.6,
    },
};

const brandGreen = '#2F855A';
const formatRole = (role: string) => role.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

const getLayoutedElements = (nodes: GraphNode[], edges: GraphEdge[], config: LayoutConfig) => {
    dagreGraph.setGraph({ rankdir: 'TB', nodesep: config.nodesep, ranksep: config.ranksep, marginx: config.margin, marginy: config.margin });
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

const parseGraphML = (xmlString: string, config: LayoutConfig): { nodes: GraphNode[], edges: GraphEdge[] } => {
    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(xmlString, "application/xml");
    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];
    const nodeElements = xmlDoc.getElementsByTagName('node');
    const edgeElements = xmlDoc.getElementsByTagName('edge');

    Array.from(nodeElements).forEach(nodeEl => {
        const id = nodeEl.getAttribute('id')!;
        const dataElements = nodeEl.getElementsByTagName('data');
        const role = Array.from(dataElements).find(d => d.getAttribute('key') === 'd0')?.textContent || 'node';
        const text = Array.from(dataElements).find(d => d.getAttribute('key') === 'd2')?.textContent || id;
        
        nodes.push({
            id,
            type: 'graphNode',
            data: { role: formatRole(role.trim()), text: text.trim(), cardWidth: config.nodeWidth },
            position: { x: 0, y: 0 },
            style: { width: config.nodeWidth },
            sourcePosition: Position.Bottom,
            targetPosition: Position.Top,
        });
    });

    Array.from(edgeElements).forEach((edgeEl, index) => {
        const source = edgeEl.getAttribute('source')!;
        const target = edgeEl.getAttribute('target')!;

        const dataElements = edgeEl.getElementsByTagName('data');
        const valueText = Array.from(dataElements).find(d => d.getAttribute('key') === 'weight')?.textContent;
        const value = valueText ? parseFloat(valueText) : 1;

        edges.push({
            id: edgeEl.getAttribute('id') || `edge-${source}-${target}-${index}`,
            source,
            target,
            type: 'smoothstep',
            data: { value },
            label: value.toFixed(2),
            labelStyle: { fill: brandGreen, fontSize: 11, fontWeight: 600 },
            labelBgStyle: { fill: 'white', fillOpacity: 0.7 },
            labelShowBg: true,
            markerEnd: { type: MarkerType.ArrowClosed, color: brandGreen, width: config.markerSize, height: config.markerSize },
            style: { stroke: brandGreen, strokeWidth: Math.max(1.2, value * 2), opacity: 0.85 },
        });
    });

    return getLayoutedElements(nodes, edges, config);
};

const XmlGraphViewer: React.FC<XmlGraphViewerProps> = ({ graphmlData }) => {
    const [nodes, setNodes] = useState<GraphNode[]>([]);
    const [edges, setEdges] = useState<GraphEdge[]>([]);
    const [layoutConfig, setLayoutConfig] = useState<LayoutConfig>(LAYOUT_PRESETS.default);
    const nodeTypes = useMemo(() => ({ graphNode: GraphNodeCard }), []);
    const isCompact = layoutConfig.id === 'compact';

    useEffect(() => {
        if (typeof window === 'undefined') return;
        const updateLayout = () => {
            const width = window.innerWidth;
            setLayoutConfig(width < 768 ? LAYOUT_PRESETS.compact : LAYOUT_PRESETS.default);
        };
        updateLayout();
        window.addEventListener('resize', updateLayout);
        return () => window.removeEventListener('resize', updateLayout);
    }, []);

    useEffect(() => {
        if (graphmlData) {
            try {
                const { nodes: layoutedNodes, edges: layoutedEdges } = parseGraphML(graphmlData, layoutConfig);
                setNodes(layoutedNodes);
                setEdges(layoutedEdges);
            } catch (e) {
                console.error("Failed to parse GraphML data:", e);
            }
        } else {
            setNodes([]);
            setEdges([]);
        }
    }, [graphmlData, layoutConfig]);

    const handleSaveGraph = () => {
        if (!graphmlData) return;
        const blob = new Blob([graphmlData], { type: 'application/xml' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'logic_graph.graphml';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    if (!graphmlData) {
        return (
            <div className="flex h-full w-full items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-white/60 p-6 text-center text-sm text-gray-500">
                Waiting for the knowledge graph to be generated...
            </div>
        );
    }

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
                panOnDrag={true}
                zoomOnPinch
                nodesDraggable={false}
                nodesConnectable={false}
            >
                <Controls />
            </ReactFlow>

            <button
                onClick={handleSaveGraph}
                className="absolute bottom-3 right-3 z-10 rounded-lg border border-brand-green/40 bg-white/95 px-4 py-2 text-sm font-semibold text-brand-green shadow hover:bg-brand-green hover:text-white transition"
            >
                Save GraphML
            </button>
        </div>
    );
};

export default XmlGraphViewer;

