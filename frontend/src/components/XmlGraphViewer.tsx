// frontend/src/components/XmlGraphViewer.tsx
import React, { useState, useEffect, useMemo, useRef } from 'react';
import ReactFlow, { Controls, Node, Edge, MarkerType, NodeProps, Handle, Position } from 'reactflow';
import { createPortal } from 'react-dom';
import dagre from 'dagre';
import 'reactflow/dist/style.css';

interface XmlGraphViewerProps {
    graphmlData: string | null;
}

type GraphNodeData = {
    role: string;
    text: string;
    cardWidth: number;
    credibility?: number;
    relevance?: number;
    evidence_strength?: number;
    method_rigor?: number;
    reproducibility?: number;
    citation_support?: number;
    verification_summary?: string;
    confidence_level?: string;
};

type GraphNode = Node<GraphNodeData>;
type GraphEdge = Edge;

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
    const hasMetrics = data.credibility !== undefined;
    const [isHovered, setIsHovered] = useState(false);
    const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
    const nodeRef = useRef<HTMLDivElement>(null);

    const handleMouseEnter = () => {
        if (nodeRef.current) {
            const rect = nodeRef.current.getBoundingClientRect();
            setTooltipPosition({
                x: rect.right + 16,
                y: rect.top
            });
            setIsHovered(true);
        }
    };

    const handleMouseLeave = () => {
        setIsHovered(false);
    };

    return (
        <>
            <div
                ref={nodeRef}
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
                className="relative"
            >
                <div
                    style={{ width: `${data.cardWidth}px` }}
                    className={`relative rounded-xl border ${palette.border} ${palette.bodyBg} p-3 shadow-sm hover:shadow-md transition-shadow`}
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
            </div>

            {/* Tooltip portal - renders outside ReactFlow */}
            {hasMetrics && isHovered && typeof document !== 'undefined' && createPortal(
                <div
                    style={{
                        position: 'fixed',
                        left: `${tooltipPosition.x}px`,
                        top: `${tooltipPosition.y}px`,
                        zIndex: 10000
                    }}
                    className="w-64 pointer-events-none animate-in fade-in duration-200"
                >
                    <div className="bg-white/90 backdrop-blur-xl rounded-lg shadow-xl border border-gray-200/60 p-3">
                        {/* Verification Summary */}
                        {data.verification_summary && (
                            <div className="mb-2.5 pb-2.5 border-b border-gray-200/60">
                                <div className="flex items-center justify-between mb-1">
                                    <h4 className="font-semibold text-gray-900 text-xs">Summary</h4>
                                    {data.confidence_level && (
                                        <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${
                                            data.confidence_level === 'high' ? 'bg-green-100/80 text-green-700' :
                                            data.confidence_level === 'medium' ? 'bg-yellow-100/80 text-yellow-700' :
                                            'bg-red-100/80 text-red-700'
                                        }`}>
                                            {data.confidence_level}
                                        </span>
                                    )}
                                </div>
                                <p className="text-gray-600 text-[10px] leading-relaxed">{data.verification_summary}</p>
                            </div>
                        )}

                        {/* Metrics */}
                        <h4 className="font-semibold text-gray-900 text-xs mb-2">Metrics</h4>
                        <div className="space-y-1.5">
                            <MetricBar label="Credibility" value={data.credibility!} />
                            <MetricBar label="Relevance" value={data.relevance!} />
                            <MetricBar label="Evidence" value={data.evidence_strength!} />
                            <MetricBar label="Method" value={data.method_rigor!} />
                            <MetricBar label="Reproducibility" value={data.reproducibility!} />
                            <MetricBar label="Citations" value={data.citation_support!} />
                        </div>

                        {/* Arrow pointing to node */}
                        <div className="absolute right-full top-4 mr-px">
                            <div className="border-[6px] border-transparent border-r-white/90" style={{ filter: 'drop-shadow(-1px 0px 1px rgba(0,0,0,0.08))' }}></div>
                        </div>
                    </div>
                </div>,
                document.body
            )}
        </>
    );
};

const MetricBar: React.FC<{ label: string; value: number }> = ({ label, value }) => {
    const percentage = Math.round(value * 100);
    const color = value >= 0.7 ? 'bg-green-500' : value >= 0.4 ? 'bg-yellow-500' : 'bg-red-500';
    const bgColor = value >= 0.7 ? 'bg-green-100/60' : value >= 0.4 ? 'bg-yellow-100/60' : 'bg-red-100/60';

    return (
        <div>
            <div className="flex justify-between items-center mb-0.5">
                <span className="text-gray-600 text-[9px] font-medium">{label}</span>
                <span className="text-gray-900 text-[9px] font-semibold">{percentage}%</span>
            </div>
            <div className={`h-1.5 ${bgColor} rounded-full overflow-hidden`}>
                <div className={`h-full ${color} transition-all duration-300 rounded-full`} style={{ width: `${percentage}%` }}></div>
            </div>
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

        // Parse verification metrics (d3-d10)
        const credibility = parseFloat(Array.from(dataElements).find(d => d.getAttribute('key') === 'd3')?.textContent || '');
        const relevance = parseFloat(Array.from(dataElements).find(d => d.getAttribute('key') === 'd4')?.textContent || '');
        const evidence_strength = parseFloat(Array.from(dataElements).find(d => d.getAttribute('key') === 'd5')?.textContent || '');
        const method_rigor = parseFloat(Array.from(dataElements).find(d => d.getAttribute('key') === 'd6')?.textContent || '');
        const reproducibility = parseFloat(Array.from(dataElements).find(d => d.getAttribute('key') === 'd7')?.textContent || '');
        const citation_support = parseFloat(Array.from(dataElements).find(d => d.getAttribute('key') === 'd8')?.textContent || '');
        const verification_summary = Array.from(dataElements).find(d => d.getAttribute('key') === 'd9')?.textContent || '';
        const confidence_level = Array.from(dataElements).find(d => d.getAttribute('key') === 'd10')?.textContent || '';

        nodes.push({
            id,
            type: 'graphNode',
            data: {
                role: formatRole(role.trim()),
                text: text.trim(),
                cardWidth: config.nodeWidth,
                ...(isNaN(credibility) ? {} : { credibility }),
                ...(isNaN(relevance) ? {} : { relevance }),
                ...(isNaN(evidence_strength) ? {} : { evidence_strength }),
                ...(isNaN(method_rigor) ? {} : { method_rigor }),
                ...(isNaN(reproducibility) ? {} : { reproducibility }),
                ...(isNaN(citation_support) ? {} : { citation_support }),
                ...(verification_summary ? { verification_summary } : {}),
                ...(confidence_level ? { confidence_level } : {}),
            },
            position: { x: 0, y: 0 },
            style: { width: config.nodeWidth },
            sourcePosition: Position.Bottom,
            targetPosition: Position.Top,
        });
    });

    Array.from(edgeElements).forEach((edgeEl, index) => {
        const source = edgeEl.getAttribute('source')!;
        const target = edgeEl.getAttribute('target')!;
        edges.push({
            id: edgeEl.getAttribute('id') || `edge-${source}-${target}-${index}`,
            source,
            target,
            type: 'smoothstep',
            markerEnd: { type: MarkerType.ArrowClosed, color: brandGreen, width: config.markerSize, height: config.markerSize },
            style: { stroke: brandGreen, strokeWidth: config.strokeWidth, opacity: 0.85 },
        });
    });

    return getLayoutedElements(nodes, edges, config);
};

const XmlGraphViewer: React.FC<XmlGraphViewerProps> = ({ graphmlData }) => {
    const [nodes, setNodes] = useState<GraphNode[]>([]);
    const [edges, setEdges] = useState<GraphEdge[]>([]);
    const [layoutConfig, setLayoutConfig] = useState<LayoutConfig>(LAYOUT_PRESETS.default);
    const [showFormula, setShowFormula] = useState(false);
    const nodeTypes = useMemo(() => ({ graphNode: GraphNodeCard }), []);
    const isCompact = layoutConfig.id === 'compact';

    useEffect(() => {
        if (typeof window === 'undefined') {
            return;
        }
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
        // The `relative` class on this parent div is essential for positioning the button.
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

            {/* Button group at bottom right */}
            <div className="absolute bottom-3 right-3 z-10 flex gap-2">
                {/* "?" button for formula */}
                <button
                    onClick={() => setShowFormula(true)}
                    className="rounded-lg border border-brand-green/40 bg-white/95 px-3 py-2 text-sm font-semibold text-brand-green shadow hover:bg-brand-green hover:text-white transition"
                    title="Show Integrity Score Formula"
                >
                    ?
                </button>

                {/* Save GraphML button */}
                <button
                    onClick={handleSaveGraph}
                    className="rounded-lg border border-brand-green/40 bg-white/95 px-4 py-2 text-sm font-semibold text-brand-green shadow hover:bg-brand-green hover:text-white transition"
                >
                    Save GraphML
                </button>
            </div>

            {/* Formula Modal */}
            {showFormula && (
                <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-50 p-6" onClick={() => setShowFormula(false)}>
                    <div className="bg-white rounded-2xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto p-8" onClick={(e) => e.stopPropagation()}>
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-2xl font-bold text-gray-900">Integrity Score Calculation</h2>
                            <button
                                onClick={() => setShowFormula(false)}
                                className="text-gray-400 hover:text-gray-600 transition text-2xl leading-none"
                            >
                                ×
                            </button>
                        </div>

                        <div className="space-y-6 text-sm">
                            {/* Main Formula */}
                            <div className="bg-gray-50 p-5 rounded-lg border border-gray-200">
                                <h3 className="font-semibold text-gray-900 mb-3">Graph-Level Score</h3>
                                <div className="font-mono text-xs bg-white p-4 rounded border border-gray-300 overflow-x-auto">
                                    <div className="whitespace-nowrap">
                                        score = (0.25 × bridge_coverage) +
                                    </div>
                                    <div className="whitespace-nowrap ml-8">
                                        (0.25 × best_path) +
                                    </div>
                                    <div className="whitespace-nowrap ml-8">
                                        (0.15 × redundancy) +
                                    </div>
                                    <div className="whitespace-nowrap ml-8">
                                        (-0.15 × fragility) +
                                    </div>
                                    <div className="whitespace-nowrap ml-8">
                                        (0.10 × coherence) +
                                    </div>
                                    <div className="whitespace-nowrap ml-8">
                                        (0.10 × coverage)
                                    </div>
                                    <div className="mt-2 whitespace-nowrap">
                                        score = clip(score, 0, 1)
                                    </div>
                                </div>
                            </div>

                            {/* Component Definitions */}
                            <div className="space-y-4">
                                <h3 className="font-semibold text-gray-900">Component Definitions</h3>

                                <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                                    <h4 className="font-semibold text-blue-900 mb-1">Bridge Coverage (Weight: 0.25)</h4>
                                    <p className="text-gray-700 mb-2">Ratio of nodes on any path from Hypothesis to Conclusion</p>
                                    <code className="text-xs bg-white px-2 py-1 rounded border border-blue-300">bridge_coverage = bridge_nodes / total_nodes</code>
                                </div>

                                <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                                    <h4 className="font-semibold text-green-900 mb-1">Best Path (Weight: 0.25)</h4>
                                    <p className="text-gray-700 mb-2">Maximum path strength from any Hypothesis to any Conclusion</p>
                                    <code className="text-xs bg-white px-2 py-1 rounded border border-green-300">best_path = max(product of edge confidences along paths)</code>
                                </div>

                                <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                                    <h4 className="font-semibold text-purple-900 mb-1">Redundancy (Weight: 0.15)</h4>
                                    <p className="text-gray-700 mb-2">Network flow capacity showing alternative paths (soft-capped at 1.0)</p>
                                    <code className="text-xs bg-white px-2 py-1 rounded border border-purple-300">redundancy = min(max_flow / 3.0, 1.0)</code>
                                </div>

                                <div className="bg-red-50 p-4 rounded-lg border border-red-200">
                                    <h4 className="font-semibold text-red-900 mb-1">Fragility (Weight: -0.15, penalty)</h4>
                                    <p className="text-gray-700 mb-2">Minimum cut value using inverse confidence as edge capacities</p>
                                    <code className="text-xs bg-white px-2 py-1 rounded border border-red-300">fragility = clip(cut_value / bridge_edges, 0, 1)</code>
                                </div>

                                <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
                                    <h4 className="font-semibold text-yellow-900 mb-1">Coherence (Weight: 0.10)</h4>
                                    <p className="text-gray-700 mb-2">Fraction of bridge edges with strong role transitions (role_prior ≥ 0.5)</p>
                                    <code className="text-xs bg-white px-2 py-1 rounded border border-yellow-300">coherence = strong_role_edges / total_bridge_edges</code>
                                </div>

                                <div className="bg-teal-50 p-4 rounded-lg border border-teal-200">
                                    <h4 className="font-semibold text-teal-900 mb-1">Coverage (Weight: 0.10)</h4>
                                    <p className="text-gray-700 mb-2">Presence of key roles (Method, Evidence, Result) on bridge</p>
                                    <code className="text-xs bg-white px-2 py-1 rounded border border-teal-300">coverage = roles_present ∩ {"{"} Method, Evidence, Result {"}"} / 3</code>
                                </div>
                            </div>

                            {/* Note */}
                            <div className="bg-gray-100 p-4 rounded-lg border border-gray-300">
                                <p className="text-xs text-gray-600">
                                    <strong>Note:</strong> Edge confidence is computed from node quality metrics (credibility, relevance, evidence strength, method rigor, reproducibility, citation support),
                                    role transition priors, text alignment, and pairwise synergy. Each component is calculated using graph algorithms like dynamic programming, network flow, and minimum cut.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default XmlGraphViewer;