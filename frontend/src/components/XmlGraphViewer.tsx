// frontend/src/components/XmlGraphViewer.tsx
import React, { useState, useEffect } from 'react';
import ReactFlow, { Controls, Background, Node, Edge, MarkerType } from 'reactflow';
import dagre from 'dagre';
import 'reactflow/dist/style.css';

interface XmlGraphViewerProps {
    graphmlData: string | null;
}

const dagreGraph = new dagre.graphlib.Graph();
dagreGraph.setDefaultEdgeLabel(() => ({}));

const nodeWidth = 200;
const nodeHeight = 50;

const getLayoutedElements = (nodes: Node[], edges: Edge[]) => {
    dagreGraph.setGraph({ rankdir: 'TB' });
    nodes.forEach((node) => {
        dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
    });
    edges.forEach((edge) => {
        dagreGraph.setEdge(edge.source, edge.target);
    });
    dagre.layout(dagreGraph);
    nodes.forEach((node) => {
        const nodeWithPosition = dagreGraph.node(node.id);
        node.position = {
            x: nodeWithPosition.x - nodeWidth / 2,
            y: nodeWithPosition.y - nodeHeight / 2,
        };
    });
    return { nodes, edges };
};

const parseGraphML = (xmlString: string): { nodes: Node[], edges: Edge[] } => {
    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(xmlString, "application/xml");
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    const nodeElements = xmlDoc.getElementsByTagName('node');
    const edgeElements = xmlDoc.getElementsByTagName('edge');

    Array.from(nodeElements).forEach(nodeEl => {
        const id = nodeEl.getAttribute('id')!;
        const dataElements = nodeEl.getElementsByTagName('data');
        const role = Array.from(dataElements).find(d => d.getAttribute('key') === 'd0')?.textContent || 'node';
        const text = Array.from(dataElements).find(d => d.getAttribute('key') === 'd2')?.textContent || id;
        
        nodes.push({
            id,
            data: { label: `${role.toUpperCase()}: ${text}` },
            position: { x: 0, y: 0 },
            style: { width: nodeWidth, textAlign: 'center', background: '#f0f0f0', borderColor: '#333' }
        });
    });

    Array.from(edgeElements).forEach(edgeEl => {
        const source = edgeEl.getAttribute('source')!;
        const target = edgeEl.getAttribute('target')!;
        edges.push({
            id: `e-${source}-${target}-${Math.random()}`,
            source,
            target,
            markerEnd: { type: MarkerType.ArrowClosed },
        });
    });

    return getLayoutedElements(nodes, edges);
};

const XmlGraphViewer: React.FC<XmlGraphViewerProps> = ({ graphmlData }) => {
    const [nodes, setNodes] = useState<Node[]>([]);
    const [edges, setEdges] = useState<Edge[]>([]);

    useEffect(() => {
        if (graphmlData) {
            try {
                const { nodes: layoutedNodes, edges: layoutedEdges } = parseGraphML(graphmlData);
                setNodes(layoutedNodes);
                setEdges(layoutedEdges);
            } catch (e) {
                console.error("Failed to parse GraphML data:", e);
            }
        } else {
            setNodes([]);
            setEdges([]);
        }
    }, [graphmlData]);

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
        return <div className="p-4 text-center">Waiting for logical graph to be generated...</div>;
    }

    return (
        // The `relative` class on this parent div is essential for positioning the button.
        <div className="w-full h-full rounded-md relative bg-gray-50">
            <ReactFlow nodes={nodes} edges={edges} fitView>
                <Background />
                <Controls />
            </ReactFlow>
            {/* The `absolute`, `bottom-4`, and `right-4` classes anchor the button to the bottom right. */}
            <button
                onClick={handleSaveGraph}
                className="absolute bottom-4 right-4 z-10 bg-brand-green text-white font-bold py-2 px-4 rounded-md shadow-lg hover:bg-brand-green-dark transition"
            >
                Save GraphML
            </button>
        </div>
    );
};

export default XmlGraphViewer;