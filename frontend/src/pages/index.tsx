// PlatosCave/frontend/src/pages/index.tsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { io, Socket } from 'socket.io-client';
import FileUploader from '../components/FileUploader';
import Sidebar, { ProcessStep } from '../components/Sidebar';
import XmlGraphViewer from '../components/XmlGraphViewer';

const INITIAL_STAGES: ProcessStep[] = [
    { name: "Validate", displayText: "Pending...", status: 'pending' },
    { name: "Decomposing PDF", displayText: "Pending...", status: 'pending' },
    { name: "Building Logic Tree", displayText: "Pending...", status: 'pending' },
    { name: "Organizing Agents", displayText: "Pending...", status: 'pending' },
    { name: "Compiling Evidence", displayText: "Pending...", status: 'pending' },
    { name: "Evaluating Integrity", displayText: "Pending...", status: 'pending' },
];

const IndexPage = () => {
    const [processSteps, setProcessSteps] = useState<ProcessStep[]>([]);
    const [uploadedFile, setUploadedFile] = useState<File | null>(null);
    const [viewMode, setViewMode] = useState<'uploader' | 'graph'>('uploader');
    const [finalScore, setFinalScore] = useState<number | null>(null);
    const [graphmlData, setGraphmlData] = useState<string | null>(null);

    useEffect(() => {
        if (!uploadedFile) return;
        const socket: Socket = io('http://localhost:5000');
        socket.on('connect', () => console.log('Connected to WebSocket server!'));

        socket.on('status_update', (msg: { data: string }) => {
            // --- DEBUG LOGGING 1: See the raw message from the server ---
            console.log("Received raw message from server:", msg.data);

            try {
                const update = JSON.parse(msg.data);

                if (update.type === 'UPDATE') {
                    setProcessSteps(prevSteps => {
                        let activeStageIndex = prevSteps.findIndex(s => s.name === update.stage);
                        if (activeStageIndex === -1) return prevSteps;
                        return prevSteps.map((step, index) => {
                            if (index === activeStageIndex) {
                                return { ...step, displayText: update.text, status: 'active' };
                            }
                            if (index < activeStageIndex && step.status !== 'completed') {
                                return { ...step, status: 'completed' };
                            }
                            return step;
                        });
                    });
                } else if (update.type === 'GRAPH_DATA') {
                    // --- DEBUG LOGGING 2: Confirm we are entering the correct block ---
                    console.log('GRAPH_DATA block entered. Setting state with data:', update.data.substring(0, 100) + '...');
                    setGraphmlData(update.data);
                } else if (update.type === 'DONE') {
                    setFinalScore(update.score);
                    setProcessSteps(prev => prev.map(s => ({...s, status: 'completed'})))
                    socket.disconnect();
                }
            } catch (e) { console.error('Failed to parse JSON:', msg.data, e); }
        });

        return () => { socket.disconnect(); };
    }, [uploadedFile]);

    const handleFileUpload = async (file: File) => {
        const formData = new FormData();
        formData.append('file', file);
        setProcessSteps(INITIAL_STAGES);
        setFinalScore(null);
        setGraphmlData(null);
        setViewMode('graph');
        setUploadedFile(file);
        try {
            await axios.post('http://localhost:5000/api/upload', formData);
        } catch (error) { console.error('Error uploading file:', error); }
    };
    
    const renderMainContent = () => {
        if (viewMode === 'graph' && uploadedFile) {
            return <XmlGraphViewer graphmlData={graphmlData} />;
        }
        if (!uploadedFile) {
            return <FileUploader onFileUpload={handleFileUpload} />;
        }
        return (
            <div className="flex flex-col items-center justify-center h-full text-text-secondary">
                <h2 className="text-2xl">Processing: {uploadedFile?.name}</h2>
                <p>The logical graph is being displayed.</p>
            </div>
        );
    };

    return (
        <main className="flex h-screen font-sans">
            <div className="w-1/4 max-w-sm bg-base-gray p-4 flex flex-col">
                <h1 className="text-2xl font-bold text-brand-green mb-8">Logos</h1>
                {uploadedFile && <Sidebar steps={processSteps} finalScore={finalScore} />}
            </div>
            <div className="flex-1 flex flex-col relative">
                <header className="absolute top-4 right-4 z-10">
                    {uploadedFile && (
                        <div className="flex space-x-2">
                            <button onClick={() => setViewMode('graph')} className={`w-10 h-10 rounded-md font-bold text-white transition ${viewMode === 'graph' ? 'bg-brand-green-dark' : 'bg-brand-green'}`}>G</button>
                        </div>
                    )}
                </header>
                <div className="flex-1 p-8">{renderMainContent()}</div>
            </div>
        </main>
    );
};

export default IndexPage;