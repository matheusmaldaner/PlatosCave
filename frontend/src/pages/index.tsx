// PlatosCave/frontend/src/pages/index.tsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { io, Socket } from 'socket.io-client';
import FileUploader from '../components/FileUploader';
import { ProcessStep } from '../components/Sidebar';
import XmlGraphViewer from '../components/XmlGraphViewer';
import SettingsModal, { Settings } from '../components/SettingsModal';
import ProgressBar from '../components/ProgressBar';
import platosCaveLogo from '../images/platos-cave-logo.png';

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
    const [finalScore, setFinalScore] = useState<number | null>(null);
    const [graphmlData, setGraphmlData] = useState<string | null>(null);
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [settings, setSettings] = useState<Settings>({ agentAggressiveness: 5, evidenceThreshold: 0.8 });

    useEffect(() => {
        if (!uploadedFile) return;
        const socket: Socket = io('http://localhost:5000');
        socket.on('connect', () => console.log('Connected to WebSocket server!'));
        socket.on('status_update', (msg: { data: string }) => {
            try {
                const update = JSON.parse(msg.data);
                if (update.type === 'UPDATE') {
                    setProcessSteps(prevSteps => {
                        let activeStageIndex = prevSteps.findIndex(s => s.name === update.stage);
                        if (activeStageIndex === -1) return prevSteps;
                        return prevSteps.map((step, index) => {
                            if (index === activeStageIndex) return { ...step, displayText: update.text, status: 'active' };
                            if (index < activeStageIndex && step.status !== 'completed') return { ...step, status: 'completed' };
                            return step;
                        });
                    });
                } else if (update.type === 'GRAPH_DATA') {
                    setGraphmlData(update.data);
                } else if (update.type === 'DONE') {
                    setFinalScore(update.score);
                    setProcessSteps(prev => prev.map(s => ({...s, status: 'completed'})));
                    socket.disconnect();
                }
            } catch (e) { console.error('Failed to parse JSON from server:', msg.data, e); }
        });
        return () => { socket.disconnect(); };
    }, [uploadedFile]);

    const handleFileUpload = async (file: File) => {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('agentAggressiveness', settings.agentAggressiveness.toString());
        formData.append('evidenceThreshold', settings.evidenceThreshold.toString());
        setProcessSteps(INITIAL_STAGES);
        setFinalScore(null);
        setGraphmlData(null);
        setUploadedFile(file);
        try {
            await axios.post('http://localhost:5000/api/upload', formData);
        } catch (error) { console.error('Error uploading file:', error); }
    };

    return (
        <>
            <SettingsModal isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} settings={settings} onSave={setSettings} />
            <main className="flex flex-col h-screen font-sans bg-white">
                <header className="w-full p-4 border-b border-gray-200 flex justify-between items-center">
                    <img src={platosCaveLogo} alt="Plato's Cave Logo" className="h-10" />
                    
                    {uploadedFile && (
                        <div className="flex items-center space-x-4">
                            {/* --- NEW: Final Score Display --- */}
                            {finalScore !== null && (
                                <div className="text-right">
                                    <span className="text-sm text-gray-500 font-semibold">Integrity Score</span>
                                    <p className="font-bold text-2xl text-brand-green">{finalScore.toFixed(2)}</p>
                                </div>
                            )}

                            <span className="font-mono text-sm text-gray-500">{uploadedFile.name}</span>
                            <button onClick={() => setIsSettingsOpen(true)} className="text-gray-500 hover:text-gray-800">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.096 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
                            </button>
                        </div>
                    )}
                </header>

                {!uploadedFile ? (
                    <div className="flex-grow flex items-center justify-center p-4">
                        <FileUploader onFileUpload={handleFileUpload} />
                    </div>
                ) : (
                    <>
                        <ProgressBar steps={processSteps} />
                        <div className="flex-grow p-4" style={{ height: 'calc(100vh - 150px)' }}>
                            <XmlGraphViewer graphmlData={graphmlData} />
                        </div>
                    </>
                )}
            </main>
        </>
    );
};

export default IndexPage;