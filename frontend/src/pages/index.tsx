// PlatosCave/frontend/src/pages/index.tsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { io, Socket } from 'socket.io-client';
import FileUploader from '../components/FileUploader';
import { ProcessStep } from '../components/Sidebar';
import XmlGraphViewer from '../components/XmlGraphViewer';
import BrowserViewer from '../components/BrowserViewer';
import SettingsModal, { Settings } from '../components/SettingsModal';
import ProgressBar from '../components/ProgressBar';
import ParticleBackground from '../components/ParticleBackground';
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
    const [submittedUrl, setSubmittedUrl] = useState<string | null>(null);
    const [finalScore, setFinalScore] = useState<number | null>(null);
    const [graphmlData, setGraphmlData] = useState<string | null>(null);
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [settings, setSettings] = useState<Settings>({ agentAggressiveness: 5, evidenceThreshold: 0.8 });
    const [isBrowserViewerOpen, setIsBrowserViewerOpen] = useState(false);
    const [browserSession, setBrowserSession] = useState<{ novncUrl?: string; cdpUrl?: string; cdpWebSocket?: string } | null>(null);

    // WebSocket connection for real-time updates
    useEffect(() => {
        if (!uploadedFile && !submittedUrl) return;

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
                    console.log('📊 Received graph data, length:', update.data?.length);
                    setGraphmlData(update.data);
                } else if (update.type === 'BROWSER_ADDRESS') {
                    setBrowserSession({
                        novncUrl: update.novnc_url,
                        cdpUrl: update.cdp_url,
                        cdpWebSocket: update.cdp_websocket
                    });
                    setIsBrowserViewerOpen(true);
                } else if (update.type === 'DONE') {
                    setFinalScore(update.score);
                    setProcessSteps(prev => prev.map(s => ({...s, status: 'completed'})));
                    setIsBrowserViewerOpen(false);
                    socket.disconnect();
                }
            } catch (e) {
                // Skip non-JSON lines (like browser-use logs) silently
                if (!msg.data.startsWith('{')) {
                    console.debug('Skipping non-JSON message:', msg.data);
                } else {
                    console.error('Failed to parse JSON from server:', msg.data, e);
                }
            }
        });
        return () => { socket.disconnect(); };
    }, [uploadedFile, submittedUrl]);

    const handleFileUpload = async (file: File) => {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('agentAggressiveness', settings.agentAggressiveness.toString());
        formData.append('evidenceThreshold', settings.evidenceThreshold.toString());
        setProcessSteps(INITIAL_STAGES);
        setFinalScore(null);
        setGraphmlData(null);
    setBrowserSession(null);
    setIsBrowserViewerOpen(false);
        setUploadedFile(file);
        setSubmittedUrl(null);  // Clear URL if file is uploaded
        try {
            await axios.post('http://localhost:5000/api/upload', formData);
        } catch (error) { console.error('Error uploading file:', error); }
    };

    const handleUrlSubmit = async (url: string) => {
        setProcessSteps(INITIAL_STAGES);
        setFinalScore(null);
        setGraphmlData(null);
    setBrowserSession(null);
    setIsBrowserViewerOpen(false);
        setSubmittedUrl(url);
        setUploadedFile(null);  // Clear file if URL is submitted
        try {
            await axios.post('http://localhost:5000/api/analyze-url', {
                url,
                agentAggressiveness: settings.agentAggressiveness,
                evidenceThreshold: settings.evidenceThreshold
            });
        } catch (error) { console.error('Error analyzing URL:', error); }
    };

    return (
        <>
            <SettingsModal isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} settings={settings} onSave={setSettings} />

            {/* Particle background only on landing page */}
            {!uploadedFile && !submittedUrl && <ParticleBackground />}

            <main className="flex flex-col h-screen font-sans bg-gradient-to-b from-white via-gray-50 to-white">
                <header className="w-full px-6 py-5 border-b border-gray-100 flex justify-between items-center bg-white/50 backdrop-blur-sm relative z-10">
                    <button
                        onClick={() => window.location.reload()}
                        className="cursor-pointer hover:opacity-70 transition-opacity duration-200"
                        aria-label="Return to home"
                    >
                        <img src={platosCaveLogo} alt="Plato's Cave Logo" className="h-9" />
                    </button>

                    {(uploadedFile || submittedUrl) && (
                        <div className="flex items-center space-x-6">
                            {/* --- Final Score Display --- */}
                            {finalScore !== null && (
                                <div className="text-right">
                                    <span className="text-xs text-gray-500 font-medium uppercase tracking-wide">Integrity Score</span>
                                    <p className="font-semibold text-3xl bg-gradient-to-r from-green-500 to-green-600 bg-clip-text text-transparent">
                                        {finalScore.toFixed(2)}
                                    </p>
                                </div>
                            )}

                            <span className="font-mono text-sm text-gray-600 max-w-md truncate">
                                {uploadedFile ? uploadedFile.name : submittedUrl}
                            </span>
                            <button
                                onClick={() => setIsSettingsOpen(true)}
                                className="text-gray-400 hover:text-gray-700 transition-colors duration-200 p-2 rounded-lg hover:bg-gray-100"
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.096 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
                            </button>
                        </div>
                    )}
                </header>

                <div className="relative flex-grow">
                    <div
                        className={`absolute inset-0 flex items-center justify-center p-4 transition-all duration-500 ease-in-out ${(!uploadedFile && !submittedUrl) ? 'opacity-100 translate-y-0 pointer-events-auto' : 'opacity-0 -translate-y-4 pointer-events-none'}`}
                    >
                        <div className="relative z-10 w-full">
                            <FileUploader onFileUpload={handleFileUpload} onUrlSubmit={handleUrlSubmit} />
                        </div>
                    </div>

                    <div
                        className={`absolute inset-0 flex flex-col transition-all duration-500 ease-in-out ${(!uploadedFile && !submittedUrl) ? 'opacity-0 translate-y-4 pointer-events-none' : 'opacity-100 translate-y-0 pointer-events-auto'}`}
                    >
                        <ProgressBar steps={processSteps} />
                        <div className="flex-grow p-4" style={{ height: 'calc(100vh - 150px)' }}>
                            <XmlGraphViewer graphmlData={graphmlData} />
                        </div>
                    </div>
                </div>
            </main>

            <BrowserViewer
                isOpen={isBrowserViewerOpen && !!browserSession?.novncUrl}
                onClose={() => setIsBrowserViewerOpen(false)}
                novncUrl={browserSession?.novncUrl}
                cdpUrl={browserSession?.cdpUrl}
                cdpWebSocket={browserSession?.cdpWebSocket}
            />
        </>
    );
};

export default IndexPage;