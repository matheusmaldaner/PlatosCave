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
    { name: "Building Knowledge Graph", displayText: "Pending...", status: 'pending' },
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
    const [sessionId, setSessionId] = useState<string | null>(null);

    // WebSocket connection for real-time updates
    useEffect(() => {
        console.log('[FRONTEND DEBUG] ========== useEffect TRIGGERED ==========');
        console.log('[FRONTEND DEBUG] uploadedFile:', uploadedFile?.name);
        console.log('[FRONTEND DEBUG] submittedUrl:', submittedUrl);

        if (!uploadedFile && !submittedUrl) {
            console.log('[FRONTEND DEBUG] No file or URL, skipping WebSocket connection');
            return;
        }

        console.log('[FRONTEND DEBUG] Creating WebSocket connection...');
        const socket: Socket = io('http://localhost:5000');
        socket.on('connect', () => {
            console.log('[FRONTEND DEBUG] ========== WEBSOCKET CONNECTED ==========');
            console.log('[FRONTEND DEBUG] Socket ID:', socket.id);
            setSessionId(socket.id || null);
        });
        socket.on('status_update', (msg: { data: string }) => {
            console.log('[FRONTEND DEBUG] ========== STATUS_UPDATE RECEIVED ==========');
            console.log('[FRONTEND DEBUG] Raw message:', msg.data.substring(0, 200));
            try {
                const update = JSON.parse(msg.data);
                console.log('[FRONTEND DEBUG] Parsed message type:', update.type);
                console.log('[FRONTEND DEBUG] Full update object:', update);

                if (update.type === 'UPDATE') {
                    console.log('[FRONTEND DEBUG] Processing UPDATE message');
                    console.log('[FRONTEND DEBUG] Stage:', update.stage, 'Text:', update.text);
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
                    console.log('[FRONTEND DEBUG] ========== GRAPH_DATA RECEIVED ==========');
                    console.log('[FRONTEND DEBUG] Graph data length:', update.data?.length);
                    setGraphmlData(update.data);
                } else if (update.type === 'BROWSER_ADDRESS') {
                    console.log('[FRONTEND DEBUG] ========== BROWSER_ADDRESS RECEIVED ==========');
                    console.log('[FRONTEND DEBUG] Full browser info:', update);
                    console.log('[FRONTEND DEBUG] noVNC URL:', update.novnc_url);
                    console.log('[FRONTEND DEBUG] CDP URL:', update.cdp_url);
                    console.log('[FRONTEND DEBUG] CDP WebSocket:', update.cdp_websocket);

                    setBrowserSession({
                        novncUrl: update.novnc_url,
                        cdpUrl: update.cdp_url,
                        cdpWebSocket: update.cdp_websocket
                    });
                    console.log('[FRONTEND DEBUG] Browser session state updated');
                    console.log('[FRONTEND DEBUG] Opening browser viewer modal...');
                    setIsBrowserViewerOpen(true);
                    console.log('[FRONTEND DEBUG] Browser viewer should now be open!');
                } else if (update.type === 'DONE') {
                    console.log('[FRONTEND DEBUG] ========== DONE MESSAGE RECEIVED ==========');
                    console.log('[FRONTEND DEBUG] Final score:', update.score);
                    setFinalScore(update.score);
                    setProcessSteps(prev => prev.map(s => ({...s, status: 'completed'})));
                    setIsBrowserViewerOpen(false);
                    socket.disconnect();
                }
            } catch (e) {
                // Skip non-JSON lines (like browser-use logs) silently
                if (!msg.data.startsWith('{')) {
                    console.debug('[FRONTEND DEBUG] Skipping non-JSON message:', msg.data.substring(0, 100));
                } else {
                    console.error('[FRONTEND DEBUG] ========== JSON PARSE ERROR ==========');
                    console.error('[FRONTEND DEBUG] Failed to parse:', msg.data.substring(0, 200));
                    console.error('[FRONTEND DEBUG] Error:', e);
                }
            }
        });

        socket.on('disconnect', () => {
            console.log('[FRONTEND DEBUG] ========== WEBSOCKET DISCONNECTED ==========');
        });

        return () => {
            console.log('[FRONTEND DEBUG] Cleaning up WebSocket connection');
            socket.disconnect();
        };
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
        console.log('[FRONTEND DEBUG] ========== handleUrlSubmit CALLED ==========');
        console.log('[FRONTEND DEBUG] URL:', url);
        console.log('[FRONTEND DEBUG] Settings:', settings);

        // First, cleanup any existing processes
        console.log('[FRONTEND DEBUG] Calling cleanup endpoint first...');
        try {
            await axios.post('http://localhost:5000/api/cleanup');
            console.log('[FRONTEND DEBUG] Cleanup completed');
        } catch (error) {
            console.error('[FRONTEND DEBUG] Cleanup error (continuing anyway):', error);
        }

        // Small delay to ensure cleanup completes
        await new Promise(resolve => setTimeout(resolve, 500));

        setProcessSteps(INITIAL_STAGES);
        setFinalScore(null);
        setGraphmlData(null);
        setBrowserSession(null);
        setIsBrowserViewerOpen(false);
        setSubmittedUrl(url);
        setUploadedFile(null);  // Clear file if URL is submitted

        console.log('[FRONTEND DEBUG] Sending POST to /api/analyze-url...');
        console.log('[FRONTEND DEBUG] Session ID:', sessionId);
        try {
            const response = await axios.post('http://localhost:5000/api/analyze-url', {
                url,
                agentAggressiveness: settings.agentAggressiveness,
                evidenceThreshold: settings.evidenceThreshold,
                sessionId: sessionId  // Send session ID to associate with process
            });
            console.log('[FRONTEND DEBUG] POST response:', response.data);
        } catch (error) {
            console.error('[FRONTEND DEBUG] Error analyzing URL:', error);
        }
    };

    return (
        <>
            <SettingsModal isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} settings={settings} onSave={setSettings} />

            {/* Particle background only on landing page */}
            {!uploadedFile && !submittedUrl && <ParticleBackground />}

            <main className="flex min-h-screen flex-col bg-gradient-to-b from-white via-gray-50 to-white font-sans" style={{ minHeight: '100dvh' }}>
                <header className="relative z-10 flex w-full flex-wrap items-center justify-between gap-4 border-b border-gray-100 bg-white/50 px-4 py-4 backdrop-blur-sm sm:px-6 sm:py-5">
                    <button
                        onClick={() => {
                            console.log('[FRONTEND DEBUG] Logo clicked - cleaning up...');
                            // Call cleanup endpoint before reload
                            axios.post('http://localhost:5000/api/cleanup').catch(err =>
                                console.error('[FRONTEND DEBUG] Cleanup error:', err)
                            );
                            // Give cleanup a moment, then reload
                            setTimeout(() => window.location.reload(), 500);
                        }}
                        className="cursor-pointer hover:opacity-70 transition-opacity duration-200"
                        aria-label="Return to home"
                    >
                        <img src={platosCaveLogo} alt="Plato's Cave Logo" className="h-9" />
                    </button>

                    {(uploadedFile || submittedUrl) && (
                        <div className="flex w-full flex-col items-start gap-3 sm:w-auto sm:flex-row sm:items-center sm:gap-6">
                            {/* --- Final Score Display --- */}
                            {finalScore !== null && (
                                <div className="text-left sm:text-right">
                                    <span className="text-[11px] font-medium uppercase tracking-wide text-gray-500">Integrity Score</span>
                                    <p className="text-2xl font-semibold text-transparent bg-gradient-to-r from-green-500 to-green-600 bg-clip-text sm:text-3xl">
                                        {finalScore.toFixed(2)}
                                    </p>
                                </div>
                            )}

                            <span className="max-w-full truncate font-mono text-xs text-gray-600 sm:max-w-md sm:text-sm">
                                {uploadedFile ? uploadedFile.name : submittedUrl}
                            </span>
                            <button
                                onClick={() => setIsSettingsOpen(true)}
                                className="rounded-lg p-2 text-gray-400 transition-colors duration-200 hover:bg-gray-100 hover:text-gray-700"
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.096 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
                            </button>
                        </div>
                    )}
                </header>

                <div className="relative flex-grow overflow-x-hidden">
                    <div
                        className={`absolute inset-0 flex overflow-y-auto p-4 transition-all duration-500 ease-in-out sm:p-8 ${(!uploadedFile && !submittedUrl) ? 'pointer-events-auto opacity-100 translate-y-0' : 'pointer-events-none opacity-0 -translate-y-4'}`}
                    >
                        <div className="relative z-10 flex w-full items-start justify-center">
                            <FileUploader onFileUpload={handleFileUpload} onUrlSubmit={handleUrlSubmit} />
                        </div>
                    </div>

                    <div
                        className={`absolute inset-0 flex flex-col transition-all duration-500 ease-in-out ${(!uploadedFile && !submittedUrl) ? 'pointer-events-none opacity-0 translate-y-4' : 'pointer-events-auto opacity-100 translate-y-0'}`}
                    >
                        <div className="pt-12 overflow-visible">
                            <ProgressBar steps={processSteps} />
                        </div>
                        
                        {/* Browser Viewer - Inline below progress bar */}
                        <BrowserViewer
                            isOpen={isBrowserViewerOpen && !!browserSession?.novncUrl}
                            onClose={() => setIsBrowserViewerOpen(false)}
                            novncUrl={browserSession?.novncUrl}
                            cdpUrl={browserSession?.cdpUrl}
                            cdpWebSocket={browserSession?.cdpWebSocket}
                        />
                        
                        <div className="flex-grow px-3 pb-24 pt-3 sm:px-6 sm:pb-6 overflow-hidden" style={{ minHeight: '55vh' }}>
                            <XmlGraphViewer graphmlData={graphmlData} />
                        </div>
                    </div>
                </div>
            </main>
        </>
    );
};

export default IndexPage;