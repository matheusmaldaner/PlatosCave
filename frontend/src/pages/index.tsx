// PlatosCave/frontend/src/pages/index.tsx
import React, { useState, useEffect, useRef } from 'react';
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
import caveVideo from '../videos/Cave_Background_With_White_Center.mp4';

const INITIAL_STAGES: ProcessStep[] = [
    { name: "Validate", displayText: "Pending...", status: 'pending' },
    { name: "Decomposing PDF", displayText: "Pending...", status: 'pending' },
    { name: "Building Knowledge Graph", displayText: "Pending...", status: 'pending' },
    { name: "Organizing Agents", displayText: "Pending...", status: 'pending' },
    { name: "Compiling Evidence", displayText: "Pending...", status: 'pending' },
    { name: "Evaluating Integrity", displayText: "Pending...", status: 'pending' },
];

const IndexPage = () => {
    const [introPhase, setIntroPhase] = useState<'video' | 'expand' | 'done'>("video");
    const expandTimeoutRef = useRef<number | null>(null);
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const [expandScale, setExpandScale] = useState(1);
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
                console.log('[WebSocket]', update.type, update);
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
                    console.log('🌐 BROWSER_ADDRESS received:', update);
                    setBrowserSession({
                        novncUrl: update.novnc_url,
                        cdpUrl: update.cdp_url,
                        cdpWebSocket: update.cdp_websocket
                    });
                    console.log('🌐 Opening browser viewer with:', update.novnc_url);
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

    // Cleanup pending timers
    useEffect(() => {
        return () => {
            if (expandTimeoutRef.current) window.clearTimeout(expandTimeoutRef.current);
        };
    }, []);

    // Kick off expansion animation when entering expand phase
    useEffect(() => {
        if (introPhase === 'expand') {
            setExpandScale(1);
            // trigger with slight delay, use larger final scale for full cover
            const id = window.setTimeout(() => setExpandScale(180), 60);
            // allow transform to complete, then dissolve main in
            expandTimeoutRef.current = window.setTimeout(() => setIntroPhase('done'), 1100);
            return () => { window.clearTimeout(id); };
        }
    }, [introPhase]);

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

            <main className={`flex min-h-screen flex-col bg-gradient-to-b from-white via-gray-50 to-white font-sans transition-opacity duration-400 ease-out ${introPhase === 'done' ? 'opacity-100' : 'opacity-0'}`} style={{ minHeight: '100dvh' }}>
                <header className="relative z-10 flex w-full flex-wrap items-center justify-between gap-4 border-b border-gray-100 bg-white/50 px-4 py-4 backdrop-blur-sm sm:px-6 sm:py-5">
                    <button
                        onClick={() => window.location.reload()}
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

                <div className="relative flex-grow overflow-hidden">
                    <div
                        className={`absolute inset-0 flex overflow-y-auto p-4 transition-all duration-500 ease-in-out sm:p-8 ${(!uploadedFile && !submittedUrl) ? 'pointer-events-auto opacity-100 translate-y-0' : 'pointer-events-none opacity-0 -translate-y-4'}`}
                    >
                        <div className="relative z-10 flex w-full items-start justify-center">
                            <FileUploader onFileUpload={handleFileUpload} onUrlSubmit={handleUrlSubmit} />
                        </div>
                    </div>

                    <div
                        className={`absolute inset-0 flex flex-col overflow-hidden transition-all duration-500 ease-in-out ${(!uploadedFile && !submittedUrl) ? 'pointer-events-none opacity-0 translate-y-4' : 'pointer-events-auto opacity-100 translate-y-0'}`}
                    >
                        <ProgressBar steps={processSteps} />
                        <div className="flex-grow px-3 pb-24 pt-3 sm:px-6 sm:pb-6" style={{ minHeight: '55vh' }}>
                            <XmlGraphViewer graphmlData={graphmlData} />
                        </div>
                    </div>
                </div>
            </main>

            {/* Intro overlay: video then expanding white box */}
            {introPhase !== 'done' && (
                <div className={`fixed inset-0 z-50 overflow-hidden ${introPhase === 'video' ? 'bg-black' : 'bg-white'}` }>
                    {/* Keep video visible for both phases so box aligns with last frame */}
                    <video
                        ref={videoRef}
                        src={caveVideo}
                        className="absolute inset-0 h-full w-full object-contain bg-black"
                        autoPlay={introPhase === 'video'}
                        muted
                        playsInline
                        style={{ transform: 'scale(1.12)', transformOrigin: 'center bottom', opacity: introPhase === 'expand' ? 0 : 1, transition: 'opacity 200ms ease-out' }}
                        onLoadedMetadata={() => {
                            try { if (videoRef.current) videoRef.current.playbackRate = 3.0; } catch {}
                        }}
                        onPlay={() => {
                            try { if (videoRef.current) videoRef.current.playbackRate = 3.0; } catch {}
                        }}
                        onEnded={() => {
                            setIntroPhase('expand');
                        }}
                        onError={() => setIntroPhase('done')}
                    />

                    {/* Expanding white box aligned to video coordinates (based on 1280x720) */}
                    {introPhase === 'expand' && (
                        <div className="absolute inset-0">
                            <div
                                className="bg-white transition-transform duration-1000 ease-in"
                                style={{
                                    // Coords mapped as percentages relative to 1280x720: left~30.1%, top~29.0%, width~48.6%, height~68.2%
                                    position: 'absolute',
                                    left: '26.2%',
                                    top: '22.06%',
                                    width: '48.6%',
                                    height: '68.2%',
                                    transformOrigin: '50% 50%',
                                    transform: `scale(${expandScale})`,
                                }}
                            />
                        </div>
                    )}
                </div>
            )}

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