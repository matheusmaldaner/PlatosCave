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
  const [settings, setSettings] = useState<Settings>({
    agentAggressiveness: 5,
    evidenceThreshold: 0.8,
  });
  const [isBrowserViewerOpen, setIsBrowserViewerOpen] = useState(false);
  const [browserSession, setBrowserSession] = useState<{
    novncUrl?: string;
    cdpUrl?: string;
    cdpWebSocket?: string;
  } | null>(null);

  // --- WebSocket connection for real-time updates ---
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
              if (index === activeStageIndex)
                return { ...step, displayText: update.text, status: 'active' };
              if (index < activeStageIndex && step.status !== 'completed')
                return { ...step, status: 'completed' };
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
            cdpWebSocket: update.cdp_websocket,
          });
          console.log('🌐 Opening browser viewer with:', update.novnc_url);
          setIsBrowserViewerOpen(true);
        } else if (update.type === 'DONE') {
          setFinalScore(update.score);
          setProcessSteps(prev => prev.map(s => ({ ...s, status: 'completed' })));
          setIsBrowserViewerOpen(false);
          socket.disconnect();
        }
      } catch (e) {
        if (!msg.data.startsWith('{')) {
          console.debug('Skipping non-JSON message:', msg.data);
        } else {
          console.error('Failed to parse JSON from server:', msg.data, e);
        }
      }
    });

    return () => {
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
    setSubmittedUrl(null);
    try {
      await axios.post('http://localhost:5000/api/upload', formData);
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  const handleUrlSubmit = async (url: string) => {
    setProcessSteps(INITIAL_STAGES);
    setFinalScore(null);
    setGraphmlData(null);
    setBrowserSession(null);
    setIsBrowserViewerOpen(false);
    setSubmittedUrl(url);
    setUploadedFile(null);
    try {
      await axios.post('http://localhost:5000/api/analyze-url', {
        url,
        agentAggressiveness: settings.agentAggressiveness,
        evidenceThreshold: settings.evidenceThreshold,
      });
    } catch (error) {
      console.error('Error analyzing URL:', error);
    }
  };

  return (
    <>
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSave={setSettings}
      />

      {!uploadedFile && !submittedUrl && <ParticleBackground />}

      <main className="flex min-h-screen flex-col bg-gradient-to-b from-white via-gray-50 to-white font-sans" style={{ minHeight: '100dvh' }}>
        {/* Header */}

        <header className="relative z-10 flex w-full items-center justify-between border-b border-gray-100 bg-white/50 px-4 py-4 backdrop-blur-sm sm:px-6 sm:py-5">
          <button
            onClick={() => window.location.reload()}
            className="cursor-pointer hover:opacity-70 transition-opacity duration-200"
            aria-label="Return to home"
          >
            <img src={platosCaveLogo} alt="Plato's Cave Logo" className="h-9" />
          </button>
          {(uploadedFile || submittedUrl) && (
            <div className="flex items-center gap-4">
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
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.096 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </button>
            </div>
          )}
        </header>

        {/* Main content */}
        <div className="relative flex-grow overflow-hidden">
          {/* File/URL upload landing screen */}
          <div
            className={`absolute inset-0 flex overflow-y-auto p-4 transition-all duration-500 ease-in-out sm:p-8 ${(!uploadedFile && !submittedUrl)
              ? 'pointer-events-auto opacity-100 translate-y-0'
              : 'pointer-events-none opacity-0 -translate-y-4'
              }`}
          >
            <div className="relative z-10 flex w-full items-start justify-center">
              <FileUploader onFileUpload={handleFileUpload} onUrlSubmit={handleUrlSubmit} />
            </div>
          </div>

          {/* Analysis screen */}
          <div
            className={`absolute inset-0 flex flex-col overflow-hidden transition-all duration-500 ease-in-out ${(!uploadedFile && !submittedUrl)
              ? 'pointer-events-none opacity-0 translate-y-4'
              : 'pointer-events-auto opacity-100 translate-y-0'
              }`}
          >
            <ProgressBar steps={processSteps} />
            <div className="flex-grow px-3 pb-24 pt-3 sm:px-6 sm:pb-6" style={{ minHeight: '55vh' }}>
              <XmlGraphViewer graphmlData={graphmlData} />
            </div>
          </div>
        </div>

        {/* --- Bottom-Centered URL Display (small, subtle with border) --- */}
        {(uploadedFile || submittedUrl) && (
        <div
            className={`fixed bottom-2 left-1/2 -translate-x-1/2 text-center text-gray-600 text-[10px] font-mono bg-white/60 backdrop-blur-sm px-3 py-1.5 rounded-md border border-gray-200 max-w-[80%] truncate transition-opacity duration-500 ${
            uploadedFile || submittedUrl ? 'opacity-100' : 'opacity-0'
            }`}
        >
            Query:{' '}
            <span className="text-gray-700">
            {uploadedFile ? uploadedFile.name : submittedUrl}
            </span>
        </div>
        )}

      </main>

      {/* Browser Viewer */}
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
