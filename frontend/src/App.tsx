// PlatosCave/frontend/src/App.tsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { io, Socket } from 'socket.io-client';
import { Settings as SettingsIcon, ChevronLeft, ChevronRight } from 'lucide-react';
import Navbar from './components/landing/Navbar';
import Hero from './components/landing/Hero';
import Footer from './components/landing/Footer';
import { ProcessStep } from './components/Sidebar';
import XmlGraphViewer from './components/XmlGraphViewer';
import SettingsDrawer from './components/SettingsDrawer';
import { Settings } from './components/SettingsModal';
import ProgressBar from './components/ProgressBar';
import BrowserViewer from './components/BrowserViewer';
import platosCaveLogo from './images/platos-cave-logo.png';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';
const API_KEY = import.meta.env.VITE_API_KEY || '';

// Debug: Log configuration (remove in production)
console.log('[Config] API URL:', API_BASE_URL);
console.log('[Config] API Key set:', API_KEY ? 'Yes (' + API_KEY.substring(0, 8) + '...)' : 'No');

// Configure axios defaults
axios.defaults.withCredentials = true;
if (API_KEY) {
  axios.defaults.headers.common['X-API-Key'] = API_KEY;
  console.log('[Config] X-API-Key header configured');
} else {
  console.warn('[Config] No API key configured - requests will be unauthenticated');
}

const INITIAL_STAGES: ProcessStep[] = [
  { name: "Validate", displayText: "Pending...", status: 'pending' },
  { name: "Decomposing PDF", displayText: "Pending...", status: 'pending' },
  { name: "Building Knowledge Graph", displayText: "Pending...", status: 'pending' },
  { name: "Organizing Agents", displayText: "Pending...", status: 'pending' },
  { name: "Compiling Evidence", displayText: "Pending...", status: 'pending' },
  { name: "Evaluating Integrity", displayText: "Pending...", status: 'pending' },
];

const App: React.FC = () => {
  const [processSteps, setProcessSteps] = useState<ProcessStep[]>([]);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [submittedUrl, setSubmittedUrl] = useState<string | null>(null);
  const [finalScore, setFinalScore] = useState<number | null>(null);
  const [graphmlData, setGraphmlData] = useState<string | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const socketRef = useRef<Socket | null>(null);
  
  // Browser viewer state for human-in-the-loop
  const [isBrowserOpen, setIsBrowserOpen] = useState(false);
  const [novncUrl, setNovncUrl] = useState<string | undefined>(undefined);
  const [cdpUrl, setCdpUrl] = useState<string | undefined>(undefined);
  const [cdpWebSocket, setCdpWebSocket] = useState<string | undefined>(undefined);

  const [settings, setSettings] = useState<Settings>({
    agentAggressiveness: 5,
    evidenceThreshold: 0.8,
    credibility: 1.0,
    relevance: 1.0,
    evidenceStrength: 1.0,
    methodRigor: 1.0,
    reproducibility: 1.0,
    citationSupport: 1.0,
    hypothesis: 1.0,
    claim: 1.0,
    method: 1.0,
    evidence: 1.0,
    result: 1.0,
    conclusion: 1.0,
    limitation: 1.0,
  });

  // Handle WebSocket messages
  const handleSocketMessage = useCallback((msg: { data: string }) => {
    try {
      const update = JSON.parse(msg.data);
      console.log('[WebSocket] Received:', update.type, update);
      
      if (update.type === 'UPDATE') {
        setProcessSteps(prev => {
          const activeIndex = prev.findIndex(s => s.name === update.stage);
          return prev.map((s, i) => {
            if (i === activeIndex) return { ...s, displayText: update.text, status: 'active' };
            if (i < activeIndex) return { ...s, status: 'completed' };
            return s;
          });
        });
      } else if (update.type === 'BROWSER_ADDRESS') {
        // Remote browser is ready - show the embedded viewer
        console.log('[WebSocket] Browser address received:', update);
        setNovncUrl(update.novnc_url);
        setCdpUrl(update.cdp_url);
        setCdpWebSocket(update.cdp_websocket);
        setIsBrowserOpen(true);
      } else if (update.type === 'GRAPH_DATA') {
        console.log('[WebSocket] Setting graph data');
        setGraphmlData(update.data);
      } else if (update.type === 'DONE') {
        console.log('[WebSocket] Analysis complete, score:', update.score);
        setFinalScore(update.score);
        setProcessSteps(prev => prev.map(s => ({ ...s, status: 'completed' })));
        // Optionally close browser viewer when done
        // setIsBrowserOpen(false);
      }
    } catch (e) {
      console.error('WebSocket parse error:', e);
    }
  }, []);

  // Connect WebSocket and return a promise that resolves when connected
  const connectSocket = useCallback((): Promise<Socket> => {
    return new Promise((resolve) => {
      // Disconnect existing socket if any
      if (socketRef.current) {
        socketRef.current.disconnect();
      }

      console.log('[WebSocket] Connecting to:', API_BASE_URL);
      
      const socket = io(API_BASE_URL, {
        path: '/socket.io',
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
        withCredentials: true,  // Send cookies with polling requests
      });
      
      socketRef.current = socket;

      socket.on('connect', () => {
        console.log('[WebSocket] Connected! Socket ID:', socket.id);
        resolve(socket);
      });

      socket.on('disconnect', (reason) => {
        console.log('[WebSocket] Disconnected:', reason);
      });

      socket.on('connect_error', (error) => {
        console.error('[WebSocket] Connection error:', error);
      });

      socket.on('status_update', handleSocketMessage);
    });
  }, [handleSocketMessage]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  const handleFileUpload = async (file: File) => {
    // Reset state first
    setProcessSteps(INITIAL_STAGES);
    setFinalScore(null);
    setGraphmlData(null);
    setUploadedFile(file);
    setSubmittedUrl(null);
    // Reset browser viewer state
    setIsBrowserOpen(false);
    setNovncUrl(undefined);
    setCdpUrl(undefined);
    setCdpWebSocket(undefined);

    // Connect WebSocket FIRST, then make API call
    await connectSocket();

    const formData = new FormData();
    formData.append('file', file);
    Object.entries(settings).forEach(([key, value]) =>
      formData.append(key, value.toString())
    );

    try {
      console.log('[API] Uploading file:', file.name);
      await axios.post(`${API_BASE_URL}/api/upload`, formData);
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  const handleUrlSubmit = async (url: string) => {
    // Reset state first
    setProcessSteps(INITIAL_STAGES);
    setFinalScore(null);
    setGraphmlData(null);
    setSubmittedUrl(url);
    setUploadedFile(null);
    // Reset browser viewer state
    setIsBrowserOpen(false);
    setNovncUrl(undefined);
    setCdpUrl(undefined);
    setCdpWebSocket(undefined);

    // Connect WebSocket FIRST, then make API call
    await connectSocket();

    try {
      console.log('[API] Analyzing URL:', url);
      await axios.post(`${API_BASE_URL}/api/analyze-url`, { url, ...settings });
    } catch (error) {
      console.error('Error analyzing URL:', error);
    }
  };

  const handleSettingsSave = (newSettings: Settings) => {
    setSettings(newSettings);
    setIsSettingsOpen(false);
  };

  const handleGetStarted = () => {
    const searchInput = document.querySelector('input[type="text"]') as HTMLInputElement;
    searchInput?.focus();
  };

  // Landing page (no file/URL submitted yet)
  if (!uploadedFile && !submittedUrl) {
    return (
      <div className="min-h-screen bg-gradient-hero">
        <Navbar onGetStarted={handleGetStarted} />
        <Hero
          onFileUpload={handleFileUpload}
          onUrlSubmit={handleUrlSubmit}
        />
        <Footer />
      </div>
    );
  }

  // Analysis page (file or URL submitted)
  return (
    <main className="flex min-h-screen flex-col bg-gradient-to-b from-white via-gray-50 to-white font-sans">
      {/* Header */}
      <header className="relative z-10 flex w-full items-center justify-between border-b border-gray-100 bg-white/80 px-4 py-4 backdrop-blur-md sm:px-6 sm:py-5">
        <button
          onClick={() => window.location.reload()}
          className="cursor-pointer transition-opacity duration-200 hover:opacity-70"
          aria-label="Return to home"
        >
          <img src={platosCaveLogo} alt="Plato's Cave Logo" className="h-9" />
        </button>
        <div className="flex items-center gap-4">
          {finalScore !== null && (
            <div className="text-left sm:text-right">
              <span className="text-[11px] font-medium uppercase tracking-wide text-text-secondary">
                Integrity Score
              </span>
              <p className="text-2xl font-semibold text-transparent bg-gradient-to-r from-brand-green to-brand-green-600 bg-clip-text sm:text-3xl">
                {finalScore.toFixed(2)}
              </p>
            </div>
          )}
          <span className="max-w-full truncate font-mono text-xs text-text-secondary sm:max-w-md sm:text-sm">
            {uploadedFile ? uploadedFile.name : submittedUrl}
          </span>
          {/* Show Browser button - only when browser URL is available */}
          {novncUrl && !isBrowserOpen && (
            <button
              onClick={() => setIsBrowserOpen(true)}
              className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-white bg-emerald-500 hover:bg-emerald-600 transition-colors duration-200"
              title="Show remote browser"
            >
              <div className="w-2 h-2 rounded-full bg-white animate-pulse" />
              Show Browser
            </button>
          )}
          <button
            onClick={() => setIsSettingsOpen(true)}
            className="rounded-lg p-2 text-text-muted transition-colors duration-200 hover:bg-gray-100 hover:text-text-primary"
          >
            <SettingsIcon className="h-5 w-5" />
          </button>
        </div>
      </header>

      {/* Drawer Tab */}
      <button
        onClick={() => setIsSettingsOpen(!isSettingsOpen)}
        className={`fixed left-0 top-1/2 z-40 flex h-32 w-10 -translate-y-1/2 transform flex-col items-center justify-center rounded-r-2xl bg-brand-green text-white shadow-xl transition-transform duration-300 hover:bg-brand-green-600 ${
          isSettingsOpen ? "translate-x-80" : "translate-x-0"
        }`}
      >
        {isSettingsOpen ? (
          <ChevronLeft className="h-8 w-8" strokeWidth={2.5} />
        ) : (
          <ChevronRight className="h-8 w-8" strokeWidth={2.5} />
        )}
      </button>

      {/* Drawer */}
      <SettingsDrawer
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSave={handleSettingsSave}
        graphmlData={graphmlData}
      />

      {/* Main content */}
      <div className="relative flex-grow overflow-hidden">
        <ProgressBar steps={processSteps} />
        
        {/* Browser Viewer - Human in the Loop (floats/overlays when open) */}
        <BrowserViewer
          isOpen={isBrowserOpen}
          onClose={() => setIsBrowserOpen(false)}
          novncUrl={novncUrl}
          cdpUrl={cdpUrl}
          cdpWebSocket={cdpWebSocket}
        />
        
        {/* Graph Viewer - always visible in background */}
        <div className="flex-grow p-4" style={{ height: "calc(100vh - 150px)" }}>
          <XmlGraphViewer graphmlData={graphmlData} isDrawerOpen={isSettingsOpen} />
        </div>
      </div>

      {/* Bottom status display */}
      <div
        className={`fixed bottom-2 left-1/2 -translate-x-1/2 max-w-[80%] truncate rounded-md border border-gray-200 bg-white/60 px-3 py-1.5 text-center font-mono text-[10px] text-text-secondary backdrop-blur-sm transition-opacity duration-500 ${
          uploadedFile || submittedUrl ? "opacity-100" : "opacity-0"
        }`}
      >
        Query: <span className="text-text-primary">{uploadedFile ? uploadedFile.name : submittedUrl}</span>
      </div>
    </main>
  );
};

export default App;

