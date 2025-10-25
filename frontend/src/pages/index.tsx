import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { io, Socket } from 'socket.io-client';
import FileUploader from '../components/FileUploader';
import { ProcessStep } from '../components/Sidebar';
import XmlGraphViewer from '../components/XmlGraphViewer';
import SettingsDrawer from '../components/SettingsDrawer';
import { Settings } from '../components/SettingsModal';
import ProgressBar from '../components/ProgressBar';
import ParticleBackground from '../components/ParticleBackground';
import platosCaveLogo from '../images/platos-cave-logo.png';
import { computePaperScore } from '../lib/aggregate';

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

  const exampleNodes = [
    {
      credibility: 0.9,
      relevance: 0.85,
      evidence_strength: 0.8,
      method_rigor: 0.85,
      reproducibility: 0.9,
      citation_support: 0.8,
      role: "result",
      level: 2,
    },
    {
      credibility: 0.95,
      relevance: 0.9,
      evidence_strength: 0.9,
      method_rigor: 0.9,
      reproducibility: 0.9,
      citation_support: 0.95,
      role: "hypothesis",
      level: 0,
    },
  ];

  useEffect(() => {
    if (!uploadedFile && !submittedUrl) return;

    const socket: Socket = io('http://localhost:5000');
    socket.on('connect', () => console.log('Connected to WebSocket server!'));

    socket.on('status_update', (msg: { data: string }) => {
      try {
        const update = JSON.parse(msg.data);
        if (update.type === 'UPDATE') {
          setProcessSteps(prev =>
            prev.map((s, i) =>
              s.name === update.stage
                ? { ...s, displayText: update.text, status: 'active' }
                : i < prev.findIndex(x => x.name === update.stage)
                ? { ...s, status: 'completed' }
                : s
            )
          );
        } else if (update.type === 'GRAPH_DATA') {
          setGraphmlData(update.data);
        } else if (update.type === 'DONE') {
          setFinalScore(update.score);
          setProcessSteps(prev => prev.map(s => ({ ...s, status: 'completed' })));
          socket.disconnect();
        }
      } catch (e) {
        if (!msg.data.startsWith('{')) console.debug('Skipping non-JSON:', msg.data);
        else console.error('JSON parse error:', e);
      }
    });

    return () => socket.disconnect();
  }, [uploadedFile, submittedUrl]);

  const handleFileUpload = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    Object.entries(settings).forEach(([key, value]) =>
      formData.append(key, value.toString())
    );

    setProcessSteps(INITIAL_STAGES);
    setFinalScore(null);
    setGraphmlData(null);
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
    setSubmittedUrl(url);
    setUploadedFile(null);

    try {
      await axios.post('http://localhost:5000/api/analyze-url', { url, ...settings });
    } catch (error) {
      console.error('Error analyzing URL:', error);
    }
  };

  const handleSettingsChange = (newSettings: Settings) => {
    setSettings(newSettings);

    const weightedNodes = exampleNodes.map(n => ({
      ...n,
      credibility: n.credibility * newSettings.credibility,
      relevance: n.relevance * newSettings.relevance,
      evidence_strength: n.evidence_strength * newSettings.evidenceStrength,
      method_rigor: n.method_rigor * newSettings.methodRigor,
      reproducibility: n.reproducibility * newSettings.reproducibility,
      citation_support: n.citation_support * newSettings.citationSupport,
    }));

    const newScore = computePaperScore(weightedNodes, newSettings.agentAggressiveness / 10, {
      hypothesis: newSettings.hypothesis,
      claim: newSettings.claim,
      method: newSettings.method,
      evidence: newSettings.evidence,
      result: newSettings.result,
      conclusion: newSettings.conclusion,
      limitation: newSettings.limitation,
    });

    setFinalScore(newScore);
  };

  return (
    <>
      {!uploadedFile && !submittedUrl && <ParticleBackground />}

      <main className="flex flex-col h-screen font-sans bg-gradient-to-b from-white via-gray-50 to-white">
        <header className="w-full px-6 py-5 border-b border-gray-100 flex justify-between items-center bg-white/50 backdrop-blur-sm relative z-10">
          <button
            onClick={() => window.location.reload()}
            className="cursor-pointer hover:opacity-70 transition-opacity duration-200"
            aria-label="Return to home"
          >
            <img src={platosCaveLogo} alt="Plato's Cave Logo" className="h-20" />
          </button>

          {(uploadedFile || submittedUrl) && (
            <div className="flex items-center space-x-6">
              {finalScore !== null && (
                <div className="text-right">
                  <span className="text-xs text-gray-500 font-medium uppercase tracking-wide">
                    Integrity Score
                  </span>
                  <p className="font-semibold text-3xl bg-gradient-to-r from-green-500 to-green-600 bg-clip-text text-transparent">
                    {finalScore.toFixed(2)}
                  </p>
                </div>
              )}
              <span className="font-mono text-sm text-gray-600 max-w-md truncate">
                {uploadedFile ? uploadedFile.name : submittedUrl}
              </span>
            </div>
          )}
        </header>

        {/* ✅ Only show the drawer tab after upload or submission */}
        {(uploadedFile || submittedUrl) && (
          <button
            onClick={() => setIsSettingsOpen(!isSettingsOpen)}
            className={`fixed top-1/2 left-0 transform -translate-y-1/2 z-40 flex flex-col items-center justify-center bg-green-300 text-white w-10 h-32 rounded-r-2xl shadow-xl hover:bg-green-400 transition-transform duration-300 ${
              isSettingsOpen ? "translate-x-80" : "translate-x-0"
            }`}
          >
            <span className="text-4xl font-bold leading-none">
              {isSettingsOpen ? "◁" : "▷"}
            </span>
          </button>
        )}

        {/* Drawer component */}
        <SettingsDrawer
          isOpen={isSettingsOpen}
          onClose={() => setIsSettingsOpen(false)}
          settings={settings}
          onSave={handleSettingsChange}
          graphmlData={graphmlData} 
        />

        {(!uploadedFile && !submittedUrl) ? (
          <div className="flex-grow flex items-center justify-center p-4 relative z-10">
            <FileUploader onFileUpload={handleFileUpload} onUrlSubmit={handleUrlSubmit} />
          </div>
        ) : (
          <>
            <ProgressBar steps={processSteps} />
            <div className="flex-grow p-4" style={{ height: 'calc(100vh - 150px)' }}>
              <XmlGraphViewer graphmlData={graphmlData} isDrawerOpen={isSettingsOpen} />

            </div>
          </>
        )}
      </main>
    </>
  );
};

export default IndexPage;
