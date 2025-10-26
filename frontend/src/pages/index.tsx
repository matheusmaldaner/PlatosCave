// frontend/src/pages/index.tsx
import React, { useState, useEffect } from "react";
import axios from "axios";
import { io, Socket } from "socket.io-client";
import FileUploader from "../components/FileUploader";
import { ProcessStep } from "../components/Sidebar";
import XmlGraphViewer from "../components/XmlGraphViewer";
import SettingsDrawer from "../components/SettingsDrawer";
import { Settings } from "../components/SettingsModal";
import ProgressBar from "../components/ProgressBar";
import ParticleBackground from "../components/ParticleBackground";
import platosCaveLogo from "../images/platos-cave-logo.png";
import { computePaperScore } from "../lib/aggregate";

const INITIAL_STAGES: ProcessStep[] = [
  { name: "Validate", displayText: "Pending...", status: "pending" },
  { name: "Decomposing PDF", displayText: "Pending...", status: "pending" },
  { name: "Building Logic Tree", displayText: "Pending...", status: "pending" },
  { name: "Organizing Agents", displayText: "Pending...", status: "pending" },
  { name: "Compiling Evidence", displayText: "Pending...", status: "pending" },
  { name: "Evaluating Integrity", displayText: "Pending...", status: "pending" },
];

const IndexPage: React.FC = () => {
  const [processSteps, setProcessSteps] = useState<ProcessStep[]>([]);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [submittedUrl, setSubmittedUrl] = useState<string | null>(null);
  const [finalScore, setFinalScore] = useState<number | null>(null);
  const [graphmlData, setGraphmlData] = useState<string | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [showTypeColors, setShowTypeColors] = useState(false);

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

  // exampleNodes — used only for frontend preview/score calculation
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

  // ---------- Websocket & status handling ----------
  useEffect(() => {
    if (!uploadedFile && !submittedUrl) return;

    const socket: Socket = io("http://localhost:5000");
    socket.on("connect", () => console.log("Connected to WebSocket server!"));

    socket.on("status_update", (msg: { data: string }) => {
      try {
        const update = JSON.parse(msg.data);
        if (update.type === "UPDATE") {
          setProcessSteps((prev) =>
            prev.map((s, i) =>
              s.name === update.stage
                ? { ...s, displayText: update.text, status: "active" }
                : i < prev.findIndex((x) => x.name === update.stage)
                ? { ...s, status: "completed" }
                : s
            )
          );
        } else if (update.type === "GRAPH_DATA") {
          setGraphmlData(update.data);
        } else if (update.type === "DONE") {
          setFinalScore(update.score);
          setProcessSteps((prev) => prev.map((s) => ({ ...s, status: "completed" })));
          socket.disconnect();
        }
      } catch (e) {
        // handle non-json or parse errors
        console.error("Error parsing websocket message:", e, msg);
      }
    });

    return () => socket.disconnect();
  }, [uploadedFile, submittedUrl]);

  // ---------- Upload / URL submit ----------
  const handleFileUpload = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    Object.entries(settings).forEach(([key, value]) => formData.append(key, value.toString()));

    setProcessSteps(INITIAL_STAGES);
    setFinalScore(null);
    setGraphmlData(null);
    setUploadedFile(file);
    setSubmittedUrl(null);

    try {
      await axios.post("http://localhost:5000/api/upload", formData);
    } catch (error) {
      console.error("Error uploading file:", error);
    }
  };

  const handleUrlSubmit = async (url: string) => {
    setProcessSteps(INITIAL_STAGES);
    setFinalScore(null);
    setGraphmlData(null);
    setSubmittedUrl(url);
    setUploadedFile(null);

    try {
      await axios.post("http://localhost:5000/api/analyze-url", { url, ...settings });
    } catch (error) {
      console.error("Error analyzing URL:", error);
    }
  };

  // ---------- Save GraphML (prompt for filename and download) ----------
  const saveGraphmlPrompt = (defaultName = "logic_graph.graphml") => {
    if (!graphmlData) {
      alert("No graph data available to save.");
      return;
    }

    // Prompt user for filename (simple, builtin prompt)
    const input = window.prompt("Enter filename to save GraphML as:", defaultName);
    if (!input) {
      // user cancelled or blank
      return;
    }
    let filename = input.trim();
    if (!filename.toLowerCase().endsWith(".graphml")) {
      filename = `${filename}.graphml`;
    }

    // create blob and download
    try {
      const blob = new Blob([graphmlData], { type: "application/xml" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to save GraphML:", err);
      alert("Failed to save graph file. See console for details.");
    }
  };

  // ---------- Apply settings (frontend-only re-score preview) ----------
  const handleSettingsSave = (newSettings: Settings) => {
    // update app settings
    setSettings(newSettings);

    // apply sliders to example nodes to recalc preview score
    const weightedNodes = exampleNodes.map((n) => ({
      ...n,
      credibility: n.credibility * newSettings.credibility,
      relevance: n.relevance * newSettings.relevance,
      evidence_strength: n.evidence_strength * newSettings.evidenceStrength,
      method_rigor: n.method_rigor * newSettings.methodRigor,
      reproducibility: n.reproducibility * newSettings.reproducibility,
      citation_support: n.citation_support * newSettings.citationSupport,
    }));

    const roleWeights: Record<string, number> = {
      hypothesis: newSettings.hypothesis ?? 1.0,
      claim: newSettings.claim ?? 1.0,
      method: newSettings.method ?? 1.0,
      evidence: newSettings.evidence ?? 1.0,
      result: newSettings.result ?? 1.0,
      conclusion: newSettings.conclusion ?? 1.0,
      limitation: newSettings.limitation ?? 1.0,
      other: 1.0,
    };

    // evidence threshold effect
    const threshold = newSettings.evidenceThreshold ?? 0.1;
    const filteredNodes = weightedNodes.map((n) => {
      const evidenceFactor = n.evidence_strength >= threshold ? 1 : Math.max(0, n.evidence_strength / threshold);
      return {
        ...n,
        credibility: n.credibility * evidenceFactor,
        relevance: n.relevance * evidenceFactor,
        evidence_strength: n.evidence_strength * evidenceFactor,
      };
    });

    const newScore = computePaperScore(filteredNodes, (newSettings.agentAggressiveness ?? 5) / 10, roleWeights);
    setFinalScore(newScore);
    setIsSettingsOpen(false);
  };

  // ---------- Legend component (shown when showTypeColors is true) ----------
  const Legend: React.FC = () => (
    <div
      className="fixed left-1/2 transform -translate-x-1/2 bottom-6 z-50 bg-white/95 border border-gray-200 shadow-lg rounded-lg px-4 py-2 flex gap-3 items-center text-sm"
      role="region"
      aria-label="Type legend"
      style={{ backdropFilter: "blur(6px)" }}
    >
      {/* ROYGBIV mapping */}
      {[
        ["Hypothesis", "#E53935"],
        ["Claim", "#FB8C00"],
        ["Method", "#FDD835"],
        ["Evidence", "#43A047"],
        ["Result", "#1E88E5"],
        ["Conclusion", "#5E35B1"],
        ["Limitation", "#8E24AA"],
      ].map(([label, color]) => (
        <div key={label} className="flex items-center gap-2">
          <span style={{ width: 18, height: 12, background: color, borderRadius: 3, boxShadow: `${color}66 0 0 8px` }} />
          <span className="text-xs text-gray-700">{label}</span>
        </div>
      ))}
    </div>
  );

  // ---------- Render ----------
  return (
    <>
      {!uploadedFile && !submittedUrl && <ParticleBackground />}

      <main className="flex flex-col h-screen font-sans bg-gradient-to-b from-white via-gray-50 to-white">
        {/* header */}
        <header className="w-full px-6 py-5 border-b border-gray-100 flex justify-between items-center bg-white/50 backdrop-blur-sm relative z-10">
          <button onClick={() => window.location.reload()} className="cursor-pointer hover:opacity-70 transition-opacity duration-200" aria-label="Return to home">
            <img src={platosCaveLogo} alt="Plato's Cave Logo" className="h-20" />
          </button>

          {(uploadedFile || submittedUrl) && (
            <div className="flex items-center space-x-6">
              {finalScore !== null && (
                <div className="text-right">
                  <span className="text-xs text-gray-500 font-medium uppercase tracking-wide">Integrity Score</span>
                  <p className="font-semibold text-3xl bg-gradient-to-r from-green-500 to-green-600 bg-clip-text text-transparent">
                    {finalScore.toFixed(2)}
                  </p>
                </div>
              )}
              <span className="font-mono text-sm text-gray-600 max-w-md truncate">{uploadedFile ? uploadedFile.name : submittedUrl}</span>
            </div>
          )}
        </header>

        {/* Drawer tab - shows only after upload or URL submit */}
        {(uploadedFile || submittedUrl) && (
          <button
            onClick={() => setIsSettingsOpen(!isSettingsOpen)}
            className={`fixed top-1/2 left-0 transform -translate-y-1/2 z-40 flex flex-col items-center justify-center bg-green-300 text-white w-10 h-32 rounded-r-2xl shadow-xl hover:bg-green-400 transition-transform duration-300 ${
              isSettingsOpen ? "translate-x-80" : "translate-x-0"
            }`}
            aria-label="Open settings drawer"
          >
            <span className="text-4xl font-bold leading-none">{isSettingsOpen ? "◁" : "▷"}</span>
          </button>
        )}

        {/* Drawer (pass onSaveGraph so Save GraphML uses the prompt) */}
        <SettingsDrawer
          isOpen={isSettingsOpen}
          onClose={() => setIsSettingsOpen(false)}
          settings={settings}
          onSave={handleSettingsSave}
          showTypeColors={showTypeColors}
          setShowTypeColors={setShowTypeColors}
          onSaveGraph={() => saveGraphmlPrompt()}
        />

        {/* main content */}
        {(!uploadedFile && !submittedUrl) ? (
          <div className="flex-grow flex items-center justify-center p-4 relative z-10">
            <FileUploader onFileUpload={handleFileUpload} onUrlSubmit={handleUrlSubmit} />
          </div>
        ) : (
          <>
            <ProgressBar steps={processSteps} />
            <div className="flex-grow p-4" style={{ height: "calc(100vh - 150px)" }}>
              <XmlGraphViewer graphmlData={graphmlData} isDrawerOpen={isSettingsOpen} settings={settings} showTypeColors={showTypeColors} />
            </div>
          </>
        )}

        {/* Legend: visible while toggle is ON */}
        {showTypeColors && <Legend />}
      </main>
    </>
  );
};

export default IndexPage;
