// PlatosCave/frontend/src/pages/index.tsx
import React, { useState, useEffect } from "react";
import axios from "axios";
import { io, Socket } from "socket.io-client";
import FileUploader from "../components/FileUploader";
import { ProcessStep } from "../components/Sidebar";
import XmlGraphViewer from "../components/XmlGraphViewer";
import BrowserViewer from "../components/BrowserViewer";
import SettingsDrawer from "../components/SettingsDrawer";
import { Settings } from "../components/SettingsModal";
import ProgressBar from "../components/ProgressBar";
import ParticleBackground from "../components/ParticleBackground";
import platosCaveLogo from "../images/platos-cave-logo.png";

const INITIAL_STAGES: ProcessStep[] = [
  { name: "Validate", displayText: "Pending...", status: "pending" },
  { name: "Decomposing PDF", displayText: "Pending...", status: "pending" },
  {
    name: "Building Knowledge Graph",
    displayText: "Pending...",
    status: "pending",
  },
  { name: "Organizing Agents", displayText: "Pending...", status: "pending" },
  { name: "Compiling Evidence", displayText: "Pending...", status: "pending" },
  {
    name: "Evaluating Integrity",
    displayText: "Pending...",
    status: "pending",
  },
];

const IndexPage = () => {
  const [processSteps, setProcessSteps] = useState<ProcessStep[]>([]);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [submittedUrl, setSubmittedUrl] = useState<string | null>(null);
  const [finalScore, setFinalScore] = useState<number | null>(null);
  const [graphmlData, setGraphmlData] = useState<string | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // Browser viewer state
  const [isBrowserOpen, setIsBrowserOpen] = useState(false);
  const [browserNovncUrl, setBrowserNovncUrl] = useState<string | undefined>(undefined);
  const [browserCdpUrl, setBrowserCdpUrl] = useState<string | undefined>(undefined);
  const [browserCdpWebSocket, setBrowserCdpWebSocket] = useState<string | undefined>(undefined);

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

  // WebSocket connection for updates
  useEffect(() => {
    if (!uploadedFile && !submittedUrl) return;

    const socket: Socket = io("http://localhost:5001");
    socket.on("connect", () => console.log("Connected to WebSocket server!"));

    socket.on("status_update", (msg: { data: string }) => {
      try {
        const update = JSON.parse(msg.data);
        if (update.type === "UPDATE") {
          setProcessSteps((prev) => {
            let activeIndex = prev.findIndex((s) => s.name === update.stage);
            return prev.map((s, i) => {
              if (i === activeIndex)
                return { ...s, displayText: update.text, status: "active" };
              if (i < activeIndex) return { ...s, status: "completed" };
              return s;
            });
          });
        } else if (update.type === "GRAPH_DATA") {
          setGraphmlData(update.data);
        } else if (update.type === "BROWSER_ADDRESS") {
          // Handle browser address from backend
          console.log("Received BROWSER_ADDRESS:", update);
          setBrowserNovncUrl(update.novnc_url);
          setBrowserCdpUrl(update.cdp_url);
          setBrowserCdpWebSocket(update.cdp_websocket);
          setIsBrowserOpen(true); // Automatically open browser viewer
        } else if (update.type === "DONE") {
          setFinalScore(update.score);
          setProcessSteps((prev) =>
            prev.map((s) => ({ ...s, status: "completed" }))
          );
          socket.disconnect();
        }
      } catch (e) {
        console.error("WebSocket parse error:", e);
      }
    });

    return () => socket.disconnect();
  }, [uploadedFile, submittedUrl]);

  const handleFileUpload = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    Object.entries(settings).forEach(([key, value]) =>
      formData.append(key, value.toString())
    );

    setProcessSteps(INITIAL_STAGES);
    setFinalScore(null);
    setGraphmlData(null);
    setUploadedFile(file);
    setSubmittedUrl(null);

    // Reset browser state
    setIsBrowserOpen(false);
    setBrowserNovncUrl(undefined);
    setBrowserCdpUrl(undefined);
    setBrowserCdpWebSocket(undefined);

    try {
      await axios.post("http://localhost:5001/api/upload", formData);
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

    // Reset browser state
    setIsBrowserOpen(false);
    setBrowserNovncUrl(undefined);
    setBrowserCdpUrl(undefined);
    setBrowserCdpWebSocket(undefined);

    try {
      await axios.post("http://localhost:5001/api/analyze-url", {
        url,
        ...settings,
      });
    } catch (error) {
      console.error("Error analyzing URL:", error);
    }
  };

  const handleSettingsSave = (newSettings: Settings) => {
    setSettings(newSettings);
    setIsSettingsOpen(false);
  };

  return (
    <>
      {!uploadedFile && !submittedUrl && <ParticleBackground />}

      <main className="flex min-h-screen flex-col bg-gradient-to-b from-white via-gray-50 to-white font-sans">
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
                  <span className="text-[11px] font-medium uppercase tracking-wide text-gray-500">
                    Integrity Score
                  </span>
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
              ></button>
            </div>
          )}
        </header>

        {/* Drawer Tab */}
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

        {/* Drawer */}
        <SettingsDrawer
          isOpen={isSettingsOpen}
          onClose={() => setIsSettingsOpen(false)}
          settings={settings}
          onSave={handleSettingsSave}
          graphmlData={graphmlData} // ✅ supports Save GraphML button
        />

        {/* Main content */}
        <div className="relative flex-grow overflow-hidden">
          {!uploadedFile && !submittedUrl ? (
            <div className="flex items-center justify-center p-6">
              <FileUploader
                onFileUpload={handleFileUpload}
                onUrlSubmit={handleUrlSubmit}
              />
            </div>
          ) : (
            <>
              <ProgressBar steps={processSteps} />

              {/* Browser Viewer - shows when browser info is received */}
              <BrowserViewer
                isOpen={isBrowserOpen}
                onClose={() => setIsBrowserOpen(false)}
                novncUrl={browserNovncUrl}
                cdpUrl={browserCdpUrl}
                cdpWebSocket={browserCdpWebSocket}
              />

              <div
                className="flex-grow p-4"
                style={{ height: "calc(100vh - 150px)" }}
              >
                <XmlGraphViewer
                  graphmlData={graphmlData}
                  isDrawerOpen={isSettingsOpen}
                />
              </div>
            </>
          )}
        </div>

        {/* Bottom status display */}
        {(uploadedFile || submittedUrl) && (
          <div
            className={`fixed bottom-2 left-1/2 -translate-x-1/2 text-center text-gray-600 text-[10px] font-mono bg-white/60 backdrop-blur-sm px-3 py-1.5 rounded-md border border-gray-200 max-w-[80%] truncate transition-opacity duration-500 ${
              uploadedFile || submittedUrl ? "opacity-100" : "opacity-0"
            }`}
          >
            Query:{" "}
            <span className="text-gray-700">
              {uploadedFile ? uploadedFile.name : submittedUrl}
            </span>
          </div>
        )}
      </main>
    </>
  );
};
export default IndexPage;
