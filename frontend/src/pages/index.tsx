// PlatosCave/frontend/src/pages/index.tsx
import React, { useState, useEffect } from "react";
import axios from "axios";
import { io, Socket } from "socket.io-client";
import FileUploader from "../components/FileUploader";
import XmlGraphViewer from "../components/XmlGraphViewer";
import BrowserViewer from "../components/BrowserViewer";
import SettingsPopover from "../components/SettingsPopover";
import ProgressBar from "../components/ProgressBar";
import ParticleBackground from "../components/ParticleBackground";
import NodeDetailPanel from "../components/NodeDetailPanel";
import { ProcessStep, Settings, DEFAULT_SETTINGS } from "../types";
import platosCaveLogo from "../images/platos-cave-logo.svg";

const API_URL = process.env.GATSBY_API_URL || "";

const INITIAL_STAGES: ProcessStep[] = [
  { name: "Validate", displayText: "Pending...", status: "pending" },
  { name: "Decomposing PDF", displayText: "Pending...", status: "pending" },
  {
    name: "Building Logic Tree",
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
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Browser viewer state
  const [isBrowserOpen, setIsBrowserOpen] = useState(false);
  const [browserNovncUrl, setBrowserNovncUrl] = useState<string | undefined>(
    undefined,
  );
  const [browserCdpUrl, setBrowserCdpUrl] = useState<string | undefined>(
    undefined,
  );
  const [browserCdpWebSocket, setBrowserCdpWebSocket] = useState<
    string | undefined
  >(undefined);

  // Active node being verified (for NodeToolbar indicator)
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null);

  // Force browser viewer to expand (increments to trigger effect)
  const [browserExpandTrigger, setBrowserExpandTrigger] = useState(0);

  // Edge confidence updates (real-time as verification progresses)
  // Map of "source->target" to confidence value
  const [edgeUpdates, setEdgeUpdates] = useState<Record<string, number>>({});

  // Track which nodes have been verified (set of node IDs like "1", "2", etc.)
  const [verifiedNodeIds, setVerifiedNodeIds] = useState<Set<string>>(new Set());

  // Node detail panel state (selected node)
  const [selectedNode, setSelectedNode] = useState<{
    nodeId: string;
    role: string;
    text: string;
    integrityScorePct: number | null;
  } | null>(null);

  // Settings state - only 3 settings that actually work
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);

  // WebSocket connection for updates
  useEffect(() => {
    if (!uploadedFile && !submittedUrl) return;

    const socket: Socket = io(API_URL);
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
          console.log("Received BROWSER_ADDRESS:", update);
          setBrowserNovncUrl(update.novnc_url);
          setBrowserCdpUrl(update.cdp_url);
          setBrowserCdpWebSocket(update.cdp_websocket);
          setIsBrowserOpen(true);
        } else if (update.type === "NODE_ACTIVE") {
          // Track which node is currently being verified
          setActiveNodeId(update.node_id || null);
        } else if (update.type === "EDGE_UPDATE") {
          // Real-time edge confidence update from verification
          const edgeKey = `${update.source}->${update.target}`;
          setEdgeUpdates((prev) => ({
            ...prev,
            [edgeKey]: update.confidence,
          }));
        } else if (update.type === "METRIC_UPDATE") {
          // Mark this node as verified in the graph
          if (update.node_id) {
            setVerifiedNodeIds(prev => new Set(prev).add(update.node_id));
          }
        } else if (update.type === "VERIFICATION_PROGRESS") {
          // Update progress text with verification counter
          setProcessSteps((prev) =>
            prev.map((s) =>
              s.name === "Compiling Evidence"
                ? { ...s, displayText: `Verified ${update.current}/${update.total}: ${update.node_text}`, status: "active" }
                : s
            )
          );
        } else if (update.type === "DONE") {
          setFinalScore(update.score);
          setIsAnalyzing(false);
          setActiveNodeId(null); // Clear active node when complete
          setProcessSteps((prev) =>
            prev.map((s) => ({ ...s, status: "completed" })),
          );
          socket.disconnect();
        }
      } catch (e) {
        console.error("WebSocket parse error:", e);
      }
    });

    return () => {
      socket.disconnect();
    };
  }, [uploadedFile, submittedUrl]);

  const handleFileUpload = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("maxNodes", settings.maxNodes.toString());
    formData.append(
      "agentAggressiveness",
      settings.agentAggressiveness.toString(),
    );
    formData.append("evidenceThreshold", settings.evidenceThreshold.toString());
    formData.append("useBrowserForVerification", settings.useBrowserForVerification.toString());
    formData.append("analysisMode", settings.analysisMode);

    setProcessSteps(INITIAL_STAGES);
    setFinalScore(null);
    setGraphmlData(null);
    setUploadedFile(file);
    setSubmittedUrl(null);
    setIsAnalyzing(true);
    setSelectedNode(null);

    // Reset browser state
    setIsBrowserOpen(false);
    setBrowserNovncUrl(undefined);
    setBrowserCdpUrl(undefined);
    setBrowserCdpWebSocket(undefined);
    setEdgeUpdates({});
    setVerifiedNodeIds(new Set());

    try {
      await axios.post(`${API_URL}/api/upload`, formData);
    } catch (error) {
      console.error("Error uploading file:", error);
      setIsAnalyzing(false);
    }
  };

  const handleUrlSubmit = async (url: string) => {
    setProcessSteps(INITIAL_STAGES);
    setFinalScore(null);
    setGraphmlData(null);
    setSubmittedUrl(url);
    setUploadedFile(null);
    setIsAnalyzing(true);
    setSelectedNode(null);

    // Reset browser state
    setIsBrowserOpen(false);
    setBrowserNovncUrl(undefined);
    setBrowserCdpUrl(undefined);
    setBrowserCdpWebSocket(undefined);
    setEdgeUpdates({});
    setVerifiedNodeIds(new Set());

    try {
      await axios.post(`${API_URL}/api/analyze-url`, {
        url,
        maxNodes: settings.maxNodes,
        agentAggressiveness: settings.agentAggressiveness,
        evidenceThreshold: settings.evidenceThreshold,
        useBrowserForVerification: settings.useBrowserForVerification,
        analysisMode: settings.analysisMode,
      });
    } catch (error) {
      console.error("Error analyzing URL:", error);
      setIsAnalyzing(false);
    }
  };

  const formatScorePct = (score: number) => {
    // Backend may send either 0..1 or 0..100. Display as an integer percent.
    const pct = score <= 1 ? score * 100 : score;
    return `${Math.round(pct)}%`;
  };

  return (
    <>
      {!uploadedFile && !submittedUrl && <ParticleBackground />}

      <main className="flex min-h-screen flex-col bg-[#f8f7f4] font-sans">
        {/* Header */}
        <header className="relative z-10 grid w-full grid-cols-[auto,1fr,auto] items-center gap-3 border-b border-gray-200 bg-[#f8f7f4] px-4 py-3 sm:px-6 sm:py-4">
          <button
            onClick={() => window.location.reload()}
            className="flex items-center cursor-pointer hover:opacity-70 transition-opacity duration-200"
            aria-label="Return to home"
          >
            <img
              src={platosCaveLogo}
              alt="Plato's Cave"
              className="block h-8 w-auto"
            />
          </button>

          {/* Center search / query bar */}
          <div className="flex items-center justify-center">
            {(uploadedFile || submittedUrl) ? (
              <div className="relative w-full max-w-2xl">
                <div className="pointer-events-none absolute inset-y-0 left-3 flex items-center text-gray-400">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <path d="M21 21l-4.35-4.35m1.85-5.15a7 7 0 11-14 0 7 7 0 0114 0z" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
                  </svg>
                </div>
                <input
                  value={uploadedFile ? uploadedFile.name : (submittedUrl || "")}
                  readOnly
                  className="w-full rounded-md border border-gray-200 bg-white px-10 py-2 text-xs text-gray-700 shadow-sm outline-none placeholder:text-gray-400 sm:text-sm"
                />
              </div>
            ) : (
              <div className="h-[34px] w-full max-w-2xl" />
            )}
          </div>

          {/* Right controls */}
          <div className="flex items-center gap-3 justify-self-end">
            {finalScore !== null && (
              <div className="rounded-md border border-gray-200 bg-white px-3 py-1.5 text-xs text-gray-700 shadow-sm">
                <span className="text-gray-500">Score:</span>{" "}
                <span className="font-semibold text-gray-900">{formatScorePct(finalScore)}</span>
              </div>
            )}

            <SettingsPopover settings={settings} onSettingsChange={setSettings} disabled={isAnalyzing} />
          </div>
        </header>

        {/* Main content */}
        <div className="relative flex-grow">
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
                hideMinimized={!!activeNodeId}
                expandTrigger={browserExpandTrigger}
              />

              <div
                className="relative flex-grow p-4"
                style={{ height: "calc(100vh - 150px)" }}
              >
                <div className="relative h-full w-full overflow-hidden">
                  <div
                    className="h-full transition-[padding-right] duration-300"
                    style={{
                      paddingRight: selectedNode ? "var(--detail-panel-w)" : 0,
                      // single source of truth for panel width
                      ["--detail-panel-w" as any]: "360px",
                    }}
                  >
                    <XmlGraphViewer
                      graphmlData={graphmlData}
                      activeNodeId={activeNodeId}
                      edgeUpdates={edgeUpdates}
                      verifiedNodeIds={verifiedNodeIds}
                      onBrowserClick={() => {
                        setIsBrowserOpen(true);
                        setBrowserExpandTrigger(prev => prev + 1);
                      }}
                      onNodeSelect={(payload) => {
                        setSelectedNode(payload);
                      }}
                    />
                  </div>

                  <NodeDetailPanel
                    isOpen={!!selectedNode}
                    node={selectedNode}
                    onClose={() => setSelectedNode(null)}
                    onViewSource={() => {
                      setIsBrowserOpen(true);
                      setBrowserExpandTrigger((prev) => prev + 1);
                    }}
                  />
                </div>
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
