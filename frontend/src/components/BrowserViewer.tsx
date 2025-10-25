import React, { useState, useEffect } from "react";

interface BrowserViewerProps {
  isOpen: boolean;
  onClose: () => void;
  novncUrl?: string;
  cdpUrl?: string;
  cdpWebSocket?: string;
}

const BrowserViewer: React.FC<BrowserViewerProps> = ({
  isOpen,
  onClose,
  novncUrl,
  cdpUrl,
  cdpWebSocket
}) => {
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Close on Escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className={`fixed inset-0 bg-slate-900/70 backdrop-blur-sm z-40 transition-opacity duration-300 ${
          isOpen ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
        onClick={onClose}
      />

      {/* Centered Viewer Shell */}
      <div
        className={`fixed inset-0 z-50 flex items-center justify-center p-6 transition-opacity duration-300 ${
          isOpen ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
      >
        <div
          className={`relative flex flex-col overflow-hidden border border-white/30 bg-white/95 shadow-2xl transition-all duration-300 ${
            isFullscreen
              ? "h-full w-full rounded-none"
              : "h-[75vh] w-full max-w-5xl rounded-3xl"
          }`}
          style={isFullscreen ? undefined : { backdropFilter: "blur(6px)" }}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-5 py-4 border-b border-white/40 bg-white/80">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="font-medium text-gray-700 text-sm">
              Browser Automation
            </span>
          </div>

          <div className="flex items-center space-x-2">
            {/* Fullscreen Toggle */}
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
                className="px-3 py-2 text-xs font-semibold uppercase tracking-wide text-gray-600 hover:bg-gray-200 rounded-lg transition-colors"
              title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
            >
                {isFullscreen ? "Exit" : "Full"}
            </button>

            {/* Close Button */}
            <button
              onClick={onClose}
                className="px-3 py-2 text-xs font-semibold uppercase tracking-wide text-gray-600 hover:bg-gray-200 rounded-lg transition-colors"
              title="Close"
            >
                Close
            </button>
          </div>
        </div>

          {/* Browser Content */}
          <div className="flex-1 bg-slate-950/90">
            {novncUrl ? (
              <iframe
                key={novncUrl}
                src={novncUrl}
                className="h-full w-full border-0"
                title="Browser Automation View"
                allowFullScreen
                sandbox="allow-same-origin allow-scripts allow-popups allow-forms"
              />
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-slate-200">
                Waiting for remote browser session...
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="px-5 py-3 border-t border-white/40 bg-white/80 text-xs text-gray-600">
            <div className="flex flex-col space-y-1">
              {cdpUrl && (
                <span>
                  CDP Endpoint: <a className="text-blue-600 underline" href={cdpUrl} target="_blank" rel="noreferrer">{cdpUrl}</a>
                </span>
              )}
              {cdpWebSocket && (
                <span className="break-all">WebSocket: {cdpWebSocket}</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default BrowserViewer;
