import React, { useState, useEffect } from "react";

interface BrowserViewerProps {
  isOpen: boolean;
  onClose: () => void;
  novncUrl?: string;
  cdpUrl?: string;
  cdpWebSocket?: string;
  hideMinimized?: boolean;
  expandTrigger?: number;
}

const BrowserViewer: React.FC<BrowserViewerProps> = ({
  isOpen,
  onClose,
  novncUrl,
  cdpUrl,
  cdpWebSocket,
  hideMinimized = false,
  expandTrigger = 0
}) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);

  // When expandTrigger changes (incremented), un-minimize the viewer
  useEffect(() => {
    if (expandTrigger > 0) {
      setIsMinimized(false);
    }
  }, [expandTrigger]);

  // Add scaling parameters to noVNC URL to fit container perfectly
  const getScaledVncUrl = (url?: string) => {
    if (!url) return url;
    const separator = url.includes('?') ? '&' : '?';
    return `${url}${separator}resize=scale&autoconnect=true`;
  };

  // Close on Escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (isFullscreen) {
          setIsFullscreen(false);
        }
      }
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [isFullscreen]);

  if (!isOpen) return null;

  // Fullscreen overlay
  if (isFullscreen) {
    return (
      <div className="fixed inset-0 z-50 flex flex-col bg-white">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-200 bg-white">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="font-medium text-gray-700 text-sm">
              Browser Automation
            </span>
          </div>

          <div className="flex items-center space-x-2">
            {/* Exit Fullscreen */}
            <button
              onClick={() => setIsFullscreen(false)}
              className="p-2 text-gray-600 hover:bg-gray-200 rounded-lg transition-colors"
              title="Exit fullscreen"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 9V4.5M9 9H4.5M9 9L3.75 3.75M15 9h4.5M15 9V4.5M15 9l5.25-5.25M9 15v4.5M9 15H4.5M9 15l-5.25 5.25M15 15h4.5M15 15v4.5m0-4.5l5.25 5.25" />
              </svg>
            </button>
          </div>
        </div>

        {/* Browser Content */}
        <div className="flex-1 bg-slate-950/90">
          {novncUrl ? (
            <iframe
              key={novncUrl}
              src={getScaledVncUrl(novncUrl)}
              className="h-full w-full border-0 block"
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
      </div>
    );
  }

  // Minimized state - floating indicator in top-left
  // Hide when NodeToolbar indicator is active (during node verification)
  if (isMinimized) {
    if (hideMinimized) {
      return null;
    }
    return (
      <div className="fixed top-20 left-4 z-30">
        <button
          onClick={() => setIsMinimized(false)}
          className="flex items-center space-x-2 px-4 py-3 bg-white border border-gray-200 rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105"
        >
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="font-medium text-gray-700 text-sm">
            Browser Active
          </span>
          <svg
            className="w-4 h-4 text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
        </button>
      </div>
    );
  }

  // Inline expanded view
  return (
    <div className="w-[70vw] mx-auto mb-6">
      <div className="relative flex flex-col overflow-hidden border border-gray-200 bg-white rounded-2xl" style={{ height: '80vh' }}>
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-white">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="font-medium text-gray-700 text-sm">
              Browser Automation
            </span>
          </div>

          <div className="flex items-center space-x-1">
            {/* Minimize Button */}
            <button
              onClick={() => setIsMinimized(true)}
              className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              title="Minimize"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
              </svg>
            </button>

            {/* Fullscreen Toggle */}
            <button
              onClick={() => setIsFullscreen(true)}
              className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              title="Fullscreen"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
              </svg>
            </button>

            {/* Close Button */}
            <button
              onClick={onClose}
              className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              title="Close"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Browser Content */}
        <div className="flex-1 bg-slate-950/90 overflow-hidden">
          {novncUrl ? (
            <iframe
              key={novncUrl}
              src={getScaledVncUrl(novncUrl)}
              className="h-full w-full border-0 block"
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
        <div className="px-4 py-2 border-t border-gray-200 bg-gray-50 text-xs text-gray-600">
          <div className="flex flex-col space-y-1">
            {cdpUrl && (
              <span>
                CDP: <a className="text-blue-600 underline" href={cdpUrl} target="_blank" rel="noreferrer">{cdpUrl}</a>
              </span>
            )}
            {cdpWebSocket && (
              <span className="break-all">WS: {cdpWebSocket}</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default BrowserViewer;
