// frontend/src/components/SettingsDrawer.tsx
import React from "react";
import { Settings } from "./SettingsModal";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  settings: Settings;
  onSave: (newSettings: Settings) => void;
  // toggle colors
  showTypeColors: boolean;
  setShowTypeColors: (v: boolean) => void;
  // optional: callback to export GraphML (index page passes graphmlData handler)
  onSaveGraph?: () => void;
}

const SettingsDrawer: React.FC<Props> = ({
  isOpen,
  onClose,
  settings,
  onSave,
  showTypeColors,
  setShowTypeColors,
  onSaveGraph,
}) => {
  const [local, setLocal] = React.useState<Settings>(settings);

  React.useEffect(() => {
    setLocal(settings);
  }, [settings, isOpen]);

  if (!isOpen) return null;

  const handleSave = () => {
    // perform basic normalization / bounds safety if needed
    const normalized: Settings = {
      ...local,
      // ensure numeric fields are numbers (in case)
      agentAggressiveness: Number(local.agentAggressiveness) || 1,
      evidenceThreshold: Number(local.evidenceThreshold) || 0.1,
      credibility: Number(local.credibility) || 1,
      relevance: Number(local.relevance) || 1,
      evidenceStrength: Number(local.evidenceStrength) || 1,
      methodRigor: Number(local.methodRigor) || 1,
      reproducibility: Number(local.reproducibility) || 1,
      citationSupport: Number(local.citationSupport) || 1,
      hypothesis: Number((local as any).hypothesis) || 1,
      claim: Number((local as any).claim) || 1,
      method: Number((local as any).method) || 1,
      evidence: Number((local as any).evidence) || 1,
      result: Number((local as any).result) || 1,
      conclusion: Number((local as any).conclusion) || 1,
      limitation: Number((local as any).limitation) || 1,
    };

    onSave(normalized);
  };

  return (
    <div
      className={`fixed left-0 z-50 bg-white shadow-2xl border-r border-gray-200 transform transition-transform duration-300 ease-in-out w-80 ${
        isOpen ? "translate-x-0" : "-translate-x-full"
      }`}
      style={{
        top: "6rem",
        bottom: "3rem",
        borderRadius: "0 1rem 1rem 0",
        maxHeight: "calc(100vh - 9rem)",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-300 bg-gray-50">
        <h2 className="text-lg font-semibold text-gray-800">Settings</h2>
        <button onClick={onClose} className="text-gray-500 hover:text-gray-700 font-bold text-xl">
          ×
        </button>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Toggle Type Colors */}
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-gray-700">Toggle Type</h3>
            <p className="text-xs text-gray-500">Apply role colors to node outlines</p>
          </div>

          <div className="flex items-center gap-2">
            <label htmlFor="toggle-type" className="relative inline-flex items-center cursor-pointer">
              <input
                id="toggle-type"
                type="checkbox"
                checked={showTypeColors}
                onChange={(e) => setShowTypeColors(e.target.checked)}
                className="sr-only"
              />
              <div
                className={`w-11 h-6 rounded-full transition-colors ${showTypeColors ? "bg-green-600" : "bg-gray-300"}`}
                aria-hidden
              />
              <span
                className={`absolute left-0.5 top-0.5 w-5 h-5 bg-white rounded-full shadow transform transition-transform ${
                  showTypeColors ? "translate-x-5" : "translate-x-0"
                }`}
                aria-hidden
              />
            </label>
          </div>
        </div>

        <div className="border-t border-gray-300" />

        {/* AGENT PARAMETERS */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2 uppercase">Agent Parameters</h3>

          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-600 font-medium">
                Agent Aggressiveness ({local.agentAggressiveness})
              </label>
              <input
                type="range"
                min={1}
                max={10}
                value={local.agentAggressiveness}
                onChange={(e) => setLocal({ ...local, agentAggressiveness: Number(e.target.value) })}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-600 font-medium">
                Evidence Threshold ({Number(local.evidenceThreshold).toFixed(1)})
              </label>
              <input
                type="range"
                min={0.1}
                max={1.0}
                step={0.1}
                value={local.evidenceThreshold}
                onChange={(e) => setLocal({ ...local, evidenceThreshold: Number(e.target.value) })}
                className="w-full"
              />
            </div>
          </div>
        </div>

        <div className="border-t border-gray-300" />

        {/* ROLE WEIGHTS */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2 uppercase">Role Weights</h3>

          <div className="space-y-4">
            {(
              [
                ["Hypothesis", "hypothesis"],
                ["Claim", "claim"],
                ["Method", "method"],
                ["Evidence", "evidence"],
                ["Result", "result"],
                ["Conclusion", "conclusion"],
                ["Limitation", "limitation"],
              ] as const
            ).map(([label, key]) => (
              <div key={key}>
                <label className="block text-sm text-gray-600 font-medium">
                  {label} ({(local as any)[key]?.toFixed?.(2) ?? Number((local as any)[key] ?? 1).toFixed(2)})
                </label>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.05}
                  value={(local as any)[key] ?? 1}
                  onChange={(e) => setLocal({ ...local, [key]: Number(e.target.value) } as any)}
                  className="w-full"
                />
              </div>
            ))}
          </div>
        </div>

        <div className="border-t border-gray-300" />

        {/* SCORING WEIGHTS */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2 uppercase">Scoring Weights</h3>

          <div className="space-y-4">
            {(
              [
                ["Credibility", "credibility"],
                ["Relevance", "relevance"],
                ["Evidence Strength", "evidenceStrength"],
                ["Method Rigor", "methodRigor"],
                ["Reproducibility", "reproducibility"],
                ["Citation Support", "citationSupport"],
              ] as const
            ).map(([label, key]) => (
              <div key={key}>
                <label className="block text-sm text-gray-600 font-medium">
                  {label} ({Number((local as any)[key]).toFixed(2)})
                </label>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.05}
                  value={(local as any)[key]}
                  onChange={(e) => setLocal({ ...local, [key]: Number(e.target.value) } as any)}
                  className="w-full"
                />
              </div>
            ))}
          </div>
        </div>

        {/* Extra spacing at bottom of scroll area so footer isn't overlapping content */}
        <div style={{ height: 12 }} />
      </div>

      {/* Sticky Footer with Apply + Save */}
      <div className="p-4 border-t border-gray-300 bg-gray-50 rounded-b-lg shadow-inner flex flex-col gap-2">
        <button
          onClick={handleSave}
          className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-2 rounded-lg transition-colors"
        >
          Apply Changes
        </button>

        <button
          onClick={() => {
            if (onSaveGraph) onSaveGraph();
            else {
              // fallback placeholder if parent hasn't provided an onSaveGraph
              console.log("Save GraphML: onSaveGraph not provided by parent.");
            }
          }}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded-lg transition-colors"
        >
          Save GraphML
        </button>
      </div>
    </div>
  );
};

export default SettingsDrawer;
