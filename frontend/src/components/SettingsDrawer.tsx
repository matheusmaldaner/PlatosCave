import React from "react";
import { Settings } from "./SettingsModal";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  settings: Settings;
  onSave: (newSettings: Settings) => void;
  showTypeColors: boolean;
  setShowTypeColors: (v: boolean) => void;
}

const SettingsDrawer: React.FC<Props> = ({
  isOpen,
  onClose,
  settings,
  onSave,
  showTypeColors,
  setShowTypeColors,
}) => {
  const [local, setLocal] = React.useState(settings);

  React.useEffect(() => setLocal(settings), [settings]);

  if (!isOpen) return null;

  const handleSave = () => {
    onSave(local);
  };

  return (
    <div
      className={`fixed left-0 z-50 bg-white shadow-2xl border-r border-gray-200 transform transition-transform duration-300 ease-in-out w-80 ${
        isOpen ? "translate-x-0" : "-translate-x-full"
      }`}
      style={{
        top: "8rem",
        bottom: "3rem",
        borderRadius: "0 1rem 1rem 0",
        maxHeight: "calc(100vh - 6rem)",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Drawer Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-300 bg-gray-50 rounded-t-lg shadow-sm">
        <h2 className="text-lg font-semibold text-gray-800">Settings</h2>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-700 font-bold text-xl"
        >
          ×
        </button>
      </div>

      {/* Visualization Toggle */}
      <div className="px-4 py-3 flex items-center justify-between border-b border-gray-300 bg-white sticky top-0 z-10">
        <span className="text-sm font-medium text-gray-700">Toggle Type</span>
        <label className="relative inline-flex items-center cursor-pointer">
          <input
            type="checkbox"
            checked={showTypeColors}
            onChange={() => setShowTypeColors(!showTypeColors)}
            className="sr-only peer"
          />
          <div className="w-11 h-6 bg-gray-300 rounded-full peer peer-checked:bg-green-500 transition-all duration-300"></div>
          <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform duration-300 peer-checked:translate-x-5"></div>
        </label>
      </div>

      {/* Scrollable Settings Section */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* --- AGENT PARAMETERS SECTION --- */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2 uppercase">
            Agent Parameters
          </h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-600 font-medium">
                Agent Aggressiveness ({local.agentAggressiveness})
              </label>
              <input
                type="range"
                min="1"
                max="10"
                value={local.agentAggressiveness}
                onChange={(e) =>
                  setLocal({
                    ...local,
                    agentAggressiveness: parseInt(e.target.value),
                  })
                }
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-600 font-medium">
                Evidence Threshold ({local.evidenceThreshold.toFixed(1)})
              </label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                value={local.evidenceThreshold}
                onChange={(e) =>
                  setLocal({
                    ...local,
                    evidenceThreshold: parseFloat(e.target.value),
                  })
                }
                className="w-full"
              />
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="border-t border-gray-300" />

        {/* --- ROLE WEIGHTS SECTION --- */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2 uppercase">
            Role Weights
          </h3>
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
                  {label} ({(local as any)[key]?.toFixed(2) ?? 1.0})
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={(local as any)[key] ?? 1.0}
                  onChange={(e) =>
                    setLocal({ ...local, [key]: parseFloat(e.target.value) })
                  }
                  className="w-full"
                />
              </div>
            ))}
          </div>
        </div>

        {/* Divider */}
        <div className="border-t border-gray-300" />

        {/* --- SCORING WEIGHTS SECTION --- */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2 uppercase">
            Scoring Weights
          </h3>
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
                  {label} ({(local as any)[key].toFixed(2)})
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={(local as any)[key]}
                  onChange={(e) =>
                    setLocal({ ...local, [key]: parseFloat(e.target.value) })
                  }
                  className="w-full"
                />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Sticky Footer Section */}
      <div className="p-4 border-t border-gray-300 bg-gray-50 rounded-b-lg shadow-inner flex flex-col gap-2">
        <button
          onClick={handleSave}
          className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-2 rounded-lg transition-colors"
        >
          Apply Changes
        </button>

        <button
          onClick={() => console.log("TODO: Export GraphML")}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded-lg transition-colors"
        >
          Save GraphML
        </button>
      </div>
    </div>
  );
};

export default SettingsDrawer;
