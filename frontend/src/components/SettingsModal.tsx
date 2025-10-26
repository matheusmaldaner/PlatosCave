// frontend/src/components/SettingsModal.tsx
import React from 'react';

export interface Settings {
  agentAggressiveness: number;
  evidenceThreshold: number;
  credibility: number;
  relevance: number;
  evidenceStrength: number;
  methodRigor: number;
  reproducibility: number;
  citationSupport: number;
  hypothesis: number;
  claim: number;
  method: number;
  evidence: number;
  result: number;
  conclusion: number;
  limitation: number;
}

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: Settings;
  onSave: (newSettings: Settings) => void;
}

const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onSave,
}) => {
  const [localSettings, setLocalSettings] = React.useState<Settings>(settings);

  React.useEffect(() => {
    setLocalSettings(settings);
  }, [settings]);

  if (!isOpen) return null;

  const handleSave = () => {
    onSave(localSettings);
    onClose();
  };

  const SliderRow = ({
    label,
    id,
    min,
    max,
    step,
    value,
    onChange,
  }: {
    label: string;
    id: string;
    min: number;
    max: number;
    step: number;
    value: number;
    onChange: (v: number) => void;
  }) => (
    <div className="mb-4">
      <label
        htmlFor={id}
        className="block text-gray-700 text-sm font-bold mb-1"
      >
        {label} ({min}–{max})
      </label>
      <input
        type="range"
        id={id}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-emerald-600"
      />
      <div className="text-center font-bold text-sm text-gray-700 mt-1">
        {value.toFixed(2)}
      </div>
    </div>
  );

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-40 flex justify-center items-center">
      <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-4xl overflow-y-auto max-h-[90vh]">
        <h2 className="text-2xl font-bold mb-6 text-center">Settings</h2>

        {/* Two-column layout */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left Column — Agent Behavior */}
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-2 border-b pb-1">
              Agent Behavior
            </h3>
            <SliderRow
              label="Agent Aggressiveness"
              id="agentAggressiveness"
              min={1}
              max={10}
              step={1}
              value={localSettings.agentAggressiveness}
              onChange={(v) =>
                setLocalSettings((s) => ({ ...s, agentAggressiveness: v }))
              }
            />
            <SliderRow
              label="Evidence Threshold"
              id="evidenceThreshold"
              min={0.1}
              max={1.0}
              step={0.1}
              value={localSettings.evidenceThreshold}
              onChange={(v) =>
                setLocalSettings((s) => ({ ...s, evidenceThreshold: v }))
              }
            />
          </div>

          {/* Right Column — Evaluation Weights */}
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-2 border-b pb-1">
              Evaluation Weights
            </h3>
            <SliderRow
              label="Credibility"
              id="credibility"
              min={0}
              max={2}
              step={0.1}
              value={localSettings.credibility}
              onChange={(v) =>
                setLocalSettings((s) => ({ ...s, credibility: v }))
              }
            />
            <SliderRow
              label="Relevance"
              id="relevance"
              min={0}
              max={2}
              step={0.1}
              value={localSettings.relevance}
              onChange={(v) =>
                setLocalSettings((s) => ({ ...s, relevance: v }))
              }
            />
            <SliderRow
              label="Evidence Strength"
              id="evidenceStrength"
              min={0}
              max={2}
              step={0.1}
              value={localSettings.evidenceStrength}
              onChange={(v) =>
                setLocalSettings((s) => ({ ...s, evidenceStrength: v }))
              }
            />
            <SliderRow
              label="Method Rigor"
              id="methodRigor"
              min={0}
              max={2}
              step={0.1}
              value={localSettings.methodRigor}
              onChange={(v) =>
                setLocalSettings((s) => ({ ...s, methodRigor: v }))
              }
            />
            <SliderRow
              label="Reproducibility"
              id="reproducibility"
              min={0}
              max={2}
              step={0.1}
              value={localSettings.reproducibility}
              onChange={(v) =>
                setLocalSettings((s) => ({ ...s, reproducibility: v }))
              }
            />
            <SliderRow
              label="Citation Support"
              id="citationSupport"
              min={0}
              max={2}
              step={0.1}
              value={localSettings.citationSupport}
              onChange={(v) =>
                setLocalSettings((s) => ({ ...s, citationSupport: v }))
              }
            />
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex items-center justify-end space-x-4 mt-8">
          <button
            onClick={onClose}
            className="text-gray-600 hover:text-gray-800 font-bold"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="bg-brand-green hover:bg-brand-green-dark text-white font-bold py-2 px-4 rounded transition"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
