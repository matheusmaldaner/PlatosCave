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

const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose, settings, onSave }) => {
    const [localSettings, setLocalSettings] = React.useState<Settings>(settings);

    React.useEffect(() => {
        setLocalSettings(settings);
    }, [settings]);

    if (!isOpen) {
        return null;
    }

    const handleSave = () => {
        onSave(localSettings);
        onClose();
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-40 flex justify-center items-center">
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-md">
                <h2 className="text-2xl font-bold mb-4">Settings</h2>
                
                {/* Agent Aggressiveness Setting */}
                <div className="mb-4">
                    <label htmlFor="agentAggressiveness" className="block text-gray-700 text-sm font-bold mb-2">
                        Agent Aggressiveness (1-10)
                    </label>
                    <input
                        type="range"
                        id="agentAggressiveness"
                        min="1"
                        max="10"
                        value={localSettings.agentAggressiveness}
                        onChange={(e) => setLocalSettings(s => ({ ...s, agentAggressiveness: parseInt(e.target.value) }))}
                        className="w-full"
                    />
                    <div className="text-center font-bold">{localSettings.agentAggressiveness}</div>
                </div>

                {/* Evidence Threshold Setting */}
                <div className="mb-6">
                    <label htmlFor="evidenceThreshold" className="block text-gray-700 text-sm font-bold mb-2">
                        Evidence Threshold (0.1 - 1.0)
                    </label>
                    <input
                        type="range"
                        id="evidenceThreshold"
                        min="0.1"
                        max="1.0"
                        step="0.1"
                        value={localSettings.evidenceThreshold}
                        onChange={(e) => setLocalSettings(s => ({ ...s, evidenceThreshold: parseFloat(e.target.value) }))}
                        className="w-full"
                    />
                     <div className="text-center font-bold">{localSettings.evidenceThreshold.toFixed(1)}</div>
                </div>

                {/* Action Buttons */}
                <div className="flex items-center justify-end space-x-4">
                    <button onClick={onClose} className="text-gray-600 hover:text-gray-800 font-bold">Cancel</button>
                    <button onClick={handleSave} className="bg-brand-green hover:bg-brand-green-dark text-white font-bold py-2 px-4 rounded">
                        Save
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SettingsModal;
