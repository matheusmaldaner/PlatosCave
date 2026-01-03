// PlatosCave/frontend/src/components/SettingsPopover.tsx
import React, { useState, useRef, useEffect } from 'react';
import { Settings, AnalysisMode, ANALYSIS_MODES } from '../types';

interface SettingsPopoverProps {
  settings: Settings;
  onSettingsChange: (settings: Settings) => void;
  disabled?: boolean;
}

const SettingsPopover: React.FC<SettingsPopoverProps> = ({
  settings,
  onSettingsChange,
  disabled = false,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [local, setLocal] = useState<Settings>(settings);
  const popoverRef = useRef<HTMLDivElement>(null);

  // Sync local state when settings prop changes
  useEffect(() => {
    setLocal(settings);
  }, [settings]);

  // Close popover when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  const handleChange = (key: keyof Settings, value: number | boolean | AnalysisMode) => {
    const updated = { ...local, [key]: value };
    setLocal(updated);
    onSettingsChange(updated);
  };

  return (
    <div className="relative" ref={popoverRef}>
      {/* Gear Icon Button */}
      <button
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={`p-2 rounded-lg transition-colors duration-200 ${
          disabled
            ? 'text-gray-300 cursor-not-allowed'
            : 'text-gray-500 hover:bg-gray-100 hover:text-gray-700'
        }`}
        aria-label="Settings"
        title={disabled ? 'Settings locked during analysis' : 'Analysis settings'}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-5 w-5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
          />
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
          />
        </svg>
      </button>

      {/* Popover */}
      {isOpen && (
        <div className="absolute right-0 top-full mt-2 w-80 bg-white rounded-xl shadow-2xl border border-gray-200 p-4 z-50 animate-in fade-in slide-in-from-top-2 duration-200">
          <h3 className="font-semibold text-gray-900 text-sm mb-4 flex items-center gap-2">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-4 w-4 text-green-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
              />
            </svg>
            Analysis Settings
          </h3>

          <div className="space-y-4">
            {/* Analysis Mode Selector */}
            <div>
              <label className="text-xs font-medium text-gray-700 mb-2 block">Analysis Mode</label>
              <div className="grid grid-cols-3 gap-2">
                {ANALYSIS_MODES.map((mode) => (
                  <button
                    key={mode.value}
                    onClick={() => handleChange('analysisMode', mode.value)}
                    className={`flex flex-col items-center p-2 rounded-lg border-2 transition-all duration-200 ${
                      local.analysisMode === mode.value
                        ? 'border-green-500 bg-green-50 shadow-sm'
                        : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                    }`}
                  >
                    <span className="text-lg mb-1">{mode.icon}</span>
                    <span className={`text-[10px] font-medium ${
                      local.analysisMode === mode.value ? 'text-green-700' : 'text-gray-600'
                    }`}>
                      {mode.label}
                    </span>
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-gray-400 mt-1.5 text-center">
                {ANALYSIS_MODES.find(m => m.value === local.analysisMode)?.description}
              </p>
            </div>

            <div className="border-t border-gray-100 pt-3">
              <p className="text-[10px] font-medium text-gray-500 uppercase tracking-wide mb-3">Advanced</p>
            </div>

            {/* Max Nodes */}
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-xs font-medium text-gray-700">Max Nodes</label>
                <span className="text-xs font-mono text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                  {local.maxNodes}
                </span>
              </div>
              <input
                type="range"
                min={5}
                max={20}
                step={1}
                value={local.maxNodes}
                onChange={(e) => handleChange('maxNodes', parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-600"
              />
              <div className="flex justify-between text-[10px] text-gray-400 mt-0.5">
                <span>5</span>
                <span>20</span>
              </div>
            </div>

            {/* Agent Aggressiveness */}
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-xs font-medium text-gray-700">Agent Aggressiveness</label>
                <span className="text-xs font-mono text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                  {local.agentAggressiveness}
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={10}
                step={1}
                value={local.agentAggressiveness}
                onChange={(e) => handleChange('agentAggressiveness', parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-600"
              />
              <div className="flex justify-between text-[10px] text-gray-400 mt-0.5">
                <span>1 (conservative)</span>
                <span>10 (thorough)</span>
              </div>
            </div>

            {/* Evidence Threshold */}
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-xs font-medium text-gray-700">Evidence Threshold</label>
                <span className="text-xs font-mono text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                  {local.evidenceThreshold.toFixed(1)}
                </span>
              </div>
              <input
                type="range"
                min={0.1}
                max={1.0}
                step={0.1}
                value={local.evidenceThreshold}
                onChange={(e) => handleChange('evidenceThreshold', parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-600"
              />
              <div className="flex justify-between text-[10px] text-gray-400 mt-0.5">
                <span>0.1 (lenient)</span>
                <span>1.0 (strict)</span>
              </div>
            </div>

            {/* Use Browser for Verification Toggle */}
            <div className="pt-2 border-t border-gray-100">
              <div className="flex justify-between items-center">
                <div className="flex-1">
                  <label className="text-xs font-medium text-gray-700">Visible Verification</label>
                  <p className="text-[10px] text-gray-400 mt-0.5">
                    Use browser for source checking (slower but transparent)
                  </p>
                </div>
                <button
                  onClick={() => handleChange('useBrowserForVerification', !local.useBrowserForVerification)}
                  className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-1 ${
                    local.useBrowserForVerification ? 'bg-green-600' : 'bg-gray-200'
                  }`}
                  role="switch"
                  aria-checked={local.useBrowserForVerification}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform duration-200 ${
                      local.useBrowserForVerification ? 'translate-x-4' : 'translate-x-0.5'
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>

          {/* Info text */}
          <p className="text-[10px] text-gray-400 mt-4 pt-3 border-t border-gray-100">
            Settings apply to the next analysis. Changes take effect immediately.
          </p>
        </div>
      )}
    </div>
  );
};

export default SettingsPopover;
