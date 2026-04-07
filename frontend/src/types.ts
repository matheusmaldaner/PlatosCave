// PlatosCave/frontend/src/types.ts

// Process step for the progress bar
export interface ProcessStep {
  name: string;
  displayText: string;
  status: 'pending' | 'active' | 'completed';
}

// Analysis modes for different document types
export type AnalysisMode = 'academic' | 'journal' | 'finance';

// Mode descriptions for UI
export const ANALYSIS_MODES: { value: AnalysisMode; label: string; description: string; icon: string }[] = [
  {
    value: 'academic',
    label: 'Academic',
    description: 'General papers, dissertations, preprints',
    icon: '📚',
  },
  {
    value: 'journal',
    label: 'Journal',
    description: 'Peer-reviewed articles, clinical studies',
    icon: '🔬',
  },
  {
    value: 'finance',
    label: 'Finance',
    description: 'Earnings, SEC filings, market analysis',
    icon: '📊',
  },
];

// Settings that actually affect the backend
export interface Settings {
  maxNodes: number;
  agentAggressiveness: number;
  evidenceThreshold: number;
  useBrowserForVerification: boolean;  // Force browser-use for verification (slower but visible)
  analysisMode: AnalysisMode;
}

// Default settings values
export const DEFAULT_SETTINGS: Settings = {
  maxNodes: 10,
  agentAggressiveness: 5,
  evidenceThreshold: 0.8,
  useBrowserForVerification: false,  // Default: use fast Exa path
  analysisMode: 'academic',
};
