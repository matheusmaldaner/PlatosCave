// PlatosCave/frontend/src/types.ts

// Process step for the progress bar
export interface ProcessStep {
  name: string;
  displayText: string;
  status: 'pending' | 'active' | 'completed';
}

// Settings that actually affect the backend
export interface Settings {
  maxNodes: number;
  agentAggressiveness: number;
  evidenceThreshold: number;
}

// Default settings values
export const DEFAULT_SETTINGS: Settings = {
  maxNodes: 10,
  agentAggressiveness: 5,
  evidenceThreshold: 0.8,
};
