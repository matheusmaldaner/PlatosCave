export const API_CONFIG = {
  baseUrl: import.meta.env.VITE_API_URL || 'http://localhost:5001',
  wsUrl: import.meta.env.VITE_WS_URL || import.meta.env.VITE_API_URL || 'http://localhost:5001',
};

// API endpoints
export const API_ENDPOINTS = {
  upload: `${API_CONFIG.baseUrl}/api/upload`,
  analyzeUrl: `${API_CONFIG.baseUrl}/api/analyze-url`,
  cleanup: `${API_CONFIG.baseUrl}/api/cleanup`,
};

export default API_CONFIG;
