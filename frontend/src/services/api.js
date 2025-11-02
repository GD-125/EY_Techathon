import axios from 'axios';

// API base URL - uses proxy configured in package.json for development
const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  }
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized - redirect to login
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API service methods
const apiService = {
  // Chat endpoints
  sendChatMessage: (data) => api.post('/chat', data),
  getChatHistory: (sessionId) => api.get(`/chat/history/${sessionId}`),

  // Data upload endpoints
  uploadData: (formData) => {
    return axios.post('/api/data/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 60000 // 60 seconds for file upload
    });
  },

  analyzeData: (fileId) => {
    return api.post('/data/analyze', { file_id: fileId });
  },

  listFiles: () => api.get('/data/files'),

  deleteFile: (fileId) => api.delete(`/data/files/${fileId}`),

  getSampleData: () => api.get('/data/sample'),

  // Application endpoints
  getApplications: (params) => api.get('/applications', { params }),

  getApplicationById: (id) => api.get(`/applications/${id}`),

  // Statistics endpoints
  getStats: () => api.get('/stats'),

  getDashboardData: () => api.get('/dashboard'),

  // Health check
  healthCheck: () => api.get('/health'),
};

export default apiService;
