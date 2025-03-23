import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_BASE_URL;

// Create axios instance with default config
const apiClient = axios.create({
    baseURL: API_BASE,
    timeout: 10000,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Add response interceptor for error handling
apiClient.interceptors.response.use(
    (response) => response,
    (error) => {
        console.error('API Error:', error);
        return Promise.reject(error);
    }
);

// API endpoints
export const api = {
    // Get system status
    getStatus: async () => {
        const response = await apiClient.get('/status');
        return response.data;
    },

    // Get trade log
    getTrades: async () => {
        const response = await apiClient.get('/trades');
        return response.data;
    },

    // Health check
    checkHealth: async () => {
        const response = await apiClient.get('/health');
        return response.data;
    },
};

export default api; 