import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 10000,
});

// Add response interceptor for error handling
api.interceptors.response.use(
    response => response,
    error => {
        if (error.response) {
            // Rate limit error
            if (error.response.status === 429) {
                throw new Error('Rate limit exceeded. Please try again in a few minutes.');
            }
            // Server error
            if (error.response.status >= 500) {
                throw new Error('Server error. Please try again later.');
            }
            throw new Error(error.response.data.message || 'An error occurred');
        }
        if (error.request) {
            throw new Error('Network error. Please check your connection.');
        }
        throw error;
    }
);

export const fetchDashboardData = async () => {
    try {
        const [metricsResponse, performanceResponse, signalsResponse, alertsResponse] = await Promise.all([
            api.get('/metrics'),
            api.get('/performance'),
            api.get('/signals'),
            api.get('/alerts')
        ]);
        

        return {
            metrics: metricsResponse.data,
            performance: performanceResponse.data,
            signals: signalsResponse.data,
            alerts: alertsResponse.data
        };
    } catch (error) {
        console.error('Error fetching dashboard data:', error);
        throw error;
    }
};

export const clearCache = async () => {
    try {
        const response = await api.post('/cache/clear');
        return response.data;
    } catch (error) {
        console.error('Error clearing cache:', error);
        throw error;
    }
};

export const updateStrategy = async (params) => {
    try {
        const response = await api.post('/strategy/update', params);
        return response.data;
    } catch (error) {
        console.error('Error updating strategy:', error);
        throw error;
    }
};

export const runBacktest = async (params) => {
    try {
        const response = await api.post('/backtest', params);
        return response.data;
    } catch (error) {
        console.error('Error running backtest:', error);
        throw error;
    }
};

export const getStrategyConfig = async () => {
    try {
        const response = await api.get('/strategy/config');
        return response.data;
    } catch (error) {
        console.error('Error fetching strategy config:', error);
        throw error;
    }
};

export default api; 