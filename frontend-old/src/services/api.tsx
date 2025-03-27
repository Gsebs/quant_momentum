import axios from 'axios';

const WS_BASE = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';
const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
const COINBASE_WS = process.env.REACT_APP_COINBASE_WS_URL || 'wss://ws-feed.pro.coinbase.com';
const BINANCE_WS = process.env.REACT_APP_BINANCE_WS_URL || 'wss://stream.binance.com:9443/ws';
const ML_BASE = process.env.REACT_APP_ML_MODEL_URL;

interface DashboardData {
  trades: any[];
  total_profit: number;
  win_rate: number;
  average_profit_per_trade: number;
  total_trades: number;
  total_profit_loss: number;
  success_rate: number;
  active_trades: number;
  profit_history: {
    timestamp: string;
    profit: number;
  }[];
}

interface LatencyMetrics {
  [exchange: string]: number;
}

interface Trade {
  id: string;
  timestamp: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  price: number;
  quantity: number;
  profit_loss: number;
}

interface ArbitrageOpportunity {
  buyExchange: string;
  sellExchange: string;
  symbol: string;
  buyPrice: number;
  sellPrice: number;
  maxSize: number;
  profit: number;
  timestamp: number;
}

interface ExchangeStatus {
  name: string;
  status: 'online' | 'offline' | 'maintenance';
  latency: number;
}

interface HealthCheck {
  status: string;
  latency: number;
}

interface OrderRequest {
  exchange: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  price: number;
  size: number;
}

// Create axios instance with optimized settings for HFT
const apiClient = axios.create({
    baseURL: API_BASE,
    timeout: 1000, // 1 second timeout for HFT operations
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

class WebSocketManager {
    private sockets: { [key: string]: WebSocket } = {};
    private messageHandlers: ((data: any) => void)[] = [];
    private reconnectAttempts: { [key: string]: number } = {};
    private maxReconnectAttempts = 5;

    initWebSockets(onMessage: (data: any) => void): void {
        this.messageHandlers.push(onMessage);
        
        // Initialize main WebSocket
        this.connectWebSocket('main', WS_BASE);
        
        // Initialize exchange WebSockets
        this.connectWebSocket('coinbase', COINBASE_WS);
        this.connectWebSocket('binance', BINANCE_WS);
    }

    private connectWebSocket(name: string, url: string): void {
        try {
            const ws = new WebSocket(url);
            
            ws.onopen = () => {
                console.log(`${name} WebSocket connected`);
                this.reconnectAttempts[name] = 0;
                
                // Subscribe to specific channels based on exchange
                if (name === 'coinbase') {
                    ws.send(JSON.stringify({
                        type: 'subscribe',
                        product_ids: ['BTC-USD', 'ETH-USD'],
                        channels: ['ticker']
                    }));
                } else if (name === 'binance') {
                    ws.send(JSON.stringify({
                        method: 'SUBSCRIBE',
                        params: ['btcusdt@trade', 'ethusdt@trade'],
                        id: 1
                    }));
                }
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.messageHandlers.forEach(handler => handler(data));
                } catch (error) {
                    console.error(`Error parsing ${name} WebSocket message:`, error);
                }
            };

            ws.onerror = (error) => {
                console.error(`${name} WebSocket error:`, error);
            };

            ws.onclose = () => {
                console.log(`${name} WebSocket closed`);
                this.handleReconnect(name, url);
            };

            this.sockets[name] = ws;
        } catch (error) {
            console.error(`Error connecting to ${name} WebSocket:`, error);
            this.handleReconnect(name, url);
        }
    }

    private handleReconnect(name: string, url: string): void {
        if (!this.reconnectAttempts[name]) {
            this.reconnectAttempts[name] = 0;
        }

        if (this.reconnectAttempts[name] < this.maxReconnectAttempts) {
            this.reconnectAttempts[name]++;
            console.log(`Attempting to reconnect ${name} WebSocket (${this.reconnectAttempts[name]}/${this.maxReconnectAttempts})`);
            setTimeout(() => this.connectWebSocket(name, url), 1000 * Math.pow(2, this.reconnectAttempts[name]));
        } else {
            console.error(`Max reconnection attempts reached for ${name} WebSocket`);
        }
    }

    closeWebSockets(): void {
        Object.entries(this.sockets).forEach(([name, socket]) => {
            console.log(`Closing ${name} WebSocket`);
            socket.close();
        });
        this.sockets = {};
        this.messageHandlers = [];
    }
}

const wsManager = new WebSocketManager();

type ApiService = {
  initWebSockets: (onMessage: (data: any) => void) => void;
  closeWebSockets: () => void;
  getDashboard: () => Promise<DashboardData>;
  getLatencyMetrics: () => Promise<LatencyMetrics>;
  getTrades: (params?: { limit?: number; offset?: number }) => Promise<Trade[]>;
  getArbitrageOpportunities: () => Promise<ArbitrageOpportunity[]>;
  getExchangeStatus: () => Promise<ExchangeStatus[]>;
  checkHealth: () => Promise<HealthCheck>;
  executeOrder: (order: OrderRequest) => Promise<any>;
};

export const api: ApiService = {
  initWebSockets: wsManager.initWebSockets.bind(wsManager),
  closeWebSockets: wsManager.closeWebSockets.bind(wsManager),
  getDashboard: async () => {
    const response = await apiClient.get('/api/dashboard');
    return response.data;
  },
  getLatencyMetrics: async () => {
    const response = await apiClient.get('/api/latency');
    return response.data;
  },
  getTrades: async (params) => {
    const response = await apiClient.get('/api/trades', { params });
    return response.data;
  },
  getArbitrageOpportunities: async () => {
    const response = await apiClient.get('/api/opportunities');
    return response.data;
  },
  getExchangeStatus: async () => {
    const response = await apiClient.get('/api/status');
    return response.data;
  },
  checkHealth: async () => {
    const start = Date.now();
    const response = await apiClient.get('/api/health');
    const latency = Date.now() - start;
    return { ...response.data, latency };
  },
  executeOrder: async (order: OrderRequest) => {
    const response = await apiClient.post('/api/orders', order);
    return response.data;
  }
};

export default api; 