import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { TradingDashboard } from '../TradingDashboard';

// Mock fetch
global.fetch = jest.fn();

// Mock chart.js
jest.mock('react-chartjs-2', () => ({
  Line: () => null,
}));

describe('TradingDashboard', () => {
  beforeEach(() => {
    (global.fetch as jest.Mock).mockClear();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.clearAllMocks();
  });

  it('renders without crashing', () => {
    render(<TradingDashboard />);
    expect(screen.getByText('Market Data')).toBeInTheDocument();
  });

  it('displays initial state', () => {
    render(<TradingDashboard />);
    expect(screen.getByText('$0.00')).toBeInTheDocument();
    expect(screen.getByText('0.00 ms')).toBeInTheDocument();
    expect(screen.getByText('0.0%')).toBeInTheDocument();
  });

  it('fetches and displays market data', async () => {
    const mockData = {
      trades: [
        {
          timestamp: '2024-01-01T06:00:00Z',
          exchange: 'Binance',
          symbol: 'BTC/USDT',
          side: 'buy',
          price: 50000,
          quantity: 0.1,
          profit: 1000,
          latency: 5,
        },
      ],
      market_data: [
        {
          exchange: 'Binance',
          price: 50000,
          timestamp: '2024-01-01T06:00:00Z',
        },
      ],
      total_pnl: 1000,
      avg_latency: 5,
    };

    (global.fetch as jest.Mock).mockResolvedValueOnce({
      json: () => Promise.resolve(mockData),
    });

    render(<TradingDashboard />);

    await waitFor(() => {
      const pnlElements = screen.getAllByText('$1000.00');
      expect(pnlElements[0]).toBeInTheDocument();
      expect(screen.getByText('5.00 ms')).toBeInTheDocument();
      expect(screen.getByText('1')).toBeInTheDocument();
      expect(screen.getByText('100.0%')).toBeInTheDocument();
    });
  });

  it('handles API errors gracefully', async () => {
    (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('API Error'));

    render(<TradingDashboard />);

    await waitFor(() => {
      expect(screen.getByText('$0.00')).toBeInTheDocument();
      expect(screen.getByText('0.00 ms')).toBeInTheDocument();
      expect(screen.getByText('0.0%')).toBeInTheDocument();
    });
  });

  it('updates data periodically', async () => {
    const mockData1 = {
      trades: [
        {
          timestamp: '2024-01-01T06:00:00Z',
          exchange: 'Binance',
          symbol: 'BTC/USDT',
          side: 'buy',
          price: 50000,
          quantity: 0.1,
          profit: 1000,
          latency: 5,
        },
      ],
      market_data: [
        {
          exchange: 'Binance',
          price: 50000,
          timestamp: '2024-01-01T06:00:00Z',
        },
      ],
      total_pnl: 1000,
      avg_latency: 5,
    };

    const mockData2 = {
      trades: [
        {
          timestamp: '2024-01-01T06:00:00Z',
          exchange: 'Binance',
          symbol: 'BTC/USDT',
          side: 'buy',
          price: 50000,
          quantity: 0.1,
          profit: 2000,
          latency: 10,
        },
      ],
      market_data: [
        {
          exchange: 'Binance',
          price: 50000,
          timestamp: '2024-01-01T06:00:00Z',
        },
      ],
      total_pnl: 2000,
      avg_latency: 10,
    };

    (global.fetch as jest.Mock)
      .mockResolvedValueOnce({
        json: () => Promise.resolve(mockData1),
      })
      .mockResolvedValueOnce({
        json: () => Promise.resolve(mockData2),
      });

    render(<TradingDashboard />);

    await waitFor(() => {
      const pnlElements = screen.getAllByText('$1000.00');
      expect(pnlElements[0]).toBeInTheDocument();
      expect(screen.getByText('5.00 ms')).toBeInTheDocument();
    });

    // Advance timers by 1 second to trigger the next update
    jest.advanceTimersByTime(1000);

    await waitFor(() => {
      const pnlElements = screen.getAllByText('$2000.00');
      expect(pnlElements[0]).toBeInTheDocument();
      expect(screen.getByText('10.00 ms')).toBeInTheDocument();
    });
  });
}); 