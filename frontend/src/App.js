import React, { useState, useEffect } from 'react';
import { api } from './services/api';

function App() {
  const [status, setStatus] = useState(null);
  const [trades, setTrades] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const statusData = await api.getStatus();
        setStatus(statusData);
        
        const tradesData = await api.getTrades();
        setTrades(tradesData);
      } catch (err) {
        setError(err.message);
        console.error('Error fetching data:', err);
      }
    };

    // Fetch initial data
    fetchData();

    // Set up polling interval
    const interval = setInterval(fetchData, 5000); // Poll every 5 seconds

    // Cleanup on unmount
    return () => clearInterval(interval);
  }, []);

  if (error) {
    return (
      <div className="min-h-screen bg-red-50 p-4">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-2xl font-bold text-red-600 mb-4">Error</h1>
          <p className="text-red-700">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold mb-4">HFT Latency Arbitrage Dashboard</h1>
        
        {/* Status Section */}
        {status && (
          <div className="bg-white rounded-lg shadow p-4 mb-4">
            <h2 className="text-xl font-semibold mb-2">System Status</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-gray-600">Latest Prices:</p>
                <p>Binance: ${status.latest_prices?.binance?.toFixed(2) || 'N/A'}</p>
                <p>Coinbase: ${status.latest_prices?.coinbase?.toFixed(2) || 'N/A'}</p>
              </div>
              <div>
                <p className="text-gray-600">Performance:</p>
                <p>Trades: {status.trades_count || 0}</p>
                <p>PnL: ${status.cumulative_pnl?.toFixed(2) || '0.00'}</p>
              </div>
            </div>
          </div>
        )}

        {/* Trades Section */}
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-xl font-semibold mb-2">Recent Trades</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-4 py-2">Time</th>
                  <th className="px-4 py-2">Type</th>
                  <th className="px-4 py-2">Price</th>
                  <th className="px-4 py-2">Profit/Loss</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((trade, index) => (
                  <tr key={index} className="border-t">
                    <td className="px-4 py-2">{new Date(trade.timestamp).toLocaleString()}</td>
                    <td className="px-4 py-2">{trade.type}</td>
                    <td className="px-4 py-2">${trade.price?.toFixed(2)}</td>
                    <td className={`px-4 py-2 ${trade.profit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      ${trade.profit?.toFixed(2)}
                    </td>
                  </tr>
                ))}
                {trades.length === 0 && (
                  <tr>
                    <td colSpan="4" className="px-4 py-2 text-center text-gray-500">
                      No trades yet
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
