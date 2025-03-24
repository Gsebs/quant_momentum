import React, { useState, useEffect } from 'react';
import { Grid, Paper, Typography, Box } from '@mui/material';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function Dashboard() {
  const [performance, setPerformance] = useState(null);
  const [opportunities, setOpportunities] = useState([]);
  const [trades, setTrades] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch performance metrics
        const perfResponse = await fetch('http://localhost:8000/performance');
        const perfData = await perfResponse.json();
        setPerformance(perfData);

        // Fetch recent opportunities
        const oppResponse = await fetch('http://localhost:8000/opportunities');
        const oppData = await oppResponse.json();
        setOpportunities(oppData.opportunities);

        // Fetch recent trades
        const tradesResponse = await fetch('http://localhost:8000/trades');
        const tradesData = await tradesResponse.json();
        setTrades(tradesData.trades);
      } catch (err) {
        setError(err.message);
        console.error('Error fetching dashboard data:', err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const performanceChartData = {
    labels: trades.map(trade => new Date(trade.timestamp).toLocaleTimeString()),
    datasets: [
      {
        label: 'Cumulative PnL',
        data: trades.map((_, index) => {
          return trades.slice(0, index + 1).reduce((sum, trade) => sum + (trade.profit_loss || 0), 0);
        }),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Trading Performance',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  if (error) {
    return (
      <Box p={3}>
        <Typography color="error" variant="h6">
          Error: {error}
        </Typography>
      </Box>
    );
  }

  return (
    <Box p={3}>
      <Grid container spacing={3}>
        {/* Performance Metrics */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>
              Performance Metrics
            </Typography>
            {performance && (
              <>
                <Typography variant="body2">
                  Total Trades: {performance.total_trades}
                </Typography>
                <Typography variant="body2">
                  Winning Trades: {performance.winning_trades}
                </Typography>
                <Typography variant="body2">
                  Win Rate: {(performance.win_rate * 100).toFixed(2)}%
                </Typography>
                <Typography variant="body2">
                  Total Profit: ${performance.total_profit.toFixed(2)}
                </Typography>
              </>
            )}
          </Paper>
        </Grid>

        {/* Recent Opportunities */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>
              Recent Opportunities
            </Typography>
            {opportunities.map((opp, index) => (
              <Box key={index} sx={{ mb: 1 }}>
                <Typography variant="body2">
                  {opp.symbol}: Buy @{opp.buy_exchange} (${opp.buy_price.toFixed(2)}) - 
                  Sell @{opp.sell_exchange} (${opp.sell_price.toFixed(2)})
                </Typography>
                <Typography variant="body2" color="primary">
                  Potential Profit: ${opp.price_difference.toFixed(2)}
                </Typography>
              </Box>
            ))}
          </Paper>
        </Grid>

        {/* Performance Chart */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Line options={chartOptions} data={performanceChartData} />
          </Paper>
        </Grid>

        {/* Recent Trades */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Recent Trades
            </Typography>
            <Box sx={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left', padding: '8px' }}>Time</th>
                    <th style={{ textAlign: 'left', padding: '8px' }}>Symbol</th>
                    <th style={{ textAlign: 'left', padding: '8px' }}>Type</th>
                    <th style={{ textAlign: 'right', padding: '8px' }}>Amount</th>
                    <th style={{ textAlign: 'right', padding: '8px' }}>Price</th>
                    <th style={{ textAlign: 'right', padding: '8px' }}>P/L</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.map((trade, index) => (
                    <tr key={index}>
                      <td style={{ padding: '8px' }}>
                        {new Date(trade.timestamp).toLocaleString()}
                      </td>
                      <td style={{ padding: '8px' }}>{trade.symbol}</td>
                      <td style={{ padding: '8px' }}>{trade.side}</td>
                      <td style={{ textAlign: 'right', padding: '8px' }}>
                        {trade.amount.toFixed(8)}
                      </td>
                      <td style={{ textAlign: 'right', padding: '8px' }}>
                        ${trade.price.toFixed(2)}
                      </td>
                      <td
                        style={{
                          textAlign: 'right',
                          padding: '8px',
                          color: trade.profit_loss >= 0 ? 'green' : 'red',
                        }}
                      >
                        ${trade.profit_loss?.toFixed(2) || '0.00'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard; 