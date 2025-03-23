import React, { useEffect, useState } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
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

interface Trade {
  timestamp: string;
  exchange: string;
  symbol: string;
  side: 'buy' | 'sell';
  price: number;
  quantity: number;
  profit: number;
  latency: number;
}

interface MarketData {
  exchange: string;
  price: number;
  timestamp: string;
}

export const TradingDashboard: React.FC = () => {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [totalPnL, setTotalPnL] = useState(0);
  const [avgLatency, setAvgLatency] = useState(0);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('https://quant-momentum-hft.herokuapp.com/status');
        const data = await response.json();
        setTrades(data.trades || []);
        setMarketData(data.market_data || []);
        setTotalPnL(data.total_pnl || 0);
        setAvgLatency(data.avg_latency || 0);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 1000);
    if (interval.unref) {
      interval.unref(); // Only call unref if it exists (Node.js environment)
    }
    return () => {
      clearInterval(interval);
    };
  }, []);

  const chartData = {
    labels: trades.map(trade => new Date(trade.timestamp).toLocaleTimeString()),
    datasets: [
      {
        label: 'PnL',
        data: trades.map(trade => trade.profit),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Profit & Loss Over Time',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Grid container spacing={3}>
        {/* Metrics */}
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6">Total PnL</Typography>
            <Typography variant="h4" color={totalPnL >= 0 ? 'success.main' : 'error.main'}>
              ${totalPnL.toFixed(2)}
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6">Avg Latency</Typography>
            <Typography variant="h4">{avgLatency.toFixed(2)} ms</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6">Total Trades</Typography>
            <Typography variant="h4">{trades.length}</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6">Win Rate</Typography>
            <Typography variant="h4">
              {trades.length > 0
                ? ((trades.filter(t => t.profit > 0).length / trades.length) * 100).toFixed(1)
                : '0.0'}%
            </Typography>
          </Paper>
        </Grid>

        {/* Chart */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Line data={chartData} options={chartOptions} />
          </Paper>
        </Grid>

        {/* Market Data */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Market Data</Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Exchange</TableCell>
                    <TableCell>Price</TableCell>
                    <TableCell>Time</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {marketData.map((data, index) => (
                    <TableRow key={index}>
                      <TableCell>{data.exchange}</TableCell>
                      <TableCell>${data.price.toFixed(2)}</TableCell>
                      <TableCell>{new Date(data.timestamp).toLocaleTimeString()}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>

        {/* Recent Trades */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Recent Trades</Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Time</TableCell>
                    <TableCell>Side</TableCell>
                    <TableCell>Price</TableCell>
                    <TableCell>Profit</TableCell>
                    <TableCell>Latency</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {trades.slice(-5).map((trade, index) => (
                    <TableRow key={index}>
                      <TableCell>{new Date(trade.timestamp).toLocaleTimeString()}</TableCell>
                      <TableCell>{trade.side}</TableCell>
                      <TableCell>${trade.price.toFixed(2)}</TableCell>
                      <TableCell sx={{ color: trade.profit >= 0 ? 'success.main' : 'error.main' }}>
                        ${trade.profit.toFixed(2)}
                      </TableCell>
                      <TableCell>{trade.latency.toFixed(2)}ms</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}; 