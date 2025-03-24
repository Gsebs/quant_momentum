import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  CircularProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function Performance() {
  const [performance, setPerformance] = useState(null);
  const [trades, setTrades] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeframe, setTimeframe] = useState('1d');

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch performance metrics
        const perfResponse = await fetch('http://localhost:8000/performance');
        const perfData = await perfResponse.json();
        setPerformance(perfData);

        // Fetch trades
        const tradesResponse = await fetch('http://localhost:8000/trades');
        const tradesData = await tradesResponse.json();
        setTrades(tradesData.trades);

        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const calculateTimeframeData = () => {
    if (!trades.length) return null;

    const now = new Date();
    let startTime;

    switch (timeframe) {
      case '1h':
        startTime = new Date(now - 60 * 60 * 1000);
        break;
      case '1d':
        startTime = new Date(now - 24 * 60 * 60 * 1000);
        break;
      case '1w':
        startTime = new Date(now - 7 * 24 * 60 * 60 * 1000);
        break;
      case '1m':
        startTime = new Date(now - 30 * 24 * 60 * 60 * 1000);
        break;
      default:
        startTime = new Date(now - 24 * 60 * 60 * 1000);
    }

    const filteredTrades = trades.filter(
      trade => new Date(trade.timestamp) >= startTime
    );

    return filteredTrades;
  };

  const timeframeData = calculateTimeframeData();

  const pnlChartData = {
    labels: timeframeData?.map(trade => new Date(trade.timestamp).toLocaleString()),
    datasets: [
      {
        label: 'Cumulative P/L',
        data: timeframeData?.map((_, index) =>
          timeframeData
            .slice(0, index + 1)
            .reduce((sum, trade) => sum + trade.profit_loss, 0)
        ),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
    ],
  };

  const volumeChartData = {
    labels: timeframeData?.map(trade => new Date(trade.timestamp).toLocaleString()),
    datasets: [
      {
        label: 'Trading Volume',
        data: timeframeData?.map(trade => trade.amount * trade.price),
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  if (loading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="200px"
      >
        <CircularProgress />
      </Box>
    );
  }

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
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5">Performance Analytics</Typography>
        <FormControl sx={{ minWidth: 120 }}>
          <InputLabel>Timeframe</InputLabel>
          <Select
            value={timeframe}
            label="Timeframe"
            onChange={(e) => setTimeframe(e.target.value)}
          >
            <MenuItem value="1h">1 Hour</MenuItem>
            <MenuItem value="1d">1 Day</MenuItem>
            <MenuItem value="1w">1 Week</MenuItem>
            <MenuItem value="1m">1 Month</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* Performance Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Total P/L
            </Typography>
            <Typography
              variant="h4"
              color={performance?.total_profit >= 0 ? 'success.main' : 'error.main'}
            >
              ${performance?.total_profit.toFixed(2)}
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Win Rate
            </Typography>
            <Typography variant="h4">
              {((performance?.winning_trades / performance?.total_trades) * 100).toFixed(1)}%
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Average Trade P/L
            </Typography>
            <Typography
              variant="h4"
              color={performance?.average_profit_per_trade >= 0 ? 'success.main' : 'error.main'}
            >
              ${performance?.average_profit_per_trade.toFixed(2)}
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Total Trades
            </Typography>
            <Typography variant="h4">{performance?.total_trades}</Typography>
          </Paper>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Profit/Loss Over Time
            </Typography>
            <Line options={chartOptions} data={pnlChartData} />
          </Paper>
        </Grid>
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Trading Volume
            </Typography>
            <Bar options={chartOptions} data={volumeChartData} />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Performance; 