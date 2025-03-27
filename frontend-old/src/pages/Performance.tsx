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
  Container,
  Alert,
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
import { api } from '../services/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface PerformanceData {
  daily_profit_loss: {
    date: string;
    profit: number;
  }[];
  monthly_profit_loss: {
    month: string;
    profit: number;
  }[];
  best_performing_pairs: {
    symbol: string;
    profit: number;
    trade_count: number;
  }[];
  worst_performing_pairs: {
    symbol: string;
    profit: number;
    trade_count: number;
  }[];
}

const Performance: React.FC = () => {
  const [performanceData, setPerformanceData] = useState<PerformanceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeframe, setTimeframe] = useState('1d');

  useEffect(() => {
    const fetchPerformanceData = async () => {
      try {
        const trades = await api.getTrades({ 
          limit: 1000
        });
        
        // Process trades into performance data
        const performanceData: PerformanceData = {
          daily_profit_loss: [],
          monthly_profit_loss: [],
          best_performing_pairs: [],
          worst_performing_pairs: []
        };

        // Group trades by date and calculate daily profit/loss
        const dailyProfits = trades.reduce((acc: { [date: string]: number }, trade) => {
          const date = new Date(trade.timestamp).toISOString().split('T')[0];
          acc[date] = (acc[date] || 0) + trade.profit_loss;
          return acc;
        }, {});

        performanceData.daily_profit_loss = Object.entries(dailyProfits).map(([date, profit]) => ({
          date,
          profit
        })).sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

        // Group trades by month and calculate monthly profit/loss
        const monthlyProfits = trades.reduce((acc: { [month: string]: number }, trade) => {
          const month = new Date(trade.timestamp).toISOString().slice(0, 7);
          acc[month] = (acc[month] || 0) + trade.profit_loss;
          return acc;
        }, {});

        performanceData.monthly_profit_loss = Object.entries(monthlyProfits).map(([month, profit]) => ({
          month,
          profit
        })).sort((a, b) => a.month.localeCompare(b.month));

        // Calculate performance by trading pair
        const pairPerformance = trades.reduce((acc: { [symbol: string]: { profit: number, count: number } }, trade) => {
          if (!acc[trade.symbol]) {
            acc[trade.symbol] = { profit: 0, count: 0 };
          }
          acc[trade.symbol].profit += trade.profit_loss;
          acc[trade.symbol].count += 1;
          return acc;
        }, {});

        const pairs = Object.entries(pairPerformance)
          .map(([symbol, data]) => ({
            symbol,
            profit: data.profit,
            trade_count: data.count
          }))
          .sort((a, b) => b.profit - a.profit);

        performanceData.best_performing_pairs = pairs.slice(0, 5);
        performanceData.worst_performing_pairs = pairs.slice(-5).reverse();

        setPerformanceData(performanceData);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
        setError(errorMessage);
        console.error('Error fetching performance data:', err);
      } finally {
        setLoading(false);
      }
    };

    // Initial fetch
    fetchPerformanceData();

    // Set up WebSocket connection for real-time updates
    api.initWebSockets((data) => {
      if (data.type === 'trade_update') {
        // Update performance data with new trade
        setPerformanceData(prevData => {
          if (!prevData) return null;
          
          const trade = data.payload;
          const date = new Date(trade.timestamp).toISOString().split('T')[0];
          const month = new Date(trade.timestamp).toISOString().slice(0, 7);

          // Update daily profit/loss
          const dailyIndex = prevData.daily_profit_loss.findIndex(d => d.date === date);
          if (dailyIndex >= 0) {
            prevData.daily_profit_loss[dailyIndex].profit += trade.profit_loss;
          } else {
            prevData.daily_profit_loss.push({ date, profit: trade.profit_loss });
            prevData.daily_profit_loss.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
          }

          // Update monthly profit/loss
          const monthlyIndex = prevData.monthly_profit_loss.findIndex(m => m.month === month);
          if (monthlyIndex >= 0) {
            prevData.monthly_profit_loss[monthlyIndex].profit += trade.profit_loss;
          } else {
            prevData.monthly_profit_loss.push({ month, profit: trade.profit_loss });
            prevData.monthly_profit_loss.sort((a, b) => a.month.localeCompare(b.month));
          }

          return { ...prevData };
        });
      }
    });

    // Cleanup WebSocket connection
    return () => {
      api.closeWebSockets();
    };
  }, [timeframe]);

  const calculateTimeframeData = () => {
    if (!performanceData?.daily_profit_loss.length) return null;

    const now = new Date();
    let startTime = new Date();

    switch (timeframe) {
      case '1d':
        startTime.setDate(now.getDate() - 1);
        break;
      case '1w':
        startTime.setDate(now.getDate() - 7);
        break;
      case '1m':
        startTime.setMonth(now.getMonth() - 1);
        break;
      case '3m':
        startTime.setMonth(now.getMonth() - 3);
        break;
      case '1y':
        startTime.setFullYear(now.getFullYear() - 1);
        break;
      default:
        startTime = new Date(0); // All time
    }

    const filteredDates = performanceData.daily_profit_loss
      .filter(item => new Date(item.date).getTime() >= startTime.getTime())
      .map(item => item.date);
    const filteredProfits = performanceData.daily_profit_loss
      .filter(item => new Date(item.date).getTime() >= startTime.getTime())
      .map(item => item.profit);

    return {
      dates: filteredDates,
      profits: filteredProfits,
    };
  };

  const handleTimeframeChange = (event: any) => {
    setTimeframe(event.target.value as string);
  };

  const timeframeData = calculateTimeframeData();

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const
      }
    },
    scales: {
      y: {
        beginAtZero: true
      }
    }
  };

  const dailyProfitData = {
    labels: timeframeData?.dates,
    datasets: [
      {
        label: 'Daily Profit/Loss',
        data: timeframeData?.profits,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        tension: 0.1,
      },
    ],
  };

  const cumulativeProfitData = {
    labels: timeframeData?.dates,
    datasets: [
      {
        label: 'Cumulative Profit/Loss',
        data: timeframeData?.profits?.reduce((acc: number[], profit) => 
          [...acc, (acc.length ? acc[acc.length - 1] : 0) + profit], 
          []
        ),
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
        tension: 0.1,
      },
    ],
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Container>
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      </Container>
    );
  }

  if (!performanceData) {
    return (
      <Container>
        <Alert severity="info" sx={{ mt: 2 }}>
          No performance data available.
        </Alert>
      </Container>
    );
  }

  return (
    <Container>
      <Box sx={{ mt: 4, mb: 2 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Trading Performance
        </Typography>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ mb: 2 }}>
              <FormControl>
                <InputLabel>Timeframe</InputLabel>
                <Select
                  value={timeframe}
                  label="Timeframe"
                  onChange={handleTimeframeChange}
                  sx={{ minWidth: 120 }}
                >
                  <MenuItem value="1d">1 Day</MenuItem>
                  <MenuItem value="1w">1 Week</MenuItem>
                  <MenuItem value="1m">1 Month</MenuItem>
                  <MenuItem value="3m">3 Months</MenuItem>
                  <MenuItem value="1y">1 Year</MenuItem>
                  <MenuItem value="all">All Time</MenuItem>
                </Select>
              </FormControl>
            </Box>

            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Daily Profit/Loss
                </Typography>
                <Line options={chartOptions} data={dailyProfitData} />
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Cumulative Profit/Loss
                </Typography>
                <Line options={chartOptions} data={cumulativeProfitData} />
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Best Performing Pairs
            </Typography>
            <Grid container spacing={2}>
              {performanceData.best_performing_pairs.map((pair) => (
                <Grid item xs={12} key={pair.symbol}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography>{pair.symbol}</Typography>
                    <Typography color="success.main">
                      ${pair.profit.toFixed(2)} ({pair.trade_count} trades)
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Worst Performing Pairs
            </Typography>
            <Grid container spacing={2}>
              {performanceData.worst_performing_pairs.map((pair) => (
                <Grid item xs={12} key={pair.symbol}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography>{pair.symbol}</Typography>
                    <Typography color="error.main">
                      ${pair.profit.toFixed(2)} ({pair.trade_count} trades)
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Performance; 