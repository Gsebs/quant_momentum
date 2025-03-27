import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Alert
} from '@mui/material';
import { Line } from 'react-chartjs-2';
import { api } from '../services/api';
import { hftTrading } from '../services/hftTrading';
import type { ArbitrageOpportunity } from '../services/hftTrading';

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

const Dashboard: React.FC = () => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [opportunities, setOpportunities] = useState<ArbitrageOpportunity[]>([]);
  const [isTrading, setIsTrading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await api.getDashboard();
        setDashboardData(data);
        setOpportunities(hftTrading.getOpportunities());
      } catch (err) {
        setError('Failed to fetch dashboard data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    api.initWebSockets((data: any) => {
      if (data.type === 'dashboard_update') {
        setDashboardData(data.data);
      } else if (data.type === 'opportunities_update') {
        setOpportunities(hftTrading.getOpportunities());
      }
    });

    return () => {
      api.closeWebSockets();
    };
  }, []);

  const handleStartTrading = () => {
    hftTrading.startTrading();
    setIsTrading(true);
  };

  const handleStopTrading = () => {
    hftTrading.stopTrading();
    setIsTrading(false);
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
      <Box p={3}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  const profitChartData = {
    labels: dashboardData?.profit_history.map(h => new Date(h.timestamp).toLocaleTimeString()) || [],
    datasets: [
      {
        label: 'Profit History',
        data: dashboardData?.profit_history.map(h => h.profit) || [],
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      }
    ]
  };

  return (
    <Box p={3}>
      <Grid container spacing={3}>
        {/* Trading Controls */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center">
              <Typography variant="h6">Trading Controls</Typography>
              <Button
                variant="contained"
                color={isTrading ? 'error' : 'primary'}
                onClick={isTrading ? handleStopTrading : handleStartTrading}
              >
                {isTrading ? 'Stop Trading' : 'Start Trading'}
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* Performance Metrics */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Performance Metrics</Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="subtitle2">Total Profit</Typography>
                <Typography variant="h4" color={(dashboardData?.total_profit || 0) >= 0 ? 'success.main' : 'error.main'}>
                  ${dashboardData?.total_profit?.toFixed(2) || '0.00'}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="subtitle2">Win Rate</Typography>
                <Typography variant="h4">
                  {((dashboardData?.win_rate || 0) * 100).toFixed(2)}%
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="subtitle2">Total Trades</Typography>
                <Typography variant="h4">{dashboardData?.total_trades}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="subtitle2">Active Trades</Typography>
                <Typography variant="h4">{dashboardData?.active_trades}</Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Profit Chart */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Profit History</Typography>
            <Box height={300}>
              <Line data={profitChartData} options={{ maintainAspectRatio: false }} />
            </Box>
          </Paper>
        </Grid>

        {/* Arbitrage Opportunities */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Arbitrage Opportunities</Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Time</TableCell>
                    <TableCell>Symbol</TableCell>
                    <TableCell>Buy Exchange</TableCell>
                    <TableCell>Sell Exchange</TableCell>
                    <TableCell>Buy Price</TableCell>
                    <TableCell>Sell Price</TableCell>
                    <TableCell>Size</TableCell>
                    <TableCell>Profit</TableCell>
                    <TableCell>Latency</TableCell>
                    <TableCell>Confidence</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {opportunities.map((opp, index) => (
                    <TableRow key={index}>
                      <TableCell>{new Date(opp.timestamp).toLocaleTimeString()}</TableCell>
                      <TableCell>{opp.symbol}</TableCell>
                      <TableCell>{opp.buyExchange}</TableCell>
                      <TableCell>{opp.sellExchange}</TableCell>
                      <TableCell>${opp.buyPrice.toFixed(2)}</TableCell>
                      <TableCell>${opp.sellPrice.toFixed(2)}</TableCell>
                      <TableCell>{opp.maxSize}</TableCell>
                      <TableCell color={opp.profit >= 0 ? 'success.main' : 'error.main'}>
                        ${opp.profit.toFixed(2)}
                      </TableCell>
                      <TableCell>{opp.latency.toFixed(2)}ms</TableCell>
                      <TableCell>{(opp.confidence * 100).toFixed(1)}%</TableCell>
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

export default Dashboard; 