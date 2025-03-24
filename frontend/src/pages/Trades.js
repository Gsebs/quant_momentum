import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Grid,
  CircularProgress,
  Chip,
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

function Trades() {
  const [trades, setTrades] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  useEffect(() => {
    const fetchTrades = async () => {
      try {
        const response = await fetch('http://localhost:8000/trades');
        const data = await response.json();
        setTrades(data.trades);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchTrades();
    const interval = setInterval(fetchTrades, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const chartData = {
    labels: trades.map(trade => new Date(trade.timestamp).toLocaleTimeString()),
    datasets: [
      {
        label: 'Trade Profit/Loss',
        data: trades.map(trade => trade.profit_loss),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: trades.map(trade => 
          trade.profit_loss >= 0 ? 'rgba(75, 192, 192, 0.5)' : 'rgba(255, 99, 132, 0.5)'
        ),
      },
      {
        label: 'Cumulative P/L',
        data: trades.map((_, index) => 
          trades
            .slice(0, index + 1)
            .reduce((sum, trade) => sum + trade.profit_loss, 0)
        ),
        borderColor: 'rgb(153, 102, 255)',
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

  const calculateStats = () => {
    if (trades.length === 0) return null;

    const totalTrades = trades.length;
    const winningTrades = trades.filter(t => t.profit_loss > 0).length;
    const totalPnL = trades.reduce((sum, t) => sum + t.profit_loss, 0);
    const winRate = (winningTrades / totalTrades) * 100;
    const averagePnL = totalPnL / totalTrades;
    const largestWin = Math.max(...trades.map(t => t.profit_loss));
    const largestLoss = Math.min(...trades.map(t => t.profit_loss));

    return {
      totalTrades,
      winningTrades,
      totalPnL,
      winRate,
      averagePnL,
      largestWin,
      largestLoss,
    };
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

  const stats = calculateStats();

  return (
    <Box p={3}>
      <Typography variant="h5" gutterBottom>
        Trade History
      </Typography>

      {/* Statistics Cards */}
      {stats && (
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Total Trades
              </Typography>
              <Typography variant="h4">{stats.totalTrades}</Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Win Rate
              </Typography>
              <Typography variant="h4">{stats.winRate.toFixed(1)}%</Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Total P/L
              </Typography>
              <Typography variant="h4" color={stats.totalPnL >= 0 ? 'success.main' : 'error.main'}>
                ${stats.totalPnL.toFixed(2)}
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Average P/L
              </Typography>
              <Typography variant="h4" color={stats.averagePnL >= 0 ? 'success.main' : 'error.main'}>
                ${stats.averagePnL.toFixed(2)}
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      )}

      {/* Performance Chart */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Line options={chartOptions} data={chartData} />
      </Paper>

      {/* Trades Table */}
      <Paper>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Time</TableCell>
                <TableCell>Symbol</TableCell>
                <TableCell>Type</TableCell>
                <TableCell align="right">Amount</TableCell>
                <TableCell align="right">Price</TableCell>
                <TableCell>Exchange</TableCell>
                <TableCell align="right">P/L</TableCell>
                <TableCell align="center">Status</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {trades
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((trade, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      {new Date(trade.timestamp).toLocaleString()}
                    </TableCell>
                    <TableCell>{trade.symbol}</TableCell>
                    <TableCell>{trade.side}</TableCell>
                    <TableCell align="right">
                      {trade.amount.toFixed(8)}
                    </TableCell>
                    <TableCell align="right">
                      ${trade.price.toFixed(2)}
                    </TableCell>
                    <TableCell>{trade.exchange}</TableCell>
                    <TableCell
                      align="right"
                      sx={{
                        color: trade.profit_loss >= 0 ? 'success.main' : 'error.main',
                      }}
                    >
                      ${trade.profit_loss.toFixed(2)}
                    </TableCell>
                    <TableCell align="center">
                      <Chip
                        label={trade.status}
                        color={
                          trade.status === 'completed'
                            ? 'success'
                            : trade.status === 'pending'
                            ? 'warning'
                            : 'error'
                        }
                        size="small"
                      />
                    </TableCell>
                  </TableRow>
                ))}
              {trades.length === 0 && (
                <TableRow>
                  <TableCell colSpan={8} align="center">
                    No trades found
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          rowsPerPageOptions={[5, 10, 25]}
          component="div"
          count={trades.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      </Paper>
    </Box>
  );
}

export default Trades; 