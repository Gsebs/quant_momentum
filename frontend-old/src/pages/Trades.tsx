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
  id: string;
  symbol: string;
  exchange1: string;
  exchange2: string;
  entry_price1: number;
  entry_price2: number;
  exit_price1: number;
  exit_price2: number;
  volume: number;
  profit_loss: number;
  status: 'open' | 'closed' | 'failed';
  timestamp: string;
}

const Trades: React.FC = () => {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  useEffect(() => {
    const fetchTrades = async () => {
      try {
        const response = await fetch('/api/trades');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setTrades(data);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
        setError(errorMessage);
        console.error('Error fetching trades:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchTrades();
    const interval = setInterval(fetchTrades, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const handleChangePage = (event: React.MouseEvent<HTMLButtonElement> | null, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
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
        position: 'top'
      },
      title: {
        display: true,
        text: 'Trading Performance'
      }
    },
    scales: {
      y: {
        beginAtZero: true
      }
    }
  } as const;

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

  const getStatusColor = (status: Trade['status']) => {
    switch (status) {
      case 'open':
        return 'primary';
      case 'closed':
        return 'success';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
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

  const stats = calculateStats();

  return (
    <Container>
      <Typography variant="h4" component="h1" gutterBottom sx={{ mt: 4 }}>
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
          <Table stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell>Time</TableCell>
                <TableCell>Symbol</TableCell>
                <TableCell>Exchange 1</TableCell>
                <TableCell>Entry Price 1</TableCell>
                <TableCell>Exit Price 1</TableCell>
                <TableCell>Exchange 2</TableCell>
                <TableCell>Entry Price 2</TableCell>
                <TableCell>Exit Price 2</TableCell>
                <TableCell>Volume</TableCell>
                <TableCell>P/L ($)</TableCell>
                <TableCell>Status</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {trades
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((trade) => (
                  <TableRow
                    key={trade.id}
                    sx={{
                      '&:last-child td, &:last-child th': { border: 0 },
                      backgroundColor: trade.profit_loss >= 0 ? 'rgba(76, 175, 80, 0.1)' : 'rgba(244, 67, 54, 0.1)',
                    }}
                  >
                    <TableCell>{new Date(trade.timestamp).toLocaleString()}</TableCell>
                    <TableCell>{trade.symbol}</TableCell>
                    <TableCell>{trade.exchange1}</TableCell>
                    <TableCell>${trade.entry_price1.toFixed(2)}</TableCell>
                    <TableCell>${trade.exit_price1.toFixed(2)}</TableCell>
                    <TableCell>{trade.exchange2}</TableCell>
                    <TableCell>${trade.entry_price2.toFixed(2)}</TableCell>
                    <TableCell>${trade.exit_price2.toFixed(2)}</TableCell>
                    <TableCell>{trade.volume.toFixed(8)}</TableCell>
                    <TableCell
                      align="right"
                      sx={{
                        color: trade.profit_loss >= 0 ? 'success.main' : 'error.main',
                      }}
                    >
                      ${trade.profit_loss.toFixed(2)}
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={trade.status.toUpperCase()}
                        color={getStatusColor(trade.status)}
                        size="small"
                      />
                    </TableCell>
                  </TableRow>
                ))}
              {trades.length === 0 && (
                <TableRow>
                  <TableCell colSpan={11} align="center">
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
    </Container>
  );
};

export default Trades; 