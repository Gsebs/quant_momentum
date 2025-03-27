import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Paper,
  Grid,
  Box,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import { api } from '../services/api';

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

const Arbitrage: React.FC = () => {
  const [opportunities, setOpportunities] = useState<ArbitrageOpportunity[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchOpportunities = async () => {
      try {
        const data = await api.getArbitrageOpportunities();
        setOpportunities(data);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
        setError(errorMessage);
        console.error('Error fetching opportunities:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchOpportunities();

    // Set up WebSocket connection for real-time updates
    api.initWebSockets((data) => {
      if (data.type === 'arbitrage_update') {
        setOpportunities(data.payload);
      }
    });

    return () => {
      api.closeWebSockets();
    };
  }, []);

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

  return (
    <Container>
      <Box sx={{ mt: 4, mb: 2 }}>
        <Typography variant="h4" component="h1">
          Arbitrage Opportunities
        </Typography>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Symbol</TableCell>
                    <TableCell>Buy Exchange</TableCell>
                    <TableCell>Sell Exchange</TableCell>
                    <TableCell>Buy Price</TableCell>
                    <TableCell>Sell Price</TableCell>
                    <TableCell>Size</TableCell>
                    <TableCell>Profit</TableCell>
                    <TableCell>Time</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {opportunities.map((opp, index) => (
                    <TableRow key={index}>
                      <TableCell>{opp.symbol}</TableCell>
                      <TableCell>{opp.buyExchange}</TableCell>
                      <TableCell>{opp.sellExchange}</TableCell>
                      <TableCell>${opp.buyPrice.toFixed(2)}</TableCell>
                      <TableCell>${opp.sellPrice.toFixed(2)}</TableCell>
                      <TableCell>{opp.maxSize.toFixed(4)}</TableCell>
                      <TableCell sx={{ color: 'success.main' }}>
                        ${opp.profit.toFixed(2)}
                      </TableCell>
                      <TableCell>
                        {new Date(opp.timestamp).toLocaleTimeString()}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Arbitrage; 