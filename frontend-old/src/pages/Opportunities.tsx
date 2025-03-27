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
  CircularProgress,
  Container,
  Alert
} from '@mui/material';
import api from '../services/api';

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

const Opportunities: React.FC = () => {
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

    // Initial fetch
    fetchOpportunities();

    // Set up WebSocket connection for real-time updates
    api.initWebSockets((data) => {
      if (data.type === 'arbitrage_update') {
        setOpportunities(data.payload);
      }
    });

    // Cleanup WebSocket connection
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
      <Typography variant="h4" component="h1" gutterBottom sx={{ mt: 4 }}>
        Arbitrage Opportunities
      </Typography>
      <Paper sx={{ width: '100%', overflow: 'hidden' }}>
        <TableContainer>
          <Table stickyHeader>
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
              {opportunities.map((opportunity) => (
                <TableRow key={opportunity.timestamp}>
                  <TableCell>{opportunity.symbol}</TableCell>
                  <TableCell>{opportunity.buyExchange}</TableCell>
                  <TableCell>{opportunity.sellExchange}</TableCell>
                  <TableCell>${opportunity.buyPrice.toFixed(2)}</TableCell>
                  <TableCell>${opportunity.sellPrice.toFixed(2)}</TableCell>
                  <TableCell>{opportunity.maxSize.toFixed(4)}</TableCell>
                  <TableCell sx={{ color: 'success.main' }}>
                    ${opportunity.profit.toFixed(2)}
                  </TableCell>
                  <TableCell>
                    {new Date(opportunity.timestamp).toLocaleTimeString()}
                  </TableCell>
                </TableRow>
              ))}
              {opportunities.length === 0 && (
                <TableRow>
                  <TableCell colSpan={8} align="center">
                    No arbitrage opportunities found
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    </Container>
  );
};

export default Opportunities; 