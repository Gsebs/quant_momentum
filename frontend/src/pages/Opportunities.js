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
  Chip,
  CircularProgress,
  Grid,
} from '@mui/material';

function Opportunities() {
  const [opportunities, setOpportunities] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  useEffect(() => {
    const fetchOpportunities = async () => {
      try {
        const response = await fetch('http://localhost:8000/opportunities');
        const data = await response.json();
        setOpportunities(data.opportunities);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchOpportunities();
    const interval = setInterval(fetchOpportunities, 1000); // Update every second
    return () => clearInterval(interval);
  }, []);

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
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
      <Typography variant="h5" gutterBottom>
        Arbitrage Opportunities
      </Typography>
      <Paper>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Symbol</TableCell>
                <TableCell>Buy Exchange</TableCell>
                <TableCell align="right">Buy Price</TableCell>
                <TableCell>Sell Exchange</TableCell>
                <TableCell align="right">Sell Price</TableCell>
                <TableCell align="right">Price Difference</TableCell>
                <TableCell align="right">Estimated Profit</TableCell>
                <TableCell align="center">Confidence</TableCell>
                <TableCell>Time</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {opportunities
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((opportunity, index) => (
                  <TableRow key={index}>
                    <TableCell>{opportunity.symbol}</TableCell>
                    <TableCell>{opportunity.buy_exchange}</TableCell>
                    <TableCell align="right">
                      ${opportunity.buy_price.toFixed(2)}
                    </TableCell>
                    <TableCell>{opportunity.sell_exchange}</TableCell>
                    <TableCell align="right">
                      ${opportunity.sell_price.toFixed(2)}
                    </TableCell>
                    <TableCell align="right">
                      ${opportunity.price_difference.toFixed(2)}
                    </TableCell>
                    <TableCell align="right">
                      ${opportunity.estimated_profit.toFixed(2)}
                    </TableCell>
                    <TableCell align="center">
                      <Chip
                        label={`${(opportunity.confidence * 100).toFixed(1)}%`}
                        color={
                          opportunity.confidence >= 0.8
                            ? 'success'
                            : opportunity.confidence >= 0.6
                            ? 'warning'
                            : 'error'
                        }
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      {new Date(opportunity.timestamp).toLocaleString()}
                    </TableCell>
                  </TableRow>
                ))}
              {opportunities.length === 0 && (
                <TableRow>
                  <TableCell colSpan={9} align="center">
                    No arbitrage opportunities found
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          rowsPerPageOptions={[5, 10, 25]}
          component="div"
          count={opportunities.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      </Paper>

      <Box mt={3}>
        <Typography variant="h6" gutterBottom>
          Statistics
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Total Opportunities
              </Typography>
              <Typography variant="h4">
                {opportunities.length}
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Average Profit
              </Typography>
              <Typography variant="h4">
                ${opportunities.reduce((sum, opp) => sum + opp.estimated_profit, 0) / opportunities.length || 0}
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Highest Profit Opportunity
              </Typography>
              <Typography variant="h4">
                ${Math.max(...opportunities.map(opp => opp.estimated_profit), 0).toFixed(2)}
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Average Confidence
              </Typography>
              <Typography variant="h4">
                {(opportunities.reduce((sum, opp) => sum + opp.confidence, 0) / opportunities.length * 100 || 0).toFixed(1)}%
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
}

export default Opportunities; 