import React, { useState, useEffect } from 'react';
import { Container, Grid, Paper, Box, Typography, CircularProgress } from '@mui/material';
import TradingMetrics from './TradingMetrics';
import PerformanceChart from './PerformanceChart';
import MomentumTable from './MomentumTable';
import AlertCenter from './AlertCenter';
import { fetchDashboardData } from '../services/api';

const Dashboard = () => {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [dashboardData, setDashboardData] = useState(null);

    useEffect(() => {
        const loadDashboardData = async () => {
            try {
                setLoading(true);
                const data = await fetchDashboardData();
                setDashboardData(data);
                setError(null);
            } catch (err) {
                setError('Failed to load dashboard data. Please try again later.');
                console.error('Dashboard data fetch error:', err);
            } finally {
                setLoading(false);
            }
        };

        loadDashboardData();
        const interval = setInterval(loadDashboardData, 300000); // Refresh every 5 minutes

        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <Box
                display="flex"
                justifyContent="center"
                alignItems="center"
                minHeight="80vh"
            >
                <CircularProgress />
            </Box>
        );
    }

    if (error) {
        return (
            <Box
                display="flex"
                justifyContent="center"
                alignItems="center"
                minHeight="80vh"
            >
                <Typography color="error" variant="h6">
                    {error}
                </Typography>
            </Box>
        );
    }

    return (
        <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
            <Grid container spacing={3}>
                <Grid item xs={12}>
                    <TradingMetrics data={dashboardData?.metrics} />
                </Grid>
                
                <Grid item xs={12} lg={8}>
                    <Paper
                        sx={{
                            p: 2,
                            display: 'flex',
                            flexDirection: 'column',
                            height: 400,
                        }}
                    >
                        <Typography variant="h6" gutterBottom>
                            Performance History
                        </Typography>
                        <PerformanceChart data={dashboardData?.performance} />
                    </Paper>
                </Grid>
                
                <Grid item xs={12} lg={4}>
                    <Paper
                        sx={{
                            p: 2,
                            display: 'flex',
                            flexDirection: 'column',
                            height: 400,
                            overflowY: 'auto',
                        }}
                    >
                        <AlertCenter alerts={dashboardData?.alerts} />
                    </Paper>
                </Grid>
                
                <Grid item xs={12}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6" gutterBottom>
                            Current Momentum Signals
                        </Typography>
                        <MomentumTable data={dashboardData?.signals} />
                    </Paper>
                </Grid>
            </Grid>
        </Container>
    );
};

export default Dashboard; 