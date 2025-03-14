import React, { useState, useEffect } from 'react';
import { Container, Grid, Paper, Typography, Box, Alert, Button } from '@mui/material';
import { fetchMomentumSignals, fetchPerformanceData, getChartUrl } from '../services/api';
import MomentumTable from './MomentumTable';
import PerformanceChart from './PerformanceChart';
import LoadingOverlay from './LoadingOverlay';

const Dashboard = () => {
    const [momentumData, setMomentumData] = useState([]);
    const [performanceData, setPerformanceData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const loadData = async () => {
        setLoading(true);
        setError(null);
        try {
            const [signals, performance] = await Promise.all([
                fetchMomentumSignals(),
                fetchPerformanceData()
            ]);
            setMomentumData(signals);
            setPerformanceData(performance);
        } catch (err) {
            setError(err.message || 'An error occurred while fetching data. The server might be temporarily unavailable due to rate limiting.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadData();
    }, []);

    if (loading) return <LoadingOverlay message="Loading dashboard data..." />;
    
    if (error) {
        return (
            <Container maxWidth="lg">
                <Box sx={{ my: 4 }}>
                    <Alert 
                        severity="error" 
                        action={
                            <Button color="inherit" size="small" onClick={loadData}>
                                Retry
                            </Button>
                        }
                        sx={{ mb: 2 }}
                    >
                        {error}
                    </Alert>
                    <Typography variant="body1" color="text.secondary">
                        The server might be experiencing high load or rate limiting from data providers.
                        Please try again in a few minutes.
                    </Typography>
                </Box>
            </Container>
        );
    }

    return (
        <Container maxWidth="lg">
            <Box sx={{ my: 4 }}>
                <Typography variant="h4" component="h1" gutterBottom>
                    Momentum Trading Strategy Dashboard
                </Typography>

                <Grid container spacing={3}>
                    {/* Performance Charts */}
                    <Grid item xs={12}>
                        <Paper sx={{ p: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Strategy Performance
                            </Typography>
                            <PerformanceChart data={performanceData} />
                        </Paper>
                    </Grid>

                    {/* Momentum Signals Table */}
                    <Grid item xs={12}>
                        <Paper sx={{ p: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Top Momentum Signals
                            </Typography>
                            <MomentumTable data={momentumData} />
                        </Paper>
                    </Grid>

                    {/* Static Charts */}
                    <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Correlation Heatmap
                            </Typography>
                            <img 
                                src={getChartUrl('correlation_heatmap.png')} 
                                alt="Correlation Heatmap"
                                style={{ width: '100%', height: 'auto' }}
                            />
                        </Paper>
                    </Grid>

                    <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Performance Plot
                            </Typography>
                            <img 
                                src={getChartUrl('performance_plot.png')} 
                                alt="Performance Plot"
                                style={{ width: '100%', height: 'auto' }}
                            />
                        </Paper>
                    </Grid>
                </Grid>
            </Box>
        </Container>
    );
};

export default Dashboard; 