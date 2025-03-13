import React, { useState, useEffect } from 'react';
import { Container, Grid, Paper, Typography, Box } from '@mui/material';
import { fetchMomentumSignals, fetchPerformanceData, getChartUrl } from '../services/api';
import MomentumTable from './MomentumTable';
import PerformanceChart from './PerformanceChart';

const Dashboard = () => {
    const [momentumData, setMomentumData] = useState([]);
    const [performanceData, setPerformanceData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const loadData = async () => {
            try {
                const [signals, performance] = await Promise.all([
                    fetchMomentumSignals(),
                    fetchPerformanceData()
                ]);
                setMomentumData(signals);
                setPerformanceData(performance);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        loadData();
    }, []);

    if (loading) return <Typography>Loading...</Typography>;
    if (error) return <Typography color="error">{error}</Typography>;

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