import React from 'react';
import { Grid, Paper, Typography, Box, Divider } from '@mui/material';
import {
    TrendingUp as TrendingUpIcon,
    TrendingDown as TrendingDownIcon,
    Timeline as TimelineIcon,
    ShowChart as ShowChartIcon,
    Speed as SpeedIcon,
    Assessment as AssessmentIcon
} from '@mui/icons-material';
import { formatCurrency, formatPercent, formatNumber } from '../utils/formatters';

const MetricCard = ({ title, value, secondaryValue, icon, trend, color }) => (
    <Paper
        elevation={2}
        sx={{
            p: 2,
            height: '100%',
            background: 'linear-gradient(45deg, #1a237e 30%, #283593 90%)',
            color: 'white',
            transition: 'transform 0.3s ease-in-out',
            '&:hover': {
                transform: 'translateY(-4px)',
            },
        }}
    >
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Box sx={{ mr: 2 }}>{icon}</Box>
            <Typography variant="h6" component="div">
                {title}
            </Typography>
        </Box>
        <Typography variant="h4" component="div" sx={{ mb: 1 }}>
            {value}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', color: color }}>
            {trend === 'up' ? <TrendingUpIcon /> : <TrendingDownIcon />}
            <Typography variant="body2" sx={{ ml: 1 }}>
                {secondaryValue}
            </Typography>
        </Box>
    </Paper>
);

const TradingMetrics = ({ data }) => {
    if (!data) return null;

    const {
        totalValue,
        totalReturn,
        dailyReturn,
        sharpeRatio,
        maxDrawdown,
        winRate,
        volatility,
        momentumScore,
        activePositions,
        turnoverRate
    } = data;

    const metrics = [
        {
            title: 'Portfolio Value',
            value: formatCurrency(totalValue),
            secondaryValue: `${formatPercent(totalReturn)} total return`,
            icon: <ShowChartIcon />,
            trend: totalReturn >= 0 ? 'up' : 'down',
            color: totalReturn >= 0 ? '#4caf50' : '#f44336'
        },
        {
            title: 'Daily Return',
            value: formatPercent(dailyReturn),
            secondaryValue: 'Today\'s performance',
            icon: <TimelineIcon />,
            trend: dailyReturn >= 0 ? 'up' : 'down',
            color: dailyReturn >= 0 ? '#4caf50' : '#f44336'
        },
        {
            title: 'Sharpe Ratio',
            value: formatNumber(sharpeRatio, 2),
            secondaryValue: 'Risk-adjusted return',
            icon: <AssessmentIcon />,
            trend: sharpeRatio >= 1.5 ? 'up' : 'down',
            color: sharpeRatio >= 1.5 ? '#4caf50' : '#f44336'
        },
        {
            title: 'Max Drawdown',
            value: formatPercent(maxDrawdown),
            secondaryValue: 'Peak to trough decline',
            icon: <TrendingDownIcon />,
            trend: 'down',
            color: '#f44336'
        },
        {
            title: 'Win Rate',
            value: formatPercent(winRate),
            secondaryValue: 'Profitable trades',
            icon: <SpeedIcon />,
            trend: winRate >= 0.5 ? 'up' : 'down',
            color: winRate >= 0.5 ? '#4caf50' : '#f44336'
        },
        {
            title: 'Momentum Score',
            value: formatNumber(momentumScore, 2),
            secondaryValue: 'Market strength indicator',
            icon: <TrendingUpIcon />,
            trend: momentumScore >= 0 ? 'up' : 'down',
            color: momentumScore >= 0 ? '#4caf50' : '#f44336'
        }
    ];

    return (
        <Box sx={{ flexGrow: 1, mb: 4 }}>
            <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
                Real-Time Trading Metrics
            </Typography>
            <Grid container spacing={3}>
                {metrics.map((metric, index) => (
                    <Grid item xs={12} sm={6} md={4} key={index}>
                        <MetricCard {...metric} />
                    </Grid>
                ))}
            </Grid>
            <Box sx={{ mt: 4 }}>
                <Paper elevation={2} sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>
                        Additional Statistics
                    </Typography>
                    <Grid container spacing={2}>
                        <Grid item xs={12} sm={6} md={4}>
                            <Typography variant="subtitle2" color="textSecondary">
                                Active Positions
                            </Typography>
                            <Typography variant="h6">
                                {activePositions}
                            </Typography>
                        </Grid>
                        <Grid item xs={12} sm={6} md={4}>
                            <Typography variant="subtitle2" color="textSecondary">
                                Portfolio Volatility
                            </Typography>
                            <Typography variant="h6">
                                {formatPercent(volatility)}
                            </Typography>
                        </Grid>
                        <Grid item xs={12} sm={6} md={4}>
                            <Typography variant="subtitle2" color="textSecondary">
                                Turnover Rate
                            </Typography>
                            <Typography variant="h6">
                                {formatPercent(turnoverRate)}
                            </Typography>
                        </Grid>
                    </Grid>
                </Paper>
            </Box>
        </Box>
    );
};

export default TradingMetrics; 