import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Tooltip,
  LinearProgress,
} from '@mui/material';
import {
  TrendingUp,
  AttachMoney,
  Timeline,
  ShowChart,
  Speed,
  Assessment,
} from '@mui/icons-material';
import { formatCurrency, formatPercentage } from '../utils/formatters';

const MetricCard = ({ title, value, icon, tooltip, progress, color }) => (
  <Paper
    elevation={0}
    sx={{
      p: 2,
      height: '100%',
      background: 'linear-gradient(45deg, #ffffff 30%, #f5f5f5 90%)',
      border: '1px solid rgba(0, 0, 0, 0.12)',
      borderRadius: 2,
      transition: 'transform 0.2s ease-in-out',
      '&:hover': {
        transform: 'translateY(-4px)',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
      },
    }}
  >
    <Tooltip title={tooltip} arrow placement="top">
      <Box>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            mb: 1,
          }}
        >
          <Box
            sx={{
              p: 1,
              borderRadius: 1,
              mr: 1,
              bgcolor: `${color}.light`,
              color: `${color}.main`,
            }}
          >
            {icon}
          </Box>
          <Typography
            variant="subtitle2"
            color="textSecondary"
            sx={{ fontWeight: 500 }}
          >
            {title}
          </Typography>
        </Box>
        <Typography variant="h4" sx={{ mb: 1, fontWeight: 'medium' }}>
          {value}
        </Typography>
        {progress !== undefined && (
          <Box sx={{ width: '100%', mt: 1 }}>
            <LinearProgress
              variant="determinate"
              value={progress}
              sx={{
                height: 6,
                borderRadius: 3,
                backgroundColor: `${color}.light`,
                '& .MuiLinearProgress-bar': {
                  backgroundColor: `${color}.main`,
                },
              }}
            />
          </Box>
        )}
      </Box>
    </Tooltip>
  </Paper>
);

const StrategyMetrics = ({ metrics }) => {
  const {
    totalValue,
    returns,
    sharpeRatio,
    volatility,
    momentum,
    winRate,
  } = metrics;

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} sm={6} md={4}>
        <MetricCard
          title="Portfolio Value"
          value={formatCurrency(totalValue)}
          icon={<AttachMoney />}
          tooltip="Current total portfolio value"
          color="primary"
        />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <MetricCard
          title="Total Return"
          value={formatPercentage(returns)}
          icon={<TrendingUp />}
          tooltip="Total return since inception"
          progress={Math.min(returns, 100)}
          color="success"
        />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <MetricCard
          title="Sharpe Ratio"
          value={sharpeRatio.toFixed(2)}
          icon={<Assessment />}
          tooltip="Risk-adjusted return metric"
          progress={Math.min(sharpeRatio * 25, 100)}
          color="info"
        />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <MetricCard
          title="Volatility"
          value={formatPercentage(volatility)}
          icon={<Timeline />}
          tooltip="30-day rolling volatility"
          progress={100 - Math.min(volatility * 2, 100)}
          color="warning"
        />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <MetricCard
          title="Momentum Score"
          value={formatPercentage(momentum)}
          icon={<Speed />}
          tooltip="Current portfolio momentum score"
          progress={Math.min(momentum * 100, 100)}
          color="secondary"
        />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <MetricCard
          title="Win Rate"
          value={formatPercentage(winRate)}
          icon={<ShowChart />}
          tooltip="Percentage of profitable trades"
          progress={winRate}
          color="success"
        />
      </Grid>
    </Grid>
  );
};

export default StrategyMetrics; 