import React, { useState } from 'react';
import {
    Box,
    ToggleButton,
    ToggleButtonGroup,
    Typography,
    useTheme,
} from '@mui/material';
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
} from 'recharts';
import { formatCurrency, formatPercent, formatDate } from '../utils/formatters';

const timeRanges = [
    { label: '1M', days: 30 },
    { label: '3M', days: 90 },
    { label: '6M', days: 180 },
    { label: '1Y', days: 365 },
    { label: 'All', days: Infinity },
];

const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload || !payload.length) return null;

    const data = payload[0].payload;
    return (
        <Box
            sx={{
                bgcolor: 'background.paper',
                p: 1.5,
                border: 1,
                borderColor: 'divider',
                borderRadius: 1,
                boxShadow: 1,
            }}
        >
            <Typography variant="subtitle2" gutterBottom>
                {formatDate(label)}
            </Typography>
            <Box sx={{ mt: 1 }}>
                <Typography variant="body2" color="text.secondary">
                    Portfolio Value: {formatCurrency(data.value)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    Return: {formatPercent(data.return)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    Cumulative Return: {formatPercent(data.cumulative_return)}
                </Typography>
            </Box>
        </Box>
    );
};

const PerformanceChart = ({ data = [] }) => {
    const theme = useTheme();
    const [timeRange, setTimeRange] = useState('3M');

    const handleTimeRangeChange = (event, newRange) => {
        if (newRange !== null) {
            setTimeRange(newRange);
        }
    };

    const filteredData = React.useMemo(() => {
        if (!data || data.length === 0) return [];

        const selectedRange = timeRanges.find(r => r.label === timeRange);
        if (!selectedRange) return data;

        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - selectedRange.days);

        return data.filter(item => new Date(item.date) >= cutoffDate);
    }, [data, timeRange]);

    if (!data || data.length === 0) {
        return (
            <Box sx={{ p: 2, textAlign: 'center' }}>
                <Typography color="textSecondary">
                    No performance data available
                </Typography>
            </Box>
        );
    }

    return (
        <Box sx={{ width: '100%', height: '100%' }}>
            <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <ToggleButtonGroup
                    value={timeRange}
                    exclusive
                    onChange={handleTimeRangeChange}
                    size="small"
                >
                    {timeRanges.map(range => (
                        <ToggleButton key={range.label} value={range.label}>
                            {range.label}
                        </ToggleButton>
                    ))}
                </ToggleButtonGroup>
            </Box>

            <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                    data={filteredData}
                    margin={{
                        top: 10,
                        right: 30,
                        left: 0,
                        bottom: 0,
                    }}
                >
                    <defs>
                        <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                            <stop
                                offset="5%"
                                stopColor={theme.palette.primary.main}
                                stopOpacity={0.8}
                            />
                            <stop
                                offset="95%"
                                stopColor={theme.palette.primary.main}
                                stopOpacity={0}
                            />
                        </linearGradient>
                        <linearGradient id="colorReturn" x1="0" y1="0" x2="0" y2="1">
                            <stop
                                offset="5%"
                                stopColor={theme.palette.success.main}
                                stopOpacity={0.8}
                            />
                            <stop
                                offset="95%"
                                stopColor={theme.palette.success.main}
                                stopOpacity={0}
                            />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                        dataKey="date"
                        tickFormatter={(date) => formatDate(date)}
                        tick={{ fontSize: 12 }}
                    />
                    <YAxis
                        yAxisId="left"
                        tickFormatter={(value) => formatCurrency(value)}
                        tick={{ fontSize: 12 }}
                    />
                    <YAxis
                        yAxisId="right"
                        orientation="right"
                        tickFormatter={(value) => formatPercent(value)}
                        tick={{ fontSize: 12 }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend />
                    <Area
                        yAxisId="left"
                        type="monotone"
                        dataKey="value"
                        name="Portfolio Value"
                        stroke={theme.palette.primary.main}
                        fillOpacity={1}
                        fill="url(#colorValue)"
                    />
                    <Area
                        yAxisId="right"
                        type="monotone"
                        dataKey="cumulative_return"
                        name="Cumulative Return"
                        stroke={theme.palette.success.main}
                        fillOpacity={1}
                        fill="url(#colorReturn)"
                    />
                </AreaChart>
            </ResponsiveContainer>
        </Box>
    );
};

export default PerformanceChart; 