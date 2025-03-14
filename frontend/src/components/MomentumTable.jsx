import React, { useState } from 'react';
import {
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    TableSortLabel,
    Paper,
    Box,
    LinearProgress,
    Chip,
    Typography,
    IconButton,
    Tooltip,
} from '@mui/material';
import {
    TrendingUp as TrendingUpIcon,
    TrendingDown as TrendingDownIcon,
    Info as InfoIcon,
} from '@mui/icons-material';
import { formatCurrency, formatNumber, formatPercent, formatCompactNumber } from '../utils/formatters';

const getSignalColor = (score) => {
    if (score >= 0.8) return '#2e7d32';  // Strong buy
    if (score >= 0.6) return '#4caf50';  // Buy
    if (score >= 0.4) return '#ffeb3b';  // Neutral
    if (score >= 0.2) return '#f44336';  // Sell
    return '#d32f2f';  // Strong sell
};

const getRecommendationChip = (recommendation) => {
    const config = {
        'strong_buy': { color: 'success', label: 'Strong Buy', variant: 'filled' },
        'buy': { color: 'success', label: 'Buy', variant: 'outlined' },
        'neutral': { color: 'warning', label: 'Neutral', variant: 'outlined' },
        'sell': { color: 'error', label: 'Sell', variant: 'outlined' },
        'strong_sell': { color: 'error', label: 'Strong Sell', variant: 'filled' }
    };

    const { color, label, variant } = config[recommendation] || config.neutral;
    return <Chip size="small" color={color} label={label} variant={variant} />;
};

const MomentumTable = ({ data = [] }) => {
    const [orderBy, setOrderBy] = useState('momentum_score');
    const [order, setOrder] = useState('desc');

    const handleSort = (property) => {
        const isAsc = orderBy === property && order === 'asc';
        setOrder(isAsc ? 'desc' : 'asc');
        setOrderBy(property);
    };

    const sortedData = React.useMemo(() => {
        return [...data].sort((a, b) => {
            const aValue = a[orderBy];
            const bValue = b[orderBy];
            
            if (order === 'desc') {
                return bValue - aValue;
            }
            return aValue - bValue;
        });
    }, [data, order, orderBy]);

    if (!data || data.length === 0) {
        return (
            <Box sx={{ p: 2, textAlign: 'center' }}>
                <Typography color="textSecondary">
                    No momentum signals available
                </Typography>
            </Box>
        );
    }

    return (
        <TableContainer component={Paper}>
            <Table size="small">
                <TableHead>
                    <TableRow>
                        <TableCell>
                            <TableSortLabel
                                active={orderBy === 'ticker'}
                                direction={orderBy === 'ticker' ? order : 'asc'}
                                onClick={() => handleSort('ticker')}
                            >
                                Ticker
                            </TableSortLabel>
                        </TableCell>
                        <TableCell align="right">
                            <TableSortLabel
                                active={orderBy === 'momentum_score'}
                                direction={orderBy === 'momentum_score' ? order : 'asc'}
                                onClick={() => handleSort('momentum_score')}
                            >
                                Momentum Score
                            </TableSortLabel>
                        </TableCell>
                        <TableCell align="right">
                            <TableSortLabel
                                active={orderBy === 'rank'}
                                direction={orderBy === 'rank' ? order : 'asc'}
                                onClick={() => handleSort('rank')}
                            >
                                Rank
                            </TableSortLabel>
                        </TableCell>
                        <TableCell align="right">
                            <TableSortLabel
                                active={orderBy === 'signal_strength'}
                                direction={orderBy === 'signal_strength' ? order : 'asc'}
                                onClick={() => handleSort('signal_strength')}
                            >
                                Signal Strength
                            </TableSortLabel>
                        </TableCell>
                        <TableCell align="right">Last Price</TableCell>
                        <TableCell align="right">Volume</TableCell>
                        <TableCell align="center">Recommendation</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {sortedData.map((row) => (
                        <TableRow
                            key={row.ticker}
                            sx={{
                                '&:last-child td, &:last-child th': { border: 0 },
                                '&:hover': { backgroundColor: 'action.hover' }
                            }}
                        >
                            <TableCell component="th" scope="row">
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    {row.ticker}
                                    {row.momentum_score >= 0.6 ? (
                                        <TrendingUpIcon color="success" fontSize="small" />
                                    ) : row.momentum_score <= 0.4 ? (
                                        <TrendingDownIcon color="error" fontSize="small" />
                                    ) : null}
                                </Box>
                            </TableCell>
                            <TableCell align="right">
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                    <LinearProgress
                                        variant="determinate"
                                        value={row.momentum_score * 100}
                                        sx={{
                                            width: 100,
                                            mr: 1,
                                            height: 8,
                                            borderRadius: 5,
                                            backgroundColor: 'grey.300',
                                            '& .MuiLinearProgress-bar': {
                                                backgroundColor: getSignalColor(row.momentum_score),
                                                borderRadius: 5,
                                            },
                                        }}
                                    />
                                    <Typography variant="body2">
                                        {formatNumber(row.momentum_score, 2)}
                                    </Typography>
                                </Box>
                            </TableCell>
                            <TableCell align="right">
                                {formatNumber(row.rank)}
                            </TableCell>
                            <TableCell align="right">
                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                    <LinearProgress
                                        variant="determinate"
                                        value={row.signal_strength * 100}
                                        sx={{
                                            width: 60,
                                            mr: 1,
                                            height: 8,
                                            borderRadius: 5,
                                            backgroundColor: 'grey.300',
                                            '& .MuiLinearProgress-bar': {
                                                backgroundColor: getSignalColor(row.signal_strength),
                                                borderRadius: 5,
                                            },
                                        }}
                                    />
                                    <Typography variant="body2">
                                        {formatPercent(row.signal_strength)}
                                    </Typography>
                                </Box>
                            </TableCell>
                            <TableCell align="right">
                                {formatCurrency(row.last_price)}
                            </TableCell>
                            <TableCell align="right">
                                {formatCompactNumber(row.volume)}
                            </TableCell>
                            <TableCell align="center">
                                {getRecommendationChip(row.recommendation)}
                            </TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    );
};

export default MomentumTable; 