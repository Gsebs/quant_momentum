import React from 'react';
import {
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper
} from '@mui/material';

const MomentumTable = ({ data }) => {
    if (!data || data.length === 0) {
        return <div>No momentum signals available</div>;
    }

    // Format number as percentage
    const formatPercent = (value) => {
        if (typeof value !== 'number') return 'N/A';
        return `${(value * 100).toFixed(2)}%`;
    };

    // Format number to 2 decimal places
    const formatNumber = (value) => {
        if (typeof value !== 'number') return 'N/A';
        return value.toFixed(2);
    };

    return (
        <TableContainer component={Paper}>
            <Table size="small">
                <TableHead>
                    <TableRow>
                        <TableCell>Ticker</TableCell>
                        <TableCell align="right">Composite Score</TableCell>
                        <TableCell align="right">Position Size</TableCell>
                        <TableCell align="right">1M Return</TableCell>
                        <TableCell align="right">3M Return</TableCell>
                        <TableCell align="right">6M Return</TableCell>
                        <TableCell align="right">12M Return</TableCell>
                        <TableCell align="right">Sharpe Ratio</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {data.map((row, index) => (
                        <TableRow key={index}>
                            <TableCell component="th" scope="row">
                                {row.Ticker || row.ticker}
                            </TableCell>
                            <TableCell align="right">{formatNumber(row.composite_score)}</TableCell>
                            <TableCell align="right">{formatPercent(row.position_size)}</TableCell>
                            <TableCell align="right">{formatPercent(row['1m_return'])}</TableCell>
                            <TableCell align="right">{formatPercent(row['3m_return'])}</TableCell>
                            <TableCell align="right">{formatPercent(row['6m_return'])}</TableCell>
                            <TableCell align="right">{formatPercent(row['12m_return'])}</TableCell>
                            <TableCell align="right">{formatNumber(row.risk_sharpe_ratio)}</TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    );
};

export default MomentumTable; 