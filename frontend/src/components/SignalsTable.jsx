import React from 'react';
import { Box } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

const SignalsTable = ({ signals }) => {
  const columns = [
    {
      field: 'ticker',
      headerName: 'Ticker',
      width: 120,
      renderCell: (params) => (
        <Box sx={{ fontWeight: 'bold' }}>
          {params.value}
        </Box>
      ),
    },
    {
      field: 'momentum_score',
      headerName: 'Momentum Score',
      width: 160,
      valueFormatter: (params) => params.value.toFixed(2),
      renderCell: (params) => (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            color: params.value > 0 ? 'success.main' : 'error.main',
          }}
        >
          {params.value > 0 ? <TrendingUpIcon sx={{ mr: 1 }} /> : <TrendingDownIcon sx={{ mr: 1 }} />}
          {params.value.toFixed(2)}
        </Box>
      ),
    },
    {
      field: 'signal',
      headerName: 'Signal',
      width: 120,
      renderCell: (params) => (
        <Box
          sx={{
            color: params.value === 'BUY' ? 'success.main' : 
                  params.value === 'SELL' ? 'error.main' : 'text.secondary',
            fontWeight: 'medium',
          }}
        >
          {params.value}
        </Box>
      ),
    },
    {
      field: 'price',
      headerName: 'Current Price',
      width: 140,
      valueFormatter: (params) => `$${params.value.toFixed(2)}`,
    },
    {
      field: 'volume',
      headerName: 'Volume',
      width: 140,
      valueFormatter: (params) => params.value.toLocaleString(),
    },
    {
      field: 'volatility',
      headerName: 'Volatility',
      width: 130,
      valueFormatter: (params) => `${(params.value * 100).toFixed(2)}%`,
    },
    {
      field: 'last_updated',
      headerName: 'Last Updated',
      width: 180,
      valueFormatter: (params) => new Date(params.value).toLocaleString(),
    },
  ];

  return (
    <Box sx={{ height: 400, width: '100%' }}>
      <DataGrid
        rows={signals}
        columns={columns}
        pageSize={5}
        rowsPerPageOptions={[5, 10, 20]}
        disableSelectionOnClick
        getRowId={(row) => row.ticker}
        sx={{
          '& .MuiDataGrid-cell:focus': {
            outline: 'none',
          },
          '& .MuiDataGrid-row:hover': {
            backgroundColor: 'action.hover',
          },
        }}
      />
    </Box>
  );
};

export default SignalsTable; 