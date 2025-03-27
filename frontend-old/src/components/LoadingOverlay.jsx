import React from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';

const LoadingOverlay = ({ message = 'Loading...' }) => {
  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'rgba(255, 255, 255, 0.8)',
        zIndex: 9999,
      }}
    >
      <CircularProgress
        size={60}
        thickness={4}
        sx={{
          color: (theme) => theme.palette.primary.main,
          marginBottom: 2,
        }}
      />
      <Typography
        variant="h6"
        sx={{
          color: (theme) => theme.palette.text.secondary,
          marginTop: 2,
        }}
      >
        {message}
      </Typography>
    </Box>
  );
};

export default LoadingOverlay; 