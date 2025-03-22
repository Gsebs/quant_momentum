import React from 'react';
import { Paper, Box, Typography, Icon } from '@mui/material';

const StatsCard = ({ title, value, subvalue, icon }) => {
  return (
    <Paper
      sx={{
        p: 2,
        display: 'flex',
        flexDirection: 'column',
        height: 140,
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          top: -20,
          right: -20,
          opacity: 0.1,
          transform: 'rotate(-15deg)',
        }}
      >
        <Icon sx={{ fontSize: 120 }}>{icon}</Icon>
      </Box>
      
      <Typography
        color="textSecondary"
        gutterBottom
        variant="overline"
      >
        {title}
      </Typography>
      
      <Typography
        color="textPrimary"
        variant="h4"
        sx={{
          mb: subvalue ? 1 : 0,
          fontWeight: 'medium',
        }}
      >
        {value}
      </Typography>
      
      {subvalue && (
        <Typography
          color="textSecondary"
          variant="caption"
          sx={{
            display: 'flex',
            alignItems: 'center',
            mt: 'auto',
          }}
        >
          {subvalue}
        </Typography>
      )}
    </Paper>
  );
};

export default StatsCard; 