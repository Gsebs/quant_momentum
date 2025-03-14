import React from 'react';
import {
  Box,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Typography,
  Paper,
  Chip,
} from '@mui/material';
import {
  NotificationsActive as AlertIcon,
  TrendingUp as PositiveIcon,
  TrendingDown as NegativeIcon,
  Info as InfoIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { formatDate } from '../utils/formatters';

const getAlertIcon = (type) => {
  switch (type.toLowerCase()) {
    case 'positive':
      return <PositiveIcon sx={{ color: 'success.main' }} />;
    case 'negative':
      return <NegativeIcon sx={{ color: 'error.main' }} />;
    case 'warning':
      return <WarningIcon sx={{ color: 'warning.main' }} />;
    case 'info':
    default:
      return <InfoIcon sx={{ color: 'info.main' }} />;
  }
};

const getAlertColor = (type) => {
  switch (type.toLowerCase()) {
    case 'positive':
      return 'success';
    case 'negative':
      return 'error';
    case 'warning':
      return 'warning';
    case 'info':
    default:
      return 'info';
  }
};

const AlertCenter = ({ alerts = [] }) => {
  if (!alerts || alerts.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography color="textSecondary">
          No alerts at this time
        </Typography>
      </Box>
    );
  }

  return (
    <List sx={{ width: '100%', bgcolor: 'background.paper' }}>
      {alerts.map((alert, index) => (
        <ListItem
          key={`${alert.id || index}`}
          sx={{
            mb: 1,
            borderRadius: 1,
            border: 1,
            borderColor: 'divider',
            '&:hover': {
              bgcolor: 'action.hover',
            },
          }}
        >
          <ListItemIcon>
            {getAlertIcon(alert.type)}
          </ListItemIcon>
          <ListItemText
            primary={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="subtitle1">
                  {alert.title}
                </Typography>
                <Chip
                  size="small"
                  label={alert.type}
                  color={getAlertColor(alert.type)}
                  sx={{ textTransform: 'capitalize' }}
                />
              </Box>
            }
            secondary={
              <Box sx={{ mt: 0.5 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {alert.message}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {formatDate(alert.timestamp)}
                </Typography>
              </Box>
            }
          />
        </ListItem>
      ))}
    </List>
  );
};

export default AlertCenter; 