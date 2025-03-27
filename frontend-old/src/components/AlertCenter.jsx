import React from 'react';
import {
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Typography,
  Box,
  Alert,
} from '@mui/material';
import {
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { formatDate } from '../utils/formatters';

const getAlertIcon = (type) => {
  switch (type) {
    case 'danger':
      return <ErrorIcon color="error" />;
    case 'warning':
      return <WarningIcon color="warning" />;
    case 'info':
      return <InfoIcon color="info" />;
    default:
      return <InfoIcon />;
  }
};

const getAlertSeverity = (type) => {
  switch (type) {
    case 'danger':
      return 'error';
    case 'warning':
      return 'warning';
    case 'info':
      return 'info';
    default:
      return 'info';
  }
};

const AlertCenter = ({ alerts = [] }) => {
  if (!alerts || alerts.length === 0) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography color="textSecondary">
          No active alerts
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        System Alerts
      </Typography>
      <List>
        {alerts.map((alert, index) => (
          <ListItem key={index} sx={{ mb: 1 }}>
            <Alert
              severity={getAlertSeverity(alert.type)}
              icon={getAlertIcon(alert.type)}
              sx={{
                width: '100%',
                '& .MuiAlert-message': {
                  width: '100%'
                }
              }}
            >
              <Box>
                <Typography variant="body1">
                  {alert.message}
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  {formatDate(alert.timestamp)}
                </Typography>
              </Box>
            </Alert>
          </ListItem>
        ))}
      </List>
    </Box>
  );
};

export default AlertCenter; 