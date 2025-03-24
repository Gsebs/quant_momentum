import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Alert,
  CircularProgress,
  Divider,
} from '@mui/material';

function Settings() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [settings, setSettings] = useState({
    exchanges: {
      binance: {
        enabled: true,
        api_key: '',
        api_secret: '',
      },
      coinbase: {
        enabled: true,
        api_key: '',
        api_secret: '',
      },
    },
    trading: {
      trading_pairs: ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT'],
      min_profit_threshold: 0.1,
      max_position_size: 1.0,
      max_daily_loss: 1000.0,
      max_drawdown: 0.1,
    },
    risk: {
      stop_loss_percentage: 1.0,
      take_profit_percentage: 2.0,
      max_open_positions: 5,
      position_sizing_type: 'fixed',
    },
  });

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        // In a real application, fetch settings from the backend
        // const response = await fetch('http://localhost:8000/settings');
        // const data = await response.json();
        // setSettings(data);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchSettings();
  }, []);

  const handleExchangeToggle = (exchange) => {
    setSettings((prev) => ({
      ...prev,
      exchanges: {
        ...prev.exchanges,
        [exchange]: {
          ...prev.exchanges[exchange],
          enabled: !prev.exchanges[exchange].enabled,
        },
      },
    }));
  };

  const handleExchangeKeyChange = (exchange, field, value) => {
    setSettings((prev) => ({
      ...prev,
      exchanges: {
        ...prev.exchanges,
        [exchange]: {
          ...prev.exchanges[exchange],
          [field]: value,
        },
      },
    }));
  };

  const handleTradingSettingChange = (field, value) => {
    setSettings((prev) => ({
      ...prev,
      trading: {
        ...prev.trading,
        [field]: value,
      },
    }));
  };

  const handleRiskSettingChange = (field, value) => {
    setSettings((prev) => ({
      ...prev,
      risk: {
        ...prev.risk,
        [field]: value,
      },
    }));
  };

  const handleSave = async () => {
    try {
      // In a real application, save settings to the backend
      // await fetch('http://localhost:8000/settings', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(settings),
      // });
      setSuccess('Settings saved successfully');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError('Failed to save settings: ' + err.message);
      setTimeout(() => setError(null), 3000);
    }
  };

  if (loading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="200px"
      >
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box p={3}>
      <Typography variant="h5" gutterBottom>
        Settings
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }}>
          {success}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Exchange Settings */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Exchange Settings
            </Typography>
            <Grid container spacing={3}>
              {Object.entries(settings.exchanges).map(([exchange, config]) => (
                <Grid item xs={12} md={6} key={exchange}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={config.enabled}
                          onChange={() => handleExchangeToggle(exchange)}
                        />
                      }
                      label={exchange.charAt(0).toUpperCase() + exchange.slice(1)}
                    />
                    {config.enabled && (
                      <Box sx={{ mt: 2 }}>
                        <TextField
                          fullWidth
                          label="API Key"
                          type="password"
                          value={config.api_key}
                          onChange={(e) =>
                            handleExchangeKeyChange(exchange, 'api_key', e.target.value)
                          }
                          margin="normal"
                        />
                        <TextField
                          fullWidth
                          label="API Secret"
                          type="password"
                          value={config.api_secret}
                          onChange={(e) =>
                            handleExchangeKeyChange(exchange, 'api_secret', e.target.value)
                          }
                          margin="normal"
                        />
                      </Box>
                    )}
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>

        {/* Trading Settings */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Trading Settings
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Minimum Profit Threshold ($)"
                  type="number"
                  value={settings.trading.min_profit_threshold}
                  onChange={(e) =>
                    handleTradingSettingChange('min_profit_threshold', parseFloat(e.target.value))
                  }
                  inputProps={{ step: 0.1 }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Maximum Position Size (BTC)"
                  type="number"
                  value={settings.trading.max_position_size}
                  onChange={(e) =>
                    handleTradingSettingChange('max_position_size', parseFloat(e.target.value))
                  }
                  inputProps={{ step: 0.1 }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Maximum Daily Loss ($)"
                  type="number"
                  value={settings.trading.max_daily_loss}
                  onChange={(e) =>
                    handleTradingSettingChange('max_daily_loss', parseFloat(e.target.value))
                  }
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Maximum Drawdown (%)"
                  type="number"
                  value={settings.trading.max_drawdown * 100}
                  onChange={(e) =>
                    handleTradingSettingChange('max_drawdown', parseFloat(e.target.value) / 100)
                  }
                  inputProps={{ step: 1 }}
                />
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Risk Management Settings */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Risk Management
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Stop Loss (%)"
                  type="number"
                  value={settings.risk.stop_loss_percentage}
                  onChange={(e) =>
                    handleRiskSettingChange('stop_loss_percentage', parseFloat(e.target.value))
                  }
                  inputProps={{ step: 0.1 }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Take Profit (%)"
                  type="number"
                  value={settings.risk.take_profit_percentage}
                  onChange={(e) =>
                    handleRiskSettingChange('take_profit_percentage', parseFloat(e.target.value))
                  }
                  inputProps={{ step: 0.1 }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Maximum Open Positions"
                  type="number"
                  value={settings.risk.max_open_positions}
                  onChange={(e) =>
                    handleRiskSettingChange('max_open_positions', parseInt(e.target.value))
                  }
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>Position Sizing Type</InputLabel>
                  <Select
                    value={settings.risk.position_sizing_type}
                    label="Position Sizing Type"
                    onChange={(e) =>
                      handleRiskSettingChange('position_sizing_type', e.target.value)
                    }
                  >
                    <MenuItem value="fixed">Fixed Size</MenuItem>
                    <MenuItem value="percentage">Percentage of Balance</MenuItem>
                    <MenuItem value="risk">Risk-Based</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>

      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
        <Button variant="contained" color="primary" onClick={handleSave}>
          Save Settings
        </Button>
      </Box>
    </Box>
  );
}

export default Settings; 