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
  Container,
} from '@mui/material';

interface Settings {
  binance_api_key: string;
  binance_api_secret: string;
  coinbase_api_key: string;
  coinbase_api_secret: string;
  min_profit_threshold: number;
  max_trade_amount: number;
  risk_level: number;
}

const Settings: React.FC = () => {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const response = await fetch('/api/settings');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setSettings(data);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
        setError(errorMessage);
        console.error('Error fetching settings:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchSettings();
  }, []);

  const handleChange = (field: keyof Settings) => (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!settings) return;

    const value = event.target.type === 'number' ? Number(event.target.value) : event.target.value;
    setSettings({
      ...settings,
      [field]: value,
    });
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!settings) return;

    setSaveStatus('saving');
    try {
      const response = await fetch('/api/settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(settings),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setSaveStatus('success');
      setTimeout(() => setSaveStatus('idle'), 3000);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
      setError(errorMessage);
      setSaveStatus('error');
      console.error('Error saving settings:', err);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Container>
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      </Container>
    );
  }

  if (!settings) {
    return (
      <Container>
        <Alert severity="info" sx={{ mt: 2 }}>
          No settings available.
        </Alert>
      </Container>
    );
  }

  return (
    <Container>
      <Typography variant="h4" component="h1" gutterBottom sx={{ mt: 4 }}>
        Settings
      </Typography>
      <Paper sx={{ p: 3 }}>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Exchange API Keys
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Binance API Key"
                type="password"
                value={settings.binance_api_key}
                onChange={handleChange('binance_api_key')}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Binance API Secret"
                type="password"
                value={settings.binance_api_secret}
                onChange={handleChange('binance_api_secret')}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Coinbase API Key"
                type="password"
                value={settings.coinbase_api_key}
                onChange={handleChange('coinbase_api_key')}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Coinbase API Secret"
                type="password"
                value={settings.coinbase_api_secret}
                onChange={handleChange('coinbase_api_secret')}
              />
            </Grid>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Trading Parameters
              </Typography>
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Minimum Profit Threshold (%)"
                type="number"
                inputProps={{ step: 0.01, min: 0 }}
                value={settings.min_profit_threshold}
                onChange={handleChange('min_profit_threshold')}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Maximum Trade Amount (USD)"
                type="number"
                inputProps={{ step: 1, min: 0 }}
                value={settings.max_trade_amount}
                onChange={handleChange('max_trade_amount')}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Risk Level (1-10)"
                type="number"
                inputProps={{ step: 1, min: 1, max: 10 }}
                value={settings.risk_level}
                onChange={handleChange('risk_level')}
              />
            </Grid>
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  disabled={saveStatus === 'saving'}
                >
                  {saveStatus === 'saving' ? 'Saving...' : 'Save Settings'}
                </Button>
                {saveStatus === 'success' && (
                  <Alert severity="success" sx={{ flexGrow: 1 }}>
                    Settings saved successfully!
                  </Alert>
                )}
                {saveStatus === 'error' && (
                  <Alert severity="error" sx={{ flexGrow: 1 }}>
                    Failed to save settings. Please try again.
                  </Alert>
                )}
              </Box>
            </Grid>
          </Grid>
        </form>
      </Paper>
    </Container>
  );
};

export default Settings; 