import React from 'react';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import Layout from './components/Layout';
import ErrorBoundary from './components/ErrorBoundary';
import AlertCenter from './components/AlertCenter';
import { TradingDashboard } from './components/TradingDashboard';

// Create a theme instance
const theme = createTheme({
  palette: {
    mode: 'dark',
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ErrorBoundary>
        <Layout title="Quant Momentum Strategy">
          <AlertCenter />
          <TradingDashboard />
        </Layout>
      </ErrorBoundary>
    </ThemeProvider>
  );
}

export default App;
