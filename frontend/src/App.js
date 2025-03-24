import React, { useState, useEffect } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

// Components
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Opportunities from './pages/Opportunities';
import Trades from './pages/Trades';
import Performance from './pages/Performance';
import Settings from './pages/Settings';

// Theme configuration
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#0a1929',
      paper: '#132f4c',
    },
  },
});

function App() {
  const [open, setOpen] = useState(true);
  const [systemStatus, setSystemStatus] = useState('stopped');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check system status on component mount
    checkSystemStatus();
    // Poll system status every 5 seconds
    const interval = setInterval(checkSystemStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const checkSystemStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/health');
      const data = await response.json();
      setSystemStatus(data.status);
      setIsLoading(false);
    } catch (error) {
      console.error('Error checking system status:', error);
      setSystemStatus('error');
      setIsLoading(false);
    }
  };

  const toggleDrawer = () => {
    setOpen(!open);
  };

  const startSystem = async () => {
    try {
      const response = await fetch('http://localhost:8000/start', {
        method: 'POST',
      });
      const data = await response.json();
      setSystemStatus(data.status);
    } catch (error) {
      console.error('Error starting system:', error);
    }
  };

  const stopSystem = async () => {
    try {
      const response = await fetch('http://localhost:8000/stop', {
        method: 'POST',
      });
      const data = await response.json();
      setSystemStatus(data.status);
    } catch (error) {
      console.error('Error stopping system:', error);
    }
  };

  if (isLoading) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
          Loading...
        </Box>
      </ThemeProvider>
    );
  }

  return (
    <Router>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box sx={{ display: 'flex' }}>
          <Navbar
            open={open}
            toggleDrawer={toggleDrawer}
            systemStatus={systemStatus}
            onStart={startSystem}
            onStop={stopSystem}
          />
          <Sidebar open={open} />
          <Box
            component="main"
            sx={{
              backgroundColor: theme.palette.background.default,
              flexGrow: 1,
              height: '100vh',
              overflow: 'auto',
              pt: 8,
              px: 3,
            }}
          >
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/opportunities" element={<Opportunities />} />
              <Route path="/trades" element={<Trades />} />
              <Route path="/performance" element={<Performance />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Box>
        </Box>
      </ThemeProvider>
    </Router>
  );
}

export default App;
