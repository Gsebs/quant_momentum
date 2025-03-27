import React, { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Opportunities from './pages/Opportunities';
import Trades from './pages/Trades';
import Performance from './pages/Performance';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});

const App: React.FC = () => {
  const [drawerOpen, setDrawerOpen] = useState(true);

  const handleDrawerToggle = () => {
    setDrawerOpen(!drawerOpen);
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Layout open={drawerOpen} onDrawerToggle={handleDrawerToggle}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/opportunities" element={<Opportunities />} />
          <Route path="/trades" element={<Trades />} />
          <Route path="/performance" element={<Performance />} />
        </Routes>
      </Layout>
    </ThemeProvider>
  );
};

export default App;
