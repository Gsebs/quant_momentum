import React from 'react';
import {
  Paper,
  Typography,
  Box,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  ExpandMore,
  TrendingUp,
  Timeline,
  Assessment,
  Speed,
  ShowChart,
  Info,
} from '@mui/icons-material';

const StrategyInfo = () => {
  const sections = [
    {
      title: 'Strategy Overview',
      icon: <Info />,
      content: `This quantitative momentum strategy identifies and invests in stocks showing strong price momentum 
      while managing risk through diversification and position sizing. The strategy combines traditional momentum 
      indicators with machine learning to enhance stock selection.`,
    },
    {
      title: 'Key Components',
      icon: <Assessment />,
      content: (
        <List>
          <ListItem>
            <ListItemIcon>
              <TrendingUp />
            </ListItemIcon>
            <ListItemText
              primary="Momentum Calculation"
              secondary="Uses multiple timeframes (1M, 3M, 6M, 12M) with higher weights on recent performance"
            />
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <Speed />
            </ListItemIcon>
            <ListItemText
              primary="Technical Indicators"
              secondary="Incorporates RSI, MACD, and volume analysis for signal confirmation"
            />
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <Timeline />
            </ListItemIcon>
            <ListItemText
              primary="Risk Management"
              secondary="Position sizing based on volatility and correlation analysis"
            />
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <ShowChart />
            </ListItemIcon>
            <ListItemText
              primary="Machine Learning Enhancement"
              secondary="Uses ML models to improve stock selection and reduce false signals"
            />
          </ListItem>
        </List>
      ),
    },
    {
      title: 'Investment Process',
      icon: <Timeline />,
      content: (
        <Box>
          <Typography variant="body1" paragraph>
            The strategy follows a systematic process:
          </Typography>
          <List>
            <ListItem>
              <ListItemText
                primary="1. Universe Selection"
                secondary="Starts with S&P 500 stocks, filtered for liquidity and market cap"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="2. Momentum Scoring"
                secondary="Calculates composite momentum scores using multiple timeframes"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="3. Signal Generation"
                secondary="Combines momentum scores with technical indicators"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="4. Risk Analysis"
                secondary="Evaluates volatility, correlation, and sector exposure"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="5. Portfolio Construction"
                secondary="Determines position sizes and rebalancing needs"
              />
            </ListItem>
          </List>
        </Box>
      ),
    },
    {
      title: 'Risk Management',
      icon: <Assessment />,
      content: (
        <Box>
          <Typography variant="body1" paragraph>
            The strategy employs multiple risk management techniques:
          </Typography>
          <List>
            <ListItem>
              <ListItemText
                primary="Position Limits"
                secondary="Maximum position size of 10% per stock"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Volatility Adjustment"
                secondary="Reduces position sizes for highly volatile stocks"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Sector Diversification"
                secondary="Maintains sector exposure limits to avoid concentration"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Stop Loss"
                secondary="Implements trailing stops to protect gains"
              />
            </ListItem>
          </List>
        </Box>
      ),
    },
  ];

  return (
    <Paper
      elevation={0}
      sx={{
        p: 3,
        border: '1px solid rgba(0, 0, 0, 0.12)',
        borderRadius: 2,
      }}
    >
      <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
        Strategy Documentation
      </Typography>
      <Divider sx={{ mb: 3 }} />
      {sections.map((section, index) => (
        <Accordion
          key={index}
          elevation={0}
          sx={{
            '&:before': {
              display: 'none',
            },
            border: '1px solid rgba(0, 0, 0, 0.12)',
            mb: 2,
          }}
        >
          <AccordionSummary
            expandIcon={<ExpandMore />}
            sx={{
              '&:hover': {
                bgcolor: 'rgba(0, 0, 0, 0.04)',
              },
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Box
                sx={{
                  mr: 2,
                  display: 'flex',
                  alignItems: 'center',
                  color: 'primary.main',
                }}
              >
                {section.icon}
              </Box>
              <Typography variant="subtitle1">{section.title}</Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            {typeof section.content === 'string' ? (
              <Typography variant="body1">{section.content}</Typography>
            ) : (
              section.content
            )}
          </AccordionDetails>
        </Accordion>
      ))}
    </Paper>
  );
};

export default StrategyInfo; 