import React from 'react';
import { Box, Button, Typography, Paper } from '@mui/material';
import { Warning } from '@mui/icons-material';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
    // You can also log the error to an error reporting service here
    console.error('Error caught by boundary:', error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: '100vh',
            bgcolor: '#f5f5f5',
          }}
        >
          <Paper
            elevation={3}
            sx={{
              p: 4,
              maxWidth: 600,
              textAlign: 'center',
            }}
          >
            <Warning
              color="error"
              sx={{
                fontSize: 64,
                mb: 2,
              }}
            />
            <Typography variant="h5" gutterBottom color="error">
              Oops! Something went wrong
            </Typography>
            <Typography color="textSecondary" paragraph>
              We're sorry, but there was an error loading this page.
              Our team has been notified and is working on the fix.
            </Typography>
            {process.env.NODE_ENV === 'development' && (
              <Box
                sx={{
                  mt: 2,
                  p: 2,
                  bgcolor: '#f8f9fa',
                  borderRadius: 1,
                  textAlign: 'left',
                }}
              >
                <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                  {this.state.error && this.state.error.toString()}
                  {this.state.errorInfo && this.state.errorInfo.componentStack}
                </Typography>
              </Box>
            )}
            <Button
              variant="contained"
              color="primary"
              onClick={this.handleRetry}
              sx={{ mt: 3 }}
            >
              Try Again
            </Button>
          </Paper>
        </Box>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary; 