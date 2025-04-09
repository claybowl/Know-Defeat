import { Typography, Button, Box, Container, Paper, Grid } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import MainLayout from '../components/layout/MainLayout';

export default function Home() {
  const navigate = useNavigate();

  return (
    <MainLayout>
      <Paper 
        elevation={0}
        sx={{ 
          py: 8, 
          px: 4, 
          mt: 4, 
          mb: 6, 
          borderRadius: 2, 
          bgcolor: 'primary.main', 
          color: 'white' 
        }}
      >
        <Container maxWidth="lg">
          <Typography
            component="h1"
            variant="h2"
            align="center"
            gutterBottom
          >
            Know Defeat Trading System
          </Typography>
          <Typography variant="h5" align="center" paragraph>
            Advanced algorithmic trading platform with autonomous agent intelligence
          </Typography>
          <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
            <Button 
              variant="contained" 
              sx={{ mx: 1, bgcolor: 'white', color: 'primary.main' }}
              onClick={() => navigate('/dashboard')}
            >
              Dashboard
            </Button>
            <Button 
              variant="outlined" 
              sx={{ mx: 1, borderColor: 'white', color: 'white' }}
              onClick={() => navigate('/bots')}
            >
              View Bots
            </Button>
          </Box>
        </Container>
      </Paper>

      <Container maxWidth="lg">
        <Grid container spacing={4}>
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Typography variant="h5" gutterBottom>
                Real-time Trading
              </Typography>
              <Typography paragraph>
                Monitor and manage active trades across multiple algorithms and securities. The system combines high-frequency trading capabilities with sophisticated pattern recognition.
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Typography variant="h5" gutterBottom>
                Autonomous Agents
              </Typography>
              <Typography paragraph>
                Our autonomous trading agents collaborate and compete to discover and exploit market opportunities while maintaining robust risk management.
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Typography variant="h5" gutterBottom>
                Performance Analytics
              </Typography>
              <Typography paragraph>
                Comprehensive performance tracking and analysis tools help you understand and optimize your trading strategies with detailed metrics and visualizations.
              </Typography>
            </Paper>
          </Grid>
        </Grid>

        <Box sx={{ mt: 6, mb: 4 }}>
          <Typography variant="h4" gutterBottom>
            System Capabilities
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1">Interactive Broker API Integration</Typography>
                  <Typography variant="body2">Real-time market data with advanced order routing.</Typography>
                </Paper>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1">Multiple Algorithm Support</Typography>
                  <Typography variant="body2">Breakout, mean reversion, momentum, and more.</Typography>
                </Paper>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1">Bot Ranking System</Typography>
                  <Typography variant="body2">Dynamic performance-based allocation of capital.</Typography>
                </Paper>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1">Real-time Analytics</Typography>
                  <Typography variant="body2">Constantly updated performance metrics and visualizations.</Typography>
                </Paper>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1">Risk Management</Typography>
                  <Typography variant="body2">Sophisticated stop loss and position sizing algorithms.</Typography>
                </Paper>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1">Machine Learning Integration</Typography>
                  <Typography variant="body2">AI-powered strategy optimization and adaptation.</Typography>
                </Paper>
              </Box>
            </Grid>
          </Grid>
        </Box>
      </Container>
    </MainLayout>
  );
}