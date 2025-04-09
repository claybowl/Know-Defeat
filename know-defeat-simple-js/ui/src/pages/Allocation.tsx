import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import {
  Typography,
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Button,
  CircularProgress,
  Alert,
  Grid,
  Card,
  CardContent,
  LinearProgress
} from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend
} from 'recharts';
import { getAllocationData } from '../api';
import MainLayout from '../components/layout/MainLayout';

// Format currency helper
const formatCurrency = (value: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  }).format(value);
};

// Format percent helper
const formatPercent = (value: number) => {
  return value.toFixed(2) + '%';
};

// Colors for the pie chart
const COLORS = [
  '#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8',
  '#82CA9D', '#8DD1E1', '#A4DE6C', '#D0ED57', '#FFC658'
];

// Custom tooltip for pie chart
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <Box
        component={Paper}
        elevation={3}
        sx={{ p: 2, backgroundColor: 'white' }}
      >
        <Typography variant="body2">{`${payload[0].name}`}</Typography>
        <Typography variant="body2" color="primary">{`Allocation: ${formatCurrency(payload[0].value)}`}</Typography>
        <Typography variant="body2">{`${payload[0].payload.allocation_percent.toFixed(1)}% of total`}</Typography>
      </Box>
    );
  }
  return null;
};

export default function Allocation() {
  const navigate = useNavigate();

  // Fetch allocation data
  const { data, isLoading, error } = useQuery({
    queryKey: ['allocation'],
    queryFn: getAllocationData
  });

  if (isLoading) {
    return (
      <MainLayout>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
          <CircularProgress />
        </Box>
      </MainLayout>
    );
  }

  if (error) {
    return (
      <MainLayout>
        <Alert severity="error">Error loading allocation data</Alert>
      </MainLayout>
    );
  }

  // Prepare data for pie chart
  const pieData = data?.allocations.map(allocation => ({
    name: `Bot ${allocation.bot_id} (${allocation.ticker})`,
    value: allocation.allocation,
    allocation_percent: allocation.allocation_percent,
    bot_id: allocation.bot_id
  })) || [];

  return (
    <MainLayout>
      <Typography variant="h4" sx={{ mb: 4 }}>
        Fund Allocation
      </Typography>

      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Total Allocation</Typography>
              <Typography variant="h3" color="primary" gutterBottom>
                {formatCurrency(data?.totalAllocation || 0)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Funds are allocated to the top {data?.allocations.length || 0} performing bots
                based on their ranking scores.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Allocation Strategy</Typography>
              <Typography variant="body1" paragraph>
                The current allocation strategy distributes funds equally among the top 10 ranked bots.
                Each bot in the top 10 receives {formatCurrency(2000)} for trading.
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Bots are ranked based on a composite score that considers win rate, profit factor,
                Sharpe ratio, and other performance metrics. Allocations are adjusted as bot rankings change.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        <Grid xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>Allocation Distribution</Typography>
            <Box sx={{ height: 400 }}>
              {pieData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      outerRadius={150}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {pieData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', mt: 10 }}>
                  No allocation data available
                </Typography>
              )}
            </Box>
          </Paper>
        </Grid>

        <Grid xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>Top Performers</Typography>
            {data?.allocations.map((bot, index) => (
              <Box key={bot.bot_id} sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="body2">
                    Bot {bot.bot_id} ({bot.ticker} - {bot.algorithm_type})
                  </Typography>
                  <Typography variant="body2">
                    <strong>{formatCurrency(bot.allocation)}</strong>
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Box sx={{ width: '100%', mr: 1 }}>
                    <LinearProgress 
                      variant="determinate" 
                      value={bot.allocation_percent} 
                      sx={{ 
                        height: 8, 
                        borderRadius: 5,
                        backgroundColor: 'rgba(0, 0, 0, 0.1)',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: COLORS[index % COLORS.length]
                        }
                      }}
                    />
                  </Box>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      {formatPercent(bot.allocation_percent)}
                    </Typography>
                  </Box>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                  <Typography variant="caption" color="text.secondary">
                    Rank Score: {bot.rank_score.toFixed(2)}
                  </Typography>
                  <Button 
                    size="small" 
                    onClick={() => navigate(`/bots/${bot.bot_id}`)}
                  >
                    Details
                  </Button>
                </Box>
              </Box>
            ))}
          </Paper>
        </Grid>

        <Grid xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Allocation Table</Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Rank</TableCell>
                    <TableCell>Bot ID</TableCell>
                    <TableCell>Name</TableCell>
                    <TableCell>Ticker</TableCell>
                    <TableCell>Algorithm</TableCell>
                    <TableCell>Rank Score</TableCell>
                    <TableCell>Allocation</TableCell>
                    <TableCell>Percentage</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {data?.allocations.map((allocation, index) => (
                    <TableRow key={allocation.bot_id}>
                      <TableCell>{index + 1}</TableCell>
                      <TableCell>{allocation.bot_id}</TableCell>
                      <TableCell>{allocation.name}</TableCell>
                      <TableCell>{allocation.ticker}</TableCell>
                      <TableCell>{allocation.algorithm_type}</TableCell>
                      <TableCell>
                        <Chip 
                          size="small"
                          label={allocation.rank_score.toFixed(2)} 
                          color={
                            allocation.rank_score > 0.8 ? 'success' : 
                            allocation.rank_score > 0.5 ? 'primary' : 
                            'default'
                          } 
                        />
                      </TableCell>
                      <TableCell>{formatCurrency(allocation.allocation)}</TableCell>
                      <TableCell>{formatPercent(allocation.allocation_percent)}</TableCell>
                      <TableCell align="right">
                        <Button 
                          variant="outlined"
                          size="small"
                          onClick={() => navigate(`/bots/${allocation.bot_id}`)}
                        >
                          View Bot
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                  {(!data?.allocations || data.allocations.length === 0) && (
                    <TableRow>
                      <TableCell colSpan={9} align="center">
                        No allocation data available
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>
    </MainLayout>
  );
}