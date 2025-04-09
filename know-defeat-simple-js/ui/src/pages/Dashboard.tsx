import { useQuery } from '@tanstack/react-query';
import { 
  Typography, 
  Grid, 
  Card, 
  CardContent, 
  Box, 
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  CircularProgress,
  Alert
} from '@mui/material';
import { 
  PieChart, 
  Pie, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  Cell 
} from 'recharts';
import { getDashboardData } from '../api';
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
  return (value * 100).toFixed(2) + '%';
};

// Pie chart colors
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

export default function Dashboard() {
  // Fetch dashboard data
  const { data, isLoading, error } = useQuery({
    queryKey: ['dashboardData'],
    queryFn: getDashboardData
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
        <Alert severity="error">Error loading dashboard data</Alert>
      </MainLayout>
    );
  }

  // Create allocation data for the pie chart
  const allocationData = data?.topBots.map((bot) => ({
    name: `Bot ${bot.bot_id}`,
    value: parseFloat(bot.total_pnl.toString())
  })) || [];

  // Sample data for the activity chart (would be replaced with real data)
  const activityData = [
    { date: '1/3', trades: 4, pnl: 124 },
    { date: '2/3', trades: 7, pnl: -56 },
    { date: '3/3', trades: 2, pnl: 78 },
    { date: '4/3', trades: 5, pnl: 143 },
    { date: '5/3', trades: 3, pnl: -89 },
    { date: '6/3', trades: 6, pnl: 176 },
    { date: '7/3', trades: 8, pnl: 209 },
    { date: '8/3', trades: 4, pnl: 123 },
    { date: '9/3', trades: 3, pnl: -45 },
    { date: '10/3', trades: 6, pnl: 56 },
  ];

  return (
    <MainLayout>
      <Typography variant="h4" sx={{ mb: 4 }}>
        Trading Dashboard
      </Typography>

      {/* Stats Overview */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" color="textSecondary">
                Total Bots
              </Typography>
              <Typography variant="h4">
                {data?.summary.totalBots}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {data?.summary.activeBots} active
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" color="textSecondary">
                Open Trades
              </Typography>
              <Typography variant="h4">
                {data?.summary.totalOpenTrades}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Across {data?.openTrades.length} bots
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" color="textSecondary">
                Total P&L
              </Typography>
              <Typography variant="h4" color={(data?.summary.totalPnl || 0) >= 0 ? 'success.main' : 'error.main'}>
                {formatCurrency(data?.summary.totalPnl || 0)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                From all closed trades
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" color="textSecondary">
                Avg Win Rate
              </Typography>
              <Typography variant="h4">
                {formatPercent(data?.summary.avgWinRate || 0)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                System-wide average
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Trade Activity
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={activityData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="trades"
                    stroke="#8884d8"
                    activeDot={{ r: 8 }}
                  />
                  <Line yAxisId="right" type="monotone" dataKey="pnl" stroke="#82ca9d" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Top Bot Allocation
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={allocationData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                  >
                    {allocationData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => formatCurrency(Number(value))} />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tables */}
      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Trades
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Trade ID</TableCell>
                      <TableCell>Bot</TableCell>
                      <TableCell>Ticker</TableCell>
                      <TableCell>Direction</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell align="right">P&L</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {data?.recentTrades.map((trade) => (
                      <TableRow key={trade.trade_id}>
                        <TableCell>{trade.trade_id}</TableCell>
                        <TableCell>{trade.bot_name || `Bot ${trade.bot_id}`}</TableCell>
                        <TableCell>{trade.ticker}</TableCell>
                        <TableCell>{trade.trade_direction}</TableCell>
                        <TableCell>
                          <Chip
                            size="small"
                            label={trade.trade_status}
                            color={
                              trade.trade_status === 'open'
                                ? 'primary'
                                : trade.trade_status === 'closed' && (trade.pnl || 0) > 0
                                ? 'success'
                                : 'error'
                            }
                          />
                        </TableCell>
                        <TableCell align="right" sx={{ color: (trade.pnl || 0) > 0 ? 'success.main' : 'error.main' }}>
                          {trade.pnl ? formatCurrency(trade.pnl) : '-'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Active Trades
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Ticker</TableCell>
                      <TableCell>Direction</TableCell>
                      <TableCell align="right">Entry Price</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {data?.openTrades.map((trade) => (
                      <TableRow key={trade.trade_id}>
                        <TableCell>{trade.ticker}</TableCell>
                        <TableCell>{trade.trade_direction}</TableCell>
                        <TableCell align="right">${parseFloat(trade.entry_price.toString()).toFixed(2)}</TableCell>
                      </TableRow>
                    ))}
                    {data?.openTrades.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={3} align="center">No active trades</TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </MainLayout>
  );
}