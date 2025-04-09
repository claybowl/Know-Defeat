import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  Typography,
  Box,
  Grid,
  Paper,
  Chip,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Alert,
  List,
  ListItem,
  ListItemText
} from '@mui/material';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { getBotById } from '../api';
import MainLayout from '../components/layout/MainLayout';

// Format currency helper
const formatCurrency = (value: number | undefined) => {
  if (value === undefined) return '-';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  }).format(value);
};

// Format date helper
const formatDate = (dateString: string | undefined) => {
  if (!dateString) return '-';
  return new Date(dateString).toLocaleString();
};

// Format percent helper
const formatPercent = (value: number | undefined) => {
  if (value === undefined) return '-';
  return (value * 100).toFixed(2) + '%';
};

// Colors for charts
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

export default function BotDetail() {
  const { id } = useParams<{ id: string }>();

  // Fetch bot data
  const { data, isLoading, error } = useQuery({
    queryKey: ['bot', id],
    queryFn: () => getBotById(id!)
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
        <Alert severity="error">Error loading bot data</Alert>
      </MainLayout>
    );
  }

  // Prepare trade data for chart
  const tradeData = data?.trades
    .filter(trade => trade.trade_status === 'closed')
    .map((trade, index) => ({
      id: index,
      tradeId: trade.trade_id,
      pnl: trade.pnl || 0,
      entryDate: new Date(trade.entry_time).toLocaleDateString(),
      exitDate: trade.exit_time ? new Date(trade.exit_time).toLocaleDateString() : null
    }));

  // Prepare win/loss data for pie chart
  const winLossData = [
    { name: 'Winning Trades', value: data?.metrics?.winning_trades || 0 },
    { name: 'Losing Trades', value: data?.metrics?.losing_trades || 0 }
  ];
  
  // We're not using radar chart for now
  // Keeping this simple for compatibility
  

  return (
    <MainLayout>
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
          <Typography variant="h4">{data?.name}</Typography>
          <Chip 
            label={data?.is_active ? 'Active' : 'Inactive'} 
            color={data?.is_active ? 'success' : 'default'} 
          />
        </Box>
        <Typography variant="subtitle1" color="text.secondary" gutterBottom>
          {data?.description}
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Bot Details */}
        <Grid xs={12} md={4}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>Bot Details</Typography>
            <Divider sx={{ mb: 2 }} />
            <List dense>
              <ListItem>
                <ListItemText 
                  primary="Bot ID" 
                  secondary={data?.bot_id} 
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Ticker" 
                  secondary={data?.ticker} 
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Algorithm Type" 
                  secondary={data?.algorithm_type} 
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Trade Direction" 
                  secondary={data?.trade_direction} 
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Position Size" 
                  secondary={formatCurrency(data?.position_size)} 
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Trailing Stop %" 
                  secondary={formatPercent(data?.trailing_stop_pct)} 
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Version" 
                  secondary={data?.version} 
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Created" 
                  secondary={formatDate(data?.created_at)} 
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Last Updated" 
                  secondary={formatDate(data?.last_updated)} 
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>

        {/* Performance Metrics */}
        <Grid xs={12} md={4}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>Performance Metrics</Typography>
            <Divider sx={{ mb: 2 }} />
            {data?.metrics ? (
              <List dense>
                <ListItem>
                  <ListItemText 
                    primary="Total Trades" 
                    secondary={data.metrics.total_trades} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Win Rate" 
                    secondary={formatPercent(data.metrics.win_rate)} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Total P&L" 
                    secondary={formatCurrency(data.metrics.total_pnl)} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Average P&L per Trade" 
                    secondary={formatCurrency(data.metrics.average_pnl_per_trade)} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Average Win Amount" 
                    secondary={formatCurrency(data.metrics.average_win_amount)} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Average Loss Amount" 
                    secondary={formatCurrency(data.metrics.average_loss_amount)} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Profit Factor" 
                    secondary={data.metrics.profit_factor.toFixed(2)} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Maximum Drawdown" 
                    secondary={formatCurrency(data.metrics.max_drawdown)} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Sharpe Ratio" 
                    secondary={data.metrics.sharpe_ratio.toFixed(2)} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Risk/Reward Ratio" 
                    secondary={data.metrics.risk_reward_ratio.toFixed(2)} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Expectancy" 
                    secondary={data.metrics.expectancy.toFixed(3)} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Rank Score" 
                    secondary={data.metrics.rank_score.toFixed(3)} 
                  />
                </ListItem>
              </List>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No metrics available for this bot yet
              </Typography>
            )}
          </Paper>
        </Grid>

        {/* Algorithm Parameters */}
        <Grid xs={12} md={4}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>Algorithm Parameters</Typography>
            <Divider sx={{ mb: 2 }} />
            {data?.parameters ? (
              <List dense>
                {Object.entries(data.parameters).map(([key, value]) => (
                  <ListItem key={key}>
                    <ListItemText 
                      primary={key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} 
                      secondary={value} 
                    />
                  </ListItem>
                ))}
              </List>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No parameters available for this bot
              </Typography>
            )}
          </Paper>
        </Grid>

        {/* Performance Charts */}
        <Grid xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Trade Performance</Typography>
            <Divider sx={{ mb: 2 }} />
            {tradeData && tradeData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={tradeData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="entryDate" />
                  <YAxis />
                  <Tooltip formatter={(value) => formatCurrency(Number(value))} />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="pnl" 
                    stroke="#8884d8" 
                    activeDot={{ r: 8 }} 
                    name="P&L"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No trade performance data available
              </Typography>
            )}
          </Paper>
        </Grid>

        {/* Win/Loss & Performance Radar */}
        <Grid xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Win/Loss Ratio</Typography>
            <Divider sx={{ mb: 2 }} />
            {data?.metrics && (data.metrics.winning_trades > 0 || data.metrics.losing_trades > 0) ? (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={winLossData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                  >
                    {winLossData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No win/loss data available
              </Typography>
            )}
          </Paper>
        </Grid>

        {/* Trade History */}
        <Grid xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Trade History</Typography>
            <Divider sx={{ mb: 2 }} />
            {data?.trades && data.trades.length > 0 ? (
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>ID</TableCell>
                      <TableCell>Ticker</TableCell>
                      <TableCell>Direction</TableCell>
                      <TableCell>Entry Price</TableCell>
                      <TableCell>Exit Price</TableCell>
                      <TableCell>Size</TableCell>
                      <TableCell>Entry Time</TableCell>
                      <TableCell>Exit Time</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Exit Reason</TableCell>
                      <TableCell align="right">P&L</TableCell>
                      <TableCell align="right">P&L %</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {data.trades.map((trade) => (
                      <TableRow
                        key={trade.trade_id}
                        sx={{ '&:hover': { backgroundColor: 'rgba(0, 0, 0, 0.04)' } }}
                      >
                        <TableCell>{trade.trade_id}</TableCell>
                        <TableCell>{trade.ticker}</TableCell>
                        <TableCell>{trade.trade_direction}</TableCell>
                        <TableCell>${parseFloat(trade.entry_price.toString()).toFixed(2)}</TableCell>
                        <TableCell>
                          {trade.exit_price 
                            ? `$${parseFloat(trade.exit_price.toString()).toFixed(2)}` 
                            : '-'
                          }
                        </TableCell>
                        <TableCell>{formatCurrency(trade.trade_size)}</TableCell>
                        <TableCell>{formatDate(trade.entry_time)}</TableCell>
                        <TableCell>{formatDate(trade.exit_time)}</TableCell>
                        <TableCell>
                          <Chip 
                            size="small"
                            label={trade.trade_status} 
                            color={
                              trade.trade_status === 'open'
                                ? 'primary'
                                : trade.pnl && trade.pnl > 0
                                ? 'success'
                                : 'error'
                            } 
                          />
                        </TableCell>
                        <TableCell>{trade.exit_reason || '-'}</TableCell>
                        <TableCell align="right" sx={{ 
                          color: trade.pnl 
                            ? trade.pnl > 0 
                              ? 'success.main' 
                              : 'error.main'
                            : 'inherit' 
                        }}>
                          {trade.pnl ? formatCurrency(trade.pnl) : '-'}
                        </TableCell>
                        <TableCell align="right" sx={{ 
                          color: trade.pnl_percent 
                            ? trade.pnl_percent > 0 
                              ? 'success.main' 
                              : 'error.main'
                            : 'inherit' 
                        }}>
                          {trade.pnl_percent ? formatPercent(trade.pnl_percent) : '-'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No trade history available
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>
    </MainLayout>
  );
}