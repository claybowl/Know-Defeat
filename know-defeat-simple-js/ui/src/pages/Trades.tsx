import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Typography,
  Box,
  Tabs,
  Tab,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  TextField,
  InputAdornment,
  CircularProgress,
  Alert,
  Grid
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import { getAllTrades, getOpenTrades } from '../api';
import MainLayout from '../components/layout/MainLayout';
import { Trade } from '../types';

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

export default function Trades() {
  const [tabValue, setTabValue] = useState(0);
  const [search, setSearch] = useState('');

  // Fetch trades data (active or all based on tab)
  const allTradesQuery = useQuery({
    queryKey: ['trades', 'all'],
    queryFn: () => getAllTrades(500),
    enabled: tabValue === 1
  });

  const openTradesQuery = useQuery({
    queryKey: ['trades', 'open'],
    queryFn: getOpenTrades
  });

  // Set current data based on selected tab
  const { data, isLoading, error } = tabValue === 0 ? openTradesQuery : allTradesQuery;

  // Calculate statistics for the current view
  const calculateStats = (trades: Trade[]) => {
    if (!trades || trades.length === 0) {
      return {
        totalTrades: 0,
        openTrades: 0,
        closedTrades: 0,
        totalPnl: 0,
        winRate: 0,
        avgTradeSize: 0
      };
    }

    const openTrades = trades.filter(trade => trade.trade_status === 'open').length;
    const closedTrades = trades.filter(trade => trade.trade_status === 'closed').length;
    const profitableTrades = trades.filter(trade => trade.pnl !== undefined && trade.pnl > 0).length;
    const totalPnl = trades.reduce((sum, trade) => sum + (trade.pnl || 0), 0);
    const avgTradeSize = trades.reduce((sum, trade) => sum + trade.trade_size, 0) / trades.length;
    const winRate = closedTrades > 0 ? (profitableTrades / closedTrades) : 0;

    return {
      totalTrades: trades.length,
      openTrades,
      closedTrades,
      totalPnl,
      winRate,
      avgTradeSize
    };
  };

  // Filter trades based on search
  const filteredTrades = data?.filter(trade => 
    (trade.ticker?.toLowerCase().includes(search.toLowerCase()) ||
    trade.bot_name?.toLowerCase().includes(search.toLowerCase()) ||
    trade.trade_id.toString().includes(search) ||
    trade.bot_id.toString().includes(search))
  ) || [];

  const stats = calculateStats(filteredTrades);

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
        <Alert severity="error">Error loading trades data</Alert>
      </MainLayout>
    );
  }

  return (
    <MainLayout>
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4">Trades</Typography>
        <TextField
          variant="outlined"
          placeholder="Search trades..."
          size="small"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
        />
      </Box>

      <Box sx={{ mb: 4 }}>
        <Tabs 
          value={tabValue} 
          onChange={(_, newValue) => setTabValue(newValue)}
          aria-label="trade tabs"
          sx={{ mb: 2 }}
        >
          <Tab label="Active Trades" />
          <Tab label="All Trades" />
        </Tabs>

        <Grid container spacing={3}>
          <Grid xs={12} sm={6} md={2}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="h6">{stats.totalTrades}</Typography>
              <Typography variant="body2" color="textSecondary">Total Trades</Typography>
            </Paper>
          </Grid>
          <Grid xs={12} sm={6} md={2}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="h6">{stats.openTrades}</Typography>
              <Typography variant="body2" color="textSecondary">Open Trades</Typography>
            </Paper>
          </Grid>
          <Grid xs={12} sm={6} md={2}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="h6">{stats.closedTrades}</Typography>
              <Typography variant="body2" color="textSecondary">Closed Trades</Typography>
            </Paper>
          </Grid>
          <Grid xs={12} sm={6} md={2}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <Typography 
                variant="h6" 
                color={stats.totalPnl >= 0 ? 'success.main' : 'error.main'}
              >
                {formatCurrency(stats.totalPnl)}
              </Typography>
              <Typography variant="body2" color="textSecondary">Total P&L</Typography>
            </Paper>
          </Grid>
          <Grid xs={12} sm={6} md={2}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="h6">{(stats.winRate * 100).toFixed(1)}%</Typography>
              <Typography variant="body2" color="textSecondary">Win Rate</Typography>
            </Paper>
          </Grid>
          <Grid xs={12} sm={6} md={2}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="h6">{formatCurrency(stats.avgTradeSize)}</Typography>
              <Typography variant="body2" color="textSecondary">Avg Size</Typography>
            </Paper>
          </Grid>
        </Grid>
      </Box>

      <TableContainer component={Paper}>
        <Table sx={{ minWidth: 650 }}>
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>Bot</TableCell>
              <TableCell>Ticker</TableCell>
              <TableCell>Direction</TableCell>
              <TableCell>Entry Price</TableCell>
              <TableCell>Exit Price</TableCell>
              <TableCell>Entry Time</TableCell>
              <TableCell>Exit Time</TableCell>
              <TableCell>Status</TableCell>
              <TableCell align="right">P&L</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredTrades.map((trade) => (
              <TableRow
                key={trade.trade_id}
                sx={{ '&:hover': { backgroundColor: 'rgba(0, 0, 0, 0.04)' } }}
              >
                <TableCell>{trade.trade_id}</TableCell>
                <TableCell>{trade.bot_name || `Bot ${trade.bot_id}`}</TableCell>
                <TableCell>{trade.ticker}</TableCell>
                <TableCell>{trade.trade_direction}</TableCell>
                <TableCell>${parseFloat(trade.entry_price.toString()).toFixed(2)}</TableCell>
                <TableCell>
                  {trade.exit_price 
                    ? `$${parseFloat(trade.exit_price.toString()).toFixed(2)}` 
                    : '-'
                  }
                </TableCell>
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
                <TableCell align="right" sx={{ 
                  color: trade.pnl 
                    ? trade.pnl > 0 
                      ? 'success.main' 
                      : 'error.main'
                    : 'inherit' 
                }}>
                  {trade.pnl ? formatCurrency(trade.pnl) : '-'}
                </TableCell>
              </TableRow>
            ))}
            {filteredTrades.length === 0 && (
              <TableRow>
                <TableCell colSpan={10} align="center">
                  No trades found matching your search criteria
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </MainLayout>
  );
}