import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
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
  TableSortLabel,
  Chip,
  Button,
  TextField,
  InputAdornment,
  CircularProgress,
  Alert,
  Grid,
  Card,
  CardContent
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import { getBotMetrics } from '../api';
import MainLayout from '../components/layout/MainLayout';
import { BotMetrics } from '../types';

// Format currency helper
const formatCurrency = (value: number | undefined) => {
  if (value === undefined) return '-';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  }).format(value);
};

// Format percent helper
const formatPercent = (value: number | undefined) => {
  if (value === undefined) return '-';
  return (value * 100).toFixed(2) + '%';
};

type SortDirection = 'asc' | 'desc';
type SortField = 'bot_id' | 'total_trades' | 'win_rate' | 'total_pnl' | 'profit_factor' | 'sharpe_ratio' | 'rank_score';

export default function Metrics() {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [sortField, setSortField] = useState<SortField>('rank_score');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  // Fetch metrics data
  const { data, isLoading, error } = useQuery({
    queryKey: ['metrics'],
    queryFn: getBotMetrics
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
        <Alert severity="error">Error loading metrics data</Alert>
      </MainLayout>
    );
  }

  // Filter metrics based on search
  const filteredMetrics = data?.filter(metric => 
    metric.bot_id.toString().includes(search)
  ) || [];

  // Sort metrics
  const sortedMetrics = [...filteredMetrics].sort((a, b) => {
    if (a[sortField] === undefined || b[sortField] === undefined) return 0;
    
    const aValue = typeof a[sortField] === 'string'
      ? parseFloat(a[sortField] as string)
      : a[sortField] as number;
    
    const bValue = typeof b[sortField] === 'string'
      ? parseFloat(b[sortField] as string)
      : b[sortField] as number;
    
    return sortDirection === 'asc'
      ? aValue - bValue
      : bValue - aValue;
  });

  // Calculate system-wide metrics
  const calculateSystemMetrics = (metrics: BotMetrics[]) => {
    if (!metrics || metrics.length === 0) {
      return {
        totalBots: 0,
        activeBots: 0,
        totalTrades: 0,
        totalPnl: 0,
        avgWinRate: 0,
        avgProfitFactor: 0,
        avgSharpeRatio: 0
      };
    }

    const botsWithTrades = metrics.filter(bot => bot.total_trades > 0);
    const totalTrades = metrics.reduce((sum, bot) => sum + bot.total_trades, 0);
    const totalPnl = metrics.reduce((sum, bot) => sum + bot.total_pnl, 0);
    const avgWinRate = botsWithTrades.length > 0 
      ? botsWithTrades.reduce((sum, bot) => sum + bot.win_rate, 0) / botsWithTrades.length
      : 0;
    const avgProfitFactor = botsWithTrades.length > 0 
      ? botsWithTrades.reduce((sum, bot) => sum + bot.profit_factor, 0) / botsWithTrades.length
      : 0;
    const avgSharpeRatio = botsWithTrades.length > 0 
      ? botsWithTrades.reduce((sum, bot) => sum + bot.sharpe_ratio, 0) / botsWithTrades.length
      : 0;

    return {
      totalBots: metrics.length,
      activeBots: botsWithTrades.length,
      totalTrades,
      totalPnl,
      avgWinRate,
      avgProfitFactor,
      avgSharpeRatio
    };
  };

  const systemMetrics = calculateSystemMetrics(data || []);

  // Handle sort change
  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  return (
    <MainLayout>
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4">Bot Metrics</Typography>
        <TextField
          variant="outlined"
          placeholder="Search by Bot ID..."
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

      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>System Overview</Typography>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">Total Bots:</Typography>
                <Typography variant="body1">{systemMetrics.totalBots}</Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">Active Bots:</Typography>
                <Typography variant="body1">{systemMetrics.activeBots}</Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">Total Trades:</Typography>
                <Typography variant="body1">{systemMetrics.totalTrades}</Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Performance</Typography>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">Total P&L:</Typography>
                <Typography 
                  variant="body1" 
                  color={systemMetrics.totalPnl >= 0 ? 'success.main' : 'error.main'}
                >
                  {formatCurrency(systemMetrics.totalPnl)}
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">Avg Win Rate:</Typography>
                <Typography variant="body1">{formatPercent(systemMetrics.avgWinRate)}</Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">Avg P/L per Bot:</Typography>
                <Typography 
                  variant="body1" 
                  color={systemMetrics.totalPnl / systemMetrics.activeBots >= 0 ? 'success.main' : 'error.main'}
                >
                  {formatCurrency(systemMetrics.totalPnl / systemMetrics.activeBots)}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Risk Metrics</Typography>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">Avg Profit Factor:</Typography>
                <Typography variant="body1">{systemMetrics.avgProfitFactor.toFixed(2)}</Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">Avg Sharpe Ratio:</Typography>
                <Typography variant="body1">{systemMetrics.avgSharpeRatio.toFixed(2)}</Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">Best Bot Rank Score:</Typography>
                <Typography variant="body1">
                  {data && data.length > 0 
                    ? Math.max(...data.map(bot => bot.rank_score)).toFixed(2)
                    : '-'
                  }
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <TableContainer component={Paper}>
        <Table sx={{ minWidth: 650 }}>
          <TableHead>
            <TableRow>
              <TableCell>
                <TableSortLabel
                  active={sortField === 'bot_id'}
                  direction={sortField === 'bot_id' ? sortDirection : 'asc'}
                  onClick={() => handleSort('bot_id')}
                >
                  Bot ID
                </TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortField === 'total_trades'}
                  direction={sortField === 'total_trades' ? sortDirection : 'asc'}
                  onClick={() => handleSort('total_trades')}
                >
                  Trades
                </TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortField === 'win_rate'}
                  direction={sortField === 'win_rate' ? sortDirection : 'asc'}
                  onClick={() => handleSort('win_rate')}
                >
                  Win Rate
                </TableSortLabel>
              </TableCell>
              <TableCell>Wins/Losses</TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortField === 'total_pnl'}
                  direction={sortField === 'total_pnl' ? sortDirection : 'asc'}
                  onClick={() => handleSort('total_pnl')}
                >
                  Total P&L
                </TableSortLabel>
              </TableCell>
              <TableCell>Avg P&L/Trade</TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortField === 'profit_factor'}
                  direction={sortField === 'profit_factor' ? sortDirection : 'asc'}
                  onClick={() => handleSort('profit_factor')}
                >
                  Profit Factor
                </TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortField === 'sharpe_ratio'}
                  direction={sortField === 'sharpe_ratio' ? sortDirection : 'asc'}
                  onClick={() => handleSort('sharpe_ratio')}
                >
                  Sharpe Ratio
                </TableSortLabel>
              </TableCell>
              <TableCell>Max Drawdown</TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortField === 'rank_score'}
                  direction={sortField === 'rank_score' ? sortDirection : 'asc'}
                  onClick={() => handleSort('rank_score')}
                >
                  Rank Score
                </TableSortLabel>
              </TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {sortedMetrics.map((metric) => (
              <TableRow
                key={metric.bot_id}
                sx={{ '&:hover': { backgroundColor: 'rgba(0, 0, 0, 0.04)' } }}
              >
                <TableCell>{metric.bot_id}</TableCell>
                <TableCell>{metric.total_trades}</TableCell>
                <TableCell>{formatPercent(metric.win_rate)}</TableCell>
                <TableCell>{`${metric.winning_trades}/${metric.losing_trades}`}</TableCell>
                <TableCell sx={{ color: metric.total_pnl >= 0 ? 'success.main' : 'error.main' }}>
                  {formatCurrency(metric.total_pnl)}
                </TableCell>
                <TableCell sx={{ color: metric.average_pnl_per_trade >= 0 ? 'success.main' : 'error.main' }}>
                  {formatCurrency(metric.average_pnl_per_trade)}
                </TableCell>
                <TableCell>{metric.profit_factor.toFixed(2)}</TableCell>
                <TableCell>{metric.sharpe_ratio.toFixed(2)}</TableCell>
                <TableCell sx={{ color: 'error.main' }}>
                  {formatCurrency(metric.max_drawdown)}
                </TableCell>
                <TableCell>
                  <Chip 
                    size="small"
                    label={metric.rank_score.toFixed(2)} 
                    color={
                      metric.rank_score > 0.8 ? 'success' : 
                      metric.rank_score > 0.5 ? 'primary' : 
                      'default'
                    } 
                  />
                </TableCell>
                <TableCell align="right">
                  <Button 
                    variant="outlined"
                    size="small"
                    onClick={() => navigate(`/bots/${metric.bot_id}`)}
                  >
                    View Bot
                  </Button>
                </TableCell>
              </TableRow>
            ))}
            {sortedMetrics.length === 0 && (
              <TableRow>
                <TableCell colSpan={11} align="center">
                  No bot metrics found matching your search criteria
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </MainLayout>
  );
}