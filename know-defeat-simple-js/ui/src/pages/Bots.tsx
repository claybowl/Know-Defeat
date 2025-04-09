import { useState } from 'react';
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
  TextField,
  InputAdornment,
  CircularProgress,
  Alert,
  Grid
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import { getAllBots } from '../api';
import MainLayout from '../components/layout/MainLayout';

export default function Bots() {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');

  // Fetch bots data
  const { data, isLoading, error } = useQuery({
    queryKey: ['bots'],
    queryFn: getAllBots
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
        <Alert severity="error">Error loading bots data</Alert>
      </MainLayout>
    );
  }

  // Filter bots based on search
  const filteredBots = data?.filter(bot => 
    bot.name.toLowerCase().includes(search.toLowerCase()) ||
    bot.ticker.toLowerCase().includes(search.toLowerCase()) ||
    bot.algorithm_type.toLowerCase().includes(search.toLowerCase()) ||
    bot.bot_id.toString().includes(search)
  ) || [];

  return (
    <MainLayout>
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4">Trading Bots</Typography>
        <TextField
          variant="outlined"
          placeholder="Search bots..."
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
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 3, bgcolor: 'primary.main', color: 'white' }}>
            <Typography variant="h4">{data?.length || 0}</Typography>
            <Typography variant="subtitle2">Total Bots</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 3, bgcolor: 'success.main', color: 'white' }}>
            <Typography variant="h4">{data?.filter(bot => bot.is_active).length || 0}</Typography>
            <Typography variant="subtitle2">Active Bots</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 3, bgcolor: 'info.main', color: 'white' }}>
            <Typography variant="h4">{new Set(data?.map(bot => bot.ticker)).size || 0}</Typography>
            <Typography variant="subtitle2">Unique Tickers</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 3, bgcolor: 'warning.main', color: 'white' }}>
            <Typography variant="h4">{new Set(data?.map(bot => bot.algorithm_type)).size || 0}</Typography>
            <Typography variant="subtitle2">Algorithm Types</Typography>
          </Paper>
        </Grid>
      </Grid>

      <TableContainer component={Paper}>
        <Table sx={{ minWidth: 650 }}>
          <TableHead>
            <TableRow>
              <TableCell>Bot ID</TableCell>
              <TableCell>Name</TableCell>
              <TableCell>Ticker</TableCell>
              <TableCell>Algorithm</TableCell>
              <TableCell>Direction</TableCell>
              <TableCell>Status</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredBots.map((bot) => (
              <TableRow
                key={bot.bot_id}
                sx={{ '&:hover': { backgroundColor: 'rgba(0, 0, 0, 0.04)' } }}
              >
                <TableCell>{bot.bot_id}</TableCell>
                <TableCell>{bot.name}</TableCell>
                <TableCell>{bot.ticker}</TableCell>
                <TableCell>{bot.algorithm_type}</TableCell>
                <TableCell>{bot.trade_direction}</TableCell>
                <TableCell>
                  <Chip 
                    size="small"
                    label={bot.is_active ? 'Active' : 'Inactive'} 
                    color={bot.is_active ? 'success' : 'default'} 
                  />
                </TableCell>
                <TableCell align="right">
                  <Button 
                    variant="outlined"
                    size="small"
                    onClick={() => navigate(`/bots/${bot.bot_id}`)}
                  >
                    View Details
                  </Button>
                </TableCell>
              </TableRow>
            ))}
            {filteredBots.length === 0 && (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  No bots found matching your search criteria
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </MainLayout>
  );
}