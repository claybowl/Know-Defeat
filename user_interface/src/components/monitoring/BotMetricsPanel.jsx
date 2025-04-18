import React, { useState, useEffect } from 'react';
import {
  Box,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Heading,
  Text,
  Flex,
  Input,
  Select,
  Badge,
  Spinner,
  useColorModeValue,
  Button,
  IconButton,
  HStack,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper
} from '@chakra-ui/react';
import { SearchIcon, RepeatIcon } from '@chakra-ui/icons';

export const BotMetricsPanel = ({ wsData }) => {
  const [metrics, setMetrics] = useState([]);
  const [filteredMetrics, setFilteredMetrics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState({
    search: '',
    ticker: '',
    algo_id: '',
    min_win_rate: 0,
    max_drawdown: 100
  });
  
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const highlightColor = useColorModeValue('blue.50', 'blue.900');
  
  // Fetch initial metrics data
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/metrics/live');
        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}`);
        }
        const data = await response.json();
        setMetrics(data.metrics || []);
        setFilteredMetrics(data.metrics || []);
        setError(null);
      } catch (err) {
        setError(`Failed to fetch metrics: ${err.message}`);
        console.error('Error fetching metrics:', err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchMetrics();
  }, []);
  
  // Update with WebSocket data when available
  useEffect(() => {
    if (wsData && wsData.data) {
      // Update metrics with new data
      const newMetricsData = Array.isArray(wsData.data) ? wsData.data : [wsData.data];
      
      setMetrics(prevMetrics => {
        const metricsMap = new Map(prevMetrics.map(m => [m.bot_id, m]));
        newMetricsData.forEach(newMetric => {
          // Ensure algo_id is present
          if (newMetric.algo_id === undefined) {
             console.warn("Received metric without algo_id:", newMetric);
          }
          metricsMap.set(newMetric.bot_id, newMetric);
        });
        return Array.from(metricsMap.values());
      });
    }
  }, [wsData]);
  
  // Apply filters when metrics or filter changes
  useEffect(() => {
    const applyFilters = () => {
      let result = [...metrics];
      
      // Text search across bot_id, ticker, and algo_id
      if (filter.search) {
        const searchLower = filter.search.toLowerCase();
        result = result.filter(m => 
          (m.bot_id?.toString().includes(searchLower)) || 
          (m.ticker?.toLowerCase().includes(searchLower)) ||
          (m.algo_id?.toString().includes(searchLower))
        );
      }
      
      // Filter by ticker
      if (filter.ticker) {
        result = result.filter(m => m.ticker === filter.ticker);
      }
      
      // Filter by algorithm ID
      if (filter.algo_id) {
        result = result.filter(m => m.algo_id === parseInt(filter.algo_id, 10));
      }
      
      // Filter by win rate
      if (filter.min_win_rate > 0) {
        result = result.filter(m => m.avg_win_rate >= filter.min_win_rate);
      }
      
      // Filter by max drawdown
      if (filter.max_drawdown < 100) {
        result = result.filter(m => m.max_drawdown <= filter.max_drawdown);
      }
      
      setFilteredMetrics(result);
    };
    
    applyFilters();
  }, [metrics, filter]);
  
  // Extract unique values for filters
  const uniqueTickers = [...new Set(metrics.map(m => m.ticker).filter(Boolean))];
  const uniqueAlgoIds = [...new Set(metrics.map(m => m.algo_id).filter(id => id !== null && id !== undefined))];
  
  // Handle filter changes
  const handleFilterChange = (field, value) => {
    setFilter(prev => ({ ...prev, [field]: value }));
  };
  
  // Format numeric values
  const formatNumber = (num, decimals = 2) => {
    if (num === null || num === undefined) return 'N/A';
    return parseFloat(num).toFixed(decimals);
  };
  
  // Get style for win rate
  const getWinRateStyle = (rate) => {
    if (rate >= 0.7) return "green.500";
    if (rate >= 0.5) return "blue.500";
    if (rate >= 0.3) return "orange.500";
    return "red.500";
  };
  
  // Get style for P&L
  const getPnlStyle = (pnl) => {
    if (pnl > 0) return "green.500";
    if (pnl < 0) return "red.500";
    return "gray.500";
  };
  
  // Refresh data manually
  const handleRefresh = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/metrics/live');
      const data = await response.json();
      setMetrics(data.metrics || []);
      setError(null);
    } catch (err) {
      setError(`Failed to refresh: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Box>
      <Flex justify="space-between" align="center" mb={4}>
        <Heading size="md">Bot Metrics</Heading>
        <HStack>
          {loading && <Spinner size="sm" />}
          <Text fontSize="sm">
            {filteredMetrics.length} of {metrics.length} bots
          </Text>
          <IconButton 
            icon={<RepeatIcon />} 
            onClick={handleRefresh} 
            size="sm" 
            aria-label="Refresh data" 
            isLoading={loading}
          />
        </HStack>
      </Flex>
      
      {/* Filters */}
      <Flex wrap="wrap" gap={4} mb={4} p={4} bg="gray.50" borderRadius="md">
        <Box flex="1" minW="200px">
          <Text fontSize="sm" mb={1}>Search</Text>
          <Flex>
            <Input
              placeholder="Search bot ID, ticker..."
              value={filter.search}
              onChange={e => handleFilterChange('search', e.target.value)}
              borderRightRadius={0}
            />
            <IconButton
              aria-label="Search"
              icon={<SearchIcon />}
              borderLeftRadius={0}
            />
          </Flex>
        </Box>
        
        <Box minW="150px">
          <Text fontSize="sm" mb={1}>Ticker</Text>
          <Select
            placeholder="All tickers"
            value={filter.ticker}
            onChange={e => handleFilterChange('ticker', e.target.value)}
          >
            {uniqueTickers.map(ticker => (
              <option key={ticker} value={ticker}>{ticker}</option>
            ))}
          </Select>
        </Box>
        
        <Box minW="180px">
          <Text fontSize="sm" mb={1}>Algorithm ID</Text>
          <Select
            placeholder="All algorithms"
            value={filter.algo_id}
            onChange={e => handleFilterChange('algo_id', e.target.value)}
          >
            {uniqueAlgoIds.map(id => (
              <option key={id} value={id}>{id}</option>
            ))}
          </Select>
        </Box>
        
        <Box minW="150px">
          <Text fontSize="sm" mb={1}>Min Win Rate (%)</Text>
          <NumberInput
            min={0}
            max={100}
            value={filter.min_win_rate * 100}
            onChange={val => handleFilterChange('min_win_rate', parseFloat(val) / 100)}
          >
            <NumberInputField />
            <NumberInputStepper>
              <NumberIncrementStepper />
              <NumberDecrementStepper />
            </NumberInputStepper>
          </NumberInput>
        </Box>
        
        <Box minW="150px">
          <Text fontSize="sm" mb={1}>Max Drawdown (%)</Text>
          <NumberInput
            min={0}
            max={100}
            value={filter.max_drawdown}
            onChange={val => handleFilterChange('max_drawdown', parseFloat(val))}
          >
            <NumberInputField />
            <NumberInputStepper>
              <NumberIncrementStepper />
              <NumberDecrementStepper />
            </NumberInputStepper>
          </NumberInput>
        </Box>
        
        <Box alignSelf="flex-end">
          <Button 
            colorScheme="gray" 
            variant="outline" 
            onClick={() => setFilter({
              search: '',
              ticker: '',
              algo_id: '',
              min_win_rate: 0,
              max_drawdown: 100
            })}
          >
            Clear Filters
          </Button>
        </Box>
      </Flex>
      
      {/* Error message */}
      {error && (
        <Box p={4} bg="red.100" color="red.800" borderRadius="md" mb={4}>
          <Text>{error}</Text>
        </Box>
      )}
      
      {/* Metrics table */}
      <Box overflowX="auto">
        <Table variant="simple" size="sm" borderWidth="1px" borderColor={borderColor}>
          <Thead bg="gray.50">
            <Tr>
              <Th>Bot ID</Th>
              <Th>Ticker</Th>
              <Th>Algo ID</Th>
              <Th isNumeric>Win Rate</Th>
              <Th isNumeric>P&L</Th>
              <Th isNumeric>Trades</Th>
              <Th isNumeric>Avg Profit</Th>
              <Th isNumeric>Max Drawdown</Th>
              <Th isNumeric>Rank Score</Th>
              <Th>Last Updated</Th>
            </Tr>
          </Thead>
          <Tbody>
            {loading && filteredMetrics.length === 0 ? (
              <Tr>
                <Td colSpan={10} textAlign="center" py={4}>
                  <Spinner size="sm" mr={2} />
                  Loading metrics...
                </Td>
              </Tr>
            ) : filteredMetrics.length === 0 ? (
              <Tr>
                <Td colSpan={10} textAlign="center" py={4}>
                  No metrics found matching the filter criteria.
                </Td>
              </Tr>
            ) : (
              filteredMetrics.map(metric => (
                <Tr 
                  key={metric.bot_id}
                  _hover={{ bg: highlightColor }}
                >
                  <Td fontWeight="bold">{metric.bot_id}</Td>
                  <Td>
                    <Badge colorScheme="blue">{metric.ticker}</Badge>
                  </Td>
                  <Td>{metric.algo_id}</Td>
                  <Td isNumeric color={getWinRateStyle(metric.avg_win_rate)}>
                    {formatNumber(metric.avg_win_rate * 100)}%
                  </Td>
                  <Td isNumeric color={getPnlStyle(metric.total_pnl)}>
                    ${formatNumber(metric.total_pnl)}
                  </Td>
                  <Td isNumeric>{metric.total_trades || 0}</Td>
                  <Td isNumeric>${formatNumber(metric.avg_profit_per_trade)}</Td>
                  <Td isNumeric color="red.500">{formatNumber(metric.max_drawdown)}%</Td>
                  <Td isNumeric>{formatNumber(metric.rank_score, 1)}</Td>
                  <Td fontSize="sm">
                    {metric.timestamp ? new Date(metric.timestamp).toLocaleString() : 'N/A'}
                  </Td>
                </Tr>
              ))
            )}
          </Tbody>
        </Table>
      </Box>
    </Box>
  );
}; 