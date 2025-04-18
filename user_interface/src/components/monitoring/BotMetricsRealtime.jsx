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
  NumberDecrementStepper,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  CloseButton
} from '@chakra-ui/react';
import { SearchIcon, RepeatIcon, WarningIcon, CheckCircleIcon } from '@chakra-ui/icons';
import useMetricsWebSocket from '../hooks/useMetricsWebSocket';

/**
 * Real-time Bot Metrics Panel using WebSocket for live updates
 */
export const BotMetricsRealtime = () => {
  // WebSocket connection for real-time metrics updates
  const {
    isConnected,
    metrics,
    error: wsError,
    connect,
    disconnect,
    reconnectAttempts
  } = useMetricsWebSocket();

  // Local state for filtering and UI
  const [filteredMetrics, setFilteredMetrics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState({
    search: '',
    ticker: '',
    algo_id: '',
    min_win_rate: 0,
    max_drawdown: 100
  });
  
  // Colors for light/dark mode
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const highlightColor = useColorModeValue('blue.50', 'blue.900');
  const successColor = useColorModeValue('green.500', 'green.300');
  const errorColor = useColorModeValue('red.500', 'red.300');
  
  // Initialize with WebSocket data when available
  useEffect(() => {
    if (metrics.length > 0) {
      setLoading(false);
    }
  }, [metrics]);
  
  // Apply filters when metrics or filter criteria change
  useEffect(() => {
    const applyFilters = () => {
      const filtered = metrics.filter(metric => {
        // Text search filter
        const searchTerm = filter.search.toLowerCase();
        const matchesSearch = 
          !searchTerm || 
          (metric.bot_id && metric.bot_id.toString().includes(searchTerm)) ||
          (metric.ticker && metric.ticker.toLowerCase().includes(searchTerm)) ||
          (metric.algorithm_type && metric.algorithm_type.toLowerCase().includes(searchTerm));
          
        // Ticker filter
        const matchesTicker = !filter.ticker || metric.ticker === filter.ticker;
        
        // Algorithm filter
        const matchesAlgo = !filter.algo_id || metric.algo_id === parseInt(filter.algo_id);
        
        // Win rate filter
        const winRate = parseFloat(metric.win_rate || 0);
        const matchesWinRate = winRate >= filter.min_win_rate / 100;
        
        // Max drawdown filter
        const drawdown = parseFloat(metric.max_drawdown || 0);
        const matchesDrawdown = drawdown <= filter.max_drawdown;
        
        return matchesSearch && matchesTicker && matchesAlgo && matchesWinRate && matchesDrawdown;
      });
      
      setFilteredMetrics(filtered);
    };
    
    applyFilters();
  }, [metrics, filter]);
  
  // Extract unique values for filter dropdowns
  const tickers = [...new Set(metrics.map(m => m.ticker).filter(Boolean))];
  const algoTypes = [...new Set(metrics.map(m => m.algorithm_type).filter(Boolean))];
  
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
  
  // Manually reconnect the WebSocket
  const handleReconnect = () => {
    connect();
  };
  
  return (
    <Box>
      <Flex justify="space-between" align="center" mb={4}>
        <Heading size="md">Bot Performance Metrics</Heading>
        <Flex align="center">
          <Badge 
            colorScheme={isConnected ? "green" : "red"} 
            mr={3}
            p={1}
            borderRadius="md"
          >
            {isConnected ? "Live Updates" : "Disconnected"}
          </Badge>
          <Button 
            size="sm" 
            leftIcon={<RepeatIcon />}
            onClick={handleReconnect}
            isDisabled={isConnected}
            colorScheme="blue"
          >
            {reconnectAttempts > 0 ? `Reconnect (${reconnectAttempts})` : "Reconnect"}
          </Button>
        </Flex>
      </Flex>
      
      {wsError && (
        <Alert status="error" mb={4} borderRadius="md">
          <AlertIcon />
          <AlertTitle mr={2}>Connection Error!</AlertTitle>
          <AlertDescription>{wsError}</AlertDescription>
          <CloseButton position="absolute" right="8px" top="8px" />
        </Alert>
      )}
      
      {/* Filter controls */}
      <Box p={4} borderWidth="1px" borderRadius="md" borderColor={borderColor} mb={4}>
        <Flex direction={{ base: "column", md: "row" }} gap={4} wrap="wrap">
          <Box flex="1" minW={{ base: "100%", md: "200px" }}>
            <Text fontSize="sm" mb={1}>Search</Text>
            <InputGroup>
              <Input
                placeholder="Search by bot ID, ticker, or algorithm..."
                value={filter.search}
                onChange={(e) => handleFilterChange('search', e.target.value)}
              />
            </InputGroup>
          </Box>
          
          <Box width={{ base: "100%", md: "150px" }}>
            <Text fontSize="sm" mb={1}>Ticker</Text>
            <Select
              placeholder="All tickers"
              value={filter.ticker}
              onChange={(e) => handleFilterChange('ticker', e.target.value)}
            >
              {tickers.map(ticker => (
                <option key={ticker} value={ticker}>{ticker}</option>
              ))}
            </Select>
          </Box>
          
          <Box width={{ base: "100%", md: "180px" }}>
            <Text fontSize="sm" mb={1}>Algorithm</Text>
            <Select
              placeholder="All algorithms"
              value={filter.algo_id}
              onChange={(e) => handleFilterChange('algo_id', e.target.value)}
            >
              {algoTypes.map(algo => (
                <option key={algo} value={algo}>{algo}</option>
              ))}
            </Select>
          </Box>
          
          <Box width={{ base: "100%", md: "150px" }}>
            <Text fontSize="sm" mb={1}>Min Win Rate (%)</Text>
            <NumberInput
              min={0}
              max={100}
              value={filter.min_win_rate}
              onChange={(valueString) => handleFilterChange('min_win_rate', parseFloat(valueString))}
            >
              <NumberInputField />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
          </Box>
          
          <Box width={{ base: "100%", md: "150px" }}>
            <Text fontSize="sm" mb={1}>Max Drawdown</Text>
            <NumberInput
              min={0}
              max={100}
              value={filter.max_drawdown}
              onChange={(valueString) => handleFilterChange('max_drawdown', parseFloat(valueString))}
            >
              <NumberInputField />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
          </Box>
        </Flex>
      </Box>
      
      {/* Results summary */}
      <Flex justify="space-between" align="center" mb={2}>
        <Text>
          Showing {filteredMetrics.length} of {metrics.length} bots
        </Text>
        <HStack>
          <Text fontSize="sm" color={isConnected ? successColor : errorColor}>
            {isConnected ? (
              <>
                <CheckCircleIcon mr={1} />
                Real-time updates active
              </>
            ) : (
              <>
                <WarningIcon mr={1} />
                Updates paused
              </>
            )}
          </Text>
        </HStack>
      </Flex>
      
      {/* Main metrics table */}
      {loading ? (
        <Flex justify="center" align="center" py={10}>
          <Spinner size="xl" />
          <Text ml={4}>Loading metrics data...</Text>
        </Flex>
      ) : filteredMetrics.length === 0 ? (
        <Box p={5} textAlign="center" borderWidth="1px" borderRadius="md" borderColor={borderColor}>
          <Text>No metrics match your filters</Text>
        </Box>
      ) : (
        <Box overflowX="auto">
          <Table variant="simple" size="sm">
            <Thead>
              <Tr>
                <Th>Bot ID</Th>
                <Th>Ticker</Th>
                <Th>Algorithm</Th>
                <Th>Win Rate</Th>
                <Th>Total PnL</Th>
                <Th>Trades</Th>
                <Th>Avg Win</Th>
                <Th>Avg Loss</Th>
                <Th>Profit Factor</Th>
                <Th>Sharpe</Th>
                <Th>Max DD</Th>
                <Th>Rank</Th>
              </Tr>
            </Thead>
            <Tbody>
              {filteredMetrics.map((metric) => (
                <Tr 
                  key={metric.bot_id}
                  bg={metric._updated ? highlightColor : undefined}
                  data-bot-id={metric.bot_id}
                  _hover={{ bg: useColorModeValue('gray.50', 'gray.700') }}
                >
                  <Td fontWeight="medium">{metric.bot_id}</Td>
                  <Td>{metric.ticker}</Td>
                  <Td>{metric.algorithm_type}</Td>
                  <Td color={getWinRateStyle(metric.win_rate)} className="win-rate">
                    {formatNumber(metric.win_rate * 100)}%
                  </Td>
                  <Td color={getPnlStyle(metric.total_pnl)} className="total-pnl">
                    ${formatNumber(metric.total_pnl)}
                  </Td>
                  <Td>{metric.total_trades || 0}</Td>
                  <Td color="green.500">${formatNumber(metric.average_win_amount)}</Td>
                  <Td color="red.500">${formatNumber(metric.average_loss_amount)}</Td>
                  <Td>{formatNumber(metric.profit_factor)}</Td>
                  <Td>{formatNumber(metric.sharpe_ratio)}</Td>
                  <Td>${formatNumber(metric.max_drawdown)}</Td>
                  <Td>
                    <Badge colorScheme={metric.rank <= 3 ? "green" : undefined}>
                      {metric.rank || "N/A"}
                    </Badge>
                  </Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </Box>
      )}
    </Box>
  );
};

export default BotMetricsRealtime; 