import React, { useState, useEffect } from 'react';
import {
  Box,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Button,
  Text,
  Heading,
  Flex,
  Input,
  InputGroup,
  InputLeftElement,
  Select,
  Stack,
  IconButton,
  useToast,
  useColorModeValue
} from '@chakra-ui/react';
import { SearchIcon, RepeatIcon } from '@chakra-ui/icons';

export const ActiveTradesTable = ({ wsData }) => {
  const [trades, setTrades] = useState([]);
  const [filteredTrades, setFilteredTrades] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterSymbol, setFilterSymbol] = useState('');
  const [uniqueSymbols, setUniqueSymbols] = useState([]);
  const toast = useToast();
  
  const tableBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  
  // Format currency with commas and dollar sign
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };
  
  // Process incoming WebSocket data
  useEffect(() => {
    if (!wsData) return;
    
    console.log('WS data received:', wsData);
    
    try {
      // Handle different message types based on action
      if (wsData.data) {
        const { data } = wsData;
        
        // Full trade update - replaces all trades
        if (data.action === 'trade_update' && Array.isArray(data.trades)) {
          setTrades(data.trades.filter(trade => trade.trade_status === 'open'));
        }
        // Single trade opened
        else if (data.action === 'trade_opened') {
          setTrades(prevTrades => [...prevTrades.filter(t => t.trade_id !== data.trade_id), data]);
        }
        // Single trade closed
        else if (data.action === 'trade_closed') {
          setTrades(prevTrades => prevTrades.filter(trade => trade.trade_id !== data.trade_id));
        }
      }
    } catch (error) {
      console.error('Error processing WebSocket data:', error);
    }
  }, [wsData]);
  
  // Extract unique symbols for filtering
  useEffect(() => {
    if (trades.length > 0) {
      const symbols = [...new Set(trades.map(trade => trade.ticker))];
      setUniqueSymbols(symbols);
    }
  }, [trades]);
  
  // Filter trades based on search term and symbol filter
  useEffect(() => {
    let filtered = trades;
    
    if (searchTerm) {
      filtered = filtered.filter(trade => 
        (trade.bot_name ? trade.bot_name.toLowerCase().includes(searchTerm.toLowerCase()) : false) ||
        (trade.ticker ? trade.ticker.toLowerCase().includes(searchTerm.toLowerCase()) : false) ||
        String(trade.trade_id).includes(searchTerm) ||
        String(trade.bot_id).includes(searchTerm)
      );
    }
    
    if (filterSymbol) {
      filtered = filtered.filter(trade => trade.ticker === filterSymbol);
    }
    
    setFilteredTrades(filtered);
  }, [trades, searchTerm, filterSymbol]);
  
  // Handle manual trade closure
  const handleCloseTrade = async (tradeId) => {
    try {
      const response = await fetch(`/api/trades/${tradeId}/close`, {
        method: 'POST',
      });
      
      if (response.ok) {
        toast({
          title: "Trade close request sent",
          description: `Trade ID: ${tradeId} close request has been sent.`,
          status: "success",
          duration: 5000,
          isClosable: true,
        });
        
        // Local state will be updated via WebSocket when the trade is actually closed
      } else {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to close trade');
      }
    } catch (error) {
      toast({
        title: "Error closing trade",
        description: error.message,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    }
  };
  
  return (
    <Box bg={tableBg} p={4} borderRadius="md" boxShadow="sm" border="1px" borderColor={borderColor}>
      <Flex justify="space-between" mb={4} align="center">
        <Heading size="md">Active Trades</Heading>
        <Text>
          {filteredTrades.length} {filteredTrades.length === 1 ? 'trade' : 'trades'} active
        </Text>
      </Flex>
      
      {/* Filters */}
      <Stack direction={["column", "row"]} spacing={4} mb={4}>
        <InputGroup maxW="300px">
          <InputLeftElement pointerEvents="none">
            <SearchIcon color="gray.400" />
          </InputLeftElement>
          <Input
            placeholder="Search by ID, bot name or ticker"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </InputGroup>
        
        <Select
          placeholder="Filter by symbol"
          value={filterSymbol}
          onChange={(e) => setFilterSymbol(e.target.value)}
          maxW="200px"
        >
          <option value="">All symbols</option>
          {uniqueSymbols.map(symbol => (
            <option key={symbol} value={symbol}>{symbol}</option>
          ))}
        </Select>
        
        <IconButton
          aria-label="Refresh data"
          icon={<RepeatIcon />}
          onClick={() => console.log("Manual refresh - not implemented")}
        />
      </Stack>
      
      {/* Trades Table */}
      <Box overflowX="auto">
        <Table variant="simple">
          <Thead>
            <Tr>
              <Th>Trade ID</Th>
              <Th>Bot ID</Th>
              <Th>Symbol</Th>
              <Th>Direction</Th>
              <Th>Entry Price</Th>
              <Th>Size</Th>
              <Th>Entry Time</Th>
              <Th>Stop Price</Th>
              <Th>Actions</Th>
            </Tr>
          </Thead>
          <Tbody>
            {filteredTrades.length > 0 ? (
              filteredTrades.map((trade) => (
                <Tr key={trade.trade_id}>
                  <Td>{trade.trade_id}</Td>
                  <Td>{trade.bot_id}</Td>
                  <Td>{trade.ticker}</Td>
                  <Td>
                    <Badge
                      colorScheme={trade.trade_direction.toLowerCase() === 'long' ? 'green' : 'red'}
                    >
                      {trade.trade_direction}
                    </Badge>
                  </Td>
                  <Td>${parseFloat(trade.entry_price).toFixed(2)}</Td>
                  <Td>{formatCurrency(trade.trade_size)}</Td>
                  <Td>{new Date(trade.entry_time).toLocaleString()}</Td>
                  <Td>
                    {trade.trailing_stop_price 
                      ? '$' + parseFloat(trade.trailing_stop_price).toFixed(2) 
                      : '-'
                    }
                  </Td>
                  <Td>
                    <Button
                      size="sm"
                      colorScheme="red"
                      onClick={() => handleCloseTrade(trade.trade_id)}
                    >
                      Close
                    </Button>
                  </Td>
                </Tr>
              ))
            ) : (
              <Tr>
                <Td colSpan={9} textAlign="center">
                  No active trades
                </Td>
              </Tr>
            )}
          </Tbody>
        </Table>
      </Box>
      
      {/* WebSocket Connection Status */}
      <Flex justify="flex-end" mt={2}>
        <Badge colorScheme={wsData ? "green" : "red"}>
          {wsData ? "WebSocket Connected" : "WebSocket Disconnected"}
        </Badge>
      </Flex>
    </Box>
  );
}; 