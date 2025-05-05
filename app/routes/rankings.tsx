import { json } from '@remix-run/node';
import { useLoaderData, Link } from '@remix-run/react';
import {
  Box,
  Heading,
  Text,
  Flex,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  HStack,
  VStack,
  SimpleGrid,
  Card,
  CardHeader,
  CardBody,
  Select,
  Button,
  Icon,
  useColorModeValue,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Tooltip,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  InputGroup,
  InputLeftElement,
  Input,
  Stack,
  Switch,
  Avatar,
  Checkbox,
  CheckboxGroup,
} from '@chakra-ui/react';
import { useState, useEffect, useRef } from 'react';
import { SearchIcon, ChevronDownIcon, ChevronUpIcon, ArrowUpIcon, ArrowDownIcon } from '@chakra-ui/icons';
import { FiFilter, FiActivity, FiPieChart, FiTrendingUp, FiTrendingDown, FiMinus, FiDollarSign, FiZap } from 'react-icons/fi';
import MainLayout from '~/components/layout/MainLayout';
import BotComparisonChart from '~/components/charts/BotComparisonChart';
import ParameterRadarChart from '~/components/charts/ParameterRadarChart';
import TradeAnalyticsChart from '~/components/charts/TradeAnalyticsChart';
import MetricInfoTooltip from '~/components/dashboard/MetricInfoTooltip';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  AreaChart,
  Area,
} from 'recharts';
import db from '~/lib/db.server';

// Custom types
interface BotMetrics {
  bot_id: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  total_pnl: number | string;
  total_pnl_percent: number | string; // Add percentage representation
  avg_profit_per_trade: number | string;
  average_pnl_per_trade_percent: number | string; // Add percentage representation
  win_rate: number | string;
  average_win_amount: number | string;
  average_loss_amount: number | string;
  profit_factor: number | string;
  max_drawdown: number | string;
  risk_reward_ratio: number | string;
  rank_score: number | string;
  last_updated: string;
  // Additional fields for historical rankings
  previous_rank?: number;
  rank_change?: number;
}

interface BotDetails {
  bot_id: number;
  name: string;
  ticker: string;
  algorithm_type: string;
  trade_direction: string;
  is_active: boolean;
  parameters?: Record<string, any>;
}

// Add a custom hook for WebSocket connection
function useTradeWebSocket(url = 'ws://localhost:8765/trades') {
  const [activeTrades, setActiveTrades] = useState<Record<number, any>>({});
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  
  useEffect(() => {
    const connect = () => {
      try {
        ws.current = new WebSocket(url);
        
        // Setup event handlers
        ws.current.onopen = () => {
          console.log('WebSocket connected to trades channel');
          setIsConnected(true);
          
          // Clear any reconnect timeout
          if (reconnectTimeout.current) {
            clearTimeout(reconnectTimeout.current);
          }
        };
        
        ws.current.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            
            // If we received trade data
            if (message.channel === 'trades' && message.data) {
              // Update active trades
              if (message.data.action === 'trade_opened') {
                // Add new trade by trade_id
                setActiveTrades(prev => ({
                  ...prev,
                  [message.data.trade_id]: message.data
                }));
              } else if (message.data.action === 'trade_closed') {
                // Remove closed trade by trade_id
                setActiveTrades(prev => {
                  const newState = { ...prev };
                  delete newState[message.data.trade_id];
                  return newState;
                });
              } else if (message.data.action === 'trade_update' && message.data.trades) {
                // Full trade update - replace entire state with current active trades
                const newActiveTrades: Record<number, any> = {};
                message.data.trades.forEach((trade: any) => {
                  if (trade.trade_status === 'open') {
                    newActiveTrades[trade.trade_id] = trade; // Key by trade_id
                  }
                });
                setActiveTrades(newActiveTrades);
              }
            }
          } catch (err) {
            console.error('Error processing WebSocket message:', err);
          }
        };
        
        ws.current.onclose = (event) => {
          console.log('WebSocket disconnected, code:', event.code);
          setIsConnected(false);
          
          // Attempt to reconnect after a delay
          reconnectTimeout.current = setTimeout(() => {
            console.log('Attempting to reconnect WebSocket...');
            connect();
          }, 2000);
        };
        
        ws.current.onerror = (error) => {
          console.error('WebSocket error:', error);
          setIsConnected(false);
        };
      } catch (error) {
        console.error('Error creating WebSocket connection:', error);
        setIsConnected(false);
      }
    };
    
    connect();
    
    // Cleanup on unmount
    return () => {
      if (ws.current) {
        ws.current.close();
      }
      
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
    };
  }, [url]);
  
  return { activeTrades, isConnected };
}

// Loader function to fetch data
export async function loader() {
  try {
    // Fetch all bots
    const bots: BotDetails[] = await db.getBots();
    
    // Fetch metrics for ranking
    const metrics: BotMetrics[] = await db.getBotMetrics();

    // Fetch today's trades count
    const todayTrades = await db.getTodayTradesCount();
    
    // Fetch daily return percentage
    const dailyReturn = await db.getDailyReturnPercentage();
    
    // Create merged bot data with metrics
    const botsWithMetrics = bots.map((bot: BotDetails) => {
      const botMetrics = metrics.find((m: BotMetrics) => m.bot_id === bot.bot_id) || {};
      return {
        ...bot,
        ...botMetrics,
      };
    });
    
    // For demo purposes, add some mock historical ranking data
    const rankedBots = botsWithMetrics.map((bot: any, index: number) => {
      const previousRank = index + 1 + (Math.random() > 0.7 ? Math.floor(Math.random() * 5) - 2 : 0);
      return {
        ...bot,
        previous_rank: previousRank,
        rank_change: previousRank - (index + 1)
      };
    });
    
    return json({ 
      bots: rankedBots,
      todayTrades: todayTrades?.count || 0,
      dailyReturn: dailyReturn?.percentage || 0
    });
  } catch (error) {
    console.error('Error loading ranking data:', error);
    return json({ 
      error: 'Failed to load ranking data', 
      bots: [],
      todayTrades: 0,
      dailyReturn: 0
    });
  }
}

// Helper functions for formatting
function formatCurrency(value: number | string) {
  const numValue = typeof value === 'string' ? parseFloat(value) : value;
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(numValue);
}

function formatPercent(value: number | string) {
  const numValue = typeof value === 'string' ? parseFloat(value) : value;
  return (numValue * 100).toFixed(2) + '%';
}

// Generate historic rank data for chart (mock data for now)
function generateHistoricRankData(botId: number) {
  const today = new Date();
  const data = [];
  
  for (let i = 30; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    
    // Generate a somewhat realistic rank fluctuation
    // Base rank is between 1-20 with small changes day to day
    let baseRank = botId <= 10 ? 
      botId + Math.floor(Math.random() * 5) : 
      10 + botId % 10 + Math.floor(Math.random() * 10);
    
    // Apply a random factor for the specific day
    const dayFactor = Math.sin(i / 5) * 3;
    const rank = Math.max(1, Math.floor(baseRank + dayFactor));
    
    data.push({
      date: date.toISOString().split('T')[0],
      rank
    });
  }
  
  return data;
}

export default function Rankings() {
  const { bots, todayTrades, dailyReturn } = useLoaderData<typeof loader>();
  const cardBg = useColorModeValue('white', 'gray.700');
  const accentColor = useColorModeValue('blue.500', 'blue.300');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const hoveredRowBg = useColorModeValue('gray.50', 'gray.600');
  const activeTradeBg = useColorModeValue('green.50', 'green.900');
  
  // Connect to WebSocket for real-time trade updates
  const { activeTrades, isConnected } = useTradeWebSocket();

  // State for filtering and sorting
  const [searchTerm, setSearchTerm] = useState('');
  const [algorithmFilter, setAlgorithmFilter] = useState('');
  const [symbolFilter, setSymbolFilter] = useState('');
  const [activeOnly, setActiveOnly] = useState(false);
  const [sortField, setSortField] = useState('rank_score');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  
  // State for selected bots in comparison
  const [selectedBotIds, setSelectedBotIds] = useState<number[]>([]);
  
  // State for ranking trends tab
  const [trendTimeframe, setTrendTimeframe] = useState('30d');
  const [trendSelectedBots, setTrendSelectedBots] = useState<number[]>([]);
  
  // Function to handle sorting
  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };
  
  // Function to render sort indicator
  const renderSortIndicator = (field: string) => {
    if (sortField !== field) return null;
    
    return sortDirection === 'asc' 
      ? <ChevronUpIcon ml={1} w={4} h={4} /> 
      : <ChevronDownIcon ml={1} w={4} h={4} />;
  };
  
  // Function to toggle bot selection for comparison
  const toggleBotSelection = (botId: number) => {
    setSelectedBotIds(prev => {
      if (prev.includes(botId)) {
        return prev.filter(id => id !== botId);
      } else {
        // Limit to 5 selections
        const newSelection = [...prev, botId];
        if (newSelection.length > 5) {
          return newSelection.slice(1);
        }
        return newSelection;
      }
    });
  };
  
  // Extract unique algorithm types and symbols for filters
  const algorithmTypes = [...new Set(bots.map((bot: any) => bot.algorithm_type))];
  const symbols = [...new Set(bots.map((bot: any) => bot.ticker))];
  
  // Filter and sort bots
  const filteredAndSortedBots = [...bots]
    .filter((bot: any) => {
      // Search filter
      if (
        searchTerm && 
        !bot.name?.toLowerCase().includes(searchTerm.toLowerCase()) && 
        !bot.ticker?.toLowerCase().includes(searchTerm.toLowerCase()) &&
        !bot.algorithm_type?.toLowerCase().includes(searchTerm.toLowerCase())
      ) {
        return false;
      }
      
      // Algorithm filter
      if (algorithmFilter && bot.algorithm_type !== algorithmFilter) {
        return false;
      }
      
      // Symbol filter
      if (symbolFilter && bot.ticker !== symbolFilter) {
        return false;
      }
      
      // Active only filter
      if (activeOnly && !bot.is_active) {
        return false;
      }
      
      return true;
    })
    .sort((a: any, b: any) => {
      let aValue: any = a[sortField as keyof typeof a];
      let bValue: any = b[sortField as keyof typeof b];
      
      // Handle numeric values stored as strings
      if (typeof aValue === 'string' && !isNaN(parseFloat(aValue))) {
        aValue = parseFloat(aValue);
      }
      
      if (typeof bValue === 'string' && !isNaN(parseFloat(bValue))) {
        bValue = parseFloat(bValue);
      }
      
      // Handle string sorting
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortDirection === 'asc' 
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }
      
      // Handle numeric sorting
      if (sortDirection === 'asc') {
        return (aValue ?? 0) - (bValue ?? 0);
      } else {
        return (bValue ?? 0) - (aValue ?? 0);
      }
    });
  
  // Prepare data for comparison chart
  const selectedBots = bots
    .filter((bot: any) => selectedBotIds.includes(bot.bot_id))
    .map((bot: any) => ({
      bot_id: bot.bot_id,
      name: `Bot ${bot.bot_id} - ${bot.ticker}`,
      win_rate: parseFloat(bot.win_rate as string || '0'),
      profit_factor: parseFloat(bot.profit_factor as string || '0'),
      max_drawdown: parseFloat(bot.max_drawdown as string || '0') / 1000, // Normalize for radar chart
      average_pnl_per_trade_percent: parseFloat(bot.avg_profit_per_trade as string || '0') * 100,
    }));
  
  // If no bots are selected, default to top 3
  const comparisonBots = selectedBots.length > 0 
    ? selectedBots 
    : bots
        .slice(0, 3)
        .map((bot: any) => ({
          bot_id: bot.bot_id,
          name: `Bot ${bot.bot_id} - ${bot.ticker}`,
          win_rate: parseFloat(bot.win_rate as string || '0'),
          profit_factor: parseFloat(bot.profit_factor as string || '0'),
          max_drawdown: parseFloat(bot.max_drawdown as string || '0') / 1000,
          average_pnl_per_trade_percent: parseFloat(bot.avg_profit_per_trade as string || '0') * 100,
        }));
        
  // Initialize trend data for selected bots or default to top 5
  useEffect(() => {
    if (trendSelectedBots.length === 0) {
      // Default to top 5 bots for trend visualization
      setTrendSelectedBots(bots.slice(0, 5).map((bot: any) => bot.bot_id));
    }
  }, [bots]);
  
  // Generate trend data for selected bots
  const generateTrendData = () => {
    // Get days based on timeframe
    const days = trendTimeframe === '7d' ? 7 : 
                trendTimeframe === '14d' ? 14 : 
                trendTimeframe === '30d' ? 30 : 90;
    
    const selectedBotsForTrend = bots
      .filter((bot: any) => trendSelectedBots.includes(bot.bot_id))
      .slice(0, 5); // Limit to 5 for readability
      
    // Generate dates for the selected timeframe
    const today = new Date();
    const dates = Array.from({ length: days }, (_, i) => {
      const date = new Date(today);
      date.setDate(date.getDate() - (days - i - 1));
      return date.toISOString().split('T')[0];
    });
    
    // Generate trend data with daily ranks for each selected bot
    return dates.map(date => {
      const dataPoint: any = { date };
      
      selectedBotsForTrend.forEach((bot: any) => {
        // Base rank calculation with some randomization to create a realistic trend
        // In a real app, this would come from historical data
        const baseRank = bots.findIndex((b: any) => b.bot_id === bot.bot_id) + 1;
        const dayOffset = parseInt(date.split('-')[2]);
        const rankVariation = Math.sin(dayOffset / 5) * 3; // Create a wave pattern
        const randomFactor = (Math.random() - 0.5) * 2; // Add some noise
        
        const rank = Math.max(1, Math.round(baseRank + rankVariation + randomFactor));
        dataPoint[`Bot ${bot.bot_id}`] = rank;
        dataPoint[`bot${bot.bot_id}Color`] = `hsl(${(bot.bot_id * 40) % 360}, 70%, 50%)`;
      });
      
      return dataPoint;
    });
  };
  
  // Generate data for fund allocation visualization
  const generateAllocationData = () => {
    // Group bots by rank tiers
    const topTier = bots.slice(0, 10);
    const midTier = bots.slice(10, 30);
    const lowTier = bots.slice(30);
    
    // Calculate allocation percentages (in a real app, this would be based on actual allocations)
    const topAllocation = topTier.length * 7; // 7% each for top 10
    const midAllocation = midTier.length * 1; // 1% each for next 20
    const lowAllocation = 100 - topAllocation - midAllocation; // Remainder for the rest
    
    return [
      { name: 'Top 10 Bots', value: topAllocation, color: '#4299E1' },
      { name: 'Mid-tier Bots (11-30)', value: midAllocation, color: '#9F7AEA' },
      { name: 'Lower-tier Bots (31+)', value: lowAllocation, color: '#ED8936' },
    ];
  };
  
  // Generate rank volatility data
  const generateVolatilityData = () => {
    return bots.slice(0, 20).map((bot: any) => {
      // Calculate a volatility score based on rank change
      // In a real app, this would be based on historical rank changes
      const volatility = Math.abs(bot.rank_change || 0) + (Math.random() * 2);
      
      return {
        name: `Bot ${bot.bot_id}`,
        volatility,
        ticker: bot.ticker,
        algorithm: bot.algorithm_type,
      };
    }).sort((a: any, b: any) => (b.volatility ?? 0) - (a.volatility ?? 0));
  };
  
  // Calculate system-wide metrics from bot data
  const totalBots = bots.length;
  const activeBots = bots.filter((bot: any) => bot.is_active).length;
  const botsWithTrades = bots.filter((bot: any) => bot.total_trades > 0);
  const totalPnl = botsWithTrades.reduce((sum: number, bot: any) => sum + parseFloat(bot.total_pnl as string || '0'), 0);
  
  // Get active trade count and list of unique bot IDs with active trades
  const activeTradeCount = Object.keys(activeTrades).length;
  const botsWithActiveTrades = new Set(Object.values(activeTrades).map((trade: any) => trade.bot_id)).size;
  
  return (
    <MainLayout>
      <Heading size="lg" mb={2}>Bot Rankings</Heading>
      <Text color="gray.500" mb={6}>Performance-based ranking of all trading bots in the system</Text>
      
      {/* Overview Stats */}
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6} mb={8}>
        <Stat
          px={4}
          py={5}
          shadow="sm"
          borderWidth="1px"
          borderRadius="lg"
          bg={cardBg}
        >
          <StatLabel fontSize="md">Total Bots</StatLabel>
          <StatNumber fontSize="3xl">{totalBots}</StatNumber>
          <StatHelpText>
            {activeBots} active ({(activeBots / totalBots * 100).toFixed(1)}%)
          </StatHelpText>
        </Stat>
        
        <Stat
          px={4}
          py={5}
          shadow="sm"
          borderWidth="1px"
          borderRadius="lg"
          bg={cardBg}
        >
          <StatLabel fontSize="md">Daily Return</StatLabel>
          <StatNumber fontSize="3xl" color={dailyReturn >= 0 ? 'green.500' : 'red.500'}>
            {formatPercent(dailyReturn)}
          </StatNumber>
          <StatHelpText>
            <StatArrow type={dailyReturn >= 0 ? 'increase' : 'decrease'} />
            Today's performance
          </StatHelpText>
        </Stat>
        
        <Stat
          px={4}
          py={5}
          shadow="sm"
          borderWidth="1px"
          borderRadius="lg"
          bg={cardBg}
        >
          <StatLabel fontSize="md">Total Trades Today</StatLabel>
          <HStack>
            <StatNumber fontSize="3xl">{todayTrades}</StatNumber>
          </HStack>
          <StatHelpText>
            Across all bots
          </StatHelpText>
        </Stat>
        
        <Stat
          px={4}
          py={5}
          shadow="sm"
          borderWidth="1px"
          borderRadius="lg"
          bg={cardBg}
        >
          <StatLabel fontSize="md">Active Trades</StatLabel>
          <HStack>
            <StatNumber fontSize="3xl">
              {activeTradeCount}
            </StatNumber>
            <Icon as={FiZap} color="orange.500" />
          </HStack>
          <StatHelpText>
            <HStack spacing={1}>
              <Icon as={isConnected ? FiActivity : FiMinus} color={isConnected ? "green.500" : "red.500"} />
              <Text>{isConnected ? `${botsWithActiveTrades} bots trading` : "Disconnected"}</Text>
            </HStack>
          </StatHelpText>
        </Stat>
      </SimpleGrid>
      
      {/* Filters */}
      <Card shadow="sm" mb={6} bg={cardBg}>
        <CardBody>
          <Stack 
            direction={{ base: 'column', md: 'row' }} 
            spacing={4}
            align={{ base: 'stretch', md: 'center' }}
          >
            <InputGroup maxW={{ md: '300px' }}>
              <InputLeftElement pointerEvents="none">
                <SearchIcon color="gray.300" />
              </InputLeftElement>
              <Input 
                placeholder="Search bots..." 
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </InputGroup>
            
            <Select 
              placeholder="Algorithm Type" 
              maxW={{ md: '200px' }}
              value={algorithmFilter}
              onChange={(e) => setAlgorithmFilter(e.target.value)}
            >
              <option value="">All Algorithms</option>
              {algorithmTypes.map((type: string | number | undefined, index: number) => (
                <option key={`${type}-${index}`} value={String(type)}>
                  {String(type)?.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
                </option>
              ))}
            </Select>
            
            <Select 
              placeholder="Symbol" 
              maxW={{ md: '150px' }}
              value={symbolFilter}
              onChange={(e) => setSymbolFilter(e.target.value)}
            >
              <option value="">All Symbols</option>
              {symbols.map((symbol: string | number | undefined, index: number) => (
                <option key={`${symbol}-${index}`} value={String(symbol)}>{String(symbol)}</option>
              ))}
            </Select>
            
            <HStack spacing={2}>
              <Text fontSize="sm">Active Only</Text>
              <Switch 
                colorScheme="green" 
                isChecked={activeOnly}
                onChange={() => setActiveOnly(prev => !prev)}
              />
            </HStack>
            
            <Box ml="auto">
              <Button 
                leftIcon={<Icon as={FiFilter} />} 
                size="sm" 
                variant="outline"
                onClick={() => {
                  setSearchTerm('');
                  setAlgorithmFilter('');
                  setSymbolFilter('');
                  setActiveOnly(false);
                }}
              >
                Clear Filters
              </Button>
            </Box>
          </Stack>
        </CardBody>
      </Card>
      
      {/* Tabs for different views */}
      <Tabs variant="enclosed" colorScheme="blue">
        <TabList>
          <Tab>Rankings Table</Tab>
        </TabList>
        
        <TabPanels>
          {/* Rankings Table Tab */}
          <TabPanel px={0}>
            <Card shadow="sm" bg={cardBg}>
              <CardHeader pb={0}>
                <Flex justify="space-between" align="center">
                  <Heading size="md">Bot Rankings</Heading>
                  <Text fontSize="sm" color="gray.500">
                    Showing {filteredAndSortedBots.length} of {bots.length} bots
                  </Text>
                </Flex>
                <Text fontSize="sm" color="gray.500" mt={1}>
                  Select up to 5 bots for comparison
                </Text>
              </CardHeader>
              <CardBody>
                <Box overflowX="auto">
                  <Table variant="simple">
                    <Thead>
                      <Tr>
                        <Th width="50px">Select</Th>
                        <Th 
                          cursor="pointer" 
                          onClick={() => handleSort('rank_score')}
                          userSelect="none"
                          width="80px"
                        >
                          <Flex align="center">
                            <MetricInfoTooltip metricKey="rank_score">
                              Rank {renderSortIndicator('rank_score')}
                            </MetricInfoTooltip>
                          </Flex>
                        </Th>
                        <Th>Bot</Th>
                        <Th 
                          cursor="pointer" 
                          onClick={() => handleSort('algorithm_type')}
                          userSelect="none"
                        >
                          <Flex align="center">
                            Strategy {renderSortIndicator('algorithm_type')}
                          </Flex>
                        </Th>
                        <Th 
                          cursor="pointer" 
                          onClick={() => handleSort('win_rate')}
                          userSelect="none"
                        >
                          <Flex align="center">
                            <MetricInfoTooltip metricKey="win_rate">
                              Win Rate {renderSortIndicator('win_rate')}
                            </MetricInfoTooltip>
                          </Flex>
                        </Th>
                        <Th 
                          cursor="pointer" 
                          onClick={() => handleSort('avg_profit_per_trade')}
                          userSelect="none"
                        >
                          <Flex align="center">
                            <MetricInfoTooltip metricKey="average_pnl_per_trade"> 
                              Avg P&L/Trade {renderSortIndicator('avg_profit_per_trade')}
                            </MetricInfoTooltip>
                          </Flex>
                        </Th>
                        <Th 
                          cursor="pointer" 
                          onClick={() => handleSort('total_pnl')}
                          userSelect="none"
                        >
                          <Flex align="center">
                            <MetricInfoTooltip metricKey="total_pnl">
                              Returns {renderSortIndicator('total_pnl')}
                            </MetricInfoTooltip>
                          </Flex>
                        </Th>
                        <Th width="100px">Status</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {filteredAndSortedBots.map((bot: any, index: number) => {
                        // Check if this bot has an active trade
                        const isInActiveTrade = Object.values(activeTrades).some(
                          (trade: any) => trade.bot_id === bot.bot_id
                        );
                        
                        // Get the active trade(s) for this bot
                        const botActiveTrades = Object.values(activeTrades).filter(
                          (trade: any) => trade.bot_id === bot.bot_id
                        );
                        
                        return (
                        <Tr 
                          key={bot.bot_id}
                          _hover={{ bg: hoveredRowBg }}
                          cursor="pointer"
                          onClick={() => toggleBotSelection(bot.bot_id)}
                          bg={
                            selectedBotIds.includes(bot.bot_id) ? 'blue.50' : 
                            isInActiveTrade ? activeTradeBg : undefined
                          }
                          _dark={{
                            bg: selectedBotIds.includes(bot.bot_id) ? 'blue.900' : 
                            isInActiveTrade ? 'green.900' : undefined,
                          }}
                        >
                          <Td>
                            <Switch 
                              colorScheme="blue" 
                              isChecked={selectedBotIds.includes(bot.bot_id)}
                              onChange={(e) => {
                                e.stopPropagation();
                                toggleBotSelection(bot.bot_id);
                              }}
                            />
                          </Td>
                          <Td>
                            <Text fontWeight="bold">{index + 1}</Text>
                          </Td>
                          <Td>
                            <HStack>
                              <Avatar 
                                size="xs" 
                                name={`Bot ${bot.bot_id}`} 
                                bg={`hsl(${(bot.bot_id * 40) % 360}, 70%, 50%)`} 
                              />
                              <Box>
                                <Text fontWeight="medium">Bot {bot.bot_id}</Text>
                                <Badge colorScheme="blue" fontSize="xs">
                                  {bot.ticker}
                                </Badge>
                              </Box>
                            </HStack>
                          </Td>
                          <Td>
                            <Text>{bot.algorithm_type?.replace(/_/g, ' ')}</Text>
                            <Text fontSize="xs" color="gray.500">
                              {bot.trade_direction}
                            </Text>
                          </Td>
                          <Td>
                            <Badge 
                              colorScheme={parseFloat(bot.win_rate as string || '0') >= 0.5 ? 'green' : 'red'}
                              p={1}
                              borderRadius="md"
                            >
                              {formatPercent(bot.win_rate || 0)}
                            </Badge>
                          </Td>
                          <Td>
                            <Text 
                              color={parseFloat(bot.avg_profit_per_trade as string || '0') >= 0 ? 'green.500' : 'red.500'}
                            >
                              {formatPercent(parseFloat(bot.avg_profit_per_trade as string || '0') / 100)}
                            </Text>
                          </Td>
                          <Td 
                            color={parseFloat(bot.total_pnl as string || '0') >= 0 ? 'green.500' : 'red.500'}
                            fontWeight="bold"
                          >
                            {formatPercent(parseFloat(bot.total_pnl as string || '0') / 100000)}
                          </Td>
                          <Td>
                            {isInActiveTrade ? (
                              <Tooltip 
                                label={
                                  botActiveTrades.length > 1
                                    ? `${botActiveTrades.length} active trades: ` + 
                                      botActiveTrades.map(t => 
                                        `${t.trade_direction.toUpperCase()} ${t.ticker} @ $${parseFloat(t.entry_price).toFixed(2)}`
                                      ).join(', ')
                                    : `${botActiveTrades[0].trade_direction.toUpperCase()} ${botActiveTrades[0].ticker} @ $${parseFloat(botActiveTrades[0].entry_price).toFixed(2)}`
                                }
                                placement="top"
                              >
                                <Badge 
                                  colorScheme="green" 
                                  p={1} 
                                  borderRadius="md"
                                  display="flex"
                                  alignItems="center"
                                >
                                  <Icon 
                                    as={botActiveTrades[0].trade_direction === 'long' ? FiTrendingUp : FiTrendingDown} 
                                    mr={1} 
                                    boxSize={3}
                                  />
                                  <Text>
                                    {botActiveTrades.length > 1 ? `${botActiveTrades.length} TRADES` : 'IN TRADE'}
                                  </Text>
                                </Badge>
                              </Tooltip>
                            ) : (
                              <Badge 
                                colorScheme="gray" 
                                p={1} 
                                borderRadius="md"
                                opacity={0.7}
                              >
                                IDLE
                              </Badge>
                            )}
                          </Td>
                        </Tr>
                      )})}
                      {filteredAndSortedBots.length === 0 && (
                        <Tr>
                          <Td colSpan={8} textAlign="center" py={6}>
                            <Text color="gray.500">No bots match your filters</Text>
                          </Td>
                        </Tr>
                      )}
                    </Tbody>
                  </Table>
                </Box>
              </CardBody>
            </Card>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </MainLayout>
  );
}