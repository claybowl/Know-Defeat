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
} from '@chakra-ui/react';
import { useState, useEffect, useRef } from 'react';
import { SearchIcon, ChevronDownIcon, ChevronUpIcon, ArrowUpIcon, ArrowDownIcon } from '@chakra-ui/icons';
import { FiFilter, FiActivity, FiPieChart, FiTrendingUp, FiTrendingDown, FiMinus } from 'react-icons/fi';
import MainLayout from '~/components/layout/MainLayout';
import BotComparisonChart from '~/components/charts/BotComparisonChart';
import ParameterRadarChart from '~/components/charts/ParameterRadarChart';
import TradeAnalyticsChart from '~/components/charts/TradeAnalyticsChart';
import db from '~/lib/db.server';

// Custom types
interface BotMetrics {
  bot_id: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  total_pnl: number | string;
  average_pnl_per_trade: number | string;
  win_rate: number | string;
  average_win_amount: number | string;
  average_loss_amount: number | string;
  profit_factor: number | string;
  max_drawdown: number | string;
  sharpe_ratio: number | string;
  risk_reward_ratio: number | string;
  expectancy: number | string;
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

// Loader function to fetch data
export async function loader() {
  try {
    // Fetch all bots
    const bots = await db.getBots();
    
    // Fetch metrics for ranking
    const metrics = await db.getBotMetrics();
    
    // Create merged bot data with metrics
    const botsWithMetrics = bots.map(bot => {
      const botMetrics = metrics.find(m => m.bot_id === bot.bot_id) || {};
      return {
        ...bot,
        ...botMetrics,
      };
    });
    
    // For demo purposes, add some mock historical ranking data
    const rankedBots = botsWithMetrics.map((bot, index) => {
      const previousRank = index + 1 + (Math.random() > 0.7 ? Math.floor(Math.random() * 5) - 2 : 0);
      return {
        ...bot,
        previous_rank: previousRank,
        rank_change: previousRank - (index + 1)
      };
    });
    
    return json({ bots: rankedBots });
  } catch (error) {
    console.error('Error loading ranking data:', error);
    return json({ error: 'Failed to load ranking data', bots: [] });
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
  const { bots } = useLoaderData<typeof loader>();
  const cardBg = useColorModeValue('white', 'gray.700');
  const accentColor = useColorModeValue('blue.500', 'blue.300');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const hoveredRowBg = useColorModeValue('gray.50', 'gray.600');

  // State for filtering and sorting
  const [searchTerm, setSearchTerm] = useState('');
  const [algorithmFilter, setAlgorithmFilter] = useState('');
  const [symbolFilter, setSymbolFilter] = useState('');
  const [activeOnly, setActiveOnly] = useState(false);
  const [sortField, setSortField] = useState('rank_score');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  
  // State for selected bots in comparison
  const [selectedBotIds, setSelectedBotIds] = useState<number[]>([]);
  
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
  const algorithmTypes = [...new Set(bots.map(bot => bot.algorithm_type))];
  const symbols = [...new Set(bots.map(bot => bot.ticker))];
  
  // Filter and sort bots
  const filteredAndSortedBots = [...bots]
    .filter(bot => {
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
    .sort((a, b) => {
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
        return aValue - bValue;
      } else {
        return bValue - aValue;
      }
    });
  
  // Prepare data for comparison chart
  const selectedBots = bots
    .filter(bot => selectedBotIds.includes(bot.bot_id))
    .map(bot => ({
      bot_id: bot.bot_id,
      name: `Bot ${bot.bot_id} - ${bot.ticker}`,
      win_rate: parseFloat(bot.win_rate as string || '0'),
      profit_factor: parseFloat(bot.profit_factor as string || '0'),
      sharpe_ratio: parseFloat(bot.sharpe_ratio as string || '0'),
      max_drawdown: parseFloat(bot.max_drawdown as string || '0') / 1000, // Normalize for radar chart
      expectancy: parseFloat(bot.expectancy as string || '0'),
    }));
  
  // If no bots are selected, default to top 3
  const comparisonBots = selectedBots.length > 0 
    ? selectedBots 
    : bots
        .slice(0, 3)
        .map(bot => ({
          bot_id: bot.bot_id,
          name: `Bot ${bot.bot_id} - ${bot.ticker}`,
          win_rate: parseFloat(bot.win_rate as string || '0'),
          profit_factor: parseFloat(bot.profit_factor as string || '0'),
          sharpe_ratio: parseFloat(bot.sharpe_ratio as string || '0'),
          max_drawdown: parseFloat(bot.max_drawdown as string || '0') / 1000,
          expectancy: parseFloat(bot.expectancy as string || '0'),
        }));
  
  // Calculate system-wide metrics from bot data
  const totalBots = bots.length;
  const activeBots = bots.filter(bot => bot.is_active).length;
  const botsWithTrades = bots.filter(bot => bot.total_trades > 0);
  const totalPnl = botsWithTrades.reduce((sum, bot) => sum + parseFloat(bot.total_pnl as string || '0'), 0);
  
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
          <StatLabel fontSize="md">Total P&L</StatLabel>
          <StatNumber fontSize="3xl" color={totalPnl >= 0 ? 'green.500' : 'red.500'}>
            {formatCurrency(totalPnl)}
          </StatNumber>
          <StatHelpText>
            <StatArrow type={totalPnl >= 0 ? 'increase' : 'decrease'} />
            System-wide performance
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
          <StatLabel fontSize="md">Top Performer</StatLabel>
          <HStack>
            <StatNumber fontSize="3xl">Bot {bots[0]?.bot_id}</StatNumber>
            <Badge colorScheme="green">#{1}</Badge>
          </HStack>
          <StatHelpText>
            Win Rate: {formatPercent(bots[0]?.win_rate || 0)}
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
          <StatLabel fontSize="md">Most Improved</StatLabel>
          <HStack>
            <StatNumber fontSize="3xl">
              Bot {bots.sort((a, b) => (b.rank_change || 0) - (a.rank_change || 0))[0]?.bot_id}
            </StatNumber>
            <Icon as={FiTrendingUp} color="green.500" />
          </HStack>
          <StatHelpText>
            Up {bots.sort((a, b) => (b.rank_change || 0) - (a.rank_change || 0))[0]?.rank_change || 0} positions
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
              {algorithmTypes.map(type => (
                <option key={type} value={type}>
                  {type?.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
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
              {symbols.map(symbol => (
                <option key={symbol} value={symbol}>{symbol}</option>
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
          <Tab>Performance Comparison</Tab>
          <Tab>Ranking Trend</Tab>
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
                            Rank {renderSortIndicator('rank_score')}
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
                            Win Rate {renderSortIndicator('win_rate')}
                          </Flex>
                        </Th>
                        <Th 
                          cursor="pointer" 
                          onClick={() => handleSort('profit_factor')}
                          userSelect="none"
                        >
                          <Flex align="center">
                            Profit Factor {renderSortIndicator('profit_factor')}
                          </Flex>
                        </Th>
                        <Th 
                          cursor="pointer" 
                          onClick={() => handleSort('sharpe_ratio')}
                          userSelect="none"
                        >
                          <Flex align="center">
                            Sharpe {renderSortIndicator('sharpe_ratio')}
                          </Flex>
                        </Th>
                        <Th 
                          cursor="pointer" 
                          onClick={() => handleSort('expectancy')}
                          userSelect="none"
                        >
                          <Flex align="center">
                            Expectancy {renderSortIndicator('expectancy')}
                          </Flex>
                        </Th>
                        <Th 
                          cursor="pointer" 
                          onClick={() => handleSort('total_pnl')}
                          userSelect="none"
                        >
                          <Flex align="center">
                            P&L {renderSortIndicator('total_pnl')}
                          </Flex>
                        </Th>
                        <Th width="100px">Trend</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {filteredAndSortedBots.map((bot, index) => (
                        <Tr 
                          key={bot.bot_id}
                          _hover={{ bg: hoveredRowBg }}
                          cursor="pointer"
                          onClick={() => toggleBotSelection(bot.bot_id)}
                          bg={selectedBotIds.includes(bot.bot_id) ? 'blue.50' : undefined}
                          _dark={{
                            bg: selectedBotIds.includes(bot.bot_id) ? 'blue.900' : undefined,
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
                            <Flex align="center">
                              <Text fontWeight="bold" mr={1}>{index + 1}</Text>
                              {bot.rank_change !== 0 && (
                                <Badge 
                                  borderRadius="full" 
                                  colorScheme={bot.rank_change > 0 ? 'green' : bot.rank_change < 0 ? 'red' : 'gray'}
                                  ml={1}
                                  display="flex"
                                  alignItems="center"
                                >
                                  {bot.rank_change > 0 ? (
                                    <Icon as={FiTrendingUp} mr={1} boxSize={3} />
                                  ) : bot.rank_change < 0 ? (
                                    <Icon as={FiTrendingDown} mr={1} boxSize={3} />
                                  ) : (
                                    <Icon as={FiMinus} mr={1} boxSize={3} />
                                  )}
                                  {Math.abs(bot.rank_change || 0)}
                                </Badge>
                              )}
                            </Flex>
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
                            {parseFloat(bot.profit_factor as string || '0').toFixed(2)}
                          </Td>
                          <Td>
                            {parseFloat(bot.sharpe_ratio as string || '0').toFixed(2)}
                          </Td>
                          <Td>
                            {parseFloat(bot.expectancy as string || '0').toFixed(3)}
                          </Td>
                          <Td 
                            color={parseFloat(bot.total_pnl as string || '0') >= 0 ? 'green.500' : 'red.500'}
                            fontWeight="bold"
                          >
                            {formatCurrency(bot.total_pnl || 0)}
                          </Td>
                          <Td>
                            <HStack spacing={1}>
                              <Tooltip label="Win Rate Trend">
                                <HStack spacing={1}>
                                  <Icon as={FiActivity} color="green.500" boxSize={4} />
                                  <Icon as={bot.rank_change > 0 ? FiTrendingUp : FiTrendingDown} 
                                    color={bot.rank_change > 0 ? "green.500" : "red.500"} 
                                    boxSize={4} 
                                  />
                                </HStack>
                              </Tooltip>
                              
                              <Tooltip label="Profit Factor Trend">
                                <HStack spacing={1}>
                                  <Icon as={FiPieChart} color="purple.500" boxSize={4} />
                                  <Icon as={parseFloat(bot.profit_factor as string || '0') > 1.5 ? FiTrendingUp : FiTrendingDown} 
                                    color={parseFloat(bot.profit_factor as string || '0') > 1.5 ? "green.500" : "red.500"} 
                                    boxSize={4} 
                                  />
                                </HStack>
                              </Tooltip>
                            </HStack>
                          </Td>
                        </Tr>
                      ))}
                      {filteredAndSortedBots.length === 0 && (
                        <Tr>
                          <Td colSpan={10} textAlign="center" py={6}>
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
          
          {/* Performance Comparison Tab */}
          <TabPanel px={0}>
            <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={6}>
              <Card shadow="sm" bg={cardBg}>
                <CardHeader>
                  <Heading size="md">Performance Metrics Comparison</Heading>
                  <Text fontSize="sm" color="gray.500" mt={1}>
                    Comparing key metrics across selected bots
                  </Text>
                </CardHeader>
                <CardBody>
                  <BotComparisonChart bots={comparisonBots} />
                </CardBody>
              </Card>
              
              <Card shadow="sm" bg={cardBg}>
                <CardHeader>
                  <Heading size="md">Strategy Parameters</Heading>
                  <Text fontSize="sm" color="gray.500" mt={1}>
                    Optimal parameter configurations
                  </Text>
                </CardHeader>
                <CardBody>
                  {selectedBotIds.length > 0 ? (
                    <Tabs variant="soft-rounded" colorScheme="blue" size="sm">
                      <TabList>
                        {selectedBotIds.map(botId => (
                          <Tab key={botId}>Bot {botId}</Tab>
                        ))}
                      </TabList>
                      <TabPanels>
                        {selectedBotIds.map(botId => {
                          const bot = bots.find(b => b.bot_id === botId);
                          // Use sample parameters if real ones aren't available
                          const parameters = bot?.parameters || {
                            lookback_period: 20,
                            volatility_threshold: 2.0,
                            profit_target_pct: 0.02,
                            stop_loss_pct: 0.01,
                            rsi_upper: 70,
                            rsi_lower: 30,
                            moving_average_period: 15,
                          };
                          
                          return (
                            <TabPanel key={botId}>
                              <ParameterRadarChart parameters={parameters} />
                            </TabPanel>
                          );
                        })}
                      </TabPanels>
                    </Tabs>
                  ) : (
                    <Flex 
                      justify="center" 
                      align="center" 
                      direction="column" 
                      h="300px"
                      color="gray.500"
                    >
                      <Text mb={3}>Select bots to compare their parameters</Text>
                      <Button 
                        colorScheme="blue" 
                        size="sm"
                        onClick={() => {
                          // Select top 3 bots by default
                          setSelectedBotIds(bots.slice(0, 3).map(b => b.bot_id));
                        }}
                      >
                        Select Top 3 Bots
                      </Button>
                    </Flex>
                  )}
                </CardBody>
              </Card>
              
              <Card shadow="sm" bg={cardBg} gridColumn={{ lg: "span 2" }}>
                <CardHeader>
                  <Heading size="md">Key Performance Indicators</Heading>
                </CardHeader>
                <CardBody>
                  <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6}>
                    {selectedBotIds.length > 0 ? selectedBotIds.map(botId => {
                      const bot = bots.find(b => b.bot_id === botId);
                      if (!bot) return null;
                      
                      return (
                        <VStack 
                          key={botId} 
                          align="stretch" 
                          p={4} 
                          borderWidth="1px" 
                          borderRadius="lg" 
                          spacing={3}
                          borderLeftWidth="4px"
                          borderLeftColor={`hsl(${(bot.bot_id * 40) % 360}, 70%, 50%)`}
                        >
                          <Flex justify="space-between" align="center">
                            <HStack>
                              <Avatar 
                                size="xs" 
                                name={`Bot ${bot.bot_id}`} 
                                bg={`hsl(${(bot.bot_id * 40) % 360}, 70%, 50%)`} 
                              />
                              <Text fontWeight="bold">Bot {bot.bot_id}</Text>
                            </HStack>
                            <Badge colorScheme="blue">{bot.ticker}</Badge>
                          </Flex>
                          
                          <SimpleGrid columns={2} spacing={3}>
                            <VStack align="start" spacing={0}>
                              <Text fontSize="xs" color="gray.500">Win Rate</Text>
                              <Text fontWeight="medium">{formatPercent(bot.win_rate || 0)}</Text>
                            </VStack>
                            
                            <VStack align="start" spacing={0}>
                              <Text fontSize="xs" color="gray.500">Profit Factor</Text>
                              <Text fontWeight="medium">{parseFloat(bot.profit_factor as string || '0').toFixed(2)}</Text>
                            </VStack>
                            
                            <VStack align="start" spacing={0}>
                              <Text fontSize="xs" color="gray.500">Sharpe Ratio</Text>
                              <Text fontWeight="medium">{parseFloat(bot.sharpe_ratio as string || '0').toFixed(2)}</Text>
                            </VStack>
                            
                            <VStack align="start" spacing={0}>
                              <Text fontSize="xs" color="gray.500">Total P&L</Text>
                              <Text 
                                fontWeight="medium"
                                color={parseFloat(bot.total_pnl as string || '0') >= 0 ? 'green.500' : 'red.500'}
                              >
                                {formatCurrency(bot.total_pnl || 0)}
                              </Text>
                            </VStack>
                          </SimpleGrid>
                          
                          <Flex justify="space-between" fontSize="sm">
                            <Text color="gray.500">Rank:</Text>
                            <HStack>
                              <Text fontWeight="bold">#{bots.findIndex(b => b.bot_id === botId) + 1}</Text>
                              {bot.rank_change !== 0 && (
                                <Badge 
                                  borderRadius="full" 
                                  colorScheme={bot.rank_change > 0 ? 'green' : 'red'}
                                  fontSize="xs"
                                >
                                  {bot.rank_change > 0 ? '+' : ''}{bot.rank_change}
                                </Badge>
                              )}
                            </HStack>
                          </Flex>
                        </VStack>
                      );
                    }) : (
                      <Box 
                        gridColumn="span 4" 
                        p={6} 
                        textAlign="center" 
                        color="gray.500"
                      >
                        <Text mb={3}>Select bots to view their key performance indicators</Text>
                        <Button 
                          colorScheme="blue" 
                          size="sm"
                          onClick={() => {
                            setSelectedBotIds(bots.slice(0, 4).map(b => b.bot_id));
                          }}
                        >
                          Select Top 4 Bots
                        </Button>
                      </Box>
                    )}
                  </SimpleGrid>
                </CardBody>
              </Card>
            </SimpleGrid>
          </TabPanel>
          
          {/* Ranking Trend Tab */}
          <TabPanel px={0}>
            <Card shadow="sm" bg={cardBg} mb={6}>
              <CardHeader>
                <Heading size="md">Ranking History</Heading>
                <Text fontSize="sm" color="gray.500" mt={1}>
                  Historical ranking positions over time
                </Text>
              </CardHeader>
              <CardBody>
                <Flex justify="center" align="center" h="400px" color="gray.500">
                  <Text>Ranking trend visualization will be implemented here</Text>
                </Flex>
              </CardBody>
            </Card>
            
            <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={6}>
              <Card shadow="sm" bg={cardBg}>
                <CardHeader>
                  <Heading size="md">Allocation Distribution</Heading>
                </CardHeader>
                <CardBody>
                  <Flex justify="center" align="center" h="300px" color="gray.500">
                    <Text>Fund allocation by rank will be shown here</Text>
                  </Flex>
                </CardBody>
              </Card>
              
              <Card shadow="sm" bg={cardBg}>
                <CardHeader>
                  <Heading size="md">Rank Volatility</Heading>
                </CardHeader>
                <CardBody>
                  <Flex justify="center" align="center" h="300px" color="gray.500">
                    <Text>Rank stability analysis will be shown here</Text>
                  </Flex>
                </CardBody>
              </Card>
            </SimpleGrid>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </MainLayout>
  );
}