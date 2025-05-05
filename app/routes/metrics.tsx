import { json } from '@remix-run/node';
import { useLoaderData, Link } from '@remix-run/react';
import {
  Box,
  Heading,
  SimpleGrid,
  Card,
  CardHeader,
  CardBody,
  Text,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  HStack,
  VStack,
  Select,
  Flex,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  useColorModeValue,
  Button,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
} from '@chakra-ui/react';
import { useState } from 'react';
import MainLayout from '~/components/layout/MainLayout';
import PerformanceMetricsChart from '~/components/charts/PerformanceMetricsChart';
import BotComparisonChart from '~/components/charts/BotComparisonChart';
import MetricInfoTooltip from '~/components/dashboard/MetricInfoTooltip';
import db from '~/lib/db.server';
import { InfoIcon } from '@chakra-ui/icons';

export async function loader() {
  try {
    console.log('Fetching bot metrics data...');
    const metrics = await db.getBotMetrics();
    console.log(`Retrieved ${metrics.length} bot metrics records`);
    
    // Log the first record for debugging
    if (metrics.length > 0) {
      console.log('Sample metrics record:', JSON.stringify(metrics[0]));
    }
    
    return json({ metrics, error: null });
  } catch (error) {
    console.error('Error in metrics loader:', error);
    return json({ 
      metrics: [], 
      error: error instanceof Error ? error.message : 'Unknown error loading metrics data' 
    });
  }
}

function formatCurrency(value: number | string | null | undefined) {
  if (value === null || value === undefined) return '$0.00';
  const numValue = typeof value === 'string' ? parseFloat(value) : value;
  if (isNaN(numValue)) return '$0.00';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(numValue);
}

function formatPercent(value: number | string | null | undefined) {
  if (value === null || value === undefined) return '0.00%';
  const numValue = typeof value === 'string' ? parseFloat(value) : value;
  if (isNaN(numValue)) return '0.00%';
  return (numValue * 100).toFixed(2) + '%';
}

// Format for percentage change that includes a positive or negative sign
function formatPercentChange(value: number | string | null | undefined) {
  if (value === null || value === undefined) return '0.00%';
  const numValue = typeof value === 'string' ? parseFloat(value) : value;
  if (isNaN(numValue)) return '0.00%';
  const sign = numValue >= 0 ? '+' : '';
  return sign + (numValue * 100).toFixed(2) + '%';
}

export default function Metrics() {
  const { metrics, error } = useLoaderData<typeof loader>();
  const cardBg = useColorModeValue('white', 'gray.700');
  const [selectedMetric, setSelectedMetric] = useState<'win_rate' | 'profit_factor'>('win_rate');
  const [sortBy, setSortBy] = useState<string>('rank_score');
  
  // Guard against empty metrics data
  if (!metrics || metrics.length === 0) {
    return (
      <MainLayout>
        <Box p={5}>
          <Heading mb={4}>Performance Metrics</Heading>
          <Card shadow="base" bg={cardBg} p={5}>
            <CardBody>
              <VStack align="center" spacing={4}>
                <Heading size="md" color="red.500">
                  {error ? 'Error Loading Metrics' : 'No Metrics Data Available'}
                </Heading>
                <Text>
                  {error || 'There are no bot metrics records in the database yet. Please ensure your bots have completed trades.'}
                </Text>
                <Button 
                  colorScheme="blue" 
                  onClick={() => window.location.reload()}
                  mt={4}
                >
                  Refresh Data
                </Button>
              </VStack>
            </CardBody>
          </Card>
        </Box>
      </MainLayout>
    );
  }
  
  // Ensure we have unique bots by bot_id
  const uniqueBotsMap = new Map();
  metrics.forEach(bot => {
    // Only keep the bot with the most recent last_updated timestamp
    if (!uniqueBotsMap.has(bot.bot_id) || 
        new Date(bot.last_updated) > new Date(uniqueBotsMap.get(bot.bot_id).last_updated)) {
      uniqueBotsMap.set(bot.bot_id, bot);
    }
  });
  
  // Convert Map to Array for rendering
  const uniqueBots = Array.from(uniqueBotsMap.values());
  console.log(`Displaying ${uniqueBots.length} unique bots out of ${metrics.length} total records`);
  
  // Calculate system-wide metrics with safeguards
  const totalBots = uniqueBots.length;
  
  // Calculate total PNL across all bots
  const totalPnl = uniqueBots.reduce((sum, m) => {
    const pnl = m.total_pnl ? parseFloat(m.total_pnl) : 0;
    return sum + pnl;
  }, 0);
  
  // Calculate total starting capital approximation (sum of position sizes)
  const totalCapital = uniqueBots.reduce((sum, m) => {
      const size = m.position_size ? parseFloat(m.position_size) : 0;
      return sum + size;
  }, 0);
  
  // Calculate Total Return Percentage
  const totalReturnPercent = totalCapital > 0 ? totalPnl / totalCapital : 0;
  
  // Calculate win rate from avg_win_rate which is in percentage format (e.g., 29.55%)
  const getWinRate = (bot) => {
    if (bot.win_rate) return parseFloat(bot.win_rate);
    if (bot.avg_win_rate) return parseFloat(bot.avg_win_rate) / 100;
    return 0;
  };
  
  // Average win rate across all bots
  const overallWinRate = uniqueBots.reduce((sum, m) => sum + getWinRate(m), 0) / Math.max(1, uniqueBots.length);
  
  // Count profitable bots
  const profitableBots = uniqueBots.filter(m => {
    const pnl = m.total_pnl ? parseFloat(m.total_pnl) : 0;
    return pnl > 0;
  }).length;
  
  // Calculate profitable bot percentage
  const profitableBotPercentage = totalBots > 0 ? profitableBots / totalBots : 0;
  
  // Sort bots based on selected sort method
  const sortBots = (bots, sortMethod) => {
    return [...bots].sort((a, b) => {
      // Helper to safely parse float
      const safeParseFloat = (val) => parseFloat(String(val || '0'));
      
      switch (sortMethod) {
        case 'rank_score':
          return safeParseFloat(b.rank_score) - safeParseFloat(a.rank_score);
        case 'total_trades':
          return safeParseFloat(b.total_trades) - safeParseFloat(a.total_trades);
        case 'win_rate':
          return getWinRate(b) - getWinRate(a);
        case 'profit_factor':
          return safeParseFloat(b.profit_factor) - safeParseFloat(a.profit_factor);
        // Add case for Avg Return/Trade %
        case 'avg_return_per_trade':
            const avgReturnA = safeParseFloat(a.position_size) !== 0 ? safeParseFloat(a.avg_profit_per_trade) / safeParseFloat(a.position_size) : 0;
            const avgReturnB = safeParseFloat(b.position_size) !== 0 ? safeParseFloat(b.avg_profit_per_trade) / safeParseFloat(b.position_size) : 0;
            return avgReturnB - avgReturnA;
        default:
          return safeParseFloat(b.rank_score) - safeParseFloat(a.rank_score);
      }
    });
  };
  
  const sortedBots = sortBots(uniqueBots, sortBy);
  
  // Get top 5 bots for comparison
  const topBots = sortedBots.slice(0, 5).map(bot => ({
    bot_id: bot.bot_id,
    name: `Bot ${bot.bot_id}`,
    win_rate: getWinRate(bot),
    profit_factor: parseFloat(bot.profit_factor || '0'),
    rank_score: parseFloat(bot.rank_score || '0'),
    expectancy: parseFloat(bot.r_multiple || '0'),
  }));
  
  const handleSortChange = (e) => {
    setSortBy(e.target.value);
  };
  
  return (
    <MainLayout>
      <Flex justify="space-between" align="center" mb={8}>
        <Heading>Performance Metrics</Heading>
        <Button
          as={Link}
          to="/metrics/documentation"
          rightIcon={<InfoIcon />}
          colorScheme="blue"
          variant="outline"
          size="sm"
        >
          Metrics Documentation
        </Button>
      </Flex>
      
      {/* System Overview Stats */}
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6} mb={8}>
        <Stat
          px={4}
          py={5}
          shadow="base"
          borderWidth="1px"
          borderRadius="lg"
          bg={cardBg}
        >
          <StatLabel fontSize="md">
            <MetricInfoTooltip metricKey="total_pnl">Total P&L</MetricInfoTooltip>
          </StatLabel>
          <StatNumber fontSize="3xl" color={totalPnl >= 0 ? 'green.500' : 'red.500'}>
            {formatCurrency(totalPnl)}
          </StatNumber>
          <StatHelpText>
            <StatArrow type={totalPnl >= 0 ? 'increase' : 'decrease'} />
            {profitableBots} out of {totalBots} bots profitable
          </StatHelpText>
        </Stat>
        
        <Stat
          px={4}
          py={5}
          shadow="base"
          borderWidth="1px"
          borderRadius="lg"
          bg={cardBg}
        >
          <StatLabel fontSize="md">
            <MetricInfoTooltip metricKey="total_pnl_percent">Total Return %</MetricInfoTooltip>
          </StatLabel>
          <StatNumber fontSize="3xl" color={totalReturnPercent >= 0 ? 'green.500' : 'red.500'}>
            {formatPercentChange(totalReturnPercent)}
          </StatNumber>
          <StatHelpText>
            <StatArrow type={totalReturnPercent >= 0 ? 'increase' : 'decrease'} />
            Based on Total PnL / Sum of Position Sizes
          </StatHelpText>
        </Stat>
        
        <Stat
          px={4}
          py={5}
          shadow="base"
          borderWidth="1px"
          borderRadius="lg"
          bg={cardBg}
        >
          <StatLabel fontSize="md">
            <MetricInfoTooltip metricKey="profitable_bots">Profitable Bots</MetricInfoTooltip>
          </StatLabel>
          <StatNumber fontSize="3xl">{formatPercent(profitableBotPercentage)}</StatNumber>
          <StatHelpText>
            {profitableBots} out of {totalBots} profitable
          </StatHelpText>
        </Stat>
        
        <Stat
          px={4}
          py={5}
          shadow="base"
          borderWidth="1px"
          borderRadius="lg"
          bg={cardBg}
        >
          <StatLabel fontSize="md">
            <MetricInfoTooltip metricKey="win_rate">Avg Win Rate</MetricInfoTooltip>
          </StatLabel>
          <StatNumber fontSize="3xl">{formatPercent(overallWinRate)}</StatNumber>
          <StatHelpText>
            Across all bots
          </StatHelpText>
        </Stat>
      </SimpleGrid>
      
      {/* Performance Metrics Table */}
      <Card shadow="base" bg={cardBg}>
        <CardHeader pb={0}>
          <Flex justify="space-between" align="center">
            <Heading size="md">Bot Performance Metrics</Heading>
            <HStack>
              <Text fontSize="sm">Sort by:</Text>
              <Select size="sm" w="150px" value={sortBy} onChange={handleSortChange}>
                <option value="rank_score">Rank Score</option>
                <option value="total_trades">Total Trades</option>
                <option value="win_rate">Win Rate</option>
                <option value="profit_factor">Profit Factor</option>
                <option value="avg_return_per_trade">Avg Return/Trade (%)</option>
              </Select>
            </HStack>
          </Flex>
        </CardHeader>
        <CardBody>
          <Box overflowX="auto">
            <Table variant="simple">
              <Thead>
                <Tr>
                  <Th cursor="pointer" onClick={() => setSortBy('rank_score')}>Rank Score</Th>
                  <Th>Bot ID</Th>
                  <Th>Ticker</Th>
                  <Th cursor="pointer" onClick={() => setSortBy('win_rate')}>Win Rate</Th>
                  <Th cursor="pointer" onClick={() => setSortBy('avg_return_per_trade')}>Avg Return/Trade (%)</Th>
                  <Th cursor="pointer" onClick={() => setSortBy('profit_factor')}>Profit Factor</Th>
                  <Th cursor="pointer" onClick={() => setSortBy('total_trades')}>Total Trades</Th>
                </Tr>
              </Thead>
              <Tbody>
                {sortedBots.map((bot) => {
                  // Calculate Avg Return %
                  const positionSize = parseFloat(bot.position_size || '0');
                  const avgProfit = parseFloat(bot.avg_profit_per_trade || '0');
                  const avgReturnPercent = positionSize !== 0 ? avgProfit / positionSize : 0;
                  
                  return (
                    <Tr key={bot.bot_id} _hover={{ bg: useColorModeValue('gray.50', 'gray.600') }}>
                      <Td>{parseFloat(bot.rank_score || '0').toFixed(2)}</Td>
                      <Td>
                        <Link to={`/bots/${bot.bot_id}`}>
                          <Button variant="link" size="sm">{bot.bot_id}</Button>
                        </Link>
                      </Td>
                      <Td>{bot.ticker}</Td>
                      <Td>{formatPercent(getWinRate(bot))}</Td>
                      <Td color={avgReturnPercent >= 0 ? 'green.500' : 'red.500'}>
                        {formatPercentChange(avgReturnPercent)}
                      </Td>
                      <Td>{parseFloat(bot.profit_factor || '0').toFixed(2)}</Td>
                      <Td>{bot.total_trades}</Td>
                    </Tr>
                  );
                })}
              </Tbody>
            </Table>
          </Box>
        </CardBody>
      </Card>
      
      {/* Performance Metrics Visualization */}
      <Box mt={8}>
        <Tabs variant="enclosed">
          <TabList>
            <Tab>Bot Performance Metrics</Tab>
            <Tab>Bot Comparison</Tab>
            <Tab as={Link} to="/metrics/documentation?tab=relationships">
              Metric Relationships
            </Tab>
          </TabList>
          
          <TabPanels>
            <TabPanel px={0}>
              <Card shadow="base" bg={cardBg}>
                <CardHeader>
                  <Flex justify="space-between" align="center">
                    <Heading size="md">Bot Performance Metrics</Heading>
                    <HStack>
                      <Text>Metric:</Text>
                      <Select 
                        value={selectedMetric} 
                        onChange={(e) => setSelectedMetric(e.target.value as any)}
                        w="180px"
                      >
                        <option value="win_rate">Win Rate</option>
                        <option value="profit_factor">Profit Factor</option>
                      </Select>
                    </HStack>
                  </Flex>
                </CardHeader>
                <CardBody>
                  <PerformanceMetricsChart 
                    data={topBots} 
                    metric={selectedMetric} 
                  />
                </CardBody>
              </Card>
            </TabPanel>
            
            <TabPanel px={0}>
              <Card shadow="base" bg={cardBg}>
                <CardHeader>
                  <Flex justify="space-between" align="center">
                    <Heading size="md">Top Bot Comparison</Heading>
                    <Button 
                      as={Link} 
                      to="/metrics/compare" 
                      size="sm" 
                      colorScheme="blue"
                    >
                      Create Custom Comparison
                    </Button>
                  </Flex>
                </CardHeader>
                <CardBody>
                  <BotComparisonChart bots={topBots} />
                </CardBody>
              </Card>
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Box>
    </MainLayout>
  );
}