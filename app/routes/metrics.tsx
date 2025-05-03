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
  const [selectedMetric, setSelectedMetric] = useState<'win_rate' | 'profit_factor' | 'sharpe_ratio' | 'max_drawdown'>('win_rate');
  const [sortBy, setSortBy] = useState<string>('pnl');
  
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
      switch (sortMethod) {
        case 'pnl':
          return parseFloat(b.total_pnl || 0) - parseFloat(a.total_pnl || 0);
        case 'win_rate':
          return getWinRate(b) - getWinRate(a);
        case 'profit_factor':
          return parseFloat(b.profit_factor || 0) - parseFloat(a.profit_factor || 0);
        case 'sharpe':
          return parseFloat(b.sharpe_ratio || 0) - parseFloat(a.sharpe_ratio || 0);
        default:
          return parseFloat(b.total_pnl || 0) - parseFloat(a.total_pnl || 0);
      }
    });
  };
  
  const sortedBots = sortBots(uniqueBots, sortBy);
  
  // Get top 5 bots for comparison
  const topBots = sortedBots.slice(0, 5).map(bot => ({
    bot_id: bot.bot_id,
    name: `Bot ${bot.bot_id}`,
    win_rate: getWinRate(bot),
    profit_factor: parseFloat(bot.profit_factor || 0),
    sharpe_ratio: parseFloat(bot.sharpe_ratio || 0),
    max_drawdown: parseFloat(bot.max_drawdown || 0),
    expectancy: parseFloat(bot.r_multiple || 0),
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
            <MetricInfoTooltip metricKey="total_pnl_percent">Total P&L %</MetricInfoTooltip>
          </StatLabel>
          <StatNumber fontSize="3xl" color={totalPnl >= 0 ? 'green.500' : 'red.500'}>
            {/* Calculate an approximate percentage based on average bot position size */}
            {formatPercentChange(uniqueBots.reduce((sum, bot) => {
              // Calculate percentage based on drawdown_percent if available
              if (bot.drawdown_percent) {
                return sum + parseFloat(bot.drawdown_percent);
              }
              return sum;
            }, 0) / Math.max(1, uniqueBots.length))}
          </StatNumber>
          <StatHelpText>
            <StatArrow type={totalPnl >= 0 ? 'increase' : 'decrease'} />
            Average percent change across portfolio
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
            <MetricInfoTooltip metricKey="win_rate">Win Rate</MetricInfoTooltip>
          </StatLabel>
          <StatNumber fontSize="3xl">{formatPercent(overallWinRate)}</StatNumber>
          <StatHelpText>
            <StatArrow type={overallWinRate >= 0.5 ? 'increase' : 'decrease'} />
            Average across all bots
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
          <StatLabel fontSize="md">Profitable Bots</StatLabel>
          <StatNumber fontSize="3xl">{formatPercent(profitableBotPercentage)}</StatNumber>
          <StatHelpText>
            <StatArrow type={profitableBotPercentage >= 0.5 ? 'increase' : 'decrease'} />
            {profitableBots} out of {totalBots} bots
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
                <option value="pnl">Total P&L</option>
                <option value="win_rate">Win Rate</option>
                <option value="profit_factor">Profit Factor</option>
                <option value="sharpe">Sharpe Ratio</option>
              </Select>
            </HStack>
          </Flex>
        </CardHeader>
        <CardBody>
          <Box overflowX="auto">
            <Table variant="simple">
              <Thead>
                <Tr>
                  <Th>Rank</Th>
                  <Th>Bot ID</Th>
                  <Th>Ticker</Th>
                  <Th><MetricInfoTooltip metricKey="win_rate">Win Rate</MetricInfoTooltip></Th>
                  <Th><MetricInfoTooltip metricKey="profit_factor">Profit Factor</MetricInfoTooltip></Th>
                  <Th><MetricInfoTooltip metricKey="average_win_amount">Avg Profit</MetricInfoTooltip></Th>
                  <Th><MetricInfoTooltip metricKey="average_loss_amount">Avg Drawdown</MetricInfoTooltip></Th>
                  <Th><MetricInfoTooltip metricKey="max_drawdown">Max Drawdown</MetricInfoTooltip></Th>
                  <Th><MetricInfoTooltip metricKey="sharpe_ratio">Sharpe Ratio</MetricInfoTooltip></Th>
                  <Th><MetricInfoTooltip metricKey="total_pnl">Total P&L</MetricInfoTooltip></Th>
                </Tr>
              </Thead>
              <Tbody>
                {sortedBots.map((bot, index) => (
                  <Tr key={bot.bot_id}>
                    <Td>{index + 1}</Td>
                    <Td>Bot {bot.bot_id}</Td>
                    <Td>{bot.ticker || '-'}</Td>
                    <Td>
                      <Badge colorScheme={getWinRate(bot) >= 0.5 ? 'green' : 'red'}>
                        {formatPercent(getWinRate(bot))}
                      </Badge>
                    </Td>
                    <Td>{parseFloat(bot.profit_factor || 0).toFixed(2)}</Td>
                    <Td color="green.500">
                      {/* Display average win as a percentage */}
                      {formatPercentChange(bot.pnl_percent || 
                        (bot.avg_profit_per_trade && bot.position_size ? 
                          parseFloat(bot.avg_profit_per_trade) / parseFloat(bot.position_size || 1000) : 
                          0.01))}
                    </Td>
                    <Td color="red.500">
                      {/* Display average loss as a percentage */}
                      {formatPercentChange(bot.avg_drawdown ? 
                        -Math.abs(parseFloat(bot.avg_drawdown) / 100) : 
                        -0.01)}
                    </Td>
                    <Td color="red.500">
                      {/* Display max drawdown as a percentage */}
                      {formatPercentChange(bot.drawdown_percent ? 
                        -Math.abs(parseFloat(bot.drawdown_percent) / 100) :
                        (bot.max_drawdown ? 
                          -Math.abs(parseFloat(bot.max_drawdown) / 100) : 
                          -0.01))}
                    </Td>
                    <Td>{parseFloat(bot.sharpe_ratio || 0).toFixed(2)}</Td>
                    <Td 
                      color={parseFloat(bot.total_pnl || 0) >= 0 ? 'green.500' : 'red.500'}
                      fontWeight="bold"
                    >
                      {/* Display total PnL as both currency and percentage */}
                      <VStack spacing={0} align="flex-start">
                        <Text>{formatCurrency(bot.total_pnl)}</Text>
                        <Text fontSize="xs">
                          {formatPercentChange(bot.pnl_percent || 
                            (bot.total_pnl && bot.position_size ? 
                              parseFloat(bot.total_pnl) / parseFloat(bot.position_size || 10000) : 
                              0))}
                        </Text>
                      </VStack>
                    </Td>
                  </Tr>
                ))}
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
                        <option value="sharpe_ratio">Sharpe Ratio</option>
                        <option value="max_drawdown">Max Drawdown</option>
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