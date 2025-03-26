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
import db from '~/lib/db.server';

export async function loader() {
  const metrics = await db.getBotMetrics();
  return json({ metrics });
}

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

export default function Metrics() {
  const { metrics } = useLoaderData<typeof loader>();
  const cardBg = useColorModeValue('white', 'gray.700');
  const [selectedMetric, setSelectedMetric] = useState<'win_rate' | 'profit_factor' | 'sharpe_ratio' | 'max_drawdown'>('win_rate');
  
  // Calculate system-wide metrics
  const totalBots = metrics.length;
  const botsWithTrades = metrics.filter(m => m.total_trades > 0);
  const totalTrades = botsWithTrades.reduce((sum, m) => sum + m.total_trades, 0);
  const totalPnl = botsWithTrades.reduce((sum, m) => sum + parseFloat(m.total_pnl), 0);
  
  const totalWinningTrades = botsWithTrades.reduce((sum, m) => sum + m.winning_trades, 0);
  const overallWinRate = totalTrades > 0 ? totalWinningTrades / totalTrades : 0;
  
  const profitableBots = botsWithTrades.filter(m => parseFloat(m.total_pnl) > 0).length;
  const profitableBotPercentage = botsWithTrades.length > 0 ? profitableBots / botsWithTrades.length : 0;
  
  // Get top 5 bots for comparison
  const topBots = [...botsWithTrades]
    .sort((a, b) => parseFloat(b.rank_score) - parseFloat(a.rank_score))
    .slice(0, 5)
    .map(bot => ({
      bot_id: bot.bot_id,
      name: `Bot ${bot.bot_id}`,
      win_rate: parseFloat(bot.win_rate),
      profit_factor: parseFloat(bot.profit_factor),
      sharpe_ratio: parseFloat(bot.sharpe_ratio),
      max_drawdown: parseFloat(bot.max_drawdown),
      expectancy: parseFloat(bot.expectancy || 0),
    }));
  
  return (
    <MainLayout>
      <Heading mb={8}>Performance Metrics</Heading>
      
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
          <StatLabel fontSize="md">Total P&L</StatLabel>
          <StatNumber fontSize="3xl" color={totalPnl >= 0 ? 'green.500' : 'red.500'}>
            {formatCurrency(totalPnl)}
          </StatNumber>
          <StatHelpText>
            <StatArrow type={totalPnl >= 0 ? 'increase' : 'decrease'} />
            {profitableBots} out of {botsWithTrades.length} bots profitable
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
          <StatLabel fontSize="md">Win Rate</StatLabel>
          <StatNumber fontSize="3xl">{formatPercent(overallWinRate)}</StatNumber>
          <StatHelpText>
            <StatArrow type={overallWinRate >= 0.5 ? 'increase' : 'decrease'} />
            {totalWinningTrades} out of {totalTrades} trades
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
          <StatLabel fontSize="md">Average P&L Per Trade</StatLabel>
          <StatNumber fontSize="3xl">
            {formatCurrency(totalTrades > 0 ? totalPnl / totalTrades : 0)}
          </StatNumber>
          <StatHelpText>
            Based on {totalTrades} closed trades
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
            {profitableBots} out of {botsWithTrades.length} bots
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
              <Select size="sm" w="150px" defaultValue="rank">
                <option value="rank">Rank Score</option>
                <option value="pnl">Total P&L</option>
                <option value="win_rate">Win Rate</option>
                <option value="profit_factor">Profit Factor</option>
                <option value="trades">Number of Trades</option>
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
                  <Th>Trades</Th>
                  <Th>Win Rate</Th>
                  <Th>Profit Factor</Th>
                  <Th>Avg Win</Th>
                  <Th>Avg Loss</Th>
                  <Th>Max Drawdown</Th>
                  <Th>Sharpe Ratio</Th>
                  <Th>Total P&L</Th>
                </Tr>
              </Thead>
              <Tbody>
                {metrics
                  .filter(bot => bot.total_trades > 0)
                  .sort((a, b) => parseFloat(b.rank_score) - parseFloat(a.rank_score))
                  .map((bot, index) => (
                    <Tr key={bot.bot_id}>
                      <Td>{index + 1}</Td>
                      <Td>Bot {bot.bot_id}</Td>
                      <Td>
                        {bot.total_trades} ({bot.winning_trades}/{bot.losing_trades})
                      </Td>
                      <Td>
                        <Badge colorScheme={parseFloat(bot.win_rate) >= 0.5 ? 'green' : 'red'}>
                          {formatPercent(bot.win_rate)}
                        </Badge>
                      </Td>
                      <Td>{parseFloat(bot.profit_factor).toFixed(2)}</Td>
                      <Td color="green.500">{formatCurrency(bot.average_win_amount)}</Td>
                      <Td color="red.500">{formatCurrency(bot.average_loss_amount)}</Td>
                      <Td>{formatCurrency(bot.max_drawdown)}</Td>
                      <Td>{parseFloat(bot.sharpe_ratio).toFixed(2)}</Td>
                      <Td 
                        color={parseFloat(bot.total_pnl) >= 0 ? 'green.500' : 'red.500'}
                        fontWeight="bold"
                      >
                        {formatCurrency(bot.total_pnl)}
                      </Td>
                    </Tr>
                  ))}
                {metrics.filter(bot => bot.total_trades > 0).length === 0 && (
                  <Tr>
                    <Td colSpan={10} textAlign="center">No metrics data available</Td>
                  </Tr>
                )}
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