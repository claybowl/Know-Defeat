import { json, LoaderFunctionArgs } from '@remix-run/node';
import { useLoaderData, Link } from '@remix-run/react';
import {
  Box,
  Heading,
  Text,
  Badge,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Card,
  CardHeader,
  CardBody,
  Button,
  HStack,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Flex,
  useColorModeValue,
  VStack,
} from '@chakra-ui/react';
import { useState } from 'react';
import { getBotById } from '~/lib/api.server';
import MainLayout from '~/components/layout/MainLayout';
import TradeHistoryChart from '~/components/charts/TradeHistoryChart';
import PerformanceMetricsChart from '~/components/charts/PerformanceMetricsChart';

export async function loader({ params }: LoaderFunctionArgs) {
  const botId = params.botId;
  if (!botId || isNaN(parseInt(botId))) {
    throw new Response('Bot ID is required', { status: 400 });
  }
  
  const bot = await getBotById(parseInt(botId));
  if (!bot) {
    throw new Response('Bot not found', { status: 404 });
  }
  
  return json({ bot });
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

export default function BotDetail() {
  const { bot } = useLoaderData<typeof loader>();
  const cardBg = useColorModeValue('white', 'gray.700');
  const [selectedMetric, setSelectedMetric] = useState<'win_rate' | 'profit_factor' | 'sharpe_ratio' | 'max_drawdown'>('win_rate');
  
  // Prepare bot metrics data for the performance chart if metrics exist
  const metricData = bot.metrics 
    ? [{
        bot_id: bot.bot_id,
        name: bot.name,
        win_rate: parseFloat(bot.metrics.win_rate),
        profit_factor: parseFloat(bot.metrics.profit_factor),
        sharpe_ratio: parseFloat(bot.metrics.sharpe_ratio),
        max_drawdown: parseFloat(bot.metrics.max_drawdown),
      }]
    : [];
  
  return (
    <MainLayout>
      <Flex justify="space-between" align="center" mb={6}>
        <Box>
          <Heading size="lg">{bot.name}</Heading>
          <Text fontSize="md" color="gray.500">ID: {bot.bot_id}</Text>
        </Box>
        <HStack>
          <Button colorScheme={bot.is_active ? 'red' : 'green'}>
            {bot.is_active ? 'Disable Bot' : 'Enable Bot'}
          </Button>
          <Button colorScheme="blue">Edit Configuration</Button>
        </HStack>
      </Flex>
      
      {/* Bot Information Card */}
      <Card mb={8} bg={cardBg} shadow="md">
        <CardHeader pb={2}>
          <Heading size="md">Bot Configuration</Heading>
        </CardHeader>
        <CardBody>
          <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={4}>
            <Box>
              <Text fontWeight="bold">Symbol</Text>
              <Text>{bot.ticker}</Text>
            </Box>
            <Box>
              <Text fontWeight="bold">Algorithm Type</Text>
              <Text>{bot.algorithm_type}</Text>
            </Box>
            <Box>
              <Text fontWeight="bold">Trade Direction</Text>
              <Badge colorScheme={
                bot.trade_direction === 'LONG' ? 'green' :
                bot.trade_direction === 'SHORT' ? 'red' : 'blue'
              }>
                {bot.trade_direction}
              </Badge>
            </Box>
            <Box>
              <Text fontWeight="bold">Position Size</Text>
              <Text>{formatCurrency(bot.position_size)}</Text>
            </Box>
            <Box>
              <Text fontWeight="bold">Trailing Stop</Text>
              <Text>{formatPercent(bot.trailing_stop_pct)}</Text>
            </Box>
            <Box>
              <Text fontWeight="bold">Status</Text>
              <Badge colorScheme={bot.is_active ? 'green' : 'gray'}>
                {bot.is_active ? 'Active' : 'Inactive'}
              </Badge>
            </Box>
            <Box>
              <Text fontWeight="bold">Version</Text>
              <Text>{bot.version || '1.0'}</Text>
            </Box>
            <Box>
              <Text fontWeight="bold">Created</Text>
              <Text>{new Date(bot.created_at).toLocaleDateString()}</Text>
            </Box>
          </SimpleGrid>
        </CardBody>
      </Card>
      
      {/* Performance Stats */}
      {bot.metrics && (
        <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6} mb={8}>
          <Stat
            px={4}
            py={5}
            shadow="base"
            borderWidth="1px"
            borderRadius="lg"
            bg={cardBg}
          >
            <StatLabel fontSize="md">Total Trades</StatLabel>
            <StatNumber fontSize="3xl">{bot.metrics.total_trades}</StatNumber>
            <StatHelpText>
              {bot.metrics.winning_trades} wins / {bot.metrics.losing_trades} losses
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
            <StatNumber fontSize="3xl">{formatPercent(bot.metrics.win_rate)}</StatNumber>
            <StatHelpText>
              <StatArrow type={bot.metrics.win_rate >= 0.5 ? 'increase' : 'decrease'} />
              {bot.metrics.winning_trades} out of {bot.metrics.total_trades}
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
            <StatLabel fontSize="md">Total P&L</StatLabel>
            <StatNumber 
              fontSize="3xl" 
              color={parseFloat(bot.metrics.total_pnl) >= 0 ? 'green.500' : 'red.500'}
            >
              {formatCurrency(bot.metrics.total_pnl)}
            </StatNumber>
            <StatHelpText>
              <StatArrow type={parseFloat(bot.metrics.total_pnl) >= 0 ? 'increase' : 'decrease'} />
              Avg: {formatCurrency(bot.metrics.average_pnl_per_trade)} / trade
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
            <StatLabel fontSize="md">Profit Factor</StatLabel>
            <StatNumber fontSize="3xl">{parseFloat(bot.metrics.profit_factor).toFixed(2)}</StatNumber>
            <StatHelpText>
              <StatArrow type={parseFloat(bot.metrics.profit_factor) >= 1 ? 'increase' : 'decrease'} />
              Risk/Reward: {parseFloat(bot.metrics.risk_reward_ratio).toFixed(2)}
            </StatHelpText>
          </Stat>
        </SimpleGrid>
      )}
      
      {/* Tabs for Trades / Parameters */}
      <Tabs colorScheme="brand" shadow="md" bg={cardBg} borderRadius="lg">
        <TabList>
          <Tab>Trade History</Tab>
          <Tab>Parameters</Tab>
          <Tab>Performance</Tab>
        </TabList>
        
        <TabPanels>
          {/* Trade History Tab */}
          <TabPanel>
            {/* Trade Visualization */}
            {bot.trades && bot.trades.length > 0 && (
              <Card shadow="md" bg={cardBg} mb={6}>
                <CardHeader pb={0}>
                  <Heading size="md">Trade Performance Visualization</Heading>
                </CardHeader>
                <CardBody>
                  <TradeHistoryChart 
                    trades={bot.trades.filter(t => t.trade_status === 'closed' && t.pnl !== null)} 
                  />
                </CardBody>
              </Card>
            )}
            
            {/* Trades Table */}
            <Card shadow="md" bg={cardBg}>
              <CardHeader pb={0}>
                <Flex justify="space-between" align="center">
                  <Heading size="md">Trades History</Heading>
                  <HStack>
                    <Button size="sm" colorScheme="blue">
                      Export Trades
                    </Button>
                  </HStack>
                </Flex>
              </CardHeader>
              <CardBody>
                <Box overflowX="auto">
                  <Table variant="simple" size="sm">
                    <Thead>
                      <Tr>
                        <Th>Trade ID</Th>
                        <Th>Direction</Th>
                        <Th>Entry Price</Th>
                        <Th>Exit Price</Th>
                        <Th>Entry Time</Th>
                        <Th>Exit Time</Th>
                        <Th>Status</Th>
                        <Th>P&L</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {bot.trades && bot.trades.map((trade: any) => (
                        <Tr key={trade.trade_id}>
                          <Td>{trade.trade_id}</Td>
                          <Td>
                            <Badge
                              colorScheme={trade.trade_direction === 'LONG' ? 'green' : 'red'}
                            >
                              {trade.trade_direction}
                            </Badge>
                          </Td>
                          <Td>${parseFloat(trade.entry_price).toFixed(2)}</Td>
                          <Td>
                            {trade.exit_price 
                              ? '$' + parseFloat(trade.exit_price).toFixed(2)
                              : '-'
                            }
                          </Td>
                          <Td>{new Date(trade.entry_time).toLocaleString()}</Td>
                          <Td>
                            {trade.exit_time 
                              ? new Date(trade.exit_time).toLocaleString()
                              : '-'
                            }
                          </Td>
                          <Td>
                            <Badge
                              colorScheme={
                                trade.trade_status === 'open'
                                  ? 'blue'
                                  : trade.trade_status === 'closed' && parseFloat(trade.pnl || 0) > 0
                                  ? 'green'
                                  : 'red'
                              }
                            >
                              {trade.trade_status}
                            </Badge>
                          </Td>
                          <Td 
                            color={
                              trade.pnl !== null
                                ? parseFloat(trade.pnl) >= 0 
                                  ? 'green.500' 
                                  : 'red.500'
                                : 'inherit'
                            }
                          >
                            {trade.pnl !== null
                              ? formatCurrency(trade.pnl)
                              : '-'
                            }
                          </Td>
                        </Tr>
                      ))}
                      {(!bot.trades || bot.trades.length === 0) && (
                        <Tr>
                          <Td colSpan={8} textAlign="center">No trades found</Td>
                        </Tr>
                      )}
                    </Tbody>
                  </Table>
                </Box>
              </CardBody>
            </Card>
          </TabPanel>
          
          {/* Parameters Tab */}
          <TabPanel>
            <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6}>
              {bot.parameters && Object.entries(bot.parameters).map(([key, value]) => (
                <Box key={key} p={4} borderWidth="1px" borderRadius="md">
                  <Text fontWeight="bold" mb={1} textTransform="capitalize">
                    {key.replace(/_/g, ' ')}
                  </Text>
                  <Text>{value}</Text>
                </Box>
              ))}
              {(!bot.parameters || Object.keys(bot.parameters).length === 0) && (
                <Text>No parameters found</Text>
              )}
            </SimpleGrid>
          </TabPanel>
          
          {/* Performance Tab */}
          <TabPanel>
            <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={6}>
              <Card shadow="md" bg={cardBg}>
                <CardHeader pb={0}>
                  <Heading size="md">Trade History Performance</Heading>
                </CardHeader>
                <CardBody>
                  {bot.trades && bot.trades.length > 0 ? (
                    <TradeHistoryChart 
                      trades={bot.trades.filter((t: any) => t.trade_status === 'closed' && t.pnl !== null)} 
                    />
                  ) : (
                    <Text>No closed trades found to display performance</Text>
                  )}
                </CardBody>
              </Card>
              
              <Card shadow="md" bg={cardBg}>
                <CardHeader pb={0}>
                  <Heading size="md">Key Performance Metrics</Heading>
                </CardHeader>
                <CardBody>
                  {bot.metrics ? (
                    <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                      <Box>
                        <Text fontWeight="bold">Win Rate</Text>
                        <Text fontSize="2xl">{formatPercent(bot.metrics.win_rate)}</Text>
                      </Box>
                      <Box>
                        <Text fontWeight="bold">Profit Factor</Text>
                        <Text fontSize="2xl">{parseFloat(bot.metrics.profit_factor).toFixed(2)}</Text>
                      </Box>
                      <Box>
                        <Text fontWeight="bold">Average Win</Text>
                        <Text fontSize="2xl" color="green.500">
                          {formatCurrency(bot.metrics.average_win_amount)}
                        </Text>
                      </Box>
                      <Box>
                        <Text fontWeight="bold">Average Loss</Text>
                        <Text fontSize="2xl" color="red.500">
                          {formatCurrency(bot.metrics.average_loss_amount)}
                        </Text>
                      </Box>
                      <Box>
                        <Text fontWeight="bold">Max Drawdown</Text>
                        <Text fontSize="2xl">
                          {formatCurrency(bot.metrics.max_drawdown)}
                        </Text>
                      </Box>
                      <Box>
                        <Text fontWeight="bold">Sharpe Ratio</Text>
                        <Text fontSize="2xl">
                          {parseFloat(bot.metrics.sharpe_ratio).toFixed(2)}
                        </Text>
                      </Box>
                    </SimpleGrid>
                  ) : (
                    <Text>No metrics available</Text>
                  )}
                </CardBody>
              </Card>
              
              {bot.metrics && (
                <Card shadow="md" bg={cardBg} gridColumn={{ base: "auto", lg: "1 / span 2" }}>
                  <CardHeader pb={0}>
                    <Heading size="md">Bot Performance Analysis</Heading>
                  </CardHeader>
                  <CardBody>
                    <Text mb={4}>
                      This bot {bot.metrics.win_rate >= 0.5 ? 'has shown consistent profitability' : 'needs improvement'} with 
                      a {formatPercent(bot.metrics.win_rate)} win rate across {bot.metrics.total_trades} trades. 
                      The profit factor of {parseFloat(bot.metrics.profit_factor).toFixed(2)} indicates 
                      {bot.metrics.profit_factor >= 1.5 ? ' strong performance with winners significantly outpacing losers.' : 
                       bot.metrics.profit_factor >= 1 ? ' positive but modest performance.' : ' that the strategy needs optimization.'}
                    </Text>
                    
                    <Text>
                      The risk-reward ratio of {parseFloat(bot.metrics.risk_reward_ratio).toFixed(2)} shows
                      {bot.metrics.risk_reward_ratio >= 1.5 ? ' excellent risk management.' : 
                       bot.metrics.risk_reward_ratio >= 1 ? ' balanced risk taking.' : ' higher risk relative to rewards.'}
                      With a Sharpe ratio of {parseFloat(bot.metrics.sharpe_ratio).toFixed(2)}, the bot's returns
                      {bot.metrics.sharpe_ratio >= 1.5 ? ' are strong compared to its volatility.' : 
                       bot.metrics.sharpe_ratio >= 1 ? ' are acceptable given its volatility.' : ' need to be improved relative to risk taken.'}
                    </Text>
                  </CardBody>
                </Card>
              )}
            </SimpleGrid>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </MainLayout>
  );
}