import React from 'react';
import {
  Box,
  Container,
  Heading,
  Text,
  SimpleGrid,
  Flex,
  Tag,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Card,
  CardHeader,
  CardBody,
  TabList,
  Tabs,
  Tab,
  TabPanels,
  TabPanel,
  Button,
  HStack,
  Icon,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  useColorModeValue,
} from '@chakra-ui/react';
import { FiSettings, FiActivity, FiPieChart, FiBarChart2, FiDollarSign } from 'react-icons/fi';
import { useLoaderData, useNavigate } from '@remix-run/react';
import MainLayout from '~/components/layout/MainLayout';
import { getBotById } from '~/lib/api.server';
import TradeHistoryChart from '~/components/charts/TradeHistoryChart';
import PerformanceMetricsChart from '~/components/charts/PerformanceMetricsChart';
import ParameterRadarChart from '~/components/charts/ParameterRadarChart';

export async function loader({ params }) {
  const bot = await getBotById(parseInt(params.botId));
  
  if (!bot) {
    throw new Response("Bot not found", { status: 404 });
  }
  
  return { bot };
}

export default function BotDetail() {
  const { bot } = useLoaderData();
  const navigate = useNavigate();
  const cardBg = useColorModeValue('white', 'gray.700');
  
  // Prepare data for charts
  const tradeHistoryData = bot.trades.map(trade => ({
    id: trade.trade_id,
    date: new Date(trade.entry_time).toLocaleDateString(),
    pnl: trade.pnl || 0,
    direction: trade.trade_direction,
    status: trade.trade_status,
    ticker: trade.ticker,
    entryPrice: trade.entry_price,
    exitPrice: trade.exit_price || 0,
  }));
  
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };
  
  const formatPercent = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };
  
  return (
    <MainLayout>
      <Box mb={8}>
        <Flex justify="space-between" align="center" mb={4}>
          <Box>
            <Heading size="lg">{bot.name}</Heading>
            <HStack mt={2} spacing={2} align="center">
              <Tag colorScheme="blue" size="md">{bot.ticker}</Tag>
              <Tag colorScheme="purple" size="md">{bot.algorithm_type.replace(/_/g, ' ')}</Tag>
              <Tag 
                colorScheme={bot.is_active ? "green" : "gray"} 
                size="md"
              >
                {bot.is_active ? "Active" : "Inactive"}
              </Tag>
            </HStack>
          </Box>
          <Button 
            leftIcon={<Icon as={FiSettings} />}
            colorScheme="blue"
            onClick={() => {
              // This would open an edit modal in a real implementation
              alert('Edit functionality would go here!');
            }}
          >
            Edit Bot
          </Button>
        </Flex>
        
        <Text color="gray.500" mt={2}>{bot.description}</Text>
      </Box>
      
      {/* Performance Overview */}
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6} mb={8}>
        <Stat
          px={4}
          py={5}
          shadow="sm"
          borderWidth="1px"
          borderRadius="lg"
          bg={cardBg}
        >
          <StatLabel fontSize="sm">Win Rate</StatLabel>
          <Flex align="center">
            <Icon as={FiActivity} color="green.500" mr={2} />
            <StatNumber fontSize="2xl">
              {bot.metrics?.win_rate 
                ? formatPercent(bot.metrics.win_rate)
                : "N/A"}
            </StatNumber>
          </Flex>
          <StatHelpText>
            {bot.metrics?.total_trades || 0} total trades
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
          <StatLabel fontSize="sm">Profit Factor</StatLabel>
          <Flex align="center">
            <Icon as={FiPieChart} color="purple.500" mr={2} />
            <StatNumber fontSize="2xl">
              {bot.metrics?.profit_factor 
                ? parseFloat(bot.metrics.profit_factor).toFixed(2)
                : "N/A"}
            </StatNumber>
          </Flex>
          <StatHelpText>
            Profit / Loss ratio
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
          <StatLabel fontSize="sm">Total P&L</StatLabel>
          <Flex align="center">
            <Icon as={FiDollarSign} color={bot.metrics?.total_pnl >= 0 ? "green.500" : "red.500"} mr={2} />
            <StatNumber fontSize="2xl">
              {bot.metrics?.total_pnl 
                ? formatCurrency(bot.metrics.total_pnl)
                : "$0.00"}
            </StatNumber>
          </Flex>
          <StatHelpText>
            All closed trades
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
          <StatLabel fontSize="sm">Sharpe Ratio</StatLabel>
          <Flex align="center">
            <Icon as={FiBarChart2} color="blue.500" mr={2} />
            <StatNumber fontSize="2xl">
              {bot.metrics?.sharpe_ratio 
                ? parseFloat(bot.metrics.sharpe_ratio).toFixed(2)
                : "N/A"}
            </StatNumber>
          </Flex>
          <StatHelpText>
            Risk-adjusted return
          </StatHelpText>
        </Stat>
      </SimpleGrid>
      
      {/* Tab Section */}
      <Box bg={cardBg} borderRadius="lg" shadow="sm" mb={8}>
        <Tabs variant="enclosed">
          <TabList px={4} pt={4}>
            <Tab>Performance</Tab>
            <Tab>Trade History</Tab>
            <Tab>Parameters</Tab>
            <Tab>Configuration</Tab>
          </TabList>
          
          <TabPanels>
            {/* Performance Tab */}
            <TabPanel p={6}>
              <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={8}>
                <Card>
                  <CardHeader pb={0}>
                    <Heading size="md">Performance Metrics</Heading>
                  </CardHeader>
                  <CardBody>
                    <Box h="300px">
                      <PerformanceMetricsChart bot={bot} />
                    </Box>
                  </CardBody>
                </Card>
                
                <Card>
                  <CardHeader pb={0}>
                    <Heading size="md">Trade Results</Heading>
                  </CardHeader>
                  <CardBody>
                    <Box h="300px">
                      <TradeHistoryChart trades={tradeHistoryData} />
                    </Box>
                  </CardBody>
                </Card>
                
                {/* Additional Metrics Table */}
                <Card gridColumn={{ lg: "span 2" }}>
                  <CardHeader pb={0}>
                    <Heading size="md">Detailed Metrics</Heading>
                  </CardHeader>
                  <CardBody>
                    <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6}>
                      <Box>
                        <Text fontWeight="bold" mb={2}>Trading Activity</Text>
                        <SimpleGrid columns={2} spacing={4}>
                          <Box>
                            <Text color="gray.500">Total Trades</Text>
                            <Text fontWeight="medium">{bot.metrics?.total_trades || 0}</Text>
                          </Box>
                          <Box>
                            <Text color="gray.500">Win Rate</Text>
                            <Text fontWeight="medium">{bot.metrics?.win_rate ? formatPercent(bot.metrics.win_rate) : "N/A"}</Text>
                          </Box>
                          <Box>
                            <Text color="gray.500">Winning Trades</Text>
                            <Text fontWeight="medium">{bot.metrics?.winning_trades || 0}</Text>
                          </Box>
                          <Box>
                            <Text color="gray.500">Losing Trades</Text>
                            <Text fontWeight="medium">{bot.metrics?.losing_trades || 0}</Text>
                          </Box>
                        </SimpleGrid>
                      </Box>
                      
                      <Box>
                        <Text fontWeight="bold" mb={2}>Profitability</Text>
                        <SimpleGrid columns={2} spacing={4}>
                          <Box>
                            <Text color="gray.500">Total P&L</Text>
                            <Text fontWeight="medium">{bot.metrics?.total_pnl ? formatCurrency(bot.metrics.total_pnl) : "$0.00"}</Text>
                          </Box>
                          <Box>
                            <Text color="gray.500">Avg P&L Per Trade</Text>
                            <Text fontWeight="medium">{bot.metrics?.average_pnl_per_trade ? formatCurrency(bot.metrics.average_pnl_per_trade) : "$0.00"}</Text>
                          </Box>
                          <Box>
                            <Text color="gray.500">Avg Win</Text>
                            <Text fontWeight="medium">{bot.metrics?.average_win_amount ? formatCurrency(bot.metrics.average_win_amount) : "$0.00"}</Text>
                          </Box>
                          <Box>
                            <Text color="gray.500">Avg Loss</Text>
                            <Text fontWeight="medium">{bot.metrics?.average_loss_amount ? formatCurrency(bot.metrics.average_loss_amount) : "$0.00"}</Text>
                          </Box>
                        </SimpleGrid>
                      </Box>
                      
                      <Box>
                        <Text fontWeight="bold" mb={2}>Risk Metrics</Text>
                        <SimpleGrid columns={2} spacing={4}>
                          <Box>
                            <Text color="gray.500">Profit Factor</Text>
                            <Text fontWeight="medium">{bot.metrics?.profit_factor ? parseFloat(bot.metrics.profit_factor).toFixed(2) : "N/A"}</Text>
                          </Box>
                          <Box>
                            <Text color="gray.500">Max Drawdown</Text>
                            <Text fontWeight="medium">{bot.metrics?.max_drawdown ? formatCurrency(bot.metrics.max_drawdown) : "N/A"}</Text>
                          </Box>
                          <Box>
                            <Text color="gray.500">Sharpe Ratio</Text>
                            <Text fontWeight="medium">{bot.metrics?.sharpe_ratio ? parseFloat(bot.metrics.sharpe_ratio).toFixed(2) : "N/A"}</Text>
                          </Box>
                          <Box>
                            <Text color="gray.500">Risk/Reward</Text>
                            <Text fontWeight="medium">{bot.metrics?.risk_reward_ratio ? parseFloat(bot.metrics.risk_reward_ratio).toFixed(2) : "N/A"}</Text>
                          </Box>
                        </SimpleGrid>
                      </Box>
                    </SimpleGrid>
                  </CardBody>
                </Card>
              </SimpleGrid>
            </TabPanel>
            
            {/* Trade History Tab */}
            <TabPanel p={6}>
              <Card>
                <CardHeader pb={0}>
                  <Heading size="md">Trade History</Heading>
                </CardHeader>
                <CardBody>
                  <Box overflowX="auto">
                    <Table variant="simple" size="sm">
                      <Thead>
                        <Tr>
                          <Th>ID</Th>
                          <Th>Date</Th>
                          <Th>Direction</Th>
                          <Th>Entry Price</Th>
                          <Th>Exit Price</Th>
                          <Th>Status</Th>
                          <Th>P&L</Th>
                        </Tr>
                      </Thead>
                      <Tbody>
                        {bot.trades.length > 0 ? (
                          bot.trades.slice().reverse().map((trade) => (
                            <Tr key={trade.trade_id}>
                              <Td>{trade.trade_id}</Td>
                              <Td>{new Date(trade.entry_time).toLocaleDateString()}</Td>
                              <Td>
                                <Badge colorScheme={trade.trade_direction === 'LONG' ? 'green' : 'red'}>
                                  {trade.trade_direction}
                                </Badge>
                              </Td>
                              <Td>${parseFloat(trade.entry_price).toFixed(2)}</Td>
                              <Td>${trade.exit_price ? parseFloat(trade.exit_price).toFixed(2) : '—'}</Td>
                              <Td>
                                <Badge
                                  colorScheme={
                                    trade.trade_status === 'open'
                                      ? 'blue'
                                      : trade.pnl > 0
                                      ? 'green'
                                      : 'red'
                                  }
                                >
                                  {trade.trade_status}
                                </Badge>
                              </Td>
                              <Td color={trade.pnl > 0 ? 'green.500' : (trade.pnl < 0 ? 'red.500' : 'gray.500')}>
                                {trade.pnl ? formatCurrency(trade.pnl) : '—'}
                              </Td>
                            </Tr>
                          ))
                        ) : (
                          <Tr>
                            <Td colSpan={7} textAlign="center" py={4}>
                              No trades found for this bot
                            </Td>
                          </Tr>
                        )}
                      </Tbody>
                    </Table>
                  </Box>
                </CardBody>
              </Card>
            </TabPanel>
            
            {/* Parameters Tab */}
            <TabPanel p={6}>
              <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={8}>
                <Card>
                  <CardHeader pb={0}>
                    <Heading size="md">Parameter Configuration</Heading>
                  </CardHeader>
                  <CardBody>
                    <SimpleGrid columns={2} spacing={4}>
                      {bot.parameters && Object.entries(bot.parameters).map(([key, value]) => (
                        <Box key={key}>
                          <Text color="gray.500">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</Text>
                          <Text fontWeight="medium">{value}</Text>
                        </Box>
                      ))}
                      {(!bot.parameters || Object.keys(bot.parameters).length === 0) && (
                        <Text gridColumn="span 2">No parameters configured for this bot</Text>
                      )}
                    </SimpleGrid>
                  </CardBody>
                </Card>
                
                <Card>
                  <CardHeader pb={0}>
                    <Heading size="md">Parameter Visualization</Heading>
                  </CardHeader>
                  <CardBody>
                    <Box h="300px">
                      <ParameterRadarChart parameters={bot.parameters || {}} />
                    </Box>
                  </CardBody>
                </Card>
              </SimpleGrid>
            </TabPanel>
            
            {/* Configuration Tab */}
            <TabPanel p={6}>
              <Card>
                <CardHeader pb={0}>
                  <Heading size="md">Bot Configuration</Heading>
                </CardHeader>
                <CardBody>
                  <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
                    <Box>
                      <Text fontWeight="bold" mb={2}>Basic Information</Text>
                      <SimpleGrid columns={2} spacing={4}>
                        <Box>
                          <Text color="gray.500">Bot ID</Text>
                          <Text fontWeight="medium">{bot.bot_id}</Text>
                        </Box>
                        <Box>
                          <Text color="gray.500">Name</Text>
                          <Text fontWeight="medium">{bot.name}</Text>
                        </Box>
                        <Box>
                          <Text color="gray.500">Ticker</Text>
                          <Text fontWeight="medium">{bot.ticker}</Text>
                        </Box>
                        <Box>
                          <Text color="gray.500">Algorithm Type</Text>
                          <Text fontWeight="medium">{bot.algorithm_type.replace(/_/g, ' ')}</Text>
                        </Box>
                      </SimpleGrid>
                    </Box>
                    
                    <Box>
                      <Text fontWeight="bold" mb={2}>Trading Configuration</Text>
                      <SimpleGrid columns={2} spacing={4}>
                        <Box>
                          <Text color="gray.500">Trade Direction</Text>
                          <Text fontWeight="medium">{bot.trade_direction}</Text>
                        </Box>
                        <Box>
                          <Text color="gray.500">Position Size</Text>
                          <Text fontWeight="medium">${bot.position_size}</Text>
                        </Box>
                        <Box>
                          <Text color="gray.500">Trailing Stop</Text>
                          <Text fontWeight="medium">{(bot.trailing_stop_pct * 100).toFixed(2)}%</Text>
                        </Box>
                        <Box>
                          <Text color="gray.500">Status</Text>
                          <Text fontWeight="medium">{bot.is_active ? 'Active' : 'Inactive'}</Text>
                        </Box>
                      </SimpleGrid>
                    </Box>
                    
                    <Box gridColumn={{ md: "span 2" }}>
                      <Text fontWeight="bold" mb={2}>Advanced</Text>
                      <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                        <Box>
                          <Text color="gray.500">Algorithm Module</Text>
                          <Text fontWeight="medium">{bot.algorithm_module}</Text>
                        </Box>
                        <Box>
                          <Text color="gray.500">Version</Text>
                          <Text fontWeight="medium">{bot.version || 'N/A'}</Text>
                        </Box>
                        <Box>
                          <Text color="gray.500">Last Updated</Text>
                          <Text fontWeight="medium">
                            {bot.last_updated 
                              ? new Date(bot.last_updated).toLocaleDateString() 
                              : 'N/A'}
                          </Text>
                        </Box>
                      </SimpleGrid>
                    </Box>
                  </SimpleGrid>
                </CardBody>
              </Card>
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Box>
    </MainLayout>
  );
}