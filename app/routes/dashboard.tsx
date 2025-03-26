import { json } from '@remix-run/node';
import { useLoaderData } from '@remix-run/react';
import {
  Box,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Grid,
  GridItem,
  Heading,
  Text,
  Card,
  CardHeader,
  CardBody,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Flex,
} from '@chakra-ui/react';
import { getDashboardData } from '~/lib/api.server';
import MainLayout from '~/components/layout/MainLayout';
import TradeActivityChart from '~/components/charts/TradeActivityChart';
import FundAllocationChart from '~/components/charts/FundAllocationChart';

export async function loader() {
  const dashboardData = await getDashboardData();
  return json(dashboardData);
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

export default function Dashboard() {
  const data = useLoaderData<typeof loader>();
  
  return (
    <MainLayout>
      <Heading mb={6}>Trading Dashboard</Heading>
      
      {/* Stats Overview */}
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6} mb={8}>
        <Stat
          px={4}
          py={5}
          shadow="base"
          borderWidth="1px"
          borderRadius="lg"
          bg="white"
        >
          <StatLabel fontSize="md">Total Bots</StatLabel>
          <StatNumber fontSize="3xl">{data.summary.totalBots}</StatNumber>
          <StatHelpText>
            <Flex align="center">
              <Text>{data.summary.activeBots} active</Text>
            </Flex>
          </StatHelpText>
        </Stat>
        
        <Stat
          px={4}
          py={5}
          shadow="base"
          borderWidth="1px"
          borderRadius="lg"
          bg="white"
        >
          <StatLabel fontSize="md">Open Trades</StatLabel>
          <StatNumber fontSize="3xl">{data.summary.totalOpenTrades}</StatNumber>
          <StatHelpText>
            <Flex align="center">
              Across {data.openTrades.length} bots
            </Flex>
          </StatHelpText>
        </Stat>
        
        <Stat
          px={4}
          py={5}
          shadow="base"
          borderWidth="1px"
          borderRadius="lg"
          bg="white"
        >
          <StatLabel fontSize="md">Total P&L</StatLabel>
          <StatNumber fontSize="3xl" color={data.summary.totalPnl >= 0 ? 'green.500' : 'red.500'}>
            {formatCurrency(data.summary.totalPnl)}
          </StatNumber>
          <StatHelpText>
            <StatArrow type={data.summary.totalPnl >= 0 ? 'increase' : 'decrease'} />
            From all closed trades
          </StatHelpText>
        </Stat>
        
        <Stat
          px={4}
          py={5}
          shadow="base"
          borderWidth="1px"
          borderRadius="lg"
          bg="white"
        >
          <StatLabel fontSize="md">Avg Win Rate</StatLabel>
          <StatNumber fontSize="3xl">{formatPercent(data.summary.avgWinRate)}</StatNumber>
          <StatHelpText>
            <Flex align="center">
              System-wide average
            </Flex>
          </StatHelpText>
        </Stat>
      </SimpleGrid>
      
      {/* Chart Section */}
      <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={6} mb={8}>
        {/* Trade Activity Chart */}
        <Card shadow="base">
          <CardHeader pb={0}>
            <Heading size="md">Trade Activity</Heading>
          </CardHeader>
          <CardBody>
            <TradeActivityChart />
          </CardBody>
        </Card>
        
        {/* Fund Allocation Chart */}
        <Card shadow="base">
          <CardHeader pb={0}>
            <Heading size="md">Fund Allocation</Heading>
          </CardHeader>
          <CardBody>
            <FundAllocationChart />
          </CardBody>
        </Card>
      </SimpleGrid>

      {/* Main Content Area */}
      <Grid 
        templateColumns={{ base: 'repeat(1, 1fr)', lg: 'repeat(3, 1fr)' }}
        gap={6}
      >
        {/* Top Performing Bots */}
        <GridItem colSpan={{ base: 1, lg: 2 }}>
          <Card shadow="base" mb={6}>
            <CardHeader pb={0}>
              <Heading size="md">Top Performing Bots</Heading>
            </CardHeader>
            <CardBody>
              <Table variant="simple" size="sm">
                <Thead>
                  <Tr>
                    <Th>Bot ID</Th>
                    <Th>Name</Th>
                    <Th>Win Rate</Th>
                    <Th>Profit Factor</Th>
                    <Th>P&L</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {data.topBots.map((bot) => (
                    <Tr key={bot.bot_id}>
                      <Td>{bot.bot_id}</Td>
                      <Td>Bot {bot.bot_id}</Td>
                      <Td>{formatPercent(bot.win_rate)}</Td>
                      <Td>{parseFloat(bot.profit_factor).toFixed(2)}</Td>
                      <Td color={parseFloat(bot.total_pnl) >= 0 ? 'green.500' : 'red.500'}>
                        {formatCurrency(bot.total_pnl)}
                      </Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            </CardBody>
          </Card>
          
          {/* Recent Trades */}
          <Card shadow="base">
            <CardHeader pb={0}>
              <Heading size="md">Recent Trades</Heading>
            </CardHeader>
            <CardBody>
              <Table variant="simple" size="sm">
                <Thead>
                  <Tr>
                    <Th>Trade ID</Th>
                    <Th>Bot</Th>
                    <Th>Ticker</Th>
                    <Th>Direction</Th>
                    <Th>Status</Th>
                    <Th>P&L</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {data.recentTrades.map((trade) => (
                    <Tr key={trade.trade_id}>
                      <Td>{trade.trade_id}</Td>
                      <Td>{trade.bot_name || `Bot ${trade.bot_id}`}</Td>
                      <Td>{trade.ticker}</Td>
                      <Td>{trade.trade_direction}</Td>
                      <Td>
                        <Badge
                          colorScheme={
                            trade.trade_status === 'open'
                              ? 'blue'
                              : trade.trade_status === 'closed' && trade.pnl > 0
                              ? 'green'
                              : 'red'
                          }
                        >
                          {trade.trade_status}
                        </Badge>
                      </Td>
                      <Td color={trade.pnl > 0 ? 'green.500' : 'red.500'}>
                        {trade.pnl 
                          ? formatCurrency(trade.pnl)
                          : '-'
                        }
                      </Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            </CardBody>
          </Card>
        </GridItem>
        
        {/* Open Trades */}
        <GridItem colSpan={1}>
          <Card shadow="base" h="100%">
            <CardHeader pb={0}>
              <Heading size="md">Active Trades</Heading>
            </CardHeader>
            <CardBody>
              <Table variant="simple" size="sm">
                <Thead>
                  <Tr>
                    <Th>Ticker</Th>
                    <Th>Direction</Th>
                    <Th>Entry Price</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {data.openTrades.map((trade) => (
                    <Tr key={trade.trade_id}>
                      <Td>{trade.ticker}</Td>
                      <Td>{trade.trade_direction}</Td>
                      <Td>${parseFloat(trade.entry_price).toFixed(2)}</Td>
                    </Tr>
                  ))}
                  {data.openTrades.length === 0 && (
                    <Tr>
                      <Td colSpan={3} textAlign="center">No active trades</Td>
                    </Tr>
                  )}
                </Tbody>
              </Table>
            </CardBody>
          </Card>
        </GridItem>
      </Grid>
    </MainLayout>
  );
}