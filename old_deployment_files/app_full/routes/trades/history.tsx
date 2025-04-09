import { json } from '@remix-run/node';
import { useLoaderData } from '@remix-run/react';
import {
  Box,
  Heading,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Card,
  CardHeader,
  CardBody,
  Text,
  Flex,
  Input,
  InputGroup,
  InputLeftElement,
  Select,
  Stack,
  HStack,
  useColorModeValue,
  Button,
} from '@chakra-ui/react';
import { SearchIcon, DownloadIcon } from '@chakra-ui/icons';
import MainLayout from '~/components/layout/MainLayout';
import db from '~/lib/db.server';

export async function loader() {
  const trades = await db.getTrades(100); // Get latest 100 trades
  return json({ trades });
}

function formatCurrency(value: number | string) {
  const numValue = typeof value === 'string' ? parseFloat(value) : value;
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(numValue);
}

export default function TradeHistory() {
  const { trades } = useLoaderData<typeof loader>();
  const cardBg = useColorModeValue('white', 'gray.700');
  
  return (
    <MainLayout>
      <Flex justify="space-between" align="center" mb={8}>
        <Box>
          <Heading size="lg">Trade History</Heading>
          <Text fontSize="md" color="gray.500">
            Showing last {trades.length} trades
          </Text>
        </Box>
        <Button leftIcon={<DownloadIcon />}>
          Export CSV
        </Button>
      </Flex>
      
      {/* Filters */}
      <Stack 
        direction={{ base: 'column', md: 'row' }} 
        mb={6} 
        spacing={4}
        align={{ base: 'stretch', md: 'center' }}
      >
        <InputGroup maxW={{ md: '300px' }}>
          <InputLeftElement pointerEvents="none">
            <SearchIcon color="gray.300" />
          </InputLeftElement>
          <Input placeholder="Search trades..." />
        </InputGroup>
        
        <Select placeholder="Symbol" maxW={{ md: '150px' }}>
          <option value="TSLA">TSLA</option>
          <option value="AAPL">AAPL</option>
          <option value="COIN">COIN</option>
          <option value="NVDA">NVDA</option>
          <option value="AMD">AMD</option>
        </Select>
        
        <Select placeholder="Status" maxW={{ md: '150px' }}>
          <option value="closed">Closed</option>
          <option value="open">Open</option>
          <option value="pending_exit">Pending Exit</option>
        </Select>
        
        <Select placeholder="Result" maxW={{ md: '150px' }}>
          <option value="profitable">Profitable</option>
          <option value="unprofitable">Unprofitable</option>
        </Select>
        
        <Select placeholder="Time Range" maxW={{ md: '200px' }}>
          <option value="today">Today</option>
          <option value="week">This Week</option>
          <option value="month">This Month</option>
          <option value="all">All Time</option>
        </Select>
      </Stack>
      
      <Card shadow="base" bg={cardBg}>
        <CardHeader pb={0}>
          <HStack justify="space-between">
            <Heading size="md">Trade Records</Heading>
            <Text fontSize="sm" color="gray.500">
              Latest trades shown first
            </Text>
          </HStack>
        </CardHeader>
        <CardBody>
          <Box overflowX="auto">
            <Table variant="simple">
              <Thead>
                <Tr>
                  <Th>ID</Th>
                  <Th>Bot</Th>
                  <Th>Symbol</Th>
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
                {trades.map((trade) => (
                  <Tr key={trade.trade_id}>
                    <Td>{trade.trade_id}</Td>
                    <Td>{trade.bot_name || `Bot ${trade.bot_id}`}</Td>
                    <Td>{trade.ticker}</Td>
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
                            : trade.trade_status === 'closed' && trade.pnl > 0
                            ? 'green'
                            : 'red'
                        }
                      >
                        {trade.trade_status}
                      </Badge>
                    </Td>
                    <Td 
                      color={
                        trade.pnl 
                          ? parseFloat(trade.pnl) >= 0 
                            ? 'green.500' 
                            : 'red.500'
                          : 'inherit'
                      }
                    >
                      {trade.pnl 
                        ? formatCurrency(trade.pnl)
                        : '-'
                      }
                    </Td>
                  </Tr>
                ))}
                {trades.length === 0 && (
                  <Tr>
                    <Td colSpan={10} textAlign="center">No trades found</Td>
                  </Tr>
                )}
              </Tbody>
            </Table>
          </Box>
        </CardBody>
      </Card>
      
      <HStack mt={6} justify="flex-end">
        <Button variant="outline">Previous</Button>
        <Text>Page 1 of 10</Text>
        <Button variant="outline">Next</Button>
      </HStack>
    </MainLayout>
  );
}