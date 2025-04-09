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
  Button,
  HStack,
  Text,
  Flex,
  Input,
  InputGroup,
  InputLeftElement,
  Select,
  Stack,
  useColorModeValue,
} from '@chakra-ui/react';
import { SearchIcon, WarningIcon } from '@chakra-ui/icons';
import MainLayout from '~/components/layout/MainLayout';
import db from '~/lib/db.server';

export async function loader() {
  const openTrades = await db.getOpenTrades();
  return json({ openTrades });
}

function formatCurrency(value: number | string) {
  const numValue = typeof value === 'string' ? parseFloat(value) : value;
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(numValue);
}

export default function ActiveTrades() {
  const { openTrades } = useLoaderData<typeof loader>();
  const cardBg = useColorModeValue('white', 'gray.700');
  
  return (
    <MainLayout>
      <Flex justify="space-between" align="center" mb={8}>
        <Box>
          <Heading size="lg">Active Trades</Heading>
          <Text fontSize="md" color="gray.500">
            {openTrades.length} open positions
          </Text>
        </Box>
        <HStack>
          <Button colorScheme="red" leftIcon={<WarningIcon />}>
            Close All Trades
          </Button>
        </HStack>
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
        
        <Select placeholder="Direction" maxW={{ md: '150px' }}>
          <option value="LONG">LONG</option>
          <option value="SHORT">SHORT</option>
        </Select>
        
        <Select placeholder="Sort By" maxW={{ md: '200px' }}>
          <option value="newest">Newest First</option>
          <option value="oldest">Oldest First</option>
          <option value="largest">Largest Size</option>
          <option value="symbol">Symbol</option>
        </Select>
      </Stack>
      
      <Card shadow="base" bg={cardBg}>
        <CardHeader pb={0}>
          <Heading size="md">Active Positions</Heading>
        </CardHeader>
        <CardBody>
          <Box overflowX="auto">
            <Table variant="simple">
              <Thead>
                <Tr>
                  <Th>Trade ID</Th>
                  <Th>Bot</Th>
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
                {openTrades.map((trade) => (
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
                    <Td>{formatCurrency(trade.trade_size)}</Td>
                    <Td>{new Date(trade.entry_time).toLocaleString()}</Td>
                    <Td>
                      {trade.trailing_stop_price 
                        ? '$' + parseFloat(trade.trailing_stop_price).toFixed(2) 
                        : '-'
                      }
                    </Td>
                    <Td>
                      <Button size="sm" colorScheme="red">
                        Close
                      </Button>
                    </Td>
                  </Tr>
                ))}
                {openTrades.length === 0 && (
                  <Tr>
                    <Td colSpan={9} textAlign="center">No active trades</Td>
                  </Tr>
                )}
              </Tbody>
            </Table>
          </Box>
        </CardBody>
      </Card>
    </MainLayout>
  );
}