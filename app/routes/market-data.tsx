import { json } from '@remix-run/node';
import { useLoaderData, useSearchParams } from '@remix-run/react';
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
  Select,
  Flex,
  Button,
  Input,
  InputGroup,
  InputRightElement,
  Stat,
  StatLabel,
  StatNumber,
  StatGroup,
  useColorModeValue,
} from '@chakra-ui/react';
import { useState, useEffect } from 'react';
import MainLayout from '~/components/layout/MainLayout';
import db from '~/lib/db.server';

export async function loader() {
  // Fetch market data
  const tickData = await db.getTickData(null, 50);
  
  // Get unique tickers for filtering
  const uniqueTickers = [...new Set(tickData.map(tick => tick.ticker))];
  
  return json({ tickData, tickers: uniqueTickers });
}

export default function MarketData() {
  const { tickData, tickers } = useLoaderData<typeof loader>();
  const [searchParams, setSearchParams] = useSearchParams();
  const [filterTicker, setFilterTicker] = useState(searchParams.get('ticker') || '');
  const [displayData, setDisplayData] = useState(tickData);
  
  // Filter data when the ticker filter changes
  useEffect(() => {
    if (filterTicker) {
      setDisplayData(tickData.filter(tick => tick.ticker === filterTicker));
    } else {
      setDisplayData(tickData);
    }
  }, [filterTicker, tickData]);
  
  // Calculate statistics for current ticker
  const tickerStats = {
    lastPrice: displayData.length > 0 ? displayData[0].price : 0,
    volume: displayData.reduce((sum, tick) => sum + (tick.volume || 0), 0),
    avgPrice: displayData.length > 0 
      ? displayData.reduce((sum, tick) => sum + tick.price, 0) / displayData.length 
      : 0,
    spread: displayData.length > 0 
      ? (displayData[0].ask - displayData[0].bid).toFixed(2) 
      : '0.00'
  };
  
  const handleTickerChange = (e) => {
    const ticker = e.target.value;
    setFilterTicker(ticker);
    if (ticker) {
      setSearchParams({ ticker });
    } else {
      setSearchParams({});
    }
  };
  
  return (
    <MainLayout>
      <Box padding="4">
        <Heading as="h1" size="xl" mb="6">
          Market Data
        </Heading>
        
        <Flex mb="4" justify="space-between" align="center">
          <Box width="250px">
            <Select 
              placeholder="Filter by ticker" 
              value={filterTicker} 
              onChange={handleTickerChange}
            >
              {tickers.map(ticker => (
                <option key={ticker} value={ticker}>
                  {ticker}
                </option>
              ))}
            </Select>
          </Box>
          
          <Button 
            colorScheme="blue" 
            onClick={() => window.location.reload()}
          >
            Refresh Data
          </Button>
        </Flex>
        
        {/* Statistics Cards */}
        <SimpleGrid columns={{ base: 1, md: 4 }} spacing="4" mb="6">
          <Card>
            <CardBody>
              <Stat>
                <StatLabel>Last Price</StatLabel>
                <StatNumber>${tickerStats.lastPrice.toFixed(2)}</StatNumber>
              </Stat>
            </CardBody>
          </Card>
          
          <Card>
            <CardBody>
              <Stat>
                <StatLabel>Volume</StatLabel>
                <StatNumber>{tickerStats.volume.toLocaleString()}</StatNumber>
              </Stat>
            </CardBody>
          </Card>
          
          <Card>
            <CardBody>
              <Stat>
                <StatLabel>Average Price</StatLabel>
                <StatNumber>${tickerStats.avgPrice.toFixed(2)}</StatNumber>
              </Stat>
            </CardBody>
          </Card>
          
          <Card>
            <CardBody>
              <Stat>
                <StatLabel>Bid-Ask Spread</StatLabel>
                <StatNumber>${tickerStats.spread}</StatNumber>
              </Stat>
            </CardBody>
          </Card>
        </SimpleGrid>
        
        {displayData.length > 0 ? (
          <Card>
            <CardHeader>
              <Heading size="md">
                {filterTicker ? `${filterTicker} Tick Data` : 'Recent Tick Data'}
              </Heading>
            </CardHeader>
            <CardBody overflowX="auto">
              <Table variant="simple">
                <Thead>
                  <Tr>
                    <Th>Ticker</Th>
                    <Th>Timestamp</Th>
                    <Th isNumeric>Price</Th>
                    <Th isNumeric>Size</Th>
                    <Th isNumeric>Bid</Th>
                    <Th isNumeric>Ask</Th>
                    <Th isNumeric>Volume</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {displayData.map((tick) => (
                    <Tr key={tick.id}>
                      <Td>
                        <Badge colorScheme="blue">{tick.ticker}</Badge>
                      </Td>
                      <Td>{new Date(tick.timestamp).toLocaleString()}</Td>
                      <Td isNumeric>${tick.price.toFixed(2)}</Td>
                      <Td isNumeric>{tick.trade_size}</Td>
                      <Td isNumeric>${tick.bid ? tick.bid.toFixed(2) : '-'}</Td>
                      <Td isNumeric>${tick.ask ? tick.ask.toFixed(2) : '-'}</Td>
                      <Td isNumeric>{tick.volume}</Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            </CardBody>
          </Card>
        ) : (
          <Card>
            <CardBody>
              <Text>No tick data available{filterTicker ? ` for ${filterTicker}` : ''}.</Text>
            </CardBody>
          </Card>
        )}
      </Box>
    </MainLayout>
  );
} 