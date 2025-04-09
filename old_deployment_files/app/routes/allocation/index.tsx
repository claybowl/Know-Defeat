import React from 'react';
import { json } from '@remix-run/node';
import { useLoaderData } from '@remix-run/react';
import {
  Box,
  Card,
  CardHeader,
  CardBody,
  Heading,
  Text,
  SimpleGrid,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Progress,
  Flex,
} from '@chakra-ui/react';
import MainLayout from '~/components/layout/MainLayout';
import FundAllocationChart from '~/components/charts/FundAllocationChart';

// Sample data - this would be replaced with actual API data
const sampleAllocationData = [
  { bot_id: 1, name: 'Bot 1 - TSLA', ticker: 'TSLA', allocation: 2000, allocation_pct: 0.2, rank_score: 0.92 },
  { bot_id: 5, name: 'Bot 5 - NVDA', ticker: 'NVDA', allocation: 2000, allocation_pct: 0.2, rank_score: 0.88 },
  { bot_id: 7, name: 'Bot 7 - AAPL', ticker: 'AAPL', allocation: 2000, allocation_pct: 0.2, rank_score: 0.85 },
  { bot_id: 12, name: 'Bot 12 - COIN', ticker: 'COIN', allocation: 2000, allocation_pct: 0.2, rank_score: 0.81 },
  { bot_id: 23, name: 'Bot 23 - NVDA', ticker: 'NVDA', allocation: 2000, allocation_pct: 0.2, rank_score: 0.78 },
];

// Loader function to get allocation data
export async function loader() {
  // In a real implementation, this would fetch data from your API
  // For now, we'll use the sample data
  
  return json({
    totalFunds: 10000,
    allocations: sampleAllocationData
  });
}

// Format currency function
function formatCurrency(value: number | string) {
  const numValue = typeof value === 'string' ? parseFloat(value) : value;
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(numValue);
}

// Format percentage function
function formatPercent(value: number | string) {
  const numValue = typeof value === 'string' ? parseFloat(value) : value;
  return (numValue * 100).toFixed(2) + '%';
}

export default function FundAllocation() {
  const data = useLoaderData<typeof loader>();
  
  return (
    <MainLayout>
      <Heading mb={6}>Fund Allocation</Heading>
      
      <Text mb={6}>
        Total Funds: <strong>{formatCurrency(data.totalFunds)}</strong>
      </Text>
      
      <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={6} mb={8}>
        {/* Fund Allocation Chart */}
        <Card shadow="base">
          <CardHeader pb={0}>
            <Heading size="md">Fund Allocation Distribution</Heading>
          </CardHeader>
          <CardBody>
            <FundAllocationChart />
          </CardBody>
        </Card>
        
        {/* Allocation Strategy */}
        <Card shadow="base">
          <CardHeader pb={0}>
            <Heading size="md">Allocation Strategy</Heading>
          </CardHeader>
          <CardBody>
            <Text mb={4}>
              Funds are allocated based on bot performance metrics including win rate, 
              profit factor, and risk-adjusted returns. Higher-ranked bots receive 
              larger allocations to maximize system returns.
            </Text>
            <Box>
              <Heading size="sm" mb={2}>Ranking Factors:</Heading>
              <SimpleGrid columns={2} spacing={4}>
                <Box>
                  <Text fontSize="sm">Win Rate</Text>
                  <Progress value={30} colorScheme="green" size="sm" mb={2} />
                </Box>
                <Box>
                  <Text fontSize="sm">Profit Factor</Text>
                  <Progress value={25} colorScheme="blue" size="sm" mb={2} />
                </Box>
                <Box>
                  <Text fontSize="sm">Risk-Adjusted Returns</Text>
                  <Progress value={20} colorScheme="purple" size="sm" mb={2} />
                </Box>
                <Box>
                  <Text fontSize="sm">Max Drawdown</Text>
                  <Progress value={15} colorScheme="orange" size="sm" mb={2} />
                </Box>
                <Box>
                  <Text fontSize="sm">Recent Performance</Text>
                  <Progress value={10} colorScheme="cyan" size="sm" mb={2} />
                </Box>
              </SimpleGrid>
            </Box>
          </CardBody>
        </Card>
      </SimpleGrid>
      
      {/* Allocation Table */}
      <Card shadow="base">
        <CardHeader pb={0}>
          <Heading size="md">Bot Allocations</Heading>
        </CardHeader>
        <CardBody>
          <Table variant="simple">
            <Thead>
              <Tr>
                <Th>Bot ID</Th>
                <Th>Name</Th>
                <Th>Ticker</Th>
                <Th>Allocation</Th>
                <Th>Percentage</Th>
                <Th>Rank Score</Th>
              </Tr>
            </Thead>
            <Tbody>
              {data.allocations.map(bot => (
                <Tr key={bot.bot_id}>
                  <Td>{bot.bot_id}</Td>
                  <Td>{bot.name}</Td>
                  <Td>{bot.ticker}</Td>
                  <Td>{formatCurrency(bot.allocation)}</Td>
                  <Td>{formatPercent(bot.allocation_pct)}</Td>
                  <Td>
                    <Flex align="center">
                      <Progress 
                        value={bot.rank_score * 100} 
                        colorScheme="blue" 
                        size="sm" 
                        width="100px" 
                        mr={2} 
                      />
                      <Text>{(bot.rank_score * 100).toFixed(0)}%</Text>
                    </Flex>
                  </Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </CardBody>
      </Card>
    </MainLayout>
  );
}