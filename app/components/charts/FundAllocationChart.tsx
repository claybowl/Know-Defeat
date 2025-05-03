import React from 'react';
import { Box, useColorModeValue, Text, VStack } from '@chakra-ui/react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';

// Color palette for the pie chart
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82ca9d', '#8dd1e1', '#a4de6c', '#d0ed57', '#ffc658'];

// Custom tooltip
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <Box 
        bg="white" 
        p={2} 
        shadow="md" 
        borderRadius="md" 
        borderWidth="1px"
      >
        <p>{`${payload[0].name}: $${payload[0].value.toLocaleString()}`}</p>
        <p>{`${(payload[0].payload.percent * 100).toFixed(2)}% of total`}</p>
      </Box>
    );
  }
  return null;
};

interface FundAllocationChartProps {
  topBots?: any[];
}

export default function FundAllocationChart({ topBots = [] }: FundAllocationChartProps) {
  // Create allocation data with fixed $2000 per bot
  const allocationData = topBots.map(bot => ({
    name: `Bot ${bot.bot_id}${bot.ticker ? ` - ${bot.ticker}` : ''}`,
    value: 2000, // Fixed $2000 per bot
    bot_id: bot.bot_id,
    ticker: bot.ticker || ''
  }));
  
  // If no bots, show placeholder message
  if (allocationData.length === 0) {
    console.log("No top bots data available for fund allocation chart");
    return (
      <VStack justify="center" align="center" h="300px">
        <Text color="gray.500">No allocation data available</Text>
        <Text fontSize="sm" color="gray.400">Check that bot metrics data exists in the database</Text>
      </VStack>
    );
  } else {
    console.log(`Showing fund allocation for ${allocationData.length} top bots`);
  }
  
  // Calculate percentages - equal for all bots
  const total = allocationData.reduce((sum, item) => sum + item.value, 0);
  const dataWithPercentage = allocationData.map(item => ({
    ...item,
    percent: item.value / total
  }));
  
  return (
    <Box h="300px" w="100%">
      <Text fontSize="sm" textAlign="center" mb={2} color="gray.600">
        Top 10 ranked bots with $2,000 allocated to each position
      </Text>
      <ResponsiveContainer width="100%" height="90%">
        <PieChart>
          <Pie
            data={dataWithPercentage}
            cx="50%"
            cy="50%"
            labelLine={false}
            outerRadius={100}
            fill="#8884d8"
            dataKey="value"
            label={({ name, percent }) => `${name.split(' - ')[0]} (${(percent * 100).toFixed(0)}%)`}
          >
            {dataWithPercentage.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
          <Legend formatter={(value) => {
            // Shorten legend text to prevent crowding
            const parts = value.split(' - ');
            return parts.length > 1 
              ? `${parts[0]} - ${parts[1]}`
              : value;
          }} />
        </PieChart>
      </ResponsiveContainer>
      <Text fontSize="sm" textAlign="center" mt={1}>
        Total Allocation: ${(total).toLocaleString()}
      </Text>
    </Box>
  );
}