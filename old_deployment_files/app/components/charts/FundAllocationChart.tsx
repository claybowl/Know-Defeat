import React from 'react';
import { Box, useColorModeValue } from '@chakra-ui/react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';

// This would be populated with real data from your API
const sampleData = [
  { name: 'Bot 1 - TSLA', value: 2000 },
  { name: 'Bot 5 - NVDA', value: 2000 },
  { name: 'Bot 7 - AAPL', value: 2000 },
  { name: 'Bot 12 - COIN', value: 2000 },
  { name: 'Bot 23 - NVDA', value: 2000 },
];

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
        <p>{`${payload[0].name}: $${payload[0].value}`}</p>
        <p>{`${(payload[0].payload.percent * 100).toFixed(2)}% of total`}</p>
      </Box>
    );
  }
  return null;
};

export default function FundAllocationChart() {
  // Calculate percentages
  const total = sampleData.reduce((sum, item) => sum + item.value, 0);
  const dataWithPercentage = sampleData.map(item => ({
    ...item,
    percent: item.value / total
  }));
  
  return (
    <Box h="300px" w="100%">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={dataWithPercentage}
            cx="50%"
            cy="50%"
            labelLine={false}
            outerRadius={100}
            fill="#8884d8"
            dataKey="value"
            label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
          >
            {dataWithPercentage.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </Box>
  );
}