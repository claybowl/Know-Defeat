import React, { useState } from 'react';
import { 
  Box, 
  Flex, 
  Text, 
  useColorModeValue, 
  ButtonGroup, 
  Button,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  HStack,
  Heading,
} from '@chakra-ui/react';
import { 
  PieChart, 
  Pie, 
  Cell, 
  Tooltip, 
  ResponsiveContainer, 
  Legend,
  Sector
} from 'recharts';

// Sample data - this would come from your API
const sampleData = [
  { name: 'Bot 1 - TSLA', value: 2000, status: 'active', returns: 12.3, strategy: 'breakout' },
  { name: 'Bot 3 - NVDA', value: 2000, status: 'active', returns: 9.5, strategy: 'momentum' },
  { name: 'Bot 7 - AAPL', value: 2000, status: 'active', returns: 7.8, strategy: 'support_resistance' },
  { name: 'Bot 2 - COIN', value: 2000, status: 'active', returns: -4.2, strategy: 'momentum' },
  { name: 'Bot 5 - AMZN', value: 2000, status: 'active', returns: 5.6, strategy: 'breakout' },
];

// Color palette based on strategy type
const strategyColors = {
  'breakout': ['#0088FE', '#4DA6FF', '#7DBEFF'],
  'momentum': ['#00C49F', '#34D3B5', '#67E1CB'],
  'support_resistance': ['#FFBB28', '#FFD166', '#FFE299'],
  'mean_reversion': ['#FF8042', '#FF9F71', '#FFBEA0'],
  'volatility_breakout': ['#8884D8', '#A296E1', '#BDB9EB'],
  'price_pattern': ['#82ca9d', '#A3D8B2', '#C3E6C7'],
};

// Get color for a strategy with index-based variation
const getStrategyColor = (strategy: string, index: number) => {
  if (strategy in strategyColors) {
    const colors = strategyColors[strategy as keyof typeof strategyColors];
    return colors[index % colors.length];
  }
  // Fallback colors
  const fallbackColors = ['#8dd1e1', '#a4de6c', '#d0ed57', '#ffc658'];
  return fallbackColors[index % fallbackColors.length];
};

// Custom tooltip
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    
    return (
      <Box 
        bg="white" 
        p={3} 
        shadow="lg" 
        borderRadius="md" 
        borderWidth="1px"
        maxW="250px"
      >
        <Text fontWeight="bold" mb={1}>{data.name}</Text>
        <Flex justify="space-between" mb={1}>
          <Text fontSize="sm" color="gray.600">Allocation:</Text>
          <Text fontSize="sm" fontWeight="medium">${data.value.toLocaleString()}</Text>
        </Flex>
        <Flex justify="space-between" mb={1}>
          <Text fontSize="sm" color="gray.600">Returns:</Text>
          <Text 
            fontSize="sm" 
            fontWeight="medium"
            color={data.returns >= 0 ? 'green.500' : 'red.500'}
          >
            {data.returns >= 0 ? '+' : ''}{data.returns}%
          </Text>
        </Flex>
        <Flex justify="space-between" mb={1}>
          <Text fontSize="sm" color="gray.600">Strategy:</Text>
          <Text fontSize="sm" fontWeight="medium" textTransform="capitalize">
            {data.strategy.replace('_', ' ')}
          </Text>
        </Flex>
        <Flex justify="space-between">
          <Text fontSize="sm" color="gray.600">Status:</Text>
          <Badge colorScheme={data.status === 'active' ? 'green' : 'gray'}>
            {data.status}
          </Badge>
        </Flex>
      </Box>
    );
  }
  return null;
};

// Active shape for pie chart
const renderActiveShape = (props: any) => {
  const { 
    cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill, payload, percent, value 
  } = props;

  return (
    <g>
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius}
        outerRadius={outerRadius + 6}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
      />
      <Sector
        cx={cx}
        cy={cy}
        startAngle={startAngle}
        endAngle={endAngle}
        innerRadius={outerRadius + 8}
        outerRadius={outerRadius + 10}
        fill={fill}
      />
      <text x={cx} y={cy} dy={-15} textAnchor="middle" fill="#333" fontSize={14} fontWeight="bold">
        {payload.name}
      </text>
      <text x={cx} y={cy} dy={15} textAnchor="middle" fill="#333" fontSize={14}>
        ${value.toLocaleString()} ({(percent * 100).toFixed(0)}%)
      </text>
    </g>
  );
};

export default function EnhancedFundAllocationChart() {
  const [activeIndex, setActiveIndex] = useState(0);
  const [viewMode, setViewMode] = useState('chart');
  
  // Calculate percentages
  const total = sampleData.reduce((sum, item) => sum + item.value, 0);
  const dataWithPercentage = sampleData.map((item, index) => ({
    ...item,
    percent: item.value / total,
    color: getStrategyColor(item.strategy, index)
  }));
  
  // Handler for pie sector hover
  const onPieEnter = (_: any, index: number) => {
    setActiveIndex(index);
  };
  
  return (
    <Box>
      <Flex justify="space-between" align="center" mb={4}>
        <Heading size="sm" fontWeight="medium">
          Fund Allocation ({total.toLocaleString()} USD)
        </Heading>
        
        <ButtonGroup size="sm" isAttached variant="outline">
          <Button
            colorScheme={viewMode === 'chart' ? 'blue' : 'gray'}
            onClick={() => setViewMode('chart')}
          >
            Chart
          </Button>
          <Button
            colorScheme={viewMode === 'table' ? 'blue' : 'gray'}
            onClick={() => setViewMode('table')}
          >
            Table
          </Button>
        </ButtonGroup>
      </Flex>
      
      {viewMode === 'chart' ? (
        <Box h="300px" w="100%">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                activeIndex={activeIndex}
                activeShape={renderActiveShape}
                data={dataWithPercentage}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={90}
                fill="#8884d8"
                dataKey="value"
                onMouseEnter={onPieEnter}
              >
                {dataWithPercentage.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </Box>
      ) : (
        <Box overflowX="auto">
          <Table variant="simple" size="sm">
            <Thead>
              <Tr>
                <Th>Bot</Th>
                <Th>Strategy</Th>
                <Th isNumeric>Allocation</Th>
                <Th isNumeric>Returns</Th>
                <Th>Status</Th>
              </Tr>
            </Thead>
            <Tbody>
              {dataWithPercentage.map((bot, index) => (
                <Tr key={index}>
                  <Td>
                    <HStack>
                      <Box 
                        w="10px" 
                        h="10px" 
                        borderRadius="full" 
                        bg={bot.color} 
                      />
                      <Text>{bot.name}</Text>
                    </HStack>
                  </Td>
                  <Td textTransform="capitalize">{bot.strategy.replace('_', ' ')}</Td>
                  <Td isNumeric>${bot.value.toLocaleString()} ({(bot.percent * 100).toFixed(1)}%)</Td>
                  <Td isNumeric color={bot.returns >= 0 ? 'green.500' : 'red.500'}>
                    {bot.returns >= 0 ? '+' : ''}{bot.returns}%
                  </Td>
                  <Td>
                    <Badge colorScheme={bot.status === 'active' ? 'green' : 'gray'}>
                      {bot.status}
                    </Badge>
                  </Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </Box>
      )}
    </Box>
  );
}