import React, { useState } from 'react';
import { 
  Box, 
  Flex, 
  Button, 
  ButtonGroup, 
  useColorModeValue,
  Text,
  HStack,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
} from '@chakra-ui/react';
import { 
  ComposedChart, 
  Area, 
  Bar, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Legend,
  ReferenceLine
} from 'recharts';

// Sample data with more metrics - this would come from your API
const generateSampleData = (period: string) => {
  // Base data - would be fetched from API based on period
  const baseData = [
    { date: '2025-03-20', trades: 12, pnl: 1540, cumulativePnl: 1540, maxDrawdown: -280 },
    { date: '2025-03-21', trades: 19, pnl: 980, cumulativePnl: 2520, maxDrawdown: -180 },
    { date: '2025-03-22', trades: 15, pnl: 1240, cumulativePnl: 3760, maxDrawdown: -320 },
    { date: '2025-03-23', trades: 21, pnl: -590, cumulativePnl: 3170, maxDrawdown: -590 },
    { date: '2025-03-24', trades: 28, pnl: 2100, cumulativePnl: 5270, maxDrawdown: -210 },
    { date: '2025-03-25', trades: 24, pnl: 1600, cumulativePnl: 6870, maxDrawdown: -250 },
    { date: '2025-03-26', trades: 18, pnl: 850, cumulativePnl: 7720, maxDrawdown: -150 },
  ];

  // Different data depending on the period selected
  if (period === '1W') {
    return baseData;
  } else if (period === '1M') {
    // Extended sample for 1 month view
    return [
      ...baseData,
      { date: '2025-03-27', trades: 22, pnl: 1250, cumulativePnl: 8970, maxDrawdown: -220 },
      { date: '2025-03-28', trades: 20, pnl: -480, cumulativePnl: 8490, maxDrawdown: -480 },
      { date: '2025-03-29', trades: 17, pnl: 890, cumulativePnl: 9380, maxDrawdown: -190 },
      { date: '2025-03-30', trades: 25, pnl: 1380, cumulativePnl: 10760, maxDrawdown: -280 },
    ];
  } else if (period === '3M') {
    // Extended sample for 3 month view
    return [
      { date: '2025-01-15', trades: 10, pnl: 850, cumulativePnl: 850, maxDrawdown: -150 },
      { date: '2025-01-30', trades: 15, pnl: 1200, cumulativePnl: 2050, maxDrawdown: -230 },
      { date: '2025-02-15', trades: 18, pnl: -420, cumulativePnl: 1630, maxDrawdown: -420 },
      { date: '2025-02-28', trades: 22, pnl: 1540, cumulativePnl: 3170, maxDrawdown: -250 },
      { date: '2025-03-15', trades: 24, pnl: 2200, cumulativePnl: 5370, maxDrawdown: -270 },
      { date: '2025-03-30', trades: 28, pnl: 1350, cumulativePnl: 6720, maxDrawdown: -180 },
    ];
  } else {
    // Default to 1D view with hourly data
    return [
      { date: '9:30 AM', trades: 3, pnl: 420, cumulativePnl: 420, maxDrawdown: -80 },
      { date: '10:30 AM', trades: 5, pnl: -150, cumulativePnl: 270, maxDrawdown: -150 },
      { date: '11:30 AM', trades: 4, pnl: 280, cumulativePnl: 550, maxDrawdown: -60 },
      { date: '12:30 PM', trades: 2, pnl: 120, cumulativePnl: 670, maxDrawdown: -40 },
      { date: '1:30 PM', trades: 7, pnl: 380, cumulativePnl: 1050, maxDrawdown: -90 },
      { date: '2:30 PM', trades: 5, pnl: -210, cumulativePnl: 840, maxDrawdown: -210 },
      { date: '3:30 PM', trades: 4, pnl: 290, cumulativePnl: 1130, maxDrawdown: -70 },
      { date: '4:00 PM', trades: 2, pnl: 120, cumulativePnl: 1250, maxDrawdown: -30 },
    ];
  }
};

// Custom tooltip
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <Box 
        bg="white" 
        p={3} 
        shadow="lg" 
        borderRadius="md" 
        borderWidth="1px"
      >
        <Text fontWeight="bold" mb={1}>{label}</Text>
        
        {payload.map((entry: any, index: number) => (
          <Flex key={`item-${index}`} align="center" mb={1}>
            <Box 
              w="12px" 
              h="12px" 
              borderRadius="sm"
              bg={entry.color} 
              mr={2} 
            />
            <Text fontSize="sm">
              {entry.name}: {entry.name === 'P&L' || entry.name === 'Cumulative P&L' || entry.name === 'Max Drawdown' 
                ? `$${entry.value.toLocaleString()}` 
                : entry.value}
            </Text>
          </Flex>
        ))}
      </Box>
    );
  }
  return null;
};

export default function EnhancedTradeActivityChart() {
  const [period, setPeriod] = useState('1W');
  const [view, setView] = useState('pnl'); // 'pnl' or 'trades' or 'combined'
  
  // Get data based on selected period
  const chartData = generateSampleData(period);
  
  // Calculate period performance
  const periodPerformance = chartData[chartData.length - 1].cumulativePnl - chartData[0].cumulativePnl;
  const periodPercentChange = periodPerformance / Math.abs(chartData[0].cumulativePnl || 1) * 100;
  
  // Colors
  const areaColor = useColorModeValue('blue.500', 'blue.200');
  const barColor = useColorModeValue('purple.500', 'purple.200');
  const profitColor = useColorModeValue('green.500', 'green.200');
  const cumulativeColor = useColorModeValue('teal.500', 'teal.200');
  const drawdownColor = useColorModeValue('red.500', 'red.200');
  
  // Background gradient
  const gradientStart = useColorModeValue('rgba(49, 130, 206, 0.1)', 'rgba(66, 153, 225, 0.1)');
  const gradientEnd = useColorModeValue('rgba(49, 130, 206, 0)', 'rgba(66, 153, 225, 0)');
  
  return (
    <Box>
      <Flex justify="space-between" align="center" mb={4}>
        <Stat>
          <StatLabel fontSize="sm" color="gray.500">Performance ({period})</StatLabel>
          <StatNumber 
            fontSize="xl" 
            color={periodPerformance >= 0 ? 'green.500' : 'red.500'}
          >
            ${periodPerformance.toLocaleString()}
          </StatNumber>
          <StatHelpText>
            <StatArrow type={periodPerformance >= 0 ? 'increase' : 'decrease'} />
            {periodPercentChange.toFixed(2)}%
          </StatHelpText>
        </Stat>
        
        <HStack spacing={4}>
          <ButtonGroup size="sm" isAttached variant="outline">
            <Button
              colorScheme={view === 'pnl' ? 'blue' : 'gray'}
              onClick={() => setView('pnl')}
            >
              P&L
            </Button>
            <Button
              colorScheme={view === 'trades' ? 'blue' : 'gray'}
              onClick={() => setView('trades')}
            >
              Trades
            </Button>
            <Button
              colorScheme={view === 'combined' ? 'blue' : 'gray'}
              onClick={() => setView('combined')}
            >
              Combined
            </Button>
          </ButtonGroup>
          
          <ButtonGroup size="sm" isAttached variant="outline">
            <Button
              colorScheme={period === '1D' ? 'blue' : 'gray'}
              onClick={() => setPeriod('1D')}
            >
              1D
            </Button>
            <Button
              colorScheme={period === '1W' ? 'blue' : 'gray'}
              onClick={() => setPeriod('1W')}
            >
              1W
            </Button>
            <Button
              colorScheme={period === '1M' ? 'blue' : 'gray'}
              onClick={() => setPeriod('1M')}
            >
              1M
            </Button>
            <Button
              colorScheme={period === '3M' ? 'blue' : 'gray'}
              onClick={() => setPeriod('3M')}
            >
              3M
            </Button>
          </ButtonGroup>
        </HStack>
      </Flex>
      
      <Box h="300px" w="100%">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={chartData}
            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
          >
            <defs>
              <linearGradient id="colorPnl" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={profitColor} stopOpacity={0.8} />
                <stop offset="95%" stopColor={profitColor} stopOpacity={0.1} />
              </linearGradient>
              <linearGradient id="colorCumulative" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={cumulativeColor} stopOpacity={0.8} />
                <stop offset="95%" stopColor={cumulativeColor} stopOpacity={0.1} />
              </linearGradient>
              <linearGradient id="colorArea" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={areaColor} stopOpacity={0.8} />
                <stop offset="95%" stopColor={areaColor} stopOpacity={0.1} />
              </linearGradient>
            </defs>
            
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            
            {(view === 'pnl' || view === 'combined') && (
              <YAxis 
                yAxisId="left" 
                orientation="left" 
                domain={['auto', 'auto']}
                tickFormatter={(value) => `$${value}`}
              />
            )}
            
            {view === 'trades' && (
              <YAxis 
                yAxisId="right" 
                orientation="right" 
              />
            )}
            
            {view === 'combined' && (
              <YAxis 
                yAxisId="right" 
                orientation="right" 
              />
            )}
            
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            
            {/* Add yAxisId to ReferenceLine */}
            {(view === 'pnl' || view === 'combined') && (
              <ReferenceLine yAxisId="left" y={0} stroke="#000" strokeDasharray="3 3" />
            )}
            
            {/* Conditionally render chart elements based on view */}
            {(view === 'pnl' || view === 'combined') && (
              <>
                <Bar 
                  yAxisId="left" 
                  dataKey="pnl" 
                  name="P&L" 
                  fill={barColor} 
                  barSize={20} 
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="cumulativePnl"
                  name="Cumulative P&L"
                  stroke={cumulativeColor}
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="maxDrawdown"
                  name="Max Drawdown"
                  stroke={drawdownColor}
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={false}
                />
              </>
            )}
            
            {(view === 'trades' || view === 'combined') && (
              <Area
                yAxisId={view === 'combined' ? 'right' : 'left'}
                type="monotone"
                dataKey="trades"
                name="Trades"
                stroke={areaColor}
                fillOpacity={1}
                fill="url(#colorArea)"
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </Box>
    </Box>
  );
}