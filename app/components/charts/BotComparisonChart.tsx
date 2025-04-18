import React from 'react';
import { Box, Text, VStack, HStack, Divider, useColorModeValue } from '@chakra-ui/react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import { getMetricDocumentation } from '~/components/dashboard/MetricInfoTooltip';

// This would be populated with real data from your API
const sampleBots = [
  {
    bot_id: 1,
    name: 'Bot 1 - TSLA',
    win_rate: 0.65,
    profit_factor: 1.8,
    sharpe_ratio: 1.2,
    max_drawdown: 0.12,
    expectancy: 0.8,
  },
  {
    bot_id: 5,
    name: 'Bot 5 - NVDA',
    win_rate: 0.72,
    profit_factor: 2.1,
    sharpe_ratio: 1.5,
    max_drawdown: 0.08,
    expectancy: 1.1,
  },
  {
    bot_id: 12,
    name: 'Bot 12 - COIN',
    win_rate: 0.58,
    profit_factor: 1.6,
    sharpe_ratio: 0.9,
    max_drawdown: 0.15,
    expectancy: 0.7,
  },
];

// Normalize bot data for radar chart comparison with safe handling of missing data
const normalizeData = (bots: any[]) => {
  // Helper function to safely get maximum value
  const safeMax = (values: any[], defaultValue = 1) => {
    const validValues = values.filter(v => typeof v === 'number' && !isNaN(v) && v !== null && v !== undefined);
    return validValues.length > 0 ? Math.max(...validValues) : defaultValue;
  };
  
  // Find max values for each metric with safety checks
  const maxValues = {
    'Win Rate': safeMax(bots.map(bot => bot.win_rate || 0), 0.7),
    'Profit Factor': safeMax(bots.map(bot => bot.profit_factor || 0), 1.5),
    'Sharpe Ratio': safeMax(bots.map(bot => bot.sharpe_ratio || 0), 1.0),
    'Risk Control': 1 - safeMax(bots.map(bot => bot.max_drawdown || 0), 0.15), // Invert drawdown
    'Expectancy': safeMax(bots.map(bot => bot.expectancy || 0), 0.5),
  };
  
  // Create normalized data for the radar chart
  return Object.keys(maxValues).map(key => {
    const result: any = { metric: key };
    
    bots.forEach(bot => {
      let value;
      // Get bot values with fallbacks
      const winRate = typeof bot.win_rate === 'number' ? bot.win_rate : 0;
      const profitFactor = typeof bot.profit_factor === 'number' ? bot.profit_factor : 0;
      const sharpeRatio = typeof bot.sharpe_ratio === 'number' ? bot.sharpe_ratio : 0;
      const maxDrawdown = typeof bot.max_drawdown === 'number' ? bot.max_drawdown : 0;
      const expectancy = typeof bot.expectancy === 'number' ? bot.expectancy : 0;
      
      switch (key) {
        case 'Win Rate':
          value = winRate / (maxValues[key] || 1); // Avoid division by zero
          break;
        case 'Profit Factor':
          value = profitFactor / (maxValues[key] || 1);
          break;
        case 'Sharpe Ratio':
          value = sharpeRatio / (maxValues[key] || 1);
          break;
        case 'Risk Control':
          value = (1 - maxDrawdown) / (maxValues[key] || 1);
          break;
        case 'Expectancy':
          value = expectancy / (maxValues[key] || 1);
          break;
        default:
          value = 0;
      }
      // Scale to 0-100 for better visualization
      result[`Bot ${bot.bot_id}`] = value * 100;
    });
    
    return result;
  });
};

// Color palette for the radar areas
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

interface BotComparisonChartProps {
  bots?: any[];
}

export default function BotComparisonChart({ bots = sampleBots }: BotComparisonChartProps) {
  const chartData = normalizeData(bots);
  const tooltipBg = useColorModeValue('white', 'gray.700');
  const tooltipBorder = useColorModeValue('gray.200', 'gray.600');
  
  // Mapping of metrics from chart to documentation keys
  const metricToDocKey = {
    'Win Rate': 'win_rate',
    'Profit Factor': 'profit_factor',
    'Sharpe Ratio': 'sharpe_ratio',
    'Risk Control': 'max_drawdown',
    'Expectancy': 'expectancy'
  };
  
  // Enhanced tooltip with metric documentation
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const metricName = payload[0].payload.metric;
      const docKey = metricToDocKey[metricName] || '';
      const metricDoc = getMetricDocumentation(docKey);
      
      return (
        <Box 
          bg={tooltipBg} 
          p={3} 
          shadow="md" 
          borderRadius="md" 
          borderWidth="1px"
          borderColor={tooltipBorder}
          maxW="300px"
        >
          <VStack align="start" spacing={2}>
            <Text fontWeight="bold">{metricName}</Text>
            <Text fontSize="sm" fontStyle="italic">{metricDoc.description}</Text>
            
            <Divider />
            
            <Text fontSize="sm" fontWeight="semibold">Bot Comparison:</Text>
            {payload.map((entry: any, index: number) => (
              <HStack key={`item-${index}`}>
                <Box w="12px" h="12px" bg={entry.color} borderRadius="sm" />
                <Text color={entry.color}>
                  {`${entry.name}: ${entry.value.toFixed(1)}%`}
                </Text>
              </HStack>
            ))}
            
            <Divider my={1} />
            
            {metricName === 'Win Rate' && (
              <Text fontSize="xs">
                Win rate is valuable when viewed together with profit factor to understand 
                overall strategy effectiveness.
              </Text>
            )}
            
            {metricName === 'Profit Factor' && (
              <Text fontSize="xs">
                Higher values indicate more profit per dollar risked. Target above 1.5 for good performance.
              </Text>
            )}
            
            {metricName === 'Risk Control' && (
              <Text fontSize="xs">
                Based on maximum drawdown. Higher values indicate better risk management.
              </Text>
            )}
          </VStack>
        </Box>
      );
    }
    return null;
  };
  
  return (
    <Box h="400px" w="100%">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData}>
          <PolarGrid />
          <PolarAngleAxis dataKey="metric" />
          <PolarRadiusAxis angle={30} domain={[0, 100]} />
          <Tooltip content={<CustomTooltip />} />
          {bots.map((bot, index) => (
            <Radar
              key={bot.bot_id}
              name={`Bot ${bot.bot_id}`}
              dataKey={`Bot ${bot.bot_id}`}
              stroke={COLORS[index % COLORS.length]}
              fill={COLORS[index % COLORS.length]}
              fillOpacity={0.3}
            />
          ))}
          <Legend />
        </RadarChart>
      </ResponsiveContainer>
    </Box>
  );
}