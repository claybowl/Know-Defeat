import React from 'react';
import { Box, useColorModeValue } from '@chakra-ui/react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

// Sample data - this would come from your API in a real implementation
const sampleData = [
  { bot_id: 1, name: 'Bot 1', win_rate: 0.65, profit_factor: 1.8, sharpe_ratio: 1.2, max_drawdown: -0.12 },
  { bot_id: 5, name: 'Bot 5', win_rate: 0.72, profit_factor: 2.1, sharpe_ratio: 1.5, max_drawdown: -0.08 },
  { bot_id: 7, name: 'Bot 7', win_rate: 0.58, profit_factor: 1.6, sharpe_ratio: 0.9, max_drawdown: -0.15 },
  { bot_id: 12, name: 'Bot 12', win_rate: 0.61, profit_factor: 1.7, sharpe_ratio: 1.1, max_drawdown: -0.13 },
  { bot_id: 23, name: 'Bot 23', win_rate: 0.69, profit_factor: 1.9, sharpe_ratio: 1.3, max_drawdown: -0.10 },
];

interface PerformanceMetricsChartProps {
  data?: any[];
  metric?: 'win_rate' | 'profit_factor' | 'sharpe_ratio' | 'max_drawdown';
}

export default function PerformanceMetricsChart({
  data = sampleData,
  metric = 'win_rate',
}: PerformanceMetricsChartProps) {
  const positiveColor = useColorModeValue('green.500', 'green.300');
  const negativeColor = useColorModeValue('red.500', 'red.300');
  
  // Normalize data for visualization and handle missing data
  const chartData = data.map(bot => {
    // Get values with fallbacks for missing data
    const winRate = typeof bot.win_rate === 'number' ? bot.win_rate : 0;
    const maxDrawdown = typeof bot.max_drawdown === 'number' ? bot.max_drawdown : 0;
    const metricValue = typeof bot[metric] === 'number' ? bot[metric] : 0;
    
    return {
      name: `Bot ${bot.bot_id}`,
      value: metric === 'win_rate'
        ? winRate * 100 // Convert to percentage
        : metric === 'max_drawdown'
          ? Math.abs(maxDrawdown * 100) // Make positive for visualization
          : metricValue,
      actualValue: metricValue,
    };
  }).sort((a, b) => b.value - a.value); // Sort in descending order
  
  // Configure metrics settings with safer calculations
  const safeMax = (values: any[], defaultValue = 1) => {
    const validValues = values.filter(v => typeof v === 'number' && !isNaN(v));
    return validValues.length > 0 ? Math.max(...validValues) : defaultValue;
  };
  
  const metricConfig = {
    win_rate: {
      label: 'Win Rate (%)',
      color: positiveColor,
      format: (value: number) => `${value.toFixed(1)}%`,
      domain: [0, 100],
    },
    profit_factor: {
      label: 'Profit Factor',
      color: positiveColor,
      format: (value: number) => value.toFixed(2),
      domain: [0, safeMax(data.map(bot => bot.profit_factor || 0), 2) * 1.1],
    },
    sharpe_ratio: {
      label: 'Sharpe Ratio',
      color: positiveColor,
      format: (value: number) => value.toFixed(2),
      domain: [0, safeMax(data.map(bot => bot.sharpe_ratio || 0), 2) * 1.1],
    },
    max_drawdown: {
      label: 'Max Drawdown (%)',
      color: negativeColor,
      format: (value: number) => `-${value.toFixed(1)}%`,
      domain: [0, safeMax(data.map(bot => Math.abs(bot.max_drawdown || 0) * 100), 10) * 1.1],
    },
  };
  
  const currentMetric = metricConfig[metric];
  
  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const value = payload[0].payload.actualValue;
      const formattedValue = metric === 'win_rate'
        ? `${(value * 100).toFixed(1)}%`
        : metric === 'max_drawdown'
          ? `-${(Math.abs(value) * 100).toFixed(1)}%`
          : value.toFixed(2);
          
      return (
        <Box 
          bg="white" 
          p={2} 
          shadow="md" 
          borderRadius="md" 
          borderWidth="1px"
        >
          <p>{`${label}: ${formattedValue}`}</p>
        </Box>
      );
    }
    return null;
  };
  
  return (
    <Box h="300px" w="100%">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis
            domain={currentMetric.domain}
            label={{ value: currentMetric.label, angle: -90, position: 'insideLeft' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          {metric === 'sharpe_ratio' && <ReferenceLine y={1} stroke="#777" strokeDasharray="3 3" />}
          {metric === 'profit_factor' && <ReferenceLine y={1} stroke="#777" strokeDasharray="3 3" />}
          <Bar
            dataKey="value"
            name={currentMetric.label}
            fill={currentMetric.color}
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </Box>
  );
}