import React from 'react';
import { Box, useColorModeValue } from '@chakra-ui/react';
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

// Normalize bot data for radar chart comparison
const normalizeData = (bots: any[]) => {
  // Find max values for each metric
  const maxValues = {
    'Win Rate': Math.max(...bots.map(bot => bot.win_rate)),
    'Profit Factor': Math.max(...bots.map(bot => bot.profit_factor)),
    'Sharpe Ratio': Math.max(...bots.map(bot => bot.sharpe_ratio)),
    'Risk Control': 1 - Math.max(...bots.map(bot => bot.max_drawdown)), // Invert drawdown
    'Expectancy': Math.max(...bots.map(bot => bot.expectancy)),
  };
  
  // Create normalized data for the radar chart
  return Object.keys(maxValues).map(key => {
    const result: any = { metric: key };
    
    bots.forEach(bot => {
      let value;
      switch (key) {
        case 'Win Rate':
          value = bot.win_rate / maxValues[key];
          break;
        case 'Profit Factor':
          value = bot.profit_factor / maxValues[key];
          break;
        case 'Sharpe Ratio':
          value = bot.sharpe_ratio / maxValues[key];
          break;
        case 'Risk Control':
          value = (1 - bot.max_drawdown) / maxValues[key];
          break;
        case 'Expectancy':
          value = bot.expectancy / maxValues[key];
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
          <p><strong>{payload[0].payload.metric}</strong></p>
          {payload.map((entry: any, index: number) => (
            <p key={`item-${index}`} style={{ color: entry.color }}>
              {`${entry.name}: ${entry.value.toFixed(1)}%`}
            </p>
          ))}
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