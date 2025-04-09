import React from 'react';
import { Box, useColorModeValue } from '@chakra-ui/react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

// Sample data - this would come from your API in a real implementation
const sampleData = [
  { date: '2025-03-20', trades: 12, pnl: 1540 },
  { date: '2025-03-21', trades: 19, pnl: 980 },
  { date: '2025-03-22', trades: 15, pnl: 1240 },
  { date: '2025-03-23', trades: 21, pnl: -590 },
  { date: '2025-03-24', trades: 28, pnl: 2100 },
  { date: '2025-03-25', trades: 24, pnl: 1600 },
  { date: '2025-03-26', trades: 18, pnl: 850 },
];

export default function TradeActivityChart() {
  const areaColor = useColorModeValue('blue.500', 'blue.200');
  const profitColor = useColorModeValue('green.500', 'green.200');
  const lossColor = useColorModeValue('red.500', 'red.200');
  
  return (
    <Box h="300px" w="100%">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={sampleData}
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id="colorTrades" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={areaColor} stopOpacity={0.8} />
              <stop offset="95%" stopColor={areaColor} stopOpacity={0.1} />
            </linearGradient>
            <linearGradient id="colorPnl" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={profitColor} stopOpacity={0.8} />
              <stop offset="95%" stopColor={profitColor} stopOpacity={0.1} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <Tooltip />
          <Legend />
          <Area
            yAxisId="left"
            type="monotone"
            dataKey="trades"
            stroke={areaColor}
            fillOpacity={1}
            fill="url(#colorTrades)"
          />
          <Area
            yAxisId="right"
            type="monotone"
            dataKey="pnl"
            stroke={profitColor}
            fillOpacity={1}
            fill="url(#colorPnl)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </Box>
  );
}