import React from 'react';
import { Box, useColorModeValue } from '@chakra-ui/react';
import {
  LineChart,
  Line,
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
  { trade_id: 1, pnl: 120, cumulative_pnl: 120, exit_time: '2025-03-01' },
  { trade_id: 2, pnl: -80, cumulative_pnl: 40, exit_time: '2025-03-02' },
  { trade_id: 3, pnl: 150, cumulative_pnl: 190, exit_time: '2025-03-03' },
  { trade_id: 4, pnl: 200, cumulative_pnl: 390, exit_time: '2025-03-05' },
  { trade_id: 5, pnl: -120, cumulative_pnl: 270, exit_time: '2025-03-07' },
  { trade_id: 6, pnl: 90, cumulative_pnl: 360, exit_time: '2025-03-08' },
  { trade_id: 7, pnl: 110, cumulative_pnl: 470, exit_time: '2025-03-10' },
  { trade_id: 8, pnl: -60, cumulative_pnl: 410, exit_time: '2025-03-12' },
  { trade_id: 9, pnl: 130, cumulative_pnl: 540, exit_time: '2025-03-15' },
  { trade_id: 10, pnl: 180, cumulative_pnl: 720, exit_time: '2025-03-18' },
];

interface TradeHistoryChartProps {
  trades?: any[];
  showIndividualTrades?: boolean;
}

export default function TradeHistoryChart({
  trades = sampleData,
  showIndividualTrades = true,
}: TradeHistoryChartProps) {
  const cumulativeLineColor = useColorModeValue('blue.500', 'blue.300');
  const profitColor = useColorModeValue('green.500', 'green.300');
  const lossColor = useColorModeValue('red.500', 'red.300');
  
  // Process data to ensure cumulative PnL is calculated correctly
  const processedData = [...trades]
    .sort((a, b) => new Date(a.exit_time).getTime() - new Date(b.exit_time).getTime())
    .map((trade, index, arr) => {
      // Calculate cumulative PnL if not already provided
      if (trade.cumulative_pnl === undefined) {
        const prevCumulative = index > 0 ? arr[index - 1].cumulative_pnl : 0;
        trade.cumulative_pnl = prevCumulative + parseFloat(trade.pnl);
      }
      return trade;
    });
  
  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      
      return (
        <Box 
          bg="white" 
          p={2} 
          shadow="md" 
          borderRadius="md" 
          borderWidth="1px"
        >
          <p><strong>Trade #{data.trade_id}</strong></p>
          <p style={{ color: data.pnl >= 0 ? profitColor : lossColor }}>
            PnL: ${data.pnl.toFixed(2)}
          </p>
          <p style={{ color: cumulativeLineColor }}>
            Cumulative PnL: ${data.cumulative_pnl.toFixed(2)}
          </p>
          <p>Date: {new Date(data.exit_time).toLocaleDateString()}</p>
        </Box>
      );
    }
    return null;
  };
  
  return (
    <Box h="300px" w="100%">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={processedData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="exit_time" 
            tickFormatter={(value) => new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} 
          />
          <YAxis />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <ReferenceLine y={0} stroke="#888" strokeDasharray="3 3" />
          
          {/* Cumulative PnL Line */}
          <Line
            type="monotone"
            dataKey="cumulative_pnl"
            name="Cumulative P&L"
            stroke={cumulativeLineColor}
            strokeWidth={2}
            dot={true}
            activeDot={{ r: 6 }}
            isAnimationActive={true}
          />
          
          {/* Individual Trade PnL Bars (optional) */}
          {showIndividualTrades && (
            <Line
              type="monotone"
              dataKey="pnl"
              name="Trade P&L"
              stroke="#888"
              strokeWidth={1}
              strokeDasharray="5 5"
              dot={(props: any) => {
                const { cx, cy, payload } = props;
                const pnl = payload.pnl;
                
                return (
                  <svg>
                    <circle
                      cx={cx}
                      cy={cy}
                      r={4}
                      fill={pnl >= 0 ? profitColor : lossColor}
                      stroke="none"
                    />
                  </svg>
                );
              }}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
}