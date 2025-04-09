import React from 'react';
import { 
  Box, 
  Flex, 
  Text, 
  SimpleGrid, 
  useColorModeValue,
  Badge
} from '@chakra-ui/react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';

interface Trade {
  trade_id: number;
  entry_price: number;
  exit_price?: number;
  pnl: number;
  pnl_percent?: number;
  trade_direction: string;
  entry_time: string;
  exit_time?: string;
  trade_status: string;
  ticker: string;
}

interface TradeAnalyticsChartProps {
  trades: Trade[];
  height?: number;
}

export default function TradeAnalyticsChart({ trades, height = 400 }: TradeAnalyticsChartProps) {
  const winColor = useColorModeValue('#38A169', '#68D391');
  const lossColor = useColorModeValue('#E53E3E', '#FC8181');
  
  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  
  // Process the trade data for visualization
  const processData = () => {
    // Filter closed trades with PnL
    const closedTrades = trades.filter(
      trade => trade.trade_status === 'closed' && trade.pnl !== null && trade.pnl !== undefined
    );
    
    // For scatter plot - separate winning and losing trades
    const winningTrades = closedTrades
      .filter(trade => parseFloat(trade.pnl.toString()) > 0)
      .map(trade => ({
        x: new Date(trade.entry_time).getTime(),
        y: parseFloat(trade.pnl.toString()),
        z: Math.abs(parseFloat(trade.pnl.toString())),
        id: trade.trade_id,
        direction: trade.trade_direction,
        entryPrice: trade.entry_price,
        exitPrice: trade.exit_price,
        entryTime: trade.entry_time,
        exitTime: trade.exit_time,
        ticker: trade.ticker,
        pnl: trade.pnl,
        pnlPercent: trade.pnl_percent,
        status: trade.trade_status
      }));
      
    const losingTrades = closedTrades
      .filter(trade => parseFloat(trade.pnl.toString()) <= 0)
      .map(trade => ({
        x: new Date(trade.entry_time).getTime(),
        y: parseFloat(trade.pnl.toString()),
        z: Math.abs(parseFloat(trade.pnl.toString())),
        id: trade.trade_id,
        direction: trade.trade_direction,
        entryPrice: trade.entry_price,
        exitPrice: trade.exit_price,
        entryTime: trade.entry_time,
        exitTime: trade.exit_time,
        ticker: trade.ticker,
        pnl: trade.pnl,
        pnlPercent: trade.pnl_percent,
        status: trade.trade_status
      }));
      
    // Calculate analytics
    const totalTrades = closedTrades.length;
    const totalPnL = closedTrades.reduce((sum, trade) => sum + parseFloat(trade.pnl.toString()), 0);
    const winningCount = winningTrades.length;
    const losingCount = losingTrades.length;
    const winRate = totalTrades > 0 ? winningCount / totalTrades : 0;
    
    const avgWin = winningTrades.length > 0 
      ? winningTrades.reduce((sum, trade) => sum + parseFloat(trade.pnl.toString()), 0) / winningTrades.length 
      : 0;
      
    const avgLoss = losingTrades.length > 0 
      ? losingTrades.reduce((sum, trade) => sum + parseFloat(trade.pnl.toString()), 0) / losingTrades.length 
      : 0;
      
    // Calculate streak information
    let currentStreak = 0;
    let maxWinStreak = 0;
    let maxLossStreak = 0;
    
    // Need to sort trades by time for streak calculation
    const sortedTrades = [...closedTrades].sort(
      (a, b) => new Date(a.entry_time).getTime() - new Date(b.entry_time).getTime()
    );
    
    sortedTrades.forEach((trade, index) => {
      const isProfitable = parseFloat(trade.pnl.toString()) > 0;
      
      if (index === 0) {
        // Initialize streak
        currentStreak = isProfitable ? 1 : -1;
      } else {
        // Update streak
        if (isProfitable) {
          if (currentStreak > 0) {
            currentStreak++; // Continue win streak
          } else {
            currentStreak = 1; // Start new win streak
          }
        } else {
          if (currentStreak < 0) {
            currentStreak--; // Continue loss streak
          } else {
            currentStreak = -1; // Start new loss streak
          }
        }
      }
      
      // Update max streaks
      if (currentStreak > 0) {
        maxWinStreak = Math.max(maxWinStreak, currentStreak);
      } else {
        maxLossStreak = Math.max(maxLossStreak, Math.abs(currentStreak));
      }
    });
    
    // Calculate current streak (for the latest trades)
    let recentStreak = 0;
    let recentStreakType = '';
    
    if (sortedTrades.length > 0) {
      const latestTrades = [...sortedTrades].reverse(); // Newest first
      
      const lastTradeWin = parseFloat(latestTrades[0].pnl.toString()) > 0;
      recentStreakType = lastTradeWin ? 'win' : 'loss';
      
      for (const trade of latestTrades) {
        const isProfitable = parseFloat(trade.pnl.toString()) > 0;
        
        if ((recentStreakType === 'win' && isProfitable) || 
            (recentStreakType === 'loss' && !isProfitable)) {
          recentStreak++;
        } else {
          break;
        }
      }
    }
    
    // Performance by time of day analysis
    const tradesByHour: Record<number, { count: number; totalPnl: number }> = {};
    closedTrades.forEach(trade => {
      const hour = new Date(trade.entry_time).getHours();
      
      if (!tradesByHour[hour]) {
        tradesByHour[hour] = { count: 0, totalPnl: 0 };
      }
      
      tradesByHour[hour].count += 1;
      tradesByHour[hour].totalPnl += parseFloat(trade.pnl.toString());
    });
    
    // Find best and worst hour
    let bestHour = -1;
    let bestHourPnl = -Infinity;
    let worstHour = -1;
    let worstHourPnl = Infinity;
    
    Object.entries(tradesByHour).forEach(([hour, data]) => {
      if (data.totalPnl > bestHourPnl) {
        bestHourPnl = data.totalPnl;
        bestHour = parseInt(hour);
      }
      
      if (data.totalPnl < worstHourPnl) {
        worstHourPnl = data.totalPnl;
        worstHour = parseInt(hour);
      }
    });
    
    const formatHour = (hour: number) => {
      const ampm = hour >= 12 ? 'PM' : 'AM';
      const hour12 = hour % 12 || 12;
      return `${hour12} ${ampm}`;
    };
    
    return {
      winningTrades,
      losingTrades,
      analytics: {
        totalTrades,
        totalPnL,
        winRate,
        avgWin,
        avgLoss,
        maxWinStreak,
        maxLossStreak,
        currentStreak: {
          count: recentStreak,
          type: recentStreakType
        },
        timeAnalysis: {
          bestHour: bestHour >= 0 ? formatHour(bestHour) : 'N/A',
          worstHour: worstHour >= 0 ? formatHour(worstHour) : 'N/A'
        }
      }
    };
  };
  
  const { winningTrades, losingTrades, analytics } = processData();
  
  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };
  
  // Custom tooltip for scatter points
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      
      return (
        <Box
          bg={cardBg}
          p={3}
          borderRadius="md"
          shadow="md"
          borderWidth="1px"
          borderColor={borderColor}
          maxW="300px"
        >
          <Flex justify="space-between" mb={2}>
            <Badge colorScheme={data.y > 0 ? 'green' : 'red'}>
              Trade #{data.id}
            </Badge>
            <Badge colorScheme={data.direction === 'LONG' ? 'green' : 'purple'}>
              {data.direction}
            </Badge>
          </Flex>
          
          <SimpleGrid columns={2} spacing={2} mb={2}>
            <Box>
              <Text fontSize="xs" color="gray.500">Entry</Text>
              <Text fontWeight="bold">${parseFloat(data.entryPrice).toFixed(2)}</Text>
            </Box>
            <Box>
              <Text fontSize="xs" color="gray.500">Exit</Text>
              <Text fontWeight="bold">${parseFloat(data.exitPrice).toFixed(2)}</Text>
            </Box>
            <Box>
              <Text fontSize="xs" color="gray.500">Entry Time</Text>
              <Text fontSize="xs">{new Date(data.entryTime).toLocaleString()}</Text>
            </Box>
            <Box>
              <Text fontSize="xs" color="gray.500">Exit Time</Text>
              <Text fontSize="xs">{new Date(data.exitTime).toLocaleString()}</Text>
            </Box>
          </SimpleGrid>
          
          <Flex justify="space-between" fontWeight="bold">
            <Text>PnL:</Text>
            <Text color={data.y > 0 ? 'green.500' : 'red.500'}>
              {formatCurrency(data.y)}
            </Text>
          </Flex>
        </Box>
      );
    }
    
    return null;
  };
  
  // Check if we have trades to display
  if (!trades || trades.length === 0) {
    return (
      <Box 
        height={`${height}px`} 
        width="100%" 
        borderWidth="1px" 
        borderRadius="md" 
        p={4}
        display="flex"
        alignItems="center"
        justifyContent="center"
      >
        <Text color="gray.500">No trade data available for analysis</Text>
      </Box>
    );
  }
  
  return (
    <Box>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={4} mb={6}>
        <Box 
          p={4} 
          borderWidth="1px" 
          borderRadius="md" 
          bg={cardBg}
          borderLeftWidth="4px"
          borderLeftColor={analytics.totalPnL >= 0 ? "green.400" : "red.400"}
        >
          <Text fontSize="sm" color="gray.500">Total P&L</Text>
          <Text 
            fontSize="2xl" 
            fontWeight="bold" 
            color={analytics.totalPnL >= 0 ? "green.500" : "red.500"}
          >
            {formatCurrency(analytics.totalPnL)}
          </Text>
          <Text fontSize="sm" color="gray.500">
            From {analytics.totalTrades} trades
          </Text>
        </Box>
        
        <Box 
          p={4} 
          borderWidth="1px" 
          borderRadius="md" 
          bg={cardBg}
          borderLeftWidth="4px"
          borderLeftColor={analytics.winRate >= 0.5 ? "green.400" : "orange.400"}
        >
          <Text fontSize="sm" color="gray.500">Win Rate</Text>
          <Text 
            fontSize="2xl" 
            fontWeight="bold"
          >
            {(analytics.winRate * 100).toFixed(1)}%
          </Text>
          <Text fontSize="sm" color="gray.500">
            {winningTrades.length} wins / {losingTrades.length} losses
          </Text>
        </Box>
        
        <Box 
          p={4} 
          borderWidth="1px" 
          borderRadius="md" 
          bg={cardBg}
          borderLeftWidth="4px"
          borderLeftColor="blue.400"
        >
          <Text fontSize="sm" color="gray.500">Avg Win/Loss</Text>
          <Flex align="center">
            <Text 
              fontSize="lg" 
              fontWeight="bold" 
              color="green.500"
              mr={2}
            >
              +{formatCurrency(analytics.avgWin)}
            </Text>
            <Text fontSize="lg">/</Text>
            <Text 
              fontSize="lg" 
              fontWeight="bold" 
              color="red.500"
              ml={2}
            >
              {formatCurrency(analytics.avgLoss)}
            </Text>
          </Flex>
          <Text fontSize="sm" color="gray.500">
            Win/Loss Ratio: {analytics.avgLoss !== 0 ? (Math.abs(analytics.avgWin / analytics.avgLoss)).toFixed(2) : 'N/A'}
          </Text>
        </Box>
        
        <Box 
          p={4} 
          borderWidth="1px" 
          borderRadius="md" 
          bg={cardBg}
          borderLeftWidth="4px"
          borderLeftColor={analytics.currentStreak.type === 'win' ? "green.400" : "red.400"}
        >
          <Text fontSize="sm" color="gray.500">Current Streak</Text>
          <Flex align="center">
            <Badge colorScheme={analytics.currentStreak.type === 'win' ? "green" : "red"} mr={2}>
              {analytics.currentStreak.type.toUpperCase()}
            </Badge>
            <Text fontSize="2xl" fontWeight="bold">
              {analytics.currentStreak.count}
            </Text>
          </Flex>
          <Flex>
            <Text fontSize="xs" color="gray.500" mr={1}>
              Max Win:
            </Text>
            <Text fontSize="xs" fontWeight="bold" color="green.500" mr={2}>
              {analytics.maxWinStreak}
            </Text>
            <Text fontSize="xs" color="gray.500" mr={1}>
              Max Loss:
            </Text>
            <Text fontSize="xs" fontWeight="bold" color="red.500">
              {analytics.maxLossStreak}
            </Text>
          </Flex>
        </Box>
      </SimpleGrid>
      
      <Box 
        height={`${height}px`} 
        width="100%"
        p={4}
        borderWidth="1px"
        borderRadius="md"
        bg={cardBg}
      >
        <Text fontWeight="medium" mb={4}>PnL by Date</Text>
        <ResponsiveContainer width="100%" height="90%">
          <ScatterChart
            margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="x" 
              name="Date" 
              tickFormatter={(date) => new Date(date).toLocaleDateString()}
              type="number"
              domain={['dataMin', 'dataMax']}
            />
            <YAxis 
              dataKey="y" 
              name="P&L" 
              unit="$" 
              tickFormatter={(pnl) => formatCurrency(pnl)}
            />
            <ZAxis 
              dataKey="z" 
              range={[40, 160]} 
              name="Size" 
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Scatter 
              name="Winning Trades" 
              data={winningTrades} 
              fill={winColor} 
            />
            <Scatter 
              name="Losing Trades" 
              data={losingTrades} 
              fill={lossColor} 
            />
          </ScatterChart>
        </ResponsiveContainer>
      </Box>
      
      <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4} mt={6}>
        <Box 
          p={4} 
          borderWidth="1px" 
          borderRadius="md" 
          bg={cardBg}
        >
          <Text fontWeight="medium" mb={2}>Time-of-day Analysis</Text>
          <Flex direction="column">
            <Flex align="center" mb={2}>
              <Box w="100px">
                <Text fontSize="sm" color="gray.500">Best Hour:</Text>
              </Box>
              <Badge colorScheme="green" fontSize="md" px={2}>
                {analytics.timeAnalysis.bestHour}
              </Badge>
            </Flex>
            <Flex align="center">
              <Box w="100px">
                <Text fontSize="sm" color="gray.500">Worst Hour:</Text>
              </Box>
              <Badge colorScheme="red" fontSize="md" px={2}>
                {analytics.timeAnalysis.worstHour}
              </Badge>
            </Flex>
          </Flex>
        </Box>
        
        <Box 
          p={4} 
          borderWidth="1px" 
          borderRadius="md" 
          bg={cardBg}
        >
          <Text fontWeight="medium" mb={2}>Performance Insights</Text>
          <Text fontSize="sm">
            This bot has a {(analytics.winRate * 100).toFixed(1)}% win rate with an average winning trade of 
            {formatCurrency(analytics.avgWin)} and average losing trade of {formatCurrency(analytics.avgLoss)}.
            {analytics.avgLoss !== 0 ? ` The win/loss ratio is ${(Math.abs(analytics.avgWin / analytics.avgLoss)).toFixed(2)}.` : ''}
            {analytics.currentStreak.count > 0 ? 
              ` Currently on a ${analytics.currentStreak.count} trade ${analytics.currentStreak.type} streak.` : 
              ''
            }
          </Text>
        </Box>
      </SimpleGrid>
    </Box>
  );
}