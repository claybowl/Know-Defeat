import React from 'react';
import {
  Box,
  Text,
  Heading,
  Grid,
  GridItem,
  VStack,
  Button,
  useColorModeValue,
  Flex,
  Badge,
} from '@chakra-ui/react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { InfoIcon } from '@chakra-ui/icons';

// Sample data - in a real app, this would come from your API
const sampleBotData = Array.from({ length: 30 }, (_, i) => {
  // Generate realistic but random data
  const winRate = 0.3 + Math.random() * 0.5; // 30-80%
  const profitFactor = (Math.random() * 2) + 0.5; // 0.5-2.5
  const maxDrawdown = Math.random() * 0.2; // 0-20%
  const sharpeRatio = Math.random() * 2; // 0-2
  
  // Make some relationships appear in the data
  // Higher win rates tend to have higher profit factors
  const adjustedProfitFactor = profitFactor * (winRate > 0.5 ? 1.2 : 0.8);
  
  // Size relates to how many trades the bot has
  const totalTrades = 20 + Math.floor(Math.random() * 180); // 20-200 trades
  
  return {
    bot_id: i + 1,
    name: `Bot ${i + 1}`,
    win_rate: winRate,
    profit_factor: adjustedProfitFactor,
    max_drawdown: maxDrawdown,
    sharpe_ratio: sharpeRatio,
    total_trades: totalTrades,
    total_pnl: totalTrades * (adjustedProfitFactor > 1 ? 10 : -5) * Math.random() * 10,
  };
});

interface MetricRelationshipChartProps {
  data?: any[];
  primaryMetric?: string;
  secondaryMetric?: string; 
}

export default function MetricRelationshipChart({
  data = sampleBotData,
  primaryMetric = 'win_rate',
  secondaryMetric = 'profit_factor',
}: MetricRelationshipChartProps) {
  const [xAxis, setXAxis] = React.useState(primaryMetric);
  const [yAxis, setYAxis] = React.useState(secondaryMetric);
  
  const cardBg = useColorModeValue('white', 'gray.700');
  const tooltipBg = useColorModeValue('white', 'gray.700');
  const tooltipBorder = useColorModeValue('gray.200', 'gray.600');
  
  // Format values for display
  const formatValue = (key: string, value: number) => {
    if (key === 'win_rate') {
      return `${(value * 100).toFixed(1)}%`;
    } else if (key === 'max_drawdown') {
      return `-${(value * 100).toFixed(1)}%`;
    } else if (key === 'total_pnl') {
      return `$${value.toFixed(2)}`;
    } else {
      return value.toFixed(2);
    }
  };
  
  // Define axis configurations
  const axisConfig: Record<string, {name: string, domain: [number, number], tickFormatter?: (value: number) => string}> = {
    win_rate: {
      name: 'Win Rate',
      domain: [0, 1],
      tickFormatter: (value) => `${(value * 100).toFixed(0)}%`
    },
    profit_factor: {
      name: 'Profit Factor',
      domain: [0, 3],
    },
    sharpe_ratio: {
      name: 'Sharpe Ratio',
      domain: [0, 3],
    },
    max_drawdown: {
      name: 'Max Drawdown',
      domain: [0, 0.3],
      tickFormatter: (value) => `${(value * 100).toFixed(0)}%`
    },
    total_pnl: {
      name: 'Total P&L',
      domain: [-500, 500],
      tickFormatter: (value) => `$${value}`
    }
  };
  
  // Enhanced tooltip with guidance
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const bot = payload[0].payload;
      
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
          <VStack align="start" spacing={1}>
            <Text fontWeight="bold">Bot {bot.bot_id}</Text>
            <Text fontSize="sm">{`${axisConfig[xAxis].name}: ${formatValue(xAxis, bot[xAxis])}`}</Text>
            <Text fontSize="sm">{`${axisConfig[yAxis].name}: ${formatValue(yAxis, bot[yAxis])}`}</Text>
            <Text fontSize="sm">{`Total Trades: ${bot.total_trades}`}</Text>
            <Text fontSize="sm">{`Total P&L: ${formatValue('total_pnl', bot.total_pnl)}`}</Text>
            
            {/* Guidance based on metrics */}
            {xAxis === 'win_rate' && yAxis === 'profit_factor' && (
              <Box mt={2} p={2} bg={useColorModeValue('blue.50', 'blue.900')} borderRadius="md" w="100%">
                {bot.win_rate > 0.5 && bot.profit_factor > 1.5 && (
                  <Text fontSize="xs" fontWeight="medium">Strong performer with good win rate and profit factor</Text>
                )}
                {bot.win_rate > 0.6 && bot.profit_factor < 1 && (
                  <Text fontSize="xs" fontWeight="medium">High win rate but losses exceed wins - check position sizing</Text>
                )}
                {bot.win_rate < 0.4 && bot.profit_factor > 1.8 && (
                  <Text fontSize="xs" fontWeight="medium">Low win rate but excellent profit factor - good risk/reward setup</Text>
                )}
                {bot.win_rate < 0.5 && bot.profit_factor < 1 && (
                  <Text fontSize="xs" fontWeight="medium">Underperforming bot - consider strategy review</Text>
                )}
              </Box>
            )}
            
            {xAxis === 'profit_factor' && yAxis === 'max_drawdown' && (
              <Box mt={2} p={2} bg={useColorModeValue('blue.50', 'blue.900')} borderRadius="md" w="100%">
                {bot.profit_factor > 1.5 && bot.max_drawdown < 0.1 && (
                  <Text fontSize="xs" fontWeight="medium">Excellent risk-adjusted returns</Text>
                )}
                {bot.profit_factor > 1.5 && bot.max_drawdown > 0.15 && (
                  <Text fontSize="xs" fontWeight="medium">Profitable but high drawdown - consider reducing position size</Text>
                )}
              </Box>
            )}
          </VStack>
        </Box>
      );
    }
    return null;
  };
  
  // Get quadrant label based on x and y axes
  const getQuadrantLabel = (xKey: string, yKey: string) => {
    if (xKey === 'win_rate' && yKey === 'profit_factor') {
      return {
        q1: 'High Win Rate, High Profit Factor - Ideal',
        q2: 'Low Win Rate, High Profit Factor - Big Winners',
        q3: 'Low Win Rate, Low Profit Factor - Problematic',
        q4: 'High Win Rate, Low Profit Factor - Small Winners/Big Losers'
      };
    }
    
    if (xKey === 'profit_factor' && yKey === 'max_drawdown') {
      return {
        q1: 'High Profit Factor, High Drawdown - Volatile but Profitable',
        q2: 'Low Profit Factor, High Drawdown - Highly Problematic',
        q3: 'Low Profit Factor, Low Drawdown - Underperforming but Stable',
        q4: 'High Profit Factor, Low Drawdown - Ideal'
      };
    }
    
    // Default labels if no specific relationship is defined
    return {
      q1: 'High X, High Y',
      q2: 'Low X, High Y',
      q3: 'Low X, Low Y',
      q4: 'High X, Low Y'
    };
  };
  
  // Get dynamic reference line positions based on metrics
  const getReferenceLines = (xKey: string, yKey: string) => {
    // Default midpoints
    let xMid = (axisConfig[xKey].domain[0] + axisConfig[xKey].domain[1]) / 2;
    let yMid = (axisConfig[yKey].domain[0] + axisConfig[yKey].domain[1]) / 2;
    
    // Specific reference points for certain metric combinations
    if (xKey === 'win_rate') xMid = 0.5; // 50% win rate
    if (yKey === 'profit_factor') yMid = 1.0; // Profit factor of 1.0
    if (xKey === 'profit_factor') xMid = 1.0; // Profit factor of 1.0
    if (yKey === 'max_drawdown') yMid = 0.15; // 15% drawdown
    
    return { xMid, yMid };
  };
  
  // Get quadrant labels 
  const quadrantLabels = getQuadrantLabel(xAxis, yAxis);
  const { xMid, yMid } = getReferenceLines(xAxis, yAxis);
  
  // Common metric combinations as presets
  const presetCombinations = [
    { x: 'win_rate', y: 'profit_factor', label: 'Win Rate vs. Profit Factor' },
    { x: 'profit_factor', y: 'max_drawdown', label: 'Profit Factor vs. Max Drawdown' },
    { x: 'sharpe_ratio', y: 'total_pnl', label: 'Sharpe Ratio vs. Total P&L' },
  ];

  return (
    <Box>
      <Flex justify="space-between" mb={4} wrap="wrap">
        <Heading size="md">Metric Relationship Analysis</Heading>
        <Flex gap={2}>
          {presetCombinations.map((combo, index) => (
            <Button 
              key={index} 
              size="sm"
              variant={xAxis === combo.x && yAxis === combo.y ? 'solid' : 'outline'}
              colorScheme="blue"
              onClick={() => {
                setXAxis(combo.x);
                setYAxis(combo.y);
              }}
            >
              {combo.label}
            </Button>
          ))}
        </Flex>
      </Flex>
      
      <Grid
        templateRows={{ base: "auto auto", md: "1fr" }}
        templateColumns={{ base: "1fr", md: "3fr 1fr" }}
        gap={4}
        h={{ base: "auto", md: "500px" }}
      >
        <GridItem p={4} bg={cardBg} borderRadius="md" shadow="sm">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                type="number" 
                dataKey={xAxis} 
                name={axisConfig[xAxis].name}
                domain={axisConfig[xAxis].domain}
                tickFormatter={axisConfig[xAxis].tickFormatter}
                label={{ value: axisConfig[xAxis].name, position: 'bottom', offset: 5 }}
              />
              <YAxis 
                type="number" 
                dataKey={yAxis} 
                name={axisConfig[yAxis].name}
                domain={axisConfig[yAxis].domain}
                tickFormatter={axisConfig[yAxis].tickFormatter}
                label={{ value: axisConfig[yAxis].name, angle: -90, position: 'left', offset: 10 }}
              />
              <ZAxis type="number" dataKey="total_trades" range={[30, 300]} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              
              {/* Reference lines */}
              <ReferenceLine x={xMid} stroke="rgba(0,0,0,0.3)" strokeDasharray="3 3" />
              <ReferenceLine y={yMid} stroke="rgba(0,0,0,0.3)" strokeDasharray="3 3" />
              
              {/* Main scatter plot */}
              <Scatter
                name="Bots"
                data={data}
                fill="#8884d8"
                fillOpacity={0.6}
                shape="circle"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </GridItem>
        
        <GridItem p={4} bg={cardBg} borderRadius="md" shadow="sm">
          <VStack align="start" spacing={4}>
            <Heading size="sm">Quadrant Analysis</Heading>
            <Text fontSize="sm" fontStyle="italic">
              This chart shows the relationship between {axisConfig[xAxis].name} and {axisConfig[yAxis].name}.
              Each point represents a bot, with size indicating trade volume.
            </Text>
            
            <Box p={2} bg={useColorModeValue('green.50', 'green.900')} borderRadius="md" w="100%">
              <Text fontSize="xs" fontWeight="bold">Quadrant 1 (Top Right)</Text>
              <Text fontSize="xs">{quadrantLabels.q1}</Text>
            </Box>
            
            <Box p={2} bg={useColorModeValue('orange.50', 'orange.900')} borderRadius="md" w="100%">
              <Text fontSize="xs" fontWeight="bold">Quadrant 2 (Top Left)</Text>
              <Text fontSize="xs">{quadrantLabels.q2}</Text>
            </Box>
            
            <Box p={2} bg={useColorModeValue('red.50', 'red.900')} borderRadius="md" w="100%">
              <Text fontSize="xs" fontWeight="bold">Quadrant 3 (Bottom Left)</Text>
              <Text fontSize="xs">{quadrantLabels.q3}</Text>
            </Box>
            
            <Box p={2} bg={useColorModeValue('blue.50', 'blue.900')} borderRadius="md" w="100%">
              <Text fontSize="xs" fontWeight="bold">Quadrant 4 (Bottom Right)</Text>
              <Text fontSize="xs">{quadrantLabels.q4}</Text>
            </Box>
            
            {xAxis === 'win_rate' && yAxis === 'profit_factor' && (
              <Box mt={2} p={3} bg={useColorModeValue('gray.50', 'gray.800')} borderRadius="md" w="100%">
                <Text fontSize="sm" fontWeight="bold">Key Insights:</Text>
                <Text fontSize="xs" mt={1}>
                  Win rate alone doesn't guarantee profitability. A lower win rate with a high profit factor 
                  (meaning winners are much larger than losers) can outperform a high win rate with a low profit factor.
                </Text>
                <Text fontSize="xs" mt={2}>
                  <Badge colorScheme="green">Ideal Zone:</Badge> High win rate (&gt;50%) + high profit factor (&gt;1.5)
                </Text>
              </Box>
            )}
            
            {xAxis === 'profit_factor' && yAxis === 'max_drawdown' && (
              <Box mt={2} p={3} bg={useColorModeValue('gray.50', 'gray.800')} borderRadius="md" w="100%">
                <Text fontSize="sm" fontWeight="bold">Key Insights:</Text>
                <Text fontSize="xs" mt={1}>
                  Profit factor must be balanced against drawdown risk. Bots with similar profit factors can have 
                  very different risk profiles based on their maximum drawdown.
                </Text>
                <Text fontSize="xs" mt={2}>
                  <Badge colorScheme="green">Ideal Zone:</Badge> High profit factor (&gt;1.5) + low drawdown (&lt;15%)
                </Text>
              </Box>
            )}
          </VStack>
        </GridItem>
      </Grid>
    </Box>
  );
} 