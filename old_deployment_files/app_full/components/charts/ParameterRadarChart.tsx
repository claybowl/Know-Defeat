import React from 'react';
import {
  Box,
  useColorModeValue
} from '@chakra-ui/react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
  Legend
} from 'recharts';

interface Parameter {
  name: string;
  value: number;
  normalized: number;
  isPercent?: boolean;
  isPeriod?: boolean;
}

// Define some standard parameter ranges to help with normalization
const parameterRanges: Record<string, { min: number; max: number; isHigherBetter?: boolean }> = {
  // Periods
  lookback_period: { min: 5, max: 50, isHigherBetter: false },
  moving_average_period: { min: 5, max: 50, isHigherBetter: false },
  
  // Percentages
  profit_target_pct: { min: 0.005, max: 0.05, isHigherBetter: true },
  stop_loss_pct: { min: 0.005, max: 0.03, isHigherBetter: false },
  trailing_stop_pct: { min: 0.005, max: 0.03, isHigherBetter: false },
  
  // Thresholds
  volatility_threshold: { min: 0.5, max: 3, isHigherBetter: true },
  
  // RSI values
  rsi_upper: { min: 65, max: 80, isHigherBetter: false },
  rsi_lower: { min: 20, max: 35, isHigherBetter: true },
  
  // Default range for unknown parameters
  default: { min: 0, max: 100, isHigherBetter: true }
};

// Function to normalize a parameter value to 0-100 scale for the radar chart
function normalizeParameterValue(name: string, value: number): number {
  const paramInfo = parameterRanges[name.toLowerCase()] || parameterRanges.default;
  const { min, max, isHigherBetter = true } = paramInfo;
  
  // Calculate normalized value between 0 and 1
  let normalized = (value - min) / (max - min);
  
  // Clamp the value between 0 and 1
  normalized = Math.max(0, Math.min(1, normalized));
  
  // For parameters where lower values are better, invert the scale
  if (!isHigherBetter) {
    normalized = 1 - normalized;
  }
  
  // Scale to 0-100 for radar chart
  return normalized * 100;
}

// Convert the raw parameters to the format needed for the radar chart
function prepareParameterData(parameters: Record<string, any>): Parameter[] {
  return Object.entries(parameters).map(([name, value]) => {
    const numValue = typeof value === 'number' ? value : parseFloat(value.toString());
    // Detect parameter types
    const isPercent = name.includes('pct') || name.includes('percent') || name.includes('threshold');
    const isPeriod = name.includes('period') || name.includes('lookback');
    
    return {
      name: name.replace(/_/g, ' ').replace(/pct/g, '%'),
      value: numValue,
      normalized: normalizeParameterValue(name, numValue),
      isPercent,
      isPeriod
    };
  });
}

// Prepare data for radar display
const prepareRadarData = (parameters: Parameter[]) => {
  // Create data in the format required by recharts
  const radarData = parameters.map(param => ({
    subject: param.name,
    A: param.normalized,
    fullMark: 100,
    originalValue: param.value,
    isPercent: param.isPercent,
    isPeriod: param.isPeriod
  }));
  
  return radarData;
};

interface ParameterRadarChartProps {
  parameters: Record<string, any>;
  height?: number;
}

export default function ParameterRadarChart({ parameters, height = 300 }: ParameterRadarChartProps) {
  const chartColor = useColorModeValue('#3182CE', '#63B3ED');
  const chartColorFill = useColorModeValue('rgba(49, 130, 206, 0.2)', 'rgba(99, 179, 237, 0.2)');
  
  // Process parameters for display
  const processedParams = prepareParameterData(parameters);
  const radarData = prepareRadarData(processedParams);
  
  // Custom tooltip for the radar chart
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      let displayValue = data.originalValue;
      
      // Format the display value based on parameter type
      if (data.isPercent) {
        displayValue = (displayValue * 100).toFixed(2) + '%';
      } else if (data.isPeriod) {
        displayValue = displayValue.toString() + (parseInt(displayValue) === 1 ? ' period' : ' periods');
      } else {
        displayValue = displayValue.toFixed(2);
      }
      
      return (
        <Box 
          bg={useColorModeValue('white', 'gray.800')} 
          p={2} 
          shadow="md" 
          borderRadius="md" 
          borderWidth="1px"
        >
          <Box fontWeight="bold" textTransform="capitalize">
            {data.subject}
          </Box>
          <Box>Value: {displayValue}</Box>
        </Box>
      );
    }
    
    return null;
  };
  
  return (
    <Box h={`${height}px`} w="100%">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
          <PolarGrid />
          <PolarAngleAxis 
            dataKey="subject" 
            tick={{ 
              fontSize: 12, 
              fill: useColorModeValue('gray.700', 'gray.300'),
              fontWeight: 'bold',
              textTransform: 'capitalize'
            }} 
          />
          <PolarRadiusAxis 
            angle={30} 
            domain={[0, 100]} 
            axisLine={false}
            tick={false}
          />
          <Radar 
            name="Parameter Values" 
            dataKey="A" 
            stroke={chartColor} 
            fill={chartColorFill} 
            fillOpacity={0.6} 
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
        </RadarChart>
      </ResponsiveContainer>
    </Box>
  );
}