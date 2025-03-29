import React from 'react';
import {
  Box,
  Text,
  Flex,
  useColorModeValue,
} from '@chakra-ui/react';
import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';

interface ParameterHistogramProps {
  name: string;
  value: number;
  range: { min: number; max: number };
  isPercent?: boolean;
  isLowerBetter?: boolean;
  height?: number;
}

export default function ParameterHistogram({ 
  name, 
  value, 
  range,
  isPercent = false,
  isLowerBetter = false,
  height = 120 
}: ParameterHistogramProps) {
  const barColor = useColorModeValue("#3182CE", "#63B3ED");
  const textColor = useColorModeValue("gray.800", "white");
  
  // Generate histogram bins data
  const generateHistogramData = () => {
    const { min, max } = range;
    const numBins = 20; // Number of bins for the histogram
    const binSize = (max - min) / numBins;
    
    let data = [];
    
    for (let i = 0; i < numBins; i++) {
      const binStart = min + i * binSize;
      const binEnd = binStart + binSize;
      const binMiddle = (binStart + binEnd) / 2;
      
      // The current value belongs to this bin
      const isValueBin = value >= binStart && value < binEnd;
      
      data.push({
        bin: binMiddle,
        value: isValueBin ? 1 : 0, // We just want to highlight the bin where the value falls
        isValueBin,
      });
    }
    
    return data;
  };
  
  const histogramData = generateHistogramData();
  
  // Format value for display
  const formatValue = (val: number) => {
    if (isPercent) {
      return (val * 100).toFixed(2) + '%';
    }
    
    // If value is integer, don't show decimal places
    if (Number.isInteger(val)) {
      return val.toString();
    }
    
    return val.toFixed(2);
  };
  
  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length && payload[0].payload.isValueBin) {
      return (
        <Box 
          bg={useColorModeValue("white", "gray.800")}
          p={2}
          borderRadius="md"
          shadow="md"
          borderWidth="1px"
        >
          <Text fontWeight="bold">Current Value</Text>
          <Text>{formatValue(value)}</Text>
        </Box>
      );
    }
    
    return null;
  };
  
  // Calculate whether the current value is in a good range
  const calculateQuality = () => {
    const { min, max } = range;
    const normalizedValue = (value - min) / (max - min); // 0 to 1
    
    // For parameters where lower is better, invert the scale
    const adjustedValue = isLowerBetter ? 1 - normalizedValue : normalizedValue;
    
    if (adjustedValue >= 0.8) return "Excellent";
    if (adjustedValue >= 0.6) return "Good";
    if (adjustedValue >= 0.4) return "Average";
    if (adjustedValue >= 0.2) return "Below Average";
    return "Poor";
  };
  
  const quality = calculateQuality();
  const qualityColors = {
    "Excellent": "green.500",
    "Good": "green.400",
    "Average": "yellow.500",
    "Below Average": "orange.500",
    "Poor": "red.500"
  };

  return (
    <Box>
      <Flex justify="space-between" align="center" mb={1}>
        <Text fontWeight="semibold" textTransform="capitalize">{name.replace(/_/g, ' ')}</Text>
        <Flex align="center">
          <Text color={qualityColors[quality as keyof typeof qualityColors]} mr={2} fontSize="sm" fontWeight="medium">
            {quality}
          </Text>
          <Text fontWeight="bold" color={textColor}>{formatValue(value)}</Text>
        </Flex>
      </Flex>
      
      <Box height={`${height}px`} width="100%">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={histogramData}
            margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis 
              dataKey="bin" 
              tickFormatter={(bin) => formatValue(bin)}
              interval="preserveEnd"
              tick={{ fontSize: 10 }}
              tickCount={5}
              axisLine={{ stroke: useColorModeValue('#E2E8F0', '#4A5568') }}
            />
            <YAxis hide={true} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="value" fill={barColor}>
              {histogramData.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={entry.isValueBin ? barColor : useColorModeValue('#EDF2F7', '#2D3748')} 
                  opacity={entry.isValueBin ? 1 : 0.3}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Box>
      
      <Flex justify="space-between" fontSize="xs" color="gray.500" mt={1}>
        <Text>{formatValue(range.min)}</Text>
        <Text>Range</Text>
        <Text>{formatValue(range.max)}</Text>
      </Flex>
    </Box>
  );
}