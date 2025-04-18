import React from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    AreaChart,
    Area
} from 'recharts';
import {
    Box,
    Text,
    Spinner,
    useColorModeValue,
    Alert,
    AlertIcon
} from '@chakra-ui/react';
import { format, parseISO } from 'date-fns';

// Define the structure of the data points expected by the chart
interface DataPoint {
    timestamp: string; // ISO 8601 string
    value: number | null; // Allow null for potential gaps
}

// Define the props for the MetricsChart component
interface MetricsChartProps {
    data: DataPoint[];
    metricName: string; // e.g., "Total PNL", "Rank Score"
    isLoading: boolean;
    error: Error | null;
    chartType?: 'line' | 'area'; // Optional: specify chart type
    color?: string; // Optional: specify line/area color
    height?: number | string; // Optional: specify chart height
}

const formatXAxis = (tickItem: string) => {
    // Format the timestamp for the X-axis
    try {
        return format(parseISO(tickItem), 'MM/dd HH:mm'); // Example format
    } catch (e) {
        return tickItem; // Fallback if parsing fails
    }
};

const formatTooltipLabel = (label: string) => {
    try {
        return format(parseISO(label), 'MMM d, yyyy HH:mm:ss');
    } catch (e) {
        return label;
    }
};

export const MetricsChart: React.FC<MetricsChartProps> = ({
    data,
    metricName,
    isLoading,
    error,
    chartType = 'line', // Default to line chart
    color,
    height = 300 // Default height
}) => {
    // Define colors based on theme
    const defaultChartColor = useColorModeValue('teal.500', 'teal.300');
    const gridColor = useColorModeValue('gray.200', 'gray.700');
    const textColor = useColorModeValue('gray.600', 'gray.400');

    const chartColor = color || defaultChartColor;

    if (isLoading) {
        return (
            <Box display="flex" justifyContent="center" alignItems="center" height={height}>
                <Spinner size="xl" color={chartColor} />
            </Box>
        );
    }

    if (error) {
        return (
            <Alert status="error" borderRadius="md" height={height}>
                <AlertIcon />
                Error loading chart data: {error.message}
            </Alert>
        );
    }

    if (!data || data.length === 0) {
        return (
            <Box display="flex" justifyContent="center" alignItems="center" height={height}>
                <Text color={textColor}>No historical data available for {metricName}.</Text>
            </Box>
        );
    }

    // Determine chart component based on type
    const ChartComponent = chartType === 'area' ? AreaChart : LineChart;
    const ChartElement = chartType === 'area' ? Area : Line;

    return (
        <Box height={height} width="100%">
            <ResponsiveContainer width="100%" height="100%">
                <ChartComponent
                    data={data}
                    margin={{
                        top: 5,
                        right: 30,
                        left: 20,
                        bottom: 5,
                    }}
                >
                    <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                    <XAxis
                        dataKey="timestamp"
                        tickFormatter={formatXAxis}
                        stroke={textColor}
                        fontSize="12px"
                    />
                    <YAxis
                        stroke={textColor}
                        fontSize="12px"
                        tickFormatter={(value) => value.toLocaleString()} // Format Y-axis numbers
                    />
                    <Tooltip
                        labelFormatter={formatTooltipLabel}
                        formatter={(value: number) => [value.toLocaleString(), metricName]} // Format tooltip content
                        contentStyle={{ backgroundColor: useColorModeValue('white', 'gray.800'), borderRadius: 'md' }}
                        labelStyle={{ color: useColorModeValue('black', 'white'), marginBottom: '5px' }}
                    />
                    {/* <Legend /> Optionally add legend if needed */}
                    <ChartElement
                        type="monotone"
                        dataKey="value"
                        name={metricName} // Name used in Tooltip/Legend
                        stroke={chartColor}
                        fill={chartType === 'area' ? chartColor : undefined}
                        fillOpacity={chartType === 'area' ? 0.3 : undefined}
                        activeDot={{ r: 6 }}
                        dot={false} // Hide dots for cleaner look on dense data
                        connectNulls={true} // Connect line/area across null data points
                    />
                </ChartComponent>
            </ResponsiveContainer>
        </Box>
    );
}; 