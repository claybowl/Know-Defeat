import React, { useState, useEffect } from 'react';
import {
    Modal,
    ModalOverlay,
    ModalContent,
    ModalHeader,
    ModalFooter,
    ModalBody,
    ModalCloseButton,
    Button,
    Box,
    Spinner,
    Alert,
    AlertIcon,
    Text,
    SimpleGrid,
    Stat,
    StatLabel,
    StatNumber,
    StatHelpText,
    VStack,
    HStack,
    Select,
    Tag,
    useColorModeValue,
    Divider,
} from '@chakra-ui/react';
import useSWR from 'swr';
import { format, parseISO, formatDistanceToNow } from 'date-fns';
import { MetricsChart } from '../charts/MetricsChart'; // Adjust path if necessary

// Match the structure returned by the GET /api/metrics/bots/{bot_id} endpoint
interface BotDetailData {
    bot_id: number;
    name: string;
    ticker: string;
    algorithm_module: string;
    algorithm_type: string;
    trade_direction: string;
    position_size: number;
    trailing_stop_pct: number;
    description?: string;
    version?: string;
    is_active: boolean;
    created_at: string; // ISO string
    total_trades?: number;
    winning_trades?: number;
    losing_trades?: number;
    total_pnl?: number;
    average_pnl_per_trade?: number;
    win_rate?: number;
    average_win_amount?: number;
    average_loss_amount?: number;
    profit_factor?: number;
    max_drawdown?: number;
    sharpe_ratio?: number;
    risk_reward_ratio?: number;
    expectancy?: number;
    rank_score?: number;
    last_updated?: string; // ISO string
}

// Match the structure returned by the GET /api/metrics/bots/{bot_id}/history endpoint
interface HistoryData {
    data: { timestamp: string; value: number | null }[];
    metric: string;
    // Add other fields if needed
}

interface BotDetailModalProps {
    isOpen: boolean;
    onClose: () => void;
    botId: number | null; // Use null when no bot is selected
}

// Basic fetcher function for useSWR
const fetcher = async (url: string) => {
    const res = await fetch(url);
    if (!res.ok) {
        let errorInfo = 'Failed to fetch data';
        try {
            errorInfo = await res.json();
        } catch (e) { /* Ignore if response is not JSON */ }
        const error = new Error('An error occurred while fetching the data.') as any;
        error.info = errorInfo;
        error.status = res.status;
        throw error;
    }
    return res.json();
};

// Helper to format numbers nicely
const formatNumber = (num: number | null | undefined, precision = 2): string => {
    if (num === null || num === undefined) return 'N/A';
    return num.toLocaleString(undefined, { minimumFractionDigits: precision, maximumFractionDigits: precision });
};

const formatPercent = (num: number | null | undefined, precision = 2): string => {
    if (num === null || num === undefined) return 'N/A';
    return `${(num * 100).toFixed(precision)}%`;
};

const formatDate = (dateString: string | null | undefined): string => {
    if (!dateString) return 'N/A';
    try {
        const date = parseISO(dateString);
        return `${format(date, 'yyyy-MM-dd HH:mm:ss')} (${formatDistanceToNow(date, { addSuffix: true })})`;
    } catch (e) {
        return 'Invalid Date';
    }
};

// Define metrics available for charting
const CHARTABLE_METRICS = {
    total_pnl: 'Total PNL',
    rank_score: 'Rank Score',
    win_rate: 'Win Rate',
    sharpe_ratio: 'Sharpe Ratio',
    total_trades: 'Total Trades',
    max_drawdown: 'Max Drawdown',
    // Add more as needed and implemented in the history endpoint
};

export const BotDetailModal: React.FC<BotDetailModalProps> = ({ isOpen, onClose, botId }) => {
    const [selectedMetric, setSelectedMetric] = useState<string>('total_pnl');
    const [selectedTimespan, setSelectedTimespan] = useState<string>('7d');

    const bgColor = useColorModeValue('gray.50', 'gray.700');
    const headerColor = useColorModeValue('gray.800', 'whiteAlpha.900');
    const statLabelColor = useColorModeValue('gray.500', 'gray.400');

    // Fetch Bot Detail Data
    const { data: botData, error: detailError, isLoading: isLoadingDetail } = useSWR<BotDetailData>(
        botId ? `/api/metrics/bots/${botId}` : null, // Only fetch if botId is not null
        fetcher
    );

    // Fetch Historical Chart Data
    const { data: historyData, error: historyError, isLoading: isLoadingHistory } = useSWR<HistoryData>(
        botId ? `/api/metrics/bots/${botId}/history?metric=${selectedMetric}&timespan=${selectedTimespan}` : null,
        fetcher,
        { refreshInterval: 60000 } // Optional: Refresh chart data periodically
    );

    // Reset chart selection when modal opens/bot changes
    useEffect(() => {
        if (isOpen) {
            setSelectedMetric('total_pnl');
            setSelectedTimespan('7d');
        }
    }, [isOpen, botId]);

    const renderContent = () => {
        if (isLoadingDetail) {
            return <Box textAlign="center" p={10}><Spinner size="xl" /></Box>;
        }

        if (detailError) {
            return (
                <Alert status="error">
                    <AlertIcon />
                    Error loading bot details: {detailError.message}
                    {detailError.info && <Text fontSize="sm">({JSON.stringify(detailError.info)})</Text>}
                </Alert>
            );
        }

        if (!botData) {
            return <Text>No bot data found.</Text>; // Should not happen if botId is valid, but good practice
        }

        return (
            <VStack spacing={6} align="stretch">
                {/* --- Basic Info --- */}
                <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                    <Stat>
                        <StatLabel color={statLabelColor}>Algorithm</StatLabel>
                        <StatNumber fontSize="lg">{botData.algorithm_type}</StatNumber>
                        <StatHelpText>{botData.algorithm_module}</StatHelpText>
                    </Stat>
                    <Stat>
                        <StatLabel color={statLabelColor}>Status</StatLabel>
                        <StatNumber fontSize="lg">
                            <Tag size="md" variant="subtle" colorScheme={botData.is_active ? 'green' : 'red'}>
                                {botData.is_active ? 'Active' : 'Inactive'}
                            </Tag>
                        </StatNumber>
                        <StatHelpText>Created: {formatDate(botData.created_at)}</StatHelpText>
                    </Stat>
                    <Stat>
                        <StatLabel color={statLabelColor}>Configuration</StatLabel>
                        <StatNumber fontSize="sm">Pos Size: {formatNumber(botData.position_size)}</StatNumber>
                        <StatHelpText>Dir: {botData.trade_direction}, Trail: {formatPercent(botData.trailing_stop_pct, 3)}</StatHelpText>
                    </Stat>
                     <Stat>
                        <StatLabel color={statLabelColor}>Last Update</StatLabel>
                        <StatNumber fontSize="lg">{formatDate(botData.last_updated)}</StatNumber>
                    </Stat>
                </SimpleGrid>

                {/* --- Performance Metrics --- */}
                <Divider />
                <Text fontSize="xl" fontWeight="semibold">Performance Metrics</Text>
                <SimpleGrid columns={{ base: 2, md: 3, lg: 4 }} spacing={4}>
                    <Stat>
                        <StatLabel color={statLabelColor}>Total PNL</StatLabel>
                        <StatNumber color={ (botData.total_pnl ?? 0) >= 0 ? 'green.500' : 'red.500' }>
                            ${formatNumber(botData.total_pnl)}
                         </StatNumber>
                    </Stat>
                    <Stat>
                        <StatLabel color={statLabelColor}>Win Rate</StatLabel>
                        <StatNumber>{formatPercent(botData.win_rate)}</StatNumber>
                        <StatHelpText>{botData.winning_trades ?? 'N/A'} Wins / {botData.losing_trades ?? 'N/A'} Losses</StatHelpText>
                    </Stat>
                    <Stat>
                        <StatLabel color={statLabelColor}>Total Trades</StatLabel>
                        <StatNumber>{botData.total_trades ?? 'N/A'}</StatNumber>
                    </Stat>
                    <Stat>
                        <StatLabel color={statLabelColor}>Profit Factor</StatLabel>
                        <StatNumber>{formatNumber(botData.profit_factor)}</StatNumber>
                    </Stat>
                    <Stat>
                        <StatLabel color={statLabelColor}>Max Drawdown</StatLabel>
                        <StatNumber color="red.500">${formatNumber(botData.max_drawdown)}</StatNumber>
                    </Stat>
                    <Stat>
                        <StatLabel color={statLabelColor}>Avg PNL/Trade</StatLabel>
                        <StatNumber>${formatNumber(botData.average_pnl_per_trade)}</StatNumber>
                    </Stat>
                    <Stat>
                        <StatLabel color={statLabelColor}>Sharpe Ratio</StatLabel>
                        <StatNumber>{formatNumber(botData.sharpe_ratio)}</StatNumber>
                    </Stat>
                     <Stat>
                        <StatLabel color={statLabelColor}>Rank Score</StatLabel>
                        <StatNumber>{formatNumber(botData.rank_score)}</StatNumber>
                    </Stat>
                     {/* Add more Stat components for other metrics as needed */}
                </SimpleGrid>

                {/* --- Historical Chart --- */}
                <Divider />
                <Text fontSize="xl" fontWeight="semibold">Historical Performance</Text>
                <HStack spacing={4} mb={4}>
                    <Select
                        value={selectedMetric}
                        onChange={(e) => setSelectedMetric(e.target.value)}
                        size="sm"
                        maxWidth="200px"
                    >
                        {Object.entries(CHARTABLE_METRICS).map(([key, value]) => (
                            <option key={key} value={key}>{value}</option>
                        ))}
                    </Select>
                    <Select
                        value={selectedTimespan}
                        onChange={(e) => setSelectedTimespan(e.target.value)}
                        size="sm"
                        maxWidth="150px"
                    >
                        <option value="1h">1 Hour</option>
                        <option value="6h">6 Hours</option>
                        <option value="1d">1 Day</option>
                        <option value="7d">7 Days</option>
                        <option value="1mo">1 Month</option>
                        <option value="3mo">3 Months</option>
                        <option value="all">All Time</option>
                    </Select>
                </HStack>
                <MetricsChart
                    data={historyData?.data || []}
                    metricName={CHARTABLE_METRICS[selectedMetric] || selectedMetric}
                    isLoading={isLoadingHistory}
                    error={historyError}
                    height={350} // Increase height for better visibility
                    // chartType='area' // Optionally change chart type
                />
            </VStack>
        );
    };

    return (
        <Modal isOpen={isOpen} onClose={onClose} size="4xl" scrollBehavior="inside">
            <ModalOverlay />
            <ModalContent mx={4}>
                <ModalHeader backgroundColor={bgColor} color={headerColor} borderTopRadius="md">
                    Bot Detail: {botData?.name ?? 'Loading...'} ({botData?.ticker ?? 'N/A'})
                    {botData?.version && <Tag size="sm" ml={2}>v{botData.version}</Tag>}
                </ModalHeader>
                <ModalCloseButton />
                <ModalBody py={6}>
                    {renderContent()}
                </ModalBody>
                <ModalFooter borderTopWidth="1px">
                    <Button onClick={onClose}>Close</Button>
                </ModalFooter>
            </ModalContent>
        </Modal>
    );
}; 