import React, { useState, useEffect } from 'react';
import {
    Box,
    Heading,
    VStack,
    useDisclosure,
    Spinner,
    Text,
    Alert,
    AlertIcon,
    Table,
    Thead,
    Tbody,
    Tr,
    Th,
    Td,
    TableContainer,
    HStack,
    Select,
    Input,
    Button,
    useToast,
    Flex,
    Spacer,
    Badge
} from '@chakra-ui/react';
import { BotDetailModal } from '../components/modals/BotDetailModal'; // Adjust path

// Data types for API response
interface BotSummary {
    bot_id: number;
    name: string;
    ticker: string;
    algorithm_type: string;
    is_active: boolean;
    total_pnl: number | null;
    win_rate: number | null;
    sharpe_ratio: number | null;
    max_drawdown: number | null;
    total_trades: number | null;
    rank_score: number | null;
    last_updated: string | null;
}

interface Pagination {
    current_page: number;
    per_page: number;
    total_items: number;
    total_pages: number;
}

interface BotListResponse {
    data: BotSummary[];
    pagination: Pagination;
}

// Props for the real table component
interface BotListTableProps {
    onViewDetails: (botId: number) => void;
}

const BotListTable: React.FC<BotListTableProps> = ({ onViewDetails }) => {
    const toast = useToast();
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [bots, setBots] = useState<BotSummary[]>([]);
    const [pagination, setPagination] = useState<Pagination>({
        current_page: 1,
        per_page: 25,
        total_items: 0,
        total_pages: 0
    });
    
    // Filters and sorting state
    const [ticker, setTicker] = useState<string>('');
    const [algorithm, setAlgorithm] = useState<string>('');
    const [sortBy, setSortBy] = useState<string>('rank_score');
    const [sortOrder, setSortOrder] = useState<string>('desc');
    
    // Function to load bots from API
    const loadBots = async () => {
        setLoading(true);
        
        try {
            // Build query parameters
            const params = new URLSearchParams({
                page: pagination.current_page.toString(),
                per_page: pagination.per_page.toString(),
                sort_by: sortBy,
                sort_order: sortOrder
            });
            
            if (ticker) params.append('ticker', ticker);
            if (algorithm) params.append('algorithm_type', algorithm);
            
            // Make API request
            const response = await fetch(`/api/metrics/bots?${params.toString()}`);
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }
            
            const data: BotListResponse = await response.json();
            setBots(data.data);
            setPagination(data.pagination);
            setError(null);
            
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
            toast({
                title: 'Error loading data',
                description: err instanceof Error ? err.message : 'Failed to load bot data',
                status: 'error',
                duration: 5000,
                isClosable: true,
            });
        } finally {
            setLoading(false);
        }
    };
    
    // Load data on initial render and when filters/sorting/pagination changes
    useEffect(() => {
        loadBots();
    }, [pagination.current_page, sortBy, sortOrder]);
    
    // Handler for applying filters
    const handleFilterApply = () => {
        setPagination(prev => ({ ...prev, current_page: 1 })); // Reset to first page
        loadBots();
    };
    
    // Handler for changing page
    const handlePageChange = (newPage: number) => {
        setPagination(prev => ({ ...prev, current_page: newPage }));
    };

    if (loading && bots.length === 0) {
        return (
            <Box textAlign="center" py={10}>
                <Spinner size="xl" />
                <Text mt={4}>Loading bot data...</Text>
            </Box>
        );
    }

    if (error && bots.length === 0) {
        return (
            <Alert status="error" borderRadius="md">
                <AlertIcon />
                <Box>
                    <Text fontWeight="bold">Failed to load bot data</Text>
                    <Text fontSize="sm">{error}</Text>
                    <Button mt={2} size="sm" onClick={loadBots}>Retry</Button>
                </Box>
            </Alert>
        );
    }

    // Format currency
    const formatCurrency = (value: number | null) => {
        if (value === null) return 'N/A';
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2
        }).format(value);
    };
    
    // Format percentage
    const formatPercent = (value: number | null) => {
        if (value === null) return 'N/A';
        return new Intl.NumberFormat('en-US', {
            style: 'percent',
            minimumFractionDigits: 2
        }).format(value);
    };

    return (
        <Box borderWidth="1px" borderRadius="lg" p={4} shadow="sm">
            {/* Filters */}
            <Flex mb={4} wrap="wrap" gap={2}>
                <Input 
                    placeholder="Filter by ticker" 
                    value={ticker} 
                    onChange={(e) => setTicker(e.target.value)}
                    maxWidth="200px"
                />
                <Input 
                    placeholder="Filter by algorithm" 
                    value={algorithm} 
                    onChange={(e) => setAlgorithm(e.target.value)}
                    maxWidth="200px"
                />
                <Button colorScheme="blue" onClick={handleFilterApply}>Apply Filters</Button>
                <Spacer />
                <Select 
                    value={sortBy} 
                    onChange={(e) => setSortBy(e.target.value)}
                    maxWidth="200px"
                >
                    <option value="rank_score">Rank Score</option>
                    <option value="total_pnl">Total PnL</option>
                    <option value="win_rate">Win Rate</option>
                    <option value="sharpe_ratio">Sharpe Ratio</option>
                    <option value="bot_id">Bot ID</option>
                </Select>
                <Select 
                    value={sortOrder} 
                    onChange={(e) => setSortOrder(e.target.value)}
                    maxWidth="120px"
                >
                    <option value="desc">Desc</option>
                    <option value="asc">Asc</option>
                </Select>
            </Flex>
            
            {/* Bot table */}
            <TableContainer>
                <Table variant="simple" size="sm">
                    <Thead>
                        <Tr>
                            <Th>ID</Th>
                            <Th>Name</Th>
                            <Th>Ticker</Th>
                            <Th>Algorithm</Th>
                            <Th isNumeric>Total PnL</Th>
                            <Th isNumeric>Win Rate</Th>
                            <Th isNumeric>Sharpe Ratio</Th>
                            <Th isNumeric>Rank Score</Th>
                            <Th>Status</Th>
                        </Tr>
                    </Thead>
                    <Tbody>
                        {bots.map((bot) => (
                            <Tr 
                                key={bot.bot_id}
                                onClick={() => onViewDetails(bot.bot_id)}
                                _hover={{ bg: "gray.50", cursor: "pointer" }}
                            >
                                <Td>{bot.bot_id}</Td>
                                <Td>{bot.name}</Td>
                                <Td>{bot.ticker}</Td>
                                <Td>{bot.algorithm_type}</Td>
                                <Td isNumeric>{bot.total_pnl !== null ? formatCurrency(bot.total_pnl) : 'N/A'}</Td>
                                <Td isNumeric>{bot.win_rate !== null ? formatPercent(bot.win_rate) : 'N/A'}</Td>
                                <Td isNumeric>{bot.sharpe_ratio !== null ? bot.sharpe_ratio.toFixed(2) : 'N/A'}</Td>
                                <Td isNumeric>{bot.rank_score !== null ? bot.rank_score.toFixed(2) : 'N/A'}</Td>
                                <Td>
                                    <Badge 
                                        colorScheme={bot.is_active ? 'green' : 'gray'}
                                    >
                                        {bot.is_active ? 'Active' : 'Inactive'}
                                    </Badge>
                                </Td>
                            </Tr>
                        ))}
                    </Tbody>
                </Table>
            </TableContainer>
            
            {/* Pagination */}
            {pagination.total_pages > 1 && (
                <Flex justify="center" mt={4}>
                    <HStack>
                        <Button 
                            size="sm" 
                            onClick={() => handlePageChange(1)}
                            isDisabled={pagination.current_page === 1}
                        >
                            First
                        </Button>
                        <Button 
                            size="sm"
                            onClick={() => handlePageChange(pagination.current_page - 1)}
                            isDisabled={pagination.current_page === 1}
                        >
                            Prev
                        </Button>
                        
                        <Text>
                            Page {pagination.current_page} of {pagination.total_pages}
                        </Text>
                        
                        <Button 
                            size="sm"
                            onClick={() => handlePageChange(pagination.current_page + 1)}
                            isDisabled={pagination.current_page === pagination.total_pages}
                        >
                            Next
                        </Button>
                        <Button 
                            size="sm"
                            onClick={() => handlePageChange(pagination.total_pages)}
                            isDisabled={pagination.current_page === pagination.total_pages}
                        >
                            Last
                        </Button>
                    </HStack>
                </Flex>
            )}
            
            {/* Summary */}
            <Text fontSize="sm" textAlign="right" mt={2} color="gray.600">
                Showing {bots.length} of {pagination.total_items} bots
            </Text>
        </Box>
    );
};


const MetricsDashboard: React.FC = () => {
    const { isOpen, onOpen, onClose } = useDisclosure(); // Hook to manage modal state
    const [selectedBotId, setSelectedBotId] = useState<number | null>(null);

    // Callback function passed to the table to open the modal for a specific bot
    const handleViewDetails = (botId: number) => {
        setSelectedBotId(botId);
        onOpen(); // Open the modal
    };

    // Function to handle closing the modal
    const handleCloseModal = () => {
        onClose();
        setSelectedBotId(null); // Clear selected ID when closing
    };

    return (
        <Box p={{ base: 4, md: 6 }}>
            <VStack spacing={6} align="stretch">
                <Heading as="h1" size="xl">
                    Bot Metrics Dashboard
                </Heading>

                {/* Render the real Bot List Table */}
                <BotListTable onViewDetails={handleViewDetails} />

            </VStack>

            {/* Render the Detail Modal (conditionally rendered is fine) */}
            {selectedBotId !== null && (
                <BotDetailModal
                    isOpen={isOpen}
                    onClose={handleCloseModal}
                    botId={selectedBotId}
                />
            )}
        </Box>
    );
};

export default MetricsDashboard; 