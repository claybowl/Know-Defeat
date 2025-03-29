import { json, LoaderFunctionArgs } from '@remix-run/node';
import { useLoaderData, Link, useNavigate, useSearchParams } from '@remix-run/react';
import {
  Box,
  Heading,
  Text,
  Badge,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Card,
  CardHeader,
  CardBody,
  Button,
  HStack,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Flex,
  useColorModeValue,
  VStack,
  Input,
  InputGroup,
  InputLeftElement,
  Select,
  Stack,
  Tooltip,
  IconButton,
  Divider,
  Progress,
  Switch,
  AlertDialog,
  AlertDialogBody,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogContent,
  AlertDialogOverlay,
  useDisclosure,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  FormControl,
  FormLabel,
  Icon,
  useToast,
} from '@chakra-ui/react';
import { useState, useRef, useEffect } from 'react';
import { getBotById } from '~/lib/api.server';
import MainLayout from '~/components/layout/MainLayout';
import TradeHistoryChart from '~/components/charts/TradeHistoryChart';
import PerformanceMetricsChart from '~/components/charts/PerformanceMetricsChart';
import ParameterRadarChart from '~/components/charts/ParameterRadarChart';
import ParameterHistogram from '~/components/charts/ParameterHistogram';
import TradeAnalyticsChart from '~/components/charts/TradeAnalyticsChart';
import { 
  FiFilter, 
  FiRefreshCw, 
  FiAlertTriangle, 
  FiSettings, 
  FiTrendingUp, 
  FiBarChart2, 
  FiArrowLeft, 
  FiEdit, 
  FiDownload, 
  FiSave,
  FiCheck,
  FiX
} from 'react-icons/fi';

// Import EditBotModal component
import EditBotModal from '~/components/bot/EditBotModal';

export async function loader({ params }: LoaderFunctionArgs) {
  const botId = params.botId;
  if (!botId || isNaN(parseInt(botId))) {
    throw new Response('Bot ID is required', { status: 400 });
  }
  
  const bot = await getBotById(parseInt(botId));
  if (!bot) {
    throw new Response('Bot not found', { status: 404 });
  }
  
  return json({ bot });
}

function formatCurrency(value: number | string) {
  const numValue = typeof value === 'string' ? parseFloat(value) : value;
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(numValue);
}

function formatPercent(value: number | string) {
  const numValue = typeof value === 'string' ? parseFloat(value) : value;
  return (numValue * 100).toFixed(2) + '%';
}

export default function BotDetail() {
  const { bot } = useLoaderData<typeof loader>();
  const navigate = useNavigate();
  const toast = useToast();
  const [searchParams] = useSearchParams();
  const cardBg = useColorModeValue('white', 'gray.700');
  const [selectedMetric, setSelectedMetric] = useState<'win_rate' | 'profit_factor' | 'sharpe_ratio' | 'max_drawdown'>('win_rate');
  const [tradeFilter, setTradeFilter] = useState('all');
  const [dateRange, setDateRange] = useState('7d');
  const [isActive, setIsActive] = useState(bot.is_active);
  const [updatedBot, setUpdatedBot] = useState(bot);
  
  // For the emergency stop dialog
  const { 
    isOpen: isStopDialogOpen, 
    onOpen: onOpenStopDialog, 
    onClose: onCloseStopDialog 
  } = useDisclosure();
  
  // For the edit configuration modal
  const { 
    isOpen: isEditModalOpen, 
    onOpen: onOpenEditModal, 
    onClose: onCloseEditModal 
  } = useDisclosure();
  
  const cancelRef = useRef(null);
  
  // Check URL parameters to see if we should open the edit modal
  useEffect(() => {
    if (searchParams.get('edit') === 'true') {
      onOpenEditModal();
    }
  }, [searchParams, onOpenEditModal]);
  
  // Prepare bot metrics data for the performance chart if metrics exist
  const metricData = bot.metrics 
    ? [{
        bot_id: bot.bot_id,
        name: bot.name,
        win_rate: parseFloat(bot.metrics.win_rate),
        profit_factor: parseFloat(bot.metrics.profit_factor),
        sharpe_ratio: parseFloat(bot.metrics.sharpe_ratio),
        max_drawdown: parseFloat(bot.metrics.max_drawdown),
      }]
    : [];
    
  // Filter trades based on selected criteria
  const filteredTrades = bot.trades ? bot.trades.filter((trade: any) => {
    // Filter by trade status
    if (tradeFilter !== 'all' && trade.trade_status !== tradeFilter) return false;
    
    // Filter by date range
    if (dateRange !== 'all') {
      const tradeDate = new Date(trade.entry_time);
      const now = new Date();
      const daysDiff = Math.floor((now.getTime() - tradeDate.getTime()) / (1000 * 60 * 60 * 24));
      
      switch (dateRange) {
        case '1d': return daysDiff < 1;
        case '7d': return daysDiff < 7;
        case '30d': return daysDiff < 30;
        case '90d': return daysDiff < 90;
      }
    }
    
    return true;
  }) : [];
  
  // Function to handle bot activation toggle
  const handleBotToggle = () => {
    const newStatus = !isActive;
    setIsActive(newStatus);
    
    // Update the bot object
    setUpdatedBot(prev => ({
      ...prev,
      is_active: newStatus
    }));
    
    // Show toast notification
    toast({
      title: `Bot ${newStatus ? 'activated' : 'deactivated'}`,
      status: newStatus ? 'success' : 'info',
      duration: 2000,
    });
    
    // In a real implementation, this would call an API to update the bot's status
  };
  
  // Function to handle saving configuration changes
  const handleSaveConfig = (updatedBotConfig: any) => {
    // Update the bot state with new configuration
    setUpdatedBot(updatedBotConfig);
    
    // If active status changed, update that too
    if (updatedBotConfig.is_active !== isActive) {
      setIsActive(updatedBotConfig.is_active);
    }
    
    // Show toast notification
    toast({
      title: "Bot configuration updated",
      description: "Changes have been saved successfully",
      status: "success",
      duration: 3000,
      isClosable: true,
    });
    
    // In a real implementation, this would call an API to update the bot's configuration
  };
  
  // Handle emergency stop
  const handleEmergencyStop = () => {
    // Deactivate the bot
    setIsActive(false);
    setUpdatedBot(prev => ({
      ...prev,
      is_active: false
    }));
    
    // Show toast notification
    toast({
      title: "Emergency Stop Activated",
      description: "Bot has been stopped and all positions will be closed",
      status: "error",
      duration: 3000,
      isClosable: true,
    });
    
    // Close the dialog
    onCloseStopDialog();
    
    // In a real implementation, this would call an API to stop the bot and close positions
  };
  
  // Daily performance stats - would come from API in real implementation
  const dailyStats = {
    trades: bot.trades ? bot.trades.filter((t: any) => {
      const tradeDate = new Date(t.entry_time);
      const now = new Date();
      return now.getDate() === tradeDate.getDate() && 
             now.getMonth() === tradeDate.getMonth() &&
             now.getFullYear() === tradeDate.getFullYear();
    }).length : 0,
    pnl: bot.trades ? bot.trades.filter((t: any) => t.pnl && t.trade_status === 'closed')
      .reduce((sum: number, t: any) => sum + parseFloat(t.pnl), 0) : 0,
    winRate: bot.metrics ? parseFloat(bot.metrics.win_rate) : 0,
    lastTrade: bot.trades && bot.trades.length > 0 ? new Date(bot.trades[0].entry_time).toLocaleString() : 'None'
  };
  
  return (
    <MainLayout>
      <Flex justify="space-between" align="center" mb={6}>
        <HStack>
          <IconButton
            aria-label="Back to bots"
            icon={<FiArrowLeft />}
            variant="ghost"
            onClick={() => navigate('/bots')}
          />
          <Box>
            <Heading size="lg">{updatedBot.name}</Heading>
            <Text fontSize="md" color="gray.500">ID: {updatedBot.bot_id}</Text>
          </Box>
        </HStack>
        <HStack spacing={4}>
          <Flex align="center" bg={cardBg} px={4} py={2} borderRadius="lg" shadow="sm">
            <Text mr={2}>Active</Text>
            <Switch 
              colorScheme="green" 
              isChecked={isActive} 
              onChange={handleBotToggle}
              size="lg"
            />
          </Flex>
          <Button 
            colorScheme="red" 
            leftIcon={<FiAlertTriangle />} 
            onClick={onOpenStopDialog}
            isDisabled={!isActive}
          >
            Emergency Stop
          </Button>
          <Button 
            colorScheme="blue" 
            leftIcon={<FiEdit />}
            onClick={onOpenEditModal}
          >
            Edit Config
          </Button>
        </HStack>
      </Flex>
      
      {/* Emergency Stop Dialog */}
      <AlertDialog
        isOpen={isStopDialogOpen}
        leastDestructiveRef={cancelRef}
        onClose={onCloseStopDialog}
      >
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              Emergency Stop Bot
            </AlertDialogHeader>

            <AlertDialogBody>
              Are you sure? This will immediately stop all trading operations and close any open positions for this bot.
            </AlertDialogBody>

            <AlertDialogFooter>
              <Button ref={cancelRef} onClick={onCloseStopDialog}>
                Cancel
              </Button>
              <Button colorScheme="red" onClick={handleEmergencyStop} ml={3}>
                Stop Bot
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
      
      {/* Edit Configuration Modal */}
      <EditBotModal
        isOpen={isEditModalOpen}
        onClose={onCloseEditModal}
        bot={updatedBot}
        onSave={handleSaveConfig}
      />
      
      {/* Daily Stats Cards */}
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6} mb={8}>
        <Card bg={cardBg} shadow="md" borderLeftWidth="4px" borderLeftColor="blue.500">
          <CardBody py={3}>
            <Flex justify="space-between" align="center">
              <Box>
                <Text fontSize="sm" color="gray.500">Today's Trades</Text>
                <Text fontSize="2xl" fontWeight="bold">{dailyStats.trades}</Text>
              </Box>
              <Box 
                bg="blue.50" 
                p={2} 
                borderRadius="full"
                color="blue.500"
              >
                <FiBarChart2 size="24px" />
              </Box>
            </Flex>
          </CardBody>
        </Card>
        
        <Card 
          bg={cardBg} 
          shadow="md" 
          borderLeftWidth="4px" 
          borderLeftColor={dailyStats.pnl >= 0 ? "green.500" : "red.500"}
        >
          <CardBody py={3}>
            <Flex justify="space-between" align="center">
              <Box>
                <Text fontSize="sm" color="gray.500">Today's P&L</Text>
                <Text 
                  fontSize="2xl" 
                  fontWeight="bold"
                  color={dailyStats.pnl >= 0 ? "green.500" : "red.500"}
                >
                  {formatCurrency(dailyStats.pnl)}
                </Text>
              </Box>
              <Box 
                bg={dailyStats.pnl >= 0 ? "green.50" : "red.50"} 
                p={2} 
                borderRadius="full"
                color={dailyStats.pnl >= 0 ? "green.500" : "red.500"}
              >
                <FiTrendingUp size="24px" />
              </Box>
            </Flex>
          </CardBody>
        </Card>
        
        <Card bg={cardBg} shadow="md" borderLeftWidth="4px" borderLeftColor="purple.500">
          <CardBody py={3}>
            <Flex justify="space-between" align="center">
              <Box>
                <Text fontSize="sm" color="gray.500">Win Rate</Text>
                <Text fontSize="2xl" fontWeight="bold">
                  {formatPercent(dailyStats.winRate)}
                </Text>
              </Box>
              <Box 
                bg="purple.50" 
                p={2} 
                borderRadius="full"
                color="purple.500"
              >
                <FiBarChart2 size="24px" />
              </Box>
            </Flex>
          </CardBody>
        </Card>
        
        <Card bg={cardBg} shadow="md" borderLeftWidth="4px" borderLeftColor="orange.500">
          <CardBody py={3}>
            <Flex justify="space-between" align="center">
              <Box>
                <Text fontSize="sm" color="gray.500">Last Trade</Text>
                <Text fontSize="md" fontWeight="bold" noOfLines={1}>
                  {dailyStats.lastTrade}
                </Text>
              </Box>
              <Box 
                bg="orange.50" 
                p={2} 
                borderRadius="full"
                color="orange.500"
              >
                <FiRefreshCw size="24px" />
              </Box>
            </Flex>
          </CardBody>
        </Card>
      </SimpleGrid>
  
      {/* Bot Information Card */}
      <Card mb={8} bg={cardBg} shadow="md">
        <CardHeader pb={2}>
          <Flex justify="space-between" align="center">
            <Heading size="md">Bot Configuration</Heading>
            <Button 
              size="sm" 
              leftIcon={<Icon as={FiEdit} />} 
              variant="ghost"
              onClick={onOpenEditModal}
            >
              Edit
            </Button>
          </Flex>
        </CardHeader>
        <CardBody>
          <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={4}>
            <Card variant="outline" p={3} borderRadius="md">
              <VStack align="start" spacing={1}>
                <Text fontSize="sm" color="gray.500">Symbol</Text>
                <Badge colorScheme="blue" fontSize="md" px={2} py={1} borderRadius="md">
                  {updatedBot.ticker}
                </Badge>
              </VStack>
            </Card>
            
            <Card variant="outline" p={3} borderRadius="md">
              <VStack align="start" spacing={1}>
                <Text fontSize="sm" color="gray.500">Algorithm Type</Text>
                <Text fontWeight="bold">{updatedBot.algorithm_type.replace(/_/g, ' ')}</Text>
              </VStack>
            </Card>
            
            <Card variant="outline" p={3} borderRadius="md">
              <VStack align="start" spacing={1}>
                <Text fontSize="sm" color="gray.500">Trade Direction</Text>
                <Badge colorScheme={
                  updatedBot.trade_direction === 'LONG' ? 'green' :
                  updatedBot.trade_direction === 'SHORT' ? 'red' : 'blue'
                } fontSize="md" px={2} py={1} borderRadius="md">
                  {updatedBot.trade_direction}
                </Badge>
              </VStack>
            </Card>
            
            <Card variant="outline" p={3} borderRadius="md">
              <VStack align="start" spacing={1}>
                <Text fontSize="sm" color="gray.500">Position Size</Text>
                <Text fontWeight="bold">{formatCurrency(updatedBot.position_size)}</Text>
              </VStack>
            </Card>
            
            <Card variant="outline" p={3} borderRadius="md">
              <VStack align="start" spacing={1}>
                <Text fontSize="sm" color="gray.500">Trailing Stop</Text>
                <Text fontWeight="bold">{formatPercent(updatedBot.trailing_stop_pct)}</Text>
              </VStack>
            </Card>
            
            <Card variant="outline" p={3} borderRadius="md">
              <VStack align="start" spacing={1}>
                <Text fontSize="sm" color="gray.500">Status</Text>
                <Badge colorScheme={isActive ? 'green' : 'gray'} fontSize="md" px={2} py={1} borderRadius="md">
                  {isActive ? 'Active' : 'Inactive'}
                </Badge>
              </VStack>
            </Card>
            
            <Card variant="outline" p={3} borderRadius="md">
              <VStack align="start" spacing={1}>
                <Text fontSize="sm" color="gray.500">Version</Text>
                <Text fontWeight="bold">{updatedBot.version || '1.0'}</Text>
              </VStack>
            </Card>
            
            <Card variant="outline" p={3} borderRadius="md">
              <VStack align="start" spacing={1}>
                <Text fontSize="sm" color="gray.500">Created</Text>
                <Text fontWeight="bold">{new Date(updatedBot.created_at).toLocaleDateString()}</Text>
              </VStack>
            </Card>
            
            {updatedBot.description && (
              <Card variant="outline" p={3} borderRadius="md" gridColumn={{ base: 'auto', md: 'span 4' }}>
                <VStack align="start" spacing={1}>
                  <Text fontSize="sm" color="gray.500">Description</Text>
                  <Text>{updatedBot.description}</Text>
                </VStack>
              </Card>
            )}
          </SimpleGrid>
        </CardBody>
      </Card>
      
      {/* Performance Stats */}
      {bot.metrics && (
        <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6} mb={8}>
          <Stat
            px={4}
            py={5}
            shadow="base"
            borderWidth="1px"
            borderRadius="lg"
            bg={cardBg}
          >
            <StatLabel fontSize="md">Total Trades</StatLabel>
            <StatNumber fontSize="3xl">{bot.metrics.total_trades}</StatNumber>
            <StatHelpText>
              {bot.metrics.winning_trades} wins / {bot.metrics.losing_trades} losses
            </StatHelpText>
          </Stat>
          
          <Stat
            px={4}
            py={5}
            shadow="base"
            borderWidth="1px"
            borderRadius="lg"
            bg={cardBg}
          >
            <StatLabel fontSize="md">Win Rate</StatLabel>
            <StatNumber fontSize="3xl">{formatPercent(bot.metrics.win_rate)}</StatNumber>
            <StatHelpText>
              <StatArrow type={bot.metrics.win_rate >= 0.5 ? 'increase' : 'decrease'} />
              {bot.metrics.winning_trades} out of {bot.metrics.total_trades}
            </StatHelpText>
          </Stat>
          
          <Stat
            px={4}
            py={5}
            shadow="base"
            borderWidth="1px"
            borderRadius="lg"
            bg={cardBg}
          >
            <StatLabel fontSize="md">Total P&L</StatLabel>
            <StatNumber 
              fontSize="3xl" 
              color={parseFloat(bot.metrics.total_pnl) >= 0 ? 'green.500' : 'red.500'}
            >
              {formatCurrency(bot.metrics.total_pnl)}
            </StatNumber>
            <StatHelpText>
              <StatArrow type={parseFloat(bot.metrics.total_pnl) >= 0 ? 'increase' : 'decrease'} />
              Avg: {formatCurrency(bot.metrics.average_pnl_per_trade)} / trade
            </StatHelpText>
          </Stat>
          
          <Stat
            px={4}
            py={5}
            shadow="base"
            borderWidth="1px"
            borderRadius="lg"
            bg={cardBg}
          >
            <StatLabel fontSize="md">Profit Factor</StatLabel>
            <StatNumber fontSize="3xl">{parseFloat(bot.metrics.profit_factor).toFixed(2)}</StatNumber>
            <StatHelpText>
              <StatArrow type={parseFloat(bot.metrics.profit_factor) >= 1 ? 'increase' : 'decrease'} />
              Risk/Reward: {parseFloat(bot.metrics.risk_reward_ratio).toFixed(2)}
            </StatHelpText>
          </Stat>
        </SimpleGrid>
      )}
      
      {/* Algorithm Parameters Visualization - Enhanced */}
      <Card shadow="md" bg={cardBg} mb={8}>
        <CardHeader pb={0}>
          <Flex justify="space-between" align="center">
            <Heading size="md">Algorithm Parameters</Heading>
            <Button 
              size="sm" 
              colorScheme="blue" 
              variant="outline" 
              leftIcon={<FiSettings />}
              onClick={onOpenEditModal}
            >
              Edit Parameters
            </Button>
          </Flex>
        </CardHeader>
        <CardBody>
          {bot.parameters && Object.keys(bot.parameters).length > 0 ? (
            <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={6}>
              {/* Parameter Radar Chart */}
              <Box 
                p={4} 
                borderWidth="1px" 
                borderRadius="lg"
                overflowX="auto"
                height="320px"
              >
                <Heading size="sm" mb={4}>Parameter Balance</Heading>
                <ParameterRadarChart parameters={bot.parameters} height={260} />
              </Box>
              
              {/* Individual Parameter Visualizations */}
              <Box>
                <Heading size="sm" mb={4}>Parameter Details</Heading>
                <SimpleGrid columns={{ base: 1, sm: 2 }} spacing={6}>
                  {Object.entries(bot.parameters).map(([key, value]) => {
                    // Determine parameter type
                    const isPercent = key.includes('pct') || key.includes('percent') || key.includes('threshold');
                    const isPeriod = key.includes('period') || key.includes('lookback');
                    const isLowerBetter = key.includes('stop_loss') || key.includes('trailing_stop') || key.includes('drawdown');
                    
                    // Define parameter ranges
                    const getRange = () => {
                      if (isPeriod) {
                        return { min: 5, max: 50 };
                      } else if (key.includes('rsi_upper')) {
                        return { min: 65, max: 85 };
                      } else if (key.includes('rsi_lower')) {
                        return { min: 15, max: 35 };
                      } else if (isPercent && key.includes('stop')) {
                        return { min: 0.005, max: 0.03 };
                      } else if (isPercent && key.includes('profit')) {
                        return { min: 0.01, max: 0.05 };
                      } else if (key.includes('volatility_threshold')) {
                        return { min: 0.5, max: 3 };
                      }
                      
                      // Default range
                      return { min: 0, max: 100 };
                    };
                    
                    const range = getRange();
                    const numValue = typeof value === 'number' ? value : parseFloat(value.toString());
                    
                    return (
                      <Box 
                        key={key} 
                        p={3} 
                        borderWidth="1px" 
                        borderRadius="md"
                        borderLeftWidth="3px"
                        borderLeftColor={isPeriod ? "blue.500" : isPercent ? "green.500" : "purple.500"}
                      >
                        <ParameterHistogram
                          name={key}
                          value={numValue}
                          range={range}
                          isPercent={isPercent}
                          isLowerBetter={isLowerBetter}
                        />
                      </Box>
                    );
                  })}
                </SimpleGrid>
              </Box>
            </SimpleGrid>
          ) : (
            <Box p={4} borderWidth="1px" borderRadius="md">
              <Text>No parameters found for this bot</Text>
            </Box>
          )}
        </CardBody>
      </Card>

      {/* Tabs for Trades / Performance */}
      <Tabs colorScheme="brand" shadow="md" bg={cardBg} borderRadius="lg">
        <TabList>
          <Tab>Trade History</Tab>
          <Tab>Performance</Tab>
        </TabList>
        
        <TabPanels>
          {/* Trade History Tab */}
          <TabPanel>
            {/* Enhanced Trade Filters */}
            <Card shadow="md" bg={cardBg} mb={6}>
              <CardHeader pb={0}>
                <Heading size="sm">Filter Trades</Heading>
              </CardHeader>
              <CardBody>
                <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
                  {/* Left column - Basic filters */}
                  <Stack spacing={4}>
                    <HStack spacing={4}>
                      <Box flex="1">
                        <Text fontSize="xs" fontWeight="medium" mb={1} color="gray.500">Trade Status</Text>
                        <Select 
                          value={tradeFilter}
                          onChange={(e) => setTradeFilter(e.target.value)}
                          size="sm"
                        >
                          <option value="all">All Statuses</option>
                          <option value="open">Open</option>
                          <option value="closed">Closed</option>
                          <option value="pending_exit">Pending Exit</option>
                        </Select>
                      </Box>
                      
                      <Box flex="1">
                        <Text fontSize="xs" fontWeight="medium" mb={1} color="gray.500">Date Range</Text>
                        <Select 
                          value={dateRange}
                          onChange={(e) => setDateRange(e.target.value)}
                          size="sm"
                        >
                          <option value="1d">Last 24 Hours</option>
                          <option value="7d">Last 7 Days</option>
                          <option value="30d">Last 30 Days</option>
                          <option value="90d">Last 90 Days</option>
                          <option value="all">All Time</option>
                        </Select>
                      </Box>
                    </HStack>
                    
                    <HStack spacing={4}>
                      <Box flex="1">
                        <Text fontSize="xs" fontWeight="medium" mb={1} color="gray.500">Direction</Text>
                        <Select size="sm" defaultValue="all">
                          <option value="all">All Directions</option>
                          <option value="LONG">Long Only</option>
                          <option value="SHORT">Short Only</option>
                        </Select>
                      </Box>
                      
                      <Box flex="1">
                        <Text fontSize="xs" fontWeight="medium" mb={1} color="gray.500">Results</Text>
                        <Select size="sm" defaultValue="all">
                          <option value="all">All Results</option>
                          <option value="winning">Winning Trades</option>
                          <option value="losing">Losing Trades</option>
                        </Select>
                      </Box>
                    </HStack>
                  </Stack>
                  
                  {/* Right column - Advanced filters */}
                  <Box borderLeft={{ base: 'none', md: '1px solid' }} borderColor="gray.200" pl={{ base: 0, md: 4 }}>
                    <HStack mb={3} justify="space-between">
                      <Text fontSize="xs" fontWeight="medium" color="gray.500">PnL Range</Text>
                      <Badge fontSize="xs">
                        {formatCurrency(-1000)} to {formatCurrency(1000)}
                      </Badge>
                    </HStack>
                    
                    <Box px={2} mb={4}>
                      {/* This would be a range slider in a real implementation */}
                      <Progress value={50} colorScheme="blue" height="8px" borderRadius="full" />
                    </Box>
                    
                    <HStack mb={3} justify="space-between">
                      <Text fontSize="xs" fontWeight="medium" color="gray.500">Trade Duration</Text>
                      <Badge fontSize="xs">
                        Any duration
                      </Badge>
                    </HStack>
                    
                    <Box px={2}>
                      {/* This would be a range slider in a real implementation */}
                      <Progress value={100} colorScheme="blue" height="8px" borderRadius="full" />
                    </Box>
                  </Box>
                </SimpleGrid>
                
                <Flex mt={6} justify="space-between">
                  <HStack>
                    <Button 
                      leftIcon={<FiFilter />}
                      size="sm"
                      colorScheme="blue"
                    >
                      Apply Filters
                    </Button>
                    <Button 
                      size="sm"
                      variant="ghost"
                    >
                      Reset
                    </Button>
                  </HStack>
                  
                  <HStack>
                    <Button 
                      leftIcon={<FiDownload />}
                      size="sm"
                      variant="outline"
                    >
                      Export Data
                    </Button>
                    <Button 
                      leftIcon={<FiRefreshCw />}
                      size="sm"
                      variant="outline"
                      colorScheme="blue"
                    >
                      Refresh
                    </Button>
                  </HStack>
                </Flex>
              </CardBody>
            </Card>
            
            {/* Filter Summary */}
            <Flex mb={6} align="center" justify="space-between">
              <HStack>
                <Text fontSize="sm" fontWeight="medium">
                  Showing {filteredTrades.length} trades
                </Text>
                {tradeFilter !== 'all' && (
                  <Badge colorScheme="blue" ml={2}>
                    Status: {tradeFilter}
                  </Badge>
                )}
                {dateRange !== 'all' && (
                  <Badge colorScheme="green" ml={2}>
                    Date: {
                      dateRange === '1d' ? 'Last 24h' :
                      dateRange === '7d' ? 'Last 7 days' :
                      dateRange === '30d' ? 'Last 30 days' : 'Last 90 days'
                    }
                  </Badge>
                )}
              </HStack>
              
              <Text fontSize="xs" color="gray.500">
                {filteredTrades.filter((t: any) => t.pnl > 0).length} winning, {filteredTrades.filter((t: any) => t.pnl <= 0).length} losing
              </Text>
            </Flex>
          
            {/* Trade Visualization */}
            {filteredTrades.length > 0 && (
              <Card shadow="md" bg={cardBg} mb={6}>
                <CardHeader pb={0}>
                  <Heading size="md">Trade Performance Visualization</Heading>
                </CardHeader>
                <CardBody>
                  <TradeHistoryChart 
                    trades={filteredTrades.filter((t: any) => t.trade_status === 'closed' && t.pnl !== null)} 
                  />
                </CardBody>
              </Card>
            )}
            
            {/* Trades Table */}
            <Card shadow="md" bg={cardBg}>
              <CardHeader pb={0}>
                <Flex justify="space-between" align="center">
                  <Heading size="md">Trades History</Heading>
                  <HStack>
                    <Text fontSize="sm" color="gray.500">
                      Showing {filteredTrades.length} trades
                    </Text>
                    <Button size="sm" colorScheme="blue" leftIcon={<FiRefreshCw />}>
                      Export CSV
                    </Button>
                  </HStack>
                </Flex>
              </CardHeader>
              <CardBody>
                <Box overflowX="auto">
                  <Table variant="simple" size="sm">
                    <Thead>
                      <Tr>
                        <Th>Trade ID</Th>
                        <Th>Direction</Th>
                        <Th>Entry Price</Th>
                        <Th>Exit Price</Th>
                        <Th>Entry Time</Th>
                        <Th>Exit Time</Th>
                        <Th>Status</Th>
                        <Th>P&L</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {filteredTrades.map((trade: any) => (
                        <Tr key={trade.trade_id}>
                          <Td>{trade.trade_id}</Td>
                          <Td>
                            <Badge
                              colorScheme={trade.trade_direction === 'LONG' ? 'green' : 'red'}
                            >
                              {trade.trade_direction}
                            </Badge>
                          </Td>
                          <Td>${parseFloat(trade.entry_price).toFixed(2)}</Td>
                          <Td>
                            {trade.exit_price 
                              ? '$' + parseFloat(trade.exit_price).toFixed(2)
                              : '-'
                            }
                          </Td>
                          <Td>{new Date(trade.entry_time).toLocaleString()}</Td>
                          <Td>
                            {trade.exit_time 
                              ? new Date(trade.exit_time).toLocaleString()
                              : '-'
                            }
                          </Td>
                          <Td>
                            <Badge
                              colorScheme={
                                trade.trade_status === 'open'
                                  ? 'blue'
                                  : trade.trade_status === 'closed' && parseFloat(trade.pnl || 0) > 0
                                  ? 'green'
                                  : 'red'
                              }
                            >
                              {trade.trade_status}
                            </Badge>
                          </Td>
                          <Td 
                            color={
                              trade.pnl !== null
                                ? parseFloat(trade.pnl) >= 0 
                                  ? 'green.500' 
                                  : 'red.500'
                                : 'inherit'
                            }
                          >
                            {trade.pnl !== null
                              ? formatCurrency(trade.pnl)
                              : '-'
                            }
                          </Td>
                        </Tr>
                      ))}
                      {filteredTrades.length === 0 && (
                        <Tr>
                          <Td colSpan={8} textAlign="center">No trades found matching the selected filters</Td>
                        </Tr>
                      )}
                    </Tbody>
                  </Table>
                </Box>
              </CardBody>
            </Card>
          </TabPanel>
          
          {/* Performance Tab - Enhanced with Advanced Analytics */}
          <TabPanel>
            <Tabs variant="soft-rounded" colorScheme="blue" mb={6}>
              <TabList>
                <Tab>Overview</Tab>
                <Tab>Advanced Analytics</Tab>
                <Tab>Performance Metrics</Tab>
              </TabList>
              
              <TabPanels mt={4}>
                {/* Overview Tab */}
                <TabPanel p={0}>
                  <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={6}>
                    <Card shadow="md" bg={cardBg}>
                      <CardHeader pb={0}>
                        <Heading size="md">Trade Performance</Heading>
                      </CardHeader>
                      <CardBody>
                        {bot.trades && bot.trades.length > 0 ? (
                          <TradeHistoryChart 
                            trades={bot.trades.filter((t: any) => t.trade_status === 'closed' && t.pnl !== null)} 
                          />
                        ) : (
                          <Text>No closed trades found to display performance</Text>
                        )}
                      </CardBody>
                    </Card>
                    
                    <Card shadow="md" bg={cardBg}>
                      <CardHeader pb={0}>
                        <Heading size="md">Performance Summary</Heading>
                      </CardHeader>
                      <CardBody>
                        {bot.metrics ? (
                          <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                            <Box 
                              p={3} 
                              borderWidth="1px" 
                              borderRadius="md"
                              bg={useColorModeValue("blue.50", "blue.900")}
                            >
                              <Text fontWeight="bold" color="blue.700" mb={1}>Win Rate</Text>
                              <Flex align="center">
                                <Text fontSize="2xl" fontWeight="bold">
                                  {formatPercent(bot.metrics.win_rate)}
                                </Text>
                                <Progress 
                                  value={parseFloat(bot.metrics.win_rate) * 100}
                                  colorScheme={parseFloat(bot.metrics.win_rate) >= 0.5 ? "green" : "orange"}
                                  height="8px"
                                  w="120px"
                                  ml={4}
                                  borderRadius="full"
                                />
                              </Flex>
                              <Text fontSize="sm" color="gray.600" mt={1}>
                                {bot.metrics.winning_trades} wins / {bot.metrics.losing_trades} losses
                              </Text>
                            </Box>
                            
                            <Box 
                              p={3} 
                              borderWidth="1px" 
                              borderRadius="md"
                              bg={useColorModeValue("purple.50", "purple.900")}
                            >
                              <Text fontWeight="bold" color="purple.700" mb={1}>Profit Factor</Text>
                              <Flex align="center">
                                <Text 
                                  fontSize="2xl" 
                                  fontWeight="bold" 
                                  color={parseFloat(bot.metrics.profit_factor) >= 1 ? "green.500" : "red.500"}
                                >
                                  {parseFloat(bot.metrics.profit_factor).toFixed(2)}
                                </Text>
                                <Badge 
                                  ml={2} 
                                  colorScheme={
                                    parseFloat(bot.metrics.profit_factor) >= 1.5 ? "green" :
                                    parseFloat(bot.metrics.profit_factor) >= 1 ? "blue" : "red"
                                  }
                                >
                                  {parseFloat(bot.metrics.profit_factor) >= 1.5 ? "Strong" :
                                   parseFloat(bot.metrics.profit_factor) >= 1 ? "Profitable" : "Needs Improvement"}
                                </Badge>
                              </Flex>
                              <Text fontSize="sm" color="gray.600" mt={1}>
                                Gross profits / gross losses
                              </Text>
                            </Box>
                            
                            <Box 
                              p={3} 
                              borderWidth="1px" 
                              borderRadius="md"
                              bg={useColorModeValue("green.50", "green.900")}
                            >
                              <Text fontWeight="bold" color="green.700" mb={1}>Average Win</Text>
                              <Text fontSize="2xl" fontWeight="bold" color="green.500">
                                {formatCurrency(bot.metrics.average_win_amount)}
                              </Text>
                              <Text fontSize="sm" color="gray.600" mt={1}>
                                Per winning trade
                              </Text>
                            </Box>
                            
                            <Box 
                              p={3} 
                              borderWidth="1px" 
                              borderRadius="md"
                              bg={useColorModeValue("red.50", "red.900")}
                            >
                              <Text fontWeight="bold" color="red.700" mb={1}>Average Loss</Text>
                              <Text fontSize="2xl" fontWeight="bold" color="red.500">
                                {formatCurrency(bot.metrics.average_loss_amount)}
                              </Text>
                              <Text fontSize="sm" color="gray.600" mt={1}>
                                Per losing trade
                              </Text>
                            </Box>
                            
                            <Box 
                              p={3} 
                              borderWidth="1px" 
                              borderRadius="md"
                              bg={useColorModeValue("orange.50", "orange.900")}
                            >
                              <Text fontWeight="bold" color="orange.700" mb={1}>Max Drawdown</Text>
                              <Text fontSize="2xl" fontWeight="bold" color={
                                parseFloat(bot.metrics.max_drawdown) > -500 ? "orange.500" : "red.500"
                              }>
                                {formatCurrency(bot.metrics.max_drawdown)}
                              </Text>
                              <Text fontSize="sm" color="gray.600" mt={1}>
                                Largest portfolio decline
                              </Text>
                            </Box>
                            
                            <Box 
                              p={3} 
                              borderWidth="1px" 
                              borderRadius="md"
                              bg={useColorModeValue("teal.50", "teal.900")}
                            >
                              <Text fontWeight="bold" color="teal.700" mb={1}>Sharpe Ratio</Text>
                              <Flex align="center">
                                <Text 
                                  fontSize="2xl" 
                                  fontWeight="bold" 
                                  color={
                                    parseFloat(bot.metrics.sharpe_ratio) >= 1.5 ? "green.500" :
                                    parseFloat(bot.metrics.sharpe_ratio) >= 1 ? "blue.500" : "red.500"
                                  }
                                >
                                  {parseFloat(bot.metrics.sharpe_ratio).toFixed(2)}
                                </Text>
                                <Badge 
                                  ml={2} 
                                  colorScheme={
                                    parseFloat(bot.metrics.sharpe_ratio) >= 1.5 ? "green" :
                                    parseFloat(bot.metrics.sharpe_ratio) >= 1 ? "blue" : "red"
                                  }
                                >
                                  {parseFloat(bot.metrics.sharpe_ratio) >= 1.5 ? "Excellent" :
                                   parseFloat(bot.metrics.sharpe_ratio) >= 1 ? "Good" : "Poor"}
                                </Badge>
                              </Flex>
                              <Text fontSize="sm" color="gray.600" mt={1}>
                                Risk-adjusted return metric
                              </Text>
                            </Box>
                          </SimpleGrid>
                        ) : (
                          <Text>No metrics available</Text>
                        )}
                      </CardBody>
                    </Card>
                    
                    {bot.metrics && (
                      <Card shadow="md" bg={cardBg} gridColumn={{ base: "auto", lg: "1 / span 2" }}>
                        <CardHeader pb={0}>
                          <Heading size="md">Bot Performance Analysis</Heading>
                        </CardHeader>
                        <CardBody>
                          <Text mb={4}>
                            This bot {bot.metrics.win_rate >= 0.5 ? 'has shown consistent profitability' : 'needs improvement'} with 
                            a {formatPercent(bot.metrics.win_rate)} win rate across {bot.metrics.total_trades} trades. 
                            The profit factor of {parseFloat(bot.metrics.profit_factor).toFixed(2)} indicates 
                            {bot.metrics.profit_factor >= 1.5 ? ' strong performance with winners significantly outpacing losers.' : 
                             bot.metrics.profit_factor >= 1 ? ' positive but modest performance.' : ' that the strategy needs optimization.'}
                          </Text>
                          
                          <Text>
                            The risk-reward ratio of {parseFloat(bot.metrics.risk_reward_ratio).toFixed(2)} shows
                            {bot.metrics.risk_reward_ratio >= 1.5 ? ' excellent risk management.' : 
                             bot.metrics.risk_reward_ratio >= 1 ? ' balanced risk taking.' : ' higher risk relative to rewards.'}
                            With a Sharpe ratio of {parseFloat(bot.metrics.sharpe_ratio).toFixed(2)}, the bot's returns
                            {bot.metrics.sharpe_ratio >= 1.5 ? ' are strong compared to its volatility.' : 
                             bot.metrics.sharpe_ratio >= 1 ? ' are acceptable given its volatility.' : ' need to be improved relative to risk taken.'}
                          </Text>
                        </CardBody>
                      </Card>
                    )}
                  </SimpleGrid>
                </TabPanel>
                
                {/* Advanced Analytics Tab */}
                <TabPanel p={0}>
                  {bot.trades && bot.trades.length > 0 ? (
                    <TradeAnalyticsChart 
                      trades={bot.trades.filter((t: any) => t.trade_status === 'closed' && t.pnl !== null)} 
                      height={500}
                    />
                  ) : (
                    <Box p={4} borderWidth="1px" borderRadius="md">
                      <Text>No closed trades found for advanced analytics</Text>
                    </Box>
                  )}
                </TabPanel>
                
                {/* Performance Metrics Tab - Detailed Analytics */}
                <TabPanel p={0}>
                  {bot.metrics ? (
                    <Card shadow="md" bg={cardBg}>
                      <CardHeader pb={0}>
                        <Heading size="md">Detailed Performance Metrics</Heading>
                      </CardHeader>
                      <CardBody>
                        <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
                          <Box>
                            <Heading size="sm" mb={4} color="blue.500">Trade Metrics</Heading>
                            <VStack align="stretch" spacing={4}>
                              <Box p={3} borderWidth="1px" borderRadius="md">
                                <Text color="gray.500" fontSize="sm">Total Trades</Text>
                                <Text fontSize="xl" fontWeight="bold">{bot.metrics.total_trades}</Text>
                              </Box>
                              <Box p={3} borderWidth="1px" borderRadius="md">
                                <Text color="gray.500" fontSize="sm">Win Rate</Text>
                                <Text fontSize="xl" fontWeight="bold">{formatPercent(bot.metrics.win_rate)}</Text>
                              </Box>
                              <Box p={3} borderWidth="1px" borderRadius="md">
                                <Text color="gray.500" fontSize="sm">Win/Loss Ratio</Text>
                                <Text fontSize="xl" fontWeight="bold">
                                  {bot.metrics.winning_trades}:{bot.metrics.losing_trades}
                                </Text>
                              </Box>
                            </VStack>
                          </Box>
                          
                          <Box>
                            <Heading size="sm" mb={4} color="green.500">Profitability Metrics</Heading>
                            <VStack align="stretch" spacing={4}>
                              <Box p={3} borderWidth="1px" borderRadius="md">
                                <Text color="gray.500" fontSize="sm">Total P&L</Text>
                                <Text 
                                  fontSize="xl" 
                                  fontWeight="bold"
                                  color={parseFloat(bot.metrics.total_pnl) >= 0 ? "green.500" : "red.500"}
                                >
                                  {formatCurrency(bot.metrics.total_pnl)}
                                </Text>
                              </Box>
                              <Box p={3} borderWidth="1px" borderRadius="md">
                                <Text color="gray.500" fontSize="sm">Profit Factor</Text>
                                <Text fontSize="xl" fontWeight="bold">{parseFloat(bot.metrics.profit_factor).toFixed(2)}</Text>
                              </Box>
                              <Box p={3} borderWidth="1px" borderRadius="md">
                                <Text color="gray.500" fontSize="sm">Expectancy</Text>
                                <Text 
                                  fontSize="xl" 
                                  fontWeight="bold"
                                  color={parseFloat(bot.metrics.expectancy) >= 0 ? "green.500" : "red.500"}
                                >
                                  {formatCurrency(bot.metrics.expectancy)}
                                </Text>
                              </Box>
                            </VStack>
                          </Box>
                          
                          <Box>
                            <Heading size="sm" mb={4} color="purple.500">Risk Metrics</Heading>
                            <VStack align="stretch" spacing={4}>
                              <Box p={3} borderWidth="1px" borderRadius="md">
                                <Text color="gray.500" fontSize="sm">Risk-Reward Ratio</Text>
                                <Text fontSize="xl" fontWeight="bold">{parseFloat(bot.metrics.risk_reward_ratio).toFixed(2)}</Text>
                              </Box>
                              <Box p={3} borderWidth="1px" borderRadius="md">
                                <Text color="gray.500" fontSize="sm">Sharpe Ratio</Text>
                                <Text fontSize="xl" fontWeight="bold">{parseFloat(bot.metrics.sharpe_ratio).toFixed(2)}</Text>
                              </Box>
                              <Box p={3} borderWidth="1px" borderRadius="md">
                                <Text color="gray.500" fontSize="sm">Max Drawdown</Text>
                                <Text fontSize="xl" fontWeight="bold">{formatCurrency(bot.metrics.max_drawdown)}</Text>
                              </Box>
                            </VStack>
                          </Box>
                        </SimpleGrid>
                        
                        <Box mt={8} p={4} borderWidth="1px" borderRadius="md" bg={useColorModeValue("gray.50", "gray.900")}>
                          <Heading size="sm" mb={4}>Performance Comparison</Heading>
                          <Text fontSize="sm">
                            This bot's metrics are displayed against benchmark standards for algorithmic trading performance.
                          </Text>
                          <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4} mt={4}>
                            <Box>
                              <Text fontWeight="bold" mb={2}>Win Rate</Text>
                              <Progress 
                                value={parseFloat(bot.metrics.win_rate) * 100} 
                                colorScheme="green" 
                                height="8px"
                                borderRadius="full"
                              />
                              <Flex justify="space-between" mt={1} fontSize="xs" color="gray.500">
                                <Text>Poor &lt;40%</Text>
                                <Text>Good 50-60%</Text>
                                <Text>Excellent &gt;70%</Text>
                              </Flex>
                            </Box>
                            
                            <Box>
                              <Text fontWeight="bold" mb={2}>Profit Factor</Text>
                              <Progress 
                                value={Math.min(parseFloat(bot.metrics.profit_factor) / 2 * 100, 100)} 
                                colorScheme="blue" 
                                height="8px"
                                borderRadius="full"
                              />
                              <Flex justify="space-between" mt={1} fontSize="xs" color="gray.500">
                                <Text>Poor &lt;1.0</Text>
                                <Text>Good 1.5-2.0</Text>
                                <Text>Excellent &gt;2.0</Text>
                              </Flex>
                            </Box>
                            
                            <Box>
                              <Text fontWeight="bold" mb={2}>Sharpe Ratio</Text>
                              <Progress 
                                value={Math.min(parseFloat(bot.metrics.sharpe_ratio) / 3 * 100, 100)} 
                                colorScheme="purple" 
                                height="8px"
                                borderRadius="full"
                              />
                              <Flex justify="space-between" mt={1} fontSize="xs" color="gray.500">
                                <Text>Poor &lt;1.0</Text>
                                <Text>Good 1.0-2.0</Text>
                                <Text>Excellent &gt;2.0</Text>
                              </Flex>
                            </Box>
                          </SimpleGrid>
                        </Box>
                      </CardBody>
                    </Card>
                  ) : (
                    <Box p={4} borderWidth="1px" borderRadius="md">
                      <Text>No metrics available to display</Text>
                    </Box>
                  )}
                </TabPanel>
              </TabPanels>
            </Tabs>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </MainLayout>
  );
}