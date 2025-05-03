import { json } from '@remix-run/node';
import { useLoaderData, Link, useNavigate } from '@remix-run/react';
import {
  Box,
  Heading,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Button,
  Text,
  Badge,
  Switch,
  Flex,
  HStack,
  Select,
  Input,
  InputGroup,
  InputLeftElement,
  Stack,
  Card,
  CardHeader,
  CardBody,
  useColorModeValue,
  Icon,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  IconButton,
  Tooltip,
  useToast,
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
  AlertDialog,
  AlertDialogBody,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogContent,
  AlertDialogOverlay,
} from '@chakra-ui/react';
import { useState, useEffect, useRef } from 'react';
import { SearchIcon, ChevronDownIcon, ChevronUpIcon } from '@chakra-ui/icons';
import { FiFilter, FiPlus, FiSettings, FiEdit, FiTrash2, FiMoreVertical, FiArchive, FiPieChart, FiActivity, FiSave } from 'react-icons/fi';
import MainLayout from '~/components/layout/MainLayout';
import { getAllBots, getBotById } from '~/lib/api.server';
import EditBotModal from '~/components/bot/EditBotModal';

export async function loader() {
  const bots = await getAllBots();
  return json({ bots });
}

// Create a new bot registration component
const NewBotForm = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
  const toast = useToast();
  const formRef = useRef<HTMLFormElement>(null);
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Get form data and create a new bot
    // This would send data to your API in a real implementation
    
    toast({
      title: "Bot registered successfully",
      status: "success",
      duration: 3000,
      isClosable: true,
    });
    
    onClose();
  };
  
  return (
    <Modal isOpen={isOpen} onClose={onClose} size="xl">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Register New Trading Bot</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <form ref={formRef} onSubmit={handleSubmit}>
            <Stack spacing={4}>
              <FormControl isRequired>
                <FormLabel>Bot Name</FormLabel>
                <Input placeholder="e.g. TSLA_Momentum_Bot" />
              </FormControl>
              
              <FormControl isRequired>
                <FormLabel>Symbol</FormLabel>
                <Select placeholder="Select ticker symbol">
                  <option value="TSLA">TSLA</option>
                  <option value="AAPL">AAPL</option>
                  <option value="COIN">COIN</option>
                  <option value="NVDA">NVDA</option>
                  <option value="AMD">AMD</option>
                  <option value="META">META</option>
                  <option value="AMZN">AMZN</option>
                  <option value="GOOGL">GOOGL</option>
                </Select>
              </FormControl>
              
              <FormControl isRequired>
                <FormLabel>Algorithm Type</FormLabel>
                <Select placeholder="Select algorithm type">
                  <option value="breakout">Breakout</option>
                  <option value="mean_reversion">Mean Reversion</option>
                  <option value="momentum">Momentum</option>
                  <option value="volatility_breakout">Volatility Breakout</option>
                  <option value="support_resistance">Support/Resistance</option>
                  <option value="price_pattern">Price Pattern</option>
                  <option value="minute_momentum">Minute Momentum</option>
                  <option value="volume_surge">Volume Surge</option>
                </Select>
              </FormControl>
              
              <FormControl isRequired>
                <FormLabel>Trade Direction</FormLabel>
                <Select placeholder="Select trade direction">
                  <option value="LONG">Long Only</option>
                  <option value="SHORT">Short Only</option>
                  <option value="BOTH">Both (Long & Short)</option>
                </Select>
              </FormControl>
              
              <FormControl isRequired>
                <FormLabel>Position Size (USD)</FormLabel>
                <Input type="number" min="100" step="100" placeholder="e.g. 1000" />
              </FormControl>
              
              <FormControl isRequired>
                <FormLabel>Trailing Stop (%)</FormLabel>
                <Input type="number" min="0.1" max="10" step="0.1" placeholder="e.g. 1.0" />
              </FormControl>
              
              <FormControl>
                <FormLabel>Description</FormLabel>
                <Input placeholder="Brief description of the bot's strategy" />
              </FormControl>
              
              <HStack>
                <Text>Active</Text>
                <Switch defaultChecked colorScheme="green" />
              </HStack>
            </Stack>
          </form>
        </ModalBody>
        <ModalFooter>
          <Button variant="ghost" mr={3} onClick={onClose}>
            Cancel
          </Button>
          <Button 
            colorScheme="blue" 
            onClick={(e) => {
              if (formRef.current) {
                // Manually trigger the form submission
                handleSubmit(e);
              }
            }}
          >
            Register Bot
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default function BotsIndex() {
  const { bots } = useLoaderData<typeof loader>();
  const toast = useToast();
  const navigate = useNavigate();
  const cardBg = useColorModeValue('white', 'gray.700');
  const hoveredRowBg = useColorModeValue('gray.50', 'gray.600');
  
  // New bot modal
  const { isOpen, onOpen, onClose } = useDisclosure();
  
  // Edit bot modal
  const {
    isOpen: isEditModalOpen,
    onOpen: onOpenEditModal,
    onClose: onCloseEditModal
  } = useDisclosure();
  
  // Delete bot confirmation dialog
  const {
    isOpen: isDeleteDialogOpen,
    onOpen: onOpenDeleteDialog,
    onClose: onCloseDeleteDialog
  } = useDisclosure();
  
  // Reference for delete confirmation dialog
  const cancelRef = useRef(null);
  
  // State for the bot to be deleted
  const [botToDelete, setBotToDelete] = useState<number | null>(null);
  
  // State for the bot to be edited
  const [selectedBot, setSelectedBot] = useState<any>(null);
  
  // State for filtering and sorting
  const [searchTerm, setSearchTerm] = useState('');
  const [algorithmFilter, setAlgorithmFilter] = useState('');
  const [symbolFilter, setSymbolFilter] = useState('');
  const [activeOnly, setActiveOnly] = useState(false);
  const [sortField, setSortField] = useState('bot_id');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  
  // Local state for bot active status (in a real app, this would sync with the server)
  const [botStatus, setBotStatus] = useState<Record<number, boolean>>({});
  
  // Initialize bot status state from the loaded data
  useEffect(() => {
    const statusMap: Record<number, boolean> = {};
    bots.forEach(bot => {
      statusMap[bot.bot_id] = bot.is_active;
    });
    setBotStatus(statusMap);
  }, [bots]);
  
  // Function to toggle bot active status
  const toggleBotStatus = (botId: number) => {
    setBotStatus(prev => {
      const newStatus = !prev[botId];
      
      // In a real app, you would call an API here to update the server
      toast({
        title: `Bot ${botId} ${newStatus ? 'activated' : 'deactivated'}`,
        status: newStatus ? 'success' : 'info',
        duration: 2000,
      });
      
      return { ...prev, [botId]: newStatus };
    });
  };
  
  // Open delete confirmation dialog
  const handleDeleteClick = (botId: number, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent row click
    setBotToDelete(botId);
    onOpenDeleteDialog();
  };
  
  // Handle actual deletion
  const handleDeleteBot = () => {
    if (botToDelete === null) return;
    
    // In a real app, this would call your API to delete the bot
    toast({
      title: "Bot deleted",
      description: `Bot ID ${botToDelete} has been removed`,
      status: "success",
      duration: 3000,
      isClosable: true,
    });
    
    onCloseDeleteDialog();
    setBotToDelete(null);
  };
  
  // Handle edit bot
  const handleEditClick = (bot: any, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent row click
    setSelectedBot(bot);
    onOpenEditModal();
  };
  
  // Handle saving bot changes
  const handleSaveBot = (updatedBot: any) => {
    // In a real app, this would call your API to update the bot
    toast({
      title: "Bot updated",
      description: `Bot ${updatedBot.name} has been updated`,
      status: "success",
      duration: 3000,
      isClosable: true,
    });
    
    // Update bot status if it was changed
    if (updatedBot.is_active !== botStatus[updatedBot.bot_id]) {
      setBotStatus(prev => ({ ...prev, [updatedBot.bot_id]: updatedBot.is_active }));
    }
  };
  
  // Function to handle sorting
  const handleSort = (field: string) => {
    if (sortField === field) {
      // Toggle direction if same field
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      // Set new field and reset direction to asc
      setSortField(field);
      setSortDirection('asc');
    }
  };
  
  // Function to render sort indicator
  const renderSortIndicator = (field: string) => {
    if (sortField !== field) return null;
    
    return sortDirection === 'asc' 
      ? <ChevronUpIcon ml={1} w={4} h={4} /> 
      : <ChevronDownIcon ml={1} w={4} h={4} />;
  };
  
  // Filter and sort bots
  const filteredAndSortedBots = [...bots]
    // Apply filters
    .filter(bot => {
      // Search filter (case insensitive)
      if (searchTerm && !bot.name.toLowerCase().includes(searchTerm.toLowerCase()) && 
          !bot.ticker.toLowerCase().includes(searchTerm.toLowerCase()) &&
          !bot.algorithm_type.toLowerCase().includes(searchTerm.toLowerCase())) {
        return false;
      }
      
      // Algorithm filter
      if (algorithmFilter && bot.algorithm_type !== algorithmFilter) {
        return false;
      }
      
      // Symbol filter
      if (symbolFilter && bot.ticker !== symbolFilter) {
        return false;
      }
      
      // Active only filter
      if (activeOnly && !botStatus[bot.bot_id]) {
        return false;
      }
      
      return true;
    })
    // Apply sorting
    .sort((a, b) => {
      let aValue = a[sortField as keyof typeof a];
      let bValue = b[sortField as keyof typeof b];
      
      // Handle special case for 'is_active' field
      if (sortField === 'is_active') {
        aValue = botStatus[a.bot_id];
        bValue = botStatus[b.bot_id];
      }
      
      // Convert to comparable types if needed
      if (typeof aValue === 'string') {
        aValue = aValue.toLowerCase();
        bValue = (bValue as string).toLowerCase();
      }
      
      // Apply sort direction
      if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
      if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });
  
  // Extract unique algorithm types for the filter
  const algorithmTypes = [...new Set(bots.map(bot => bot.algorithm_type))];
  
  // Extract unique symbols for the filter
  const symbols = [...new Set(bots.map(bot => bot.ticker))];
  
  return (
    <MainLayout>
      <Flex justify="space-between" align="center" mb={8}>
        <Heading size="lg">Trading Bots</Heading>
        <Button 
          colorScheme="blue" 
          leftIcon={<Icon as={FiPlus} />}
          onClick={onOpen}
        >
          Register New Bot
        </Button>
      </Flex>
      
      {/* Filters */}
      <Card shadow="sm" mb={6} bg={cardBg}>
        <CardBody>
          <Stack 
            direction={{ base: 'column', md: 'row' }} 
            spacing={4}
            align={{ base: 'stretch', md: 'center' }}
          >
            <InputGroup maxW={{ md: '300px' }}>
              <InputLeftElement pointerEvents="none">
                <SearchIcon color="gray.300" />
              </InputLeftElement>
              <Input 
                placeholder="Search bots..." 
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </InputGroup>
            
            <Select 
              placeholder="Algorithm Type" 
              maxW={{ md: '200px' }}
              value={algorithmFilter}
              onChange={(e) => setAlgorithmFilter(e.target.value)}
            >
              <option value="">All Algorithms</option>
              {algorithmTypes.map(type => (
                <option key={type} value={type}>
                  {type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </option>
              ))}
            </Select>
            
            <Select 
              placeholder="Symbol" 
              maxW={{ md: '150px' }}
              value={symbolFilter}
              onChange={(e) => setSymbolFilter(e.target.value)}
            >
              <option value="">All Symbols</option>
              {symbols.map(symbol => (
                <option key={symbol} value={symbol}>{symbol}</option>
              ))}
            </Select>
            
            <HStack spacing={2}>
              <Text fontSize="sm">Active Only</Text>
              <Switch 
                colorScheme="green" 
                isChecked={activeOnly}
                onChange={() => setActiveOnly(prev => !prev)}
              />
            </HStack>
            
            <Box ml="auto">
              <Button 
                leftIcon={<Icon as={FiFilter} />} 
                size="sm" 
                variant="outline"
                onClick={() => {
                  // Reset filters
                  setSearchTerm('');
                  setAlgorithmFilter('');
                  setSymbolFilter('');
                  setActiveOnly(false);
                }}
              >
                Clear Filters
              </Button>
            </Box>
          </Stack>
        </CardBody>
      </Card>
      
      <Card shadow="sm" bg={cardBg}>
        <CardHeader pb={0}>
          <Flex justify="space-between" align="center">
            <Heading size="md">Bot List</Heading>
            <Text fontSize="sm" color="gray.500">
              Showing {filteredAndSortedBots.length} of {bots.length} bots
            </Text>
          </Flex>
        </CardHeader>
        <CardBody>
          <Box overflowX="auto">
            <Table variant="simple">
              <Thead>
                <Tr>
                  <Th 
                    cursor="pointer" 
                    onClick={() => handleSort('bot_id')}
                    userSelect="none"
                  >
                    <Flex align="center">
                      ID {renderSortIndicator('bot_id')}
                    </Flex>
                  </Th>
                  <Th 
                    cursor="pointer" 
                    onClick={() => handleSort('name')}
                    userSelect="none"
                  >
                    <Flex align="center">
                      Name {renderSortIndicator('name')}
                    </Flex>
                  </Th>
                  <Th 
                    cursor="pointer" 
                    onClick={() => handleSort('ticker')}
                    userSelect="none"
                  >
                    <Flex align="center">
                      Symbol {renderSortIndicator('ticker')}
                    </Flex>
                  </Th>
                  <Th 
                    cursor="pointer" 
                    onClick={() => handleSort('algorithm_type')}
                    userSelect="none"
                  >
                    <Flex align="center">
                      Algorithm {renderSortIndicator('algorithm_type')}
                    </Flex>
                  </Th>
                  <Th>Performance</Th>
                  <Th 
                    cursor="pointer" 
                    onClick={() => handleSort('is_active')}
                    userSelect="none"
                  >
                    <Flex align="center">
                      Status {renderSortIndicator('is_active')}
                    </Flex>
                  </Th>
                  <Th>Actions</Th>
                </Tr>
              </Thead>
              <Tbody>
                {filteredAndSortedBots.map((bot) => (
                  <Tr 
                    key={bot.bot_id}
                    _hover={{ bg: hoveredRowBg }}
                  >
                    <Td>{bot.bot_id}</Td>
                    <Td fontWeight="medium">
                      {bot.name}
                    </Td>
                    <Td>
                      <Badge colorScheme="blue" fontSize="sm" px={2}>
                        {bot.ticker}
                      </Badge>
                    </Td>
                    <Td>
                      <Text>
                        {bot.algorithm_type.replace(/_/g, ' ')}
                      </Text>
                      <Text fontSize="xs" color="gray.500">
                        {bot.trade_direction}
                      </Text>
                    </Td>
                    <Td>
                      {/* Display actual metrics if available, otherwise show a no data message */}
                      {bot.metrics ? (
                        <HStack spacing={3}>
                          <Tooltip label="Win Rate" placement="top">
                            <HStack spacing={1}>
                              <Icon 
                                as={FiActivity} 
                                color={parseFloat(bot.metrics.win_rate) >= 0.5 ? "green.500" : "red.500"} 
                                boxSize={4} 
                              />
                              <Text fontSize="sm">
                                {(parseFloat(bot.metrics.win_rate) * 100).toFixed(1)}%
                              </Text>
                            </HStack>
                          </Tooltip>
                          
                          <Tooltip label="Profit Factor" placement="top">
                            <HStack spacing={1}>
                              <Icon 
                                as={FiPieChart} 
                                color={parseFloat(bot.metrics.profit_factor) >= 1 ? "purple.500" : "orange.500"} 
                                boxSize={4} 
                              />
                              <Text fontSize="sm">
                                {parseFloat(bot.metrics.profit_factor).toFixed(2)}
                              </Text>
                            </HStack>
                          </Tooltip>
                        </HStack>
                      ) : (
                        <Text fontSize="sm" color="gray.500">No metrics available</Text>
                      )}
                    </Td>
                    <Td onClick={(e) => e.stopPropagation()}>
                      <Switch 
                        colorScheme="green" 
                        isChecked={botStatus[bot.bot_id]} 
                        onChange={() => toggleBotStatus(bot.bot_id)}
                      />
                    </Td>
                    <Td onClick={(e) => e.stopPropagation()}>
                      <HStack spacing={2}>
                        <IconButton
                          aria-label="Edit bot configuration"
                          icon={<FiSettings />}
                          size="sm"
                          colorScheme="blue"
                          variant="ghost"
                          onClick={(e) => handleEditClick(bot, e)}
                        />
                        
                        <Menu>
                          <MenuButton
                            as={IconButton}
                            aria-label="More options"
                            icon={<FiMoreVertical />}
                            size="sm"
                            variant="ghost"
                          />
                          <MenuList>
                            <MenuItem 
                              icon={<Icon as={FiEdit} />} 
                              onClick={(e) => handleEditClick(bot, e)}
                            >
                              Edit Configuration
                            </MenuItem>
                            <MenuItem 
                              icon={<Icon as={FiArchive} />}
                              onClick={() => toggleBotStatus(bot.bot_id)}
                            >
                              {botStatus[bot.bot_id] ? 'Deactivate' : 'Activate'}
                            </MenuItem>
                            <MenuItem 
                              icon={<Icon as={FiTrash2} />} 
                              color="red.500"
                              onClick={(e) => handleDeleteClick(bot.bot_id, e)}
                            >
                              Delete Bot
                            </MenuItem>
                          </MenuList>
                        </Menu>
                      </HStack>
                    </Td>
                  </Tr>
                ))}
                {filteredAndSortedBots.length === 0 && (
                  <Tr>
                    <Td colSpan={7} textAlign="center" py={6}>
                      <Text color="gray.500">No bots match your filters</Text>
                    </Td>
                  </Tr>
                )}
              </Tbody>
            </Table>
          </Box>
        </CardBody>
      </Card>
      
      {/* New Bot Registration Modal */}
      <NewBotForm isOpen={isOpen} onClose={onClose} />
      
      {/* Edit Bot Modal */}
      {selectedBot && (
        <EditBotModal
          isOpen={isEditModalOpen}
          onClose={onCloseEditModal}
          bot={selectedBot}
          onSave={handleSaveBot}
        />
      )}
      
      {/* Delete Bot Confirmation Dialog */}
      <AlertDialog
        isOpen={isDeleteDialogOpen}
        leastDestructiveRef={cancelRef}
        onClose={onCloseDeleteDialog}
      >
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              Delete Bot
            </AlertDialogHeader>

            <AlertDialogBody>
              Are you sure you want to delete this bot? This action cannot be undone.
              All trade history and metrics for this bot will be permanently removed.
            </AlertDialogBody>

            <AlertDialogFooter>
              <Button ref={cancelRef} onClick={onCloseDeleteDialog}>
                Cancel
              </Button>
              <Button colorScheme="red" onClick={handleDeleteBot} ml={3}>
                Delete
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    </MainLayout>
  );
}