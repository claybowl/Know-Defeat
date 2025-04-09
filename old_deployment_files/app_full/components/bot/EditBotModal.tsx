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
  FormControl,
  FormLabel,
  Input,
  Select,
  HStack,
  Switch,
  VStack,
  Box,
  Heading,
  Text,
  SimpleGrid,
  Icon,
  useColorModeValue,
} from '@chakra-ui/react';
import { FiSave } from 'react-icons/fi';

interface EditBotModalProps {
  isOpen: boolean;
  onClose: () => void;
  bot: any;
  onSave: (updatedBot: any) => void;
}

export default function EditBotModal({ isOpen, onClose, bot, onSave }: EditBotModalProps) {
  const [formValues, setFormValues] = useState({
    name: '',
    ticker: '',
    algorithm_type: '',
    trade_direction: '',
    position_size: 0,
    trailing_stop_pct: 0,
    description: '',
    is_active: true,
    parameters: {} as Record<string, any>
  });
  
  // Initialize form with bot data when modal opens
  useEffect(() => {
    if (isOpen && bot) {
      setFormValues({
        name: bot.name || '',
        ticker: bot.ticker || '',
        algorithm_type: bot.algorithm_type || '',
        trade_direction: bot.trade_direction || '',
        position_size: bot.position_size || 0,
        trailing_stop_pct: bot.trailing_stop_pct || 0,
        description: bot.description || '',
        is_active: bot.is_active !== undefined ? bot.is_active : true,
        parameters: { ...bot.parameters } || {}
      });
    }
  }, [isOpen, bot]);
  
  // Handle form changes
  const handleChange = (field: string, value: any) => {
    setFormValues(prev => ({ ...prev, [field]: value }));
  };
  
  // Handle parameter changes
  const handleParameterChange = (key: string, value: any) => {
    setFormValues(prev => ({
      ...prev,
      parameters: {
        ...prev.parameters,
        [key]: value
      }
    }));
  };
  
  // Handle form submission
  const handleSubmit = () => {
    // In a real app, this would call an API
    onSave({
      ...bot,
      ...formValues
    });
    onClose();
  };
  
  // Get algorithm types for the dropdown
  const algorithmTypes = [
    'breakout',
    'mean_reversion',
    'momentum',
    'volatility_breakout',
    'support_resistance',
    'price_pattern',
    'minute_momentum',
    'volume_surge'
  ];
  
  // Get tickers for the dropdown
  const tickers = [
    'TSLA',
    'AAPL',
    'COIN',
    'NVDA',
    'AMD',
    'META',
    'AMZN',
    'GOOGL'
  ];
  
  // Render custom modal without using Modal components
  if (!isOpen) return null;
  
  return (
    <Box
      position="fixed"
      top="0"
      left="0"
      right="0"
      bottom="0"
      bg="rgba(0, 0, 0, 0.4)"
      zIndex="1000"
      onClick={onClose}
    >
      <Box
        position="relative"
        top="50%"
        transform="translateY(-50%)"
        maxW="800px"
        mx="auto"
        bg={useColorModeValue("white", "gray.800")}
        borderRadius="md"
        boxShadow="xl"
        onClick={(e) => e.stopPropagation()}
      >
        <Box p={4} borderBottomWidth="1px">
          <Heading size="md">Edit Bot Configuration</Heading>
          <Button
            position="absolute"
            top="10px"
            right="10px"
            size="sm"
            onClick={onClose}
            variant="ghost"
          >
            ✕
          </Button>
        </Box>
        
        <Box p={6} maxH="70vh" overflowY="auto">
          <VStack spacing={6} align="stretch">
            {/* Basic Bot Information */}
            <Box>
              <Heading size="sm" mb={4}>Basic Information</Heading>
              <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                <FormControl>
                  <FormLabel>Bot Name</FormLabel>
                  <Input 
                    value={formValues.name} 
                    onChange={(e) => handleChange('name', e.target.value)}
                  />
                </FormControl>
                
                <FormControl>
                  <FormLabel>Symbol</FormLabel>
                  <Select 
                    value={formValues.ticker}
                    onChange={(e) => handleChange('ticker', e.target.value)}
                  >
                    {tickers.map(ticker => (
                      <option key={ticker} value={ticker}>{ticker}</option>
                    ))}
                  </Select>
                </FormControl>
                
                <FormControl>
                  <FormLabel>Algorithm Type</FormLabel>
                  <Select 
                    value={formValues.algorithm_type}
                    onChange={(e) => handleChange('algorithm_type', e.target.value)}
                  >
                    {algorithmTypes.map(type => (
                      <option key={type} value={type}>
                        {type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </option>
                    ))}
                  </Select>
                </FormControl>
                
                <FormControl>
                  <FormLabel>Trade Direction</FormLabel>
                  <Select 
                    value={formValues.trade_direction}
                    onChange={(e) => handleChange('trade_direction', e.target.value)}
                  >
                    <option value="LONG">Long Only</option>
                    <option value="SHORT">Short Only</option>
                    <option value="BOTH">Both (Long & Short)</option>
                  </Select>
                </FormControl>
                
                <FormControl>
                  <FormLabel>Position Size (USD)</FormLabel>
                  <Input 
                    type="number" 
                    value={formValues.position_size} 
                    onChange={(e) => handleChange('position_size', parseFloat(e.target.value))}
                    min={100}
                    step={100}
                  />
                </FormControl>
                
                <FormControl>
                  <FormLabel>Trailing Stop (%)</FormLabel>
                  <Input 
                    type="number" 
                    value={formValues.trailing_stop_pct} 
                    onChange={(e) => handleChange('trailing_stop_pct', parseFloat(e.target.value))}
                    min={0.1}
                    max={10}
                    step={0.1}
                  />
                </FormControl>
                
                <FormControl gridColumn={{ md: "span 2" }}>
                  <FormLabel>Description</FormLabel>
                  <Input 
                    value={formValues.description || ''} 
                    onChange={(e) => handleChange('description', e.target.value)}
                  />
                </FormControl>
                
                <HStack>
                  <FormLabel mb={0}>Active</FormLabel>
                  <Switch 
                    isChecked={formValues.is_active} 
                    onChange={(e) => handleChange('is_active', e.target.checked)}
                    colorScheme="green"
                  />
                </HStack>
              </SimpleGrid>
            </Box>
            
            {/* Algorithm Parameters */}
            <Box>
              <Heading size="sm" mb={4}>Algorithm Parameters</Heading>
              {Object.keys(formValues.parameters).length > 0 ? (
                <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                  {Object.entries(formValues.parameters).map(([key, value]) => {
                    // Determine parameter type
                    const isPercent = key.includes('pct') || key.includes('percent') || key.includes('threshold');
                    const isPeriod = key.includes('period') || key.includes('lookback');
                    
                    return (
                      <FormControl key={key}>
                        <FormLabel>
                          {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          {isPercent && ' (%)'}
                          {isPeriod && ' (Periods)'}
                        </FormLabel>
                        <Input 
                          type="number"
                          value={value as string | number} 
                          onChange={(e) => {
                            const newValue = isPercent 
                              ? parseFloat(e.target.value) / 100 
                              : parseFloat(e.target.value);
                            handleParameterChange(key, newValue);
                          }}
                          step={isPercent ? 0.1 : 1}
                          min={0}
                        />
                      </FormControl>
                    );
                  })}
                </SimpleGrid>
              ) : (
                <Text color="gray.500">No parameters defined for this bot.</Text>
              )}
            </Box>
          </VStack>
        </Box>
        
        <Box p={4} borderTopWidth="1px" textAlign="right">
          <Button variant="ghost" mr={3} onClick={onClose}>
            Cancel
          </Button>
          <Button 
            colorScheme="blue" 
            leftIcon={<Icon as={FiSave} />}
            onClick={handleSubmit}
          >
            Save Changes
          </Button>
        </Box>
      </Box>
    </Box>
  );
}