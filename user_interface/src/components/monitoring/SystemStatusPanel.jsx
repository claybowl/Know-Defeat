import React from 'react';
import {
  Box,
  Heading,
  Text,
  Flex,
  Badge,
  Grid,
  useColorModeValue,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatGroup,
  Divider,
  Button
} from '@chakra-ui/react';

export const SystemStatusPanel = ({ 
  heartbeatData, 
  metricsConnected,
  tradesConnected, 
  rankingsConnected 
}) => {
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  
  // Calculate last heartbeat timing
  const getLastHeartbeatText = () => {
    if (!heartbeatData || !heartbeatData.timestamp) {
      return 'No data';
    }
    
    const heartbeatTime = new Date(heartbeatData.timestamp);
    const now = new Date();
    const diffMs = now - heartbeatTime;
    const diffSeconds = Math.floor(diffMs / 1000);
    
    if (diffSeconds < 10) {
      return 'Just now';
    } else if (diffSeconds < 60) {
      return `${diffSeconds} seconds ago`;
    } else if (diffSeconds < 3600) {
      return `${Math.floor(diffSeconds / 60)} minutes ago`;
    } else {
      return heartbeatTime.toLocaleTimeString();
    }
  };
  
  // Get status badge color
  const getStatusColor = (isConnected) => {
    return isConnected ? 'green' : 'red';
  };

  // Get last poll time in friendly format
  const getLastPollText = () => {
    if (!heartbeatData || !heartbeatData.last_poll_time) {
      return 'No data';
    }
    
    const pollTime = new Date(heartbeatData.last_poll_time);
    const now = new Date();
    const diffMs = now - pollTime;
    const diffSeconds = Math.floor(diffMs / 1000);
    
    if (diffSeconds < 60) {
      return `${diffSeconds} seconds ago`;
    } else if (diffSeconds < 3600) {
      return `${Math.floor(diffSeconds / 60)} minutes ago`;
    } else {
      return pollTime.toLocaleString();
    }
  };
  
  return (
    <Box 
      p={4} 
      bg={bgColor} 
      borderWidth="1px" 
      borderColor={borderColor} 
      borderRadius="md"
      boxShadow="sm"
    >
      <Heading size="sm" mb={3}>System Health Monitor</Heading>
      
      <Divider mb={4} />
      
      <Grid templateColumns={{ base: "repeat(1, 1fr)", md: "repeat(2, 1fr)", lg: "repeat(4, 1fr)" }} gap={4}>
        {/* Database Status */}
        <Box p={3} borderWidth="1px" borderRadius="md" borderColor={borderColor}>
          <Flex justify="space-between" align="center" mb={2}>
            <Text fontWeight="bold">Database</Text>
            <Badge 
              colorScheme={heartbeatData?.database_status === 'connected' ? 'green' : 'red'}
              p={1}
              borderRadius="md"
            >
              {heartbeatData?.database_status || 'Unknown'}
            </Badge>
          </Flex>
          <Text fontSize="sm">
            Last heartbeat: {getLastHeartbeatText()}
          </Text>
        </Box>
        
        {/* Polling System */}
        <Box p={3} borderWidth="1px" borderRadius="md" borderColor={borderColor}>
          <Flex justify="space-between" align="center" mb={2}>
            <Text fontWeight="bold">Polling System</Text>
            <Badge 
              colorScheme={heartbeatData?.last_poll_time ? 'green' : 'yellow'}
              p={1}
              borderRadius="md"
            >
              {heartbeatData?.last_poll_time ? 'Active' : 'Unknown'}
            </Badge>
          </Flex>
          <Text fontSize="sm">
            Last poll: {getLastPollText()}
          </Text>
        </Box>
        
        {/* WebSocket Status */}
        <Box p={3} borderWidth="1px" borderRadius="md" borderColor={borderColor}>
          <Text fontWeight="bold" mb={2}>WebSocket Connections</Text>
          <Flex direction="column" gap={1}>
            <Flex justify="space-between">
              <Text fontSize="sm">Metrics:</Text>
              <Badge 
                colorScheme={getStatusColor(metricsConnected)}
                variant="subtle"
                px={2}
              >
                {metricsConnected ? 'Connected' : 'Disconnected'}
              </Badge>
            </Flex>
            <Flex justify="space-between">
              <Text fontSize="sm">Trades:</Text>
              <Badge 
                colorScheme={getStatusColor(tradesConnected)}
                variant="subtle"
                px={2}
              >
                {tradesConnected ? 'Connected' : 'Disconnected'}
              </Badge>
            </Flex>
            <Flex justify="space-between">
              <Text fontSize="sm">Rankings:</Text>
              <Badge 
                colorScheme={getStatusColor(rankingsConnected)}
                variant="subtle"
                px={2}
              >
                {rankingsConnected ? 'Connected' : 'Disconnected'}
              </Badge>
            </Flex>
          </Flex>
        </Box>
        
        {/* Active Bots */}
        <Box p={3} borderWidth="1px" borderRadius="md" borderColor={borderColor}>
          <Text fontWeight="bold" mb={2}>System Stats</Text>
          <StatGroup>
            <Stat>
              <StatLabel fontSize="xs">Active Bots</StatLabel>
              <StatNumber fontSize="lg">{heartbeatData?.active_bot_count || 0}</StatNumber>
              <StatHelpText fontSize="xs">Trading bots</StatHelpText>
            </Stat>
            <Stat>
              <StatLabel fontSize="xs">Connections</StatLabel>
              <StatNumber fontSize="lg">{heartbeatData?.active_websocket_connections || 0}</StatNumber>
              <StatHelpText fontSize="xs">Monitoring clients</StatHelpText>
            </Stat>
          </StatGroup>
        </Box>
      </Grid>
      
      <Text mt={4} fontSize="xs" color="gray.500" textAlign="right">
        Last updated: {heartbeatData?.timestamp ? new Date(heartbeatData.timestamp).toLocaleString() : 'N/A'}
      </Text>
    </Box>
  );
}; 