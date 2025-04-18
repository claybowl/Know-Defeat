import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Grid, 
  Heading, 
  Tabs, 
  TabList, 
  Tab, 
  TabPanels, 
  TabPanel, 
  useColorModeValue, 
  Flex,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Badge,
  Text,
  Button
} from '@chakra-ui/react';
import { BotMetricsPanel } from './monitoring/BotMetricsPanel';
import { ActiveTradesTable } from './monitoring/ActiveTradesTable';
import { BotRankingsTable } from './monitoring/BotRankingsTable';
import { SystemStatusPanel } from './monitoring/SystemStatusPanel';
import { PollingConfigPanel } from './monitoring/PollingConfigPanel';
import { PerformanceCharts } from './monitoring/PerformanceCharts';

// WebSocket connection for real-time updates
const useWebSocket = (url) => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    let ws = null;
    let reconnectTimer = null;
    
    const connect = () => {
      ws = new WebSocket(url);
      
      ws.onopen = () => {
        setIsConnected(true);
        setError(null);
        console.log('WebSocket connected');
      };
      
      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          setData(parsed);
        } catch (err) {
          console.error('Error parsing WebSocket data', err);
        }
      };
      
      ws.onerror = (err) => {
        setError(err);
        console.error('WebSocket error', err);
      };
      
      ws.onclose = () => {
        setIsConnected(false);
        
        // Try to reconnect after 3 seconds
        reconnectTimer = setTimeout(() => {
          console.log('Attempting to reconnect WebSocket...');
          connect();
        }, 3000);
      };
    };
    
    connect();
    
    // Clean up
    return () => {
      if (ws) {
        ws.close();
      }
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
    };
  }, [url]);
  
  return { data, error, isConnected };
};

const MonitoringDashboard = () => {
  const bgColor = useColorModeValue('white', 'gray.800');
  const [activeTab, setActiveTab] = useState(0);
  const [systemStatus, setSystemStatus] = useState({ status: 'loading' });
  const [heartbeatTimer, setHeartbeatTimer] = useState(null);
  
  // WebSocket connection for metrics
  const metricsWs = useWebSocket('ws://localhost:8765/bot_metrics');
  const tradesWs = useWebSocket('ws://localhost:8765/trades');
  const rankingsWs = useWebSocket('ws://localhost:8765/rankings');
  
  // Fetch system status heartbeat
  const fetchHeartbeat = async () => {
    try {
      const response = await fetch('/api/system/heartbeat');
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Error fetching system heartbeat', error);
      setSystemStatus({ status: 'error' });
    }
  };
  
  // Initialize heartbeat polling
  useEffect(() => {
    fetchHeartbeat();
    
    const timer = setInterval(fetchHeartbeat, 30000); // Every 30 seconds
    setHeartbeatTimer(timer);
    
    return () => {
      clearInterval(timer);
    };
  }, []);
  
  return (
    <Box p={5} bg={bgColor} borderRadius="md" boxShadow="sm">
      <Flex justify="space-between" align="center" mb={6}>
        <Heading size="lg">Real-Time Trading Monitoring</Heading>
        <Flex align="center">
          <Text mr={2}>System Status:</Text>
          <Badge 
            colorScheme={systemStatus.database_status === 'connected' ? 'green' : 'red'}
            p={2}
            borderRadius="md"
          >
            {systemStatus.database_status === 'connected' ? 'Online' : 'Offline'}
          </Badge>
          <Button size="sm" ml={4} onClick={fetchHeartbeat} colorScheme="blue">
            Refresh
          </Button>
        </Flex>
      </Flex>
      
      {/* Status Stats */}
      <Grid templateColumns="repeat(4, 1fr)" gap={6} mb={6}>
        <Stat bg="blue.50" p={4} borderRadius="md" boxShadow="sm">
          <StatLabel>Active Bots</StatLabel>
          <StatNumber>{systemStatus.active_bot_count || '0'}</StatNumber>
          <StatHelpText>Trading bots currently active</StatHelpText>
        </Stat>
        <Stat bg="green.50" p={4} borderRadius="md" boxShadow="sm">
          <StatLabel>WebSocket Connections</StatLabel>
          <StatNumber>{systemStatus.active_websocket_connections || '0'}</StatNumber>
          <StatHelpText>Live monitoring clients</StatHelpText>
        </Stat>
        <Stat bg="purple.50" p={4} borderRadius="md" boxShadow="sm">
          <StatLabel>Last Poll</StatLabel>
          <StatNumber>
            {systemStatus.last_poll_time ? 
              new Date(systemStatus.last_poll_time).toLocaleTimeString() : 
              'N/A'}
          </StatNumber>
          <StatHelpText>Latest metrics collection</StatHelpText>
        </Stat>
        <Stat bg="orange.50" p={4} borderRadius="md" boxShadow="sm">
          <StatLabel>Connection Status</StatLabel>
          <StatNumber>
            {metricsWs.isConnected ? 
              <Badge colorScheme="green">Connected</Badge> : 
              <Badge colorScheme="red">Disconnected</Badge>}
          </StatNumber>
          <StatHelpText>Real-time data stream</StatHelpText>
        </Stat>
      </Grid>
      
      {/* Main Content Tabs */}
      <Tabs index={activeTab} onChange={setActiveTab} variant="enclosed" colorScheme="blue">
        <TabList>
          <Tab>Bot Performance</Tab>
          <Tab>Active Trades</Tab>
          <Tab>Bot Rankings</Tab>
          <Tab>Charts</Tab>
          <Tab>System Config</Tab>
        </TabList>
        
        <TabPanels>
          <TabPanel>
            <BotMetricsPanel wsData={metricsWs.data} />
          </TabPanel>
          
          <TabPanel>
            <ActiveTradesTable wsData={tradesWs.data} />
          </TabPanel>
          
          <TabPanel>
            <BotRankingsTable wsData={rankingsWs.data} />
          </TabPanel>
          
          <TabPanel>
            <PerformanceCharts metricsData={metricsWs.data} />
          </TabPanel>
          
          <TabPanel>
            <PollingConfigPanel systemStatus={systemStatus} />
          </TabPanel>
        </TabPanels>
      </Tabs>
      
      <Box mt={4}>
        <SystemStatusPanel 
          heartbeatData={systemStatus} 
          metricsConnected={metricsWs.isConnected}
          tradesConnected={tradesWs.isConnected} 
          rankingsConnected={rankingsWs.isConnected}
        />
      </Box>
    </Box>
  );
};

export default MonitoringDashboard; 