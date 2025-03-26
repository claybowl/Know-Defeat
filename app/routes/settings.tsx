import { json } from '@remix-run/node';
import { useLoaderData } from '@remix-run/react';
import {
  Box,
  Heading,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  FormControl,
  FormLabel,
  Input,
  Switch,
  Button,
  VStack,
  HStack,
  Text,
  Card,
  CardHeader,
  CardBody,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  useColorModeValue,
  Divider,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
} from '@chakra-ui/react';
import MainLayout from '~/components/layout/MainLayout';

export async function loader() {
  // In a real app, you would load actual settings from the database
  const weights = [
    { variable_name: 'win_rate', weight: 0.25, description: 'Percentage of winning trades' },
    { variable_name: 'profit_factor', weight: 0.2, description: 'Ratio of gross profits to gross losses' },
    { variable_name: 'sharpe_ratio', weight: 0.15, description: 'Risk-adjusted return metric' },
    { variable_name: 'max_drawdown', weight: 0.1, description: 'Maximum peak-to-trough decline' },
    { variable_name: 'expectancy', weight: 0.15, description: 'Average amount you can expect to win per trade' },
    { variable_name: 'total_pnl', weight: 0.1, description: 'Total profit and loss' },
    { variable_name: 'risk_reward_ratio', weight: 0.05, description: 'Ratio of average win to average loss' },
  ];
  
  return json({ weights });
}

export default function Settings() {
  const { weights } = useLoaderData<typeof loader>();
  const cardBg = useColorModeValue('white', 'gray.700');
  
  return (
    <MainLayout>
      <Heading mb={6}>System Settings</Heading>
      
      <Tabs colorScheme="brand" shadow="md" bg={cardBg} borderRadius="lg">
        <TabList>
          <Tab>General</Tab>
          <Tab>Ranking Weights</Tab>
          <Tab>Database</Tab>
          <Tab>API</Tab>
        </TabList>
        
        <TabPanels>
          {/* General Settings */}
          <TabPanel>
            <VStack spacing={6} align="stretch">
              <Card>
                <CardHeader>
                  <Heading size="md">Trading Parameters</Heading>
                </CardHeader>
                <CardBody>
                  <VStack spacing={6} align="stretch">
                    <FormControl display="flex" alignItems="center">
                      <FormLabel mb="0">Enable Trading System</FormLabel>
                      <Switch colorScheme="brand" size="lg" defaultChecked />
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel>Default Position Size</FormLabel>
                      <NumberInput defaultValue={1000} min={100} precision={2}>
                        <NumberInputField />
                        <NumberInputStepper>
                          <NumberIncrementStepper />
                          <NumberDecrementStepper />
                        </NumberInputStepper>
                      </NumberInput>
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel>Maximum Concurrent Trades</FormLabel>
                      <NumberInput defaultValue={20} min={1} max={100}>
                        <NumberInputField />
                        <NumberInputStepper>
                          <NumberIncrementStepper />
                          <NumberDecrementStepper />
                        </NumberInputStepper>
                      </NumberInput>
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel>Default Trailing Stop Percentage</FormLabel>
                      <NumberInput defaultValue={1} min={0.1} max={10} precision={2} step={0.1}>
                        <NumberInputField />
                        <NumberInputStepper>
                          <NumberIncrementStepper />
                          <NumberDecrementStepper />
                        </NumberInputStepper>
                      </NumberInput>
                    </FormControl>
                    
                    <FormControl display="flex" alignItems="center">
                      <FormLabel mb="0">Auto-Allocate Funds</FormLabel>
                      <Switch colorScheme="brand" size="lg" defaultChecked />
                    </FormControl>
                  </VStack>
                </CardBody>
              </Card>
              
              <Card>
                <CardHeader>
                  <Heading size="md">System Settings</Heading>
                </CardHeader>
                <CardBody>
                  <VStack spacing={6} align="stretch">
                    <FormControl>
                      <FormLabel>Update Frequency (seconds)</FormLabel>
                      <NumberInput defaultValue={5} min={1} max={60}>
                        <NumberInputField />
                        <NumberInputStepper>
                          <NumberIncrementStepper />
                          <NumberDecrementStepper />
                        </NumberInputStepper>
                      </NumberInput>
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel>Metrics Calculation Interval (minutes)</FormLabel>
                      <NumberInput defaultValue={15} min={5} max={60}>
                        <NumberInputField />
                        <NumberInputStepper>
                          <NumberIncrementStepper />
                          <NumberDecrementStepper />
                        </NumberInputStepper>
                      </NumberInput>
                    </FormControl>
                    
                    <FormControl display="flex" alignItems="center">
                      <FormLabel mb="0">Enable Detailed Logging</FormLabel>
                      <Switch colorScheme="brand" size="lg" defaultChecked />
                    </FormControl>
                    
                    <FormControl display="flex" alignItems="center">
                      <FormLabel mb="0">Auto-Restart Failed Bots</FormLabel>
                      <Switch colorScheme="brand" size="lg" defaultChecked />
                    </FormControl>
                  </VStack>
                </CardBody>
              </Card>
              
              <Button size="lg" colorScheme="brand" alignSelf="end">
                Save Settings
              </Button>
            </VStack>
          </TabPanel>
          
          {/* Ranking Weights */}
          <TabPanel>
            <Card>
              <CardHeader>
                <Heading size="md">Bot Ranking Weight Adjustments</Heading>
                <Text fontSize="sm" mt={1} color="gray.500">
                  Adjust the importance of each metric in the bot ranking algorithm
                </Text>
              </CardHeader>
              <CardBody>
                <Table variant="simple">
                  <Thead>
                    <Tr>
                      <Th>Metric</Th>
                      <Th>Description</Th>
                      <Th>Weight</Th>
                      <Th>Adjustment</Th>
                    </Tr>
                  </Thead>
                  <Tbody>
                    {weights.map((weight) => (
                      <Tr key={weight.variable_name}>
                        <Td fontWeight="medium" textTransform="capitalize">
                          {weight.variable_name.replace(/_/g, ' ')}
                        </Td>
                        <Td>{weight.description}</Td>
                        <Td>{(weight.weight * 100).toFixed(0)}%</Td>
                        <Td width="300px">
                          <Slider 
                            defaultValue={weight.weight * 100}
                            min={0}
                            max={50}
                            step={5}
                            colorScheme="brand"
                          >
                            <SliderTrack>
                              <SliderFilledTrack />
                            </SliderTrack>
                            <SliderThumb boxSize={6} />
                          </Slider>
                        </Td>
                      </Tr>
                    ))}
                  </Tbody>
                </Table>
                
                <Divider my={6} />
                
                <HStack justify="space-between">
                  <Button variant="outline">Reset to Default</Button>
                  <Button colorScheme="brand">Save Changes</Button>
                </HStack>
              </CardBody>
            </Card>
          </TabPanel>
          
          {/* Database Settings */}
          <TabPanel>
            <Card>
              <CardHeader>
                <Heading size="md">Database Configuration</Heading>
              </CardHeader>
              <CardBody>
                <VStack spacing={6} align="stretch">
                  <FormControl>
                    <FormLabel>Database Host</FormLabel>
                    <Input defaultValue="localhost" />
                  </FormControl>
                  
                  <FormControl>
                    <FormLabel>Database Name</FormLabel>
                    <Input defaultValue="tick_data" />
                  </FormControl>
                  
                  <FormControl>
                    <FormLabel>Username</FormLabel>
                    <Input defaultValue="clayb" />
                  </FormControl>
                  
                  <FormControl>
                    <FormLabel>Password</FormLabel>
                    <Input type="password" defaultValue="••••••••" />
                  </FormControl>
                  
                  <FormControl>
                    <FormLabel>Port</FormLabel>
                    <Input defaultValue="5432" />
                  </FormControl>
                  
                  <HStack spacing={4}>
                    <Button colorScheme="blue">Test Connection</Button>
                    <Button colorScheme="brand">Save Connection</Button>
                  </HStack>
                </VStack>
              </CardBody>
            </Card>
          </TabPanel>
          
          {/* API Settings */}
          <TabPanel>
            <Card>
              <CardHeader>
                <Heading size="md">Interactive Brokers API</Heading>
              </CardHeader>
              <CardBody>
                <VStack spacing={6} align="stretch">
                  <FormControl>
                    <FormLabel>IB Gateway Host</FormLabel>
                    <Input defaultValue="127.0.0.1" />
                  </FormControl>
                  
                  <FormControl>
                    <FormLabel>IB Gateway Port</FormLabel>
                    <Input defaultValue="4002" />
                  </FormControl>
                  
                  <FormControl>
                    <FormLabel>Client ID</FormLabel>
                    <Input defaultValue="0" />
                  </FormControl>
                  
                  <FormControl display="flex" alignItems="center">
                    <FormLabel mb="0">Enable Auto-Reconnect</FormLabel>
                    <Switch colorScheme="brand" size="lg" defaultChecked />
                  </FormControl>
                  
                  <FormControl display="flex" alignItems="center">
                    <FormLabel mb="0">Enable API Logging</FormLabel>
                    <Switch colorScheme="brand" size="lg" defaultChecked />
                  </FormControl>
                  
                  <HStack spacing={4}>
                    <Button colorScheme="blue">Test Connection</Button>
                    <Button colorScheme="brand">Save Settings</Button>
                  </HStack>
                </VStack>
              </CardBody>
            </Card>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </MainLayout>
  );
}