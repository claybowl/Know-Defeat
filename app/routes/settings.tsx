import { json, type ActionArgs } from '@remix-run/node';
import { Form, useLoaderData, useSubmit } from '@remix-run/react';
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
  useToast,
} from '@chakra-ui/react';
import MainLayout from '~/components/layout/MainLayout';
import { useState, useEffect } from 'react';
import db from '~/lib/db.server';

// Define type for variable weights
interface VariableWeight {
  variable_name: string;
  weight: number | string; // Allow string initially from DB
  description?: string | null;
  last_updated?: string; // Keep if needed
}

export async function loader() {
  try {
    // Fetch actual weights from the database
    const weights: VariableWeight[] = await db.getVariableWeights(); 

    // Manually add descriptions (ideally store these better)
    const descriptions: Record<string, string> = {
      'two_hour_performance': 'Performance over the last 2 hours',
      'one_week_performance': 'Performance over the last week',
      'one_month_performance': 'Performance over the last month',
      'win_streak_3': 'Bonus for 3 consecutive wins',
      'win_streak_4': 'Bonus for 4 consecutive wins',
      'win_streak_5': 'Bonus for 5 consecutive wins',
      'avg_win_rate': 'Overall win rate (as percentage)',
      'one_hour_performance': 'Performance over the last hour',
      'one_day_performance': 'Performance over the last day',
      'win_streak_2': 'Bonus for 2 consecutive wins',
      // Add any other descriptions needed
    };

    const weightsWithDesc = weights.map(w => ({
        ...w,
        // Convert weight to number for consistency in the frontend
        weight: parseFloat(String(w.weight)), 
        description: descriptions[w.variable_name] || 'Metric influencing bot ranking score.' 
    }));

    return json({ weights: weightsWithDesc });
  } catch (error) {
    console.error("Error loading variable weights:", error);
    // Return empty array or default weights on error
    return json({ weights: [], error: 'Failed to load weights' }); 
  }
}

// Action function to handle saving changes
export async function action({ request }: ActionArgs) {
  const formData = await request.formData();
  const weightsData = formData.get('weights');
  
  if (typeof weightsData !== 'string') {
    return json({ error: 'Invalid weights data submitted' }, { status: 400 });
  }
  
  try {
    const weightsToUpdate: VariableWeight[] = JSON.parse(weightsData);
    
    // Prepare data for DB update (ensure weight is a number)
    const updatedWeights = weightsToUpdate.map(w => ({
        variable_name: w.variable_name,
        weight: parseFloat(String(w.weight)) 
    }));
    
    // Assume db.updateVariableWeights takes an array of {variable_name, weight}
    await db.updateVariableWeights(updatedWeights); 
    
    return json({ success: true, message: 'Weights updated successfully!' });
  } catch (error) {
    console.error("Failed to update weights:", error);
    // Check if error is an object and has a message property
    const message = (error instanceof Error) ? error.message : 'Failed to update weights';
    return json({ error: message }, { status: 500 });
  }
}

export default function Settings() {
  const { weights: initialWeights, error: loaderError } = useLoaderData<typeof loader>();
  const [weights, setWeights] = useState<VariableWeight[]>(initialWeights);
  const cardBg = useColorModeValue('white', 'gray.700');
  const submit = useSubmit();
  const toast = useToast(); // Initialize toast

  // Effect to show loader errors
  useEffect(() => {
    if (loaderError) {
      toast({
        title: "Error loading weights",
        description: loaderError,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    }
  }, [loaderError, toast]);
  
  // Effect to reset state if initialWeights change (e.g., after successful save and reload)
   useEffect(() => {
     setWeights(initialWeights);
   }, [initialWeights]);

  const handleSliderChange = (variableName: string, value: number) => {
    setWeights(prevWeights =>
      prevWeights.map(w =>
        w.variable_name === variableName ? { ...w, weight: value } : w
      )
    );
  };

  // Handle form submission via useSubmit
  const handleSaveChanges = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData();
    // Send current state weights
    formData.append('weights', JSON.stringify(weights)); 
    
    submit(formData, { 
      method: 'post',
      // Optional: optimistic update or navigation after success
      replace: true // replace history entry
    });

    // Show pending toast (optional)
     toast({
       title: "Saving changes...",
       status: "info",
       duration: 2000,
       isClosable: true,
     });
  };
  
  // Reset changes back to the initially loaded weights
  const handleResetChanges = () => {
      setWeights(initialWeights); 
       toast({
        title: "Changes reset",
        description: "Weights reverted to last saved values.",
        status: "info",
        duration: 3000,
        isClosable: true,
      });
  }

  // Determine max value for sliders dynamically
  // Ensure it's at least 100 or slightly more than the max weight
  const maxSliderValue = Math.ceil(Math.max(...weights.map(w => parseFloat(String(w.weight)) || 0), 100) / 10) * 10;

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
            {/* Wrap in Form */}
            <Form method="post" onSubmit={handleSaveChanges}>
              <Card>
                <CardHeader>
                  <Heading size="md">Bot Ranking Weight Adjustments</Heading>
                  <Text fontSize="sm" mt={1} color="gray.500">
                    Adjust the importance of each metric in the bot ranking algorithm. Weights determine influence on the final rank score.
                  </Text>
                </CardHeader>
                <CardBody>
                  <Table variant="simple">
                    <Thead>
                      <Tr>
                        <Th>Metric</Th>
                        <Th>Description</Th>
                        <Th isNumeric>Current Weight</Th> {/* Right align numeric */}
                        <Th>Adjustment</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {/* Map over state weights */}
                      {weights.map((weight) => (
                        <Tr key={weight.variable_name}>
                          <Td fontWeight="medium" textTransform="capitalize">
                            {weight.variable_name.replace(/_/g, ' ')}
                          </Td>
                          <Td fontSize="sm">{weight.description || 'N/A'}</Td>
                          {/* Display weight from state, formatted */}
                          <Td isNumeric fontWeight="medium">{parseFloat(String(weight.weight)).toFixed(1)}</Td>
                          <Td width={{ base: "150px", md: "300px" }}> {/* Responsive width */}
                            <Slider
                              // Controlled component: value linked to state
                              value={parseFloat(String(weight.weight))} 
                              min={0}
                              max={maxSliderValue} // Use dynamic max
                              step={0.5}          // Allow finer adjustments
                              colorScheme="brand"
                              // Update state on change
                              onChange={(val) => handleSliderChange(weight.variable_name, val)}
                              aria-label={`Slider for ${weight.variable_name}`}
                            >
                              <SliderTrack>
                                <SliderFilledTrack />
                              </SliderTrack>
                              <SliderThumb boxSize={5} /> {/* Slightly smaller thumb */}
                            </Slider>
                          </Td>
                        </Tr>
                      ))}
                      {/* Handle case where no weights are loaded */}
                       {weights.length === 0 && !loaderError && (
                         <Tr>
                           <Td colSpan={4} textAlign="center" py={6}>
                             <Text color="gray.500">No ranking weights found.</Text>
                           </Td>
                         </Tr>
                       )}
                       {loaderError && (
                         <Tr>
                           <Td colSpan={4} textAlign="center" py={6}>
                             <Text color="red.500">Error loading weights. Please try again later.</Text>
                           </Td>
                         </Tr>
                       )}
                    </Tbody>
                  </Table>
                  
                  <Divider my={6} />
                  
                  <HStack justify="space-between">
                    {/* Reset button */}
                    <Button variant="outline" onClick={handleResetChanges}>Reset Changes</Button> 
                    {/* Submit button */}
                    <Button colorScheme="brand" type="submit" isLoading={false /* TODO: Add loading state from useNavigation */}>
                      Save Changes
                    </Button> 
                  </HStack>
                </CardBody>
              </Card>
            </Form>
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