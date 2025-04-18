import { json } from '@remix-run/node';
import { Link, useSearchParams } from '@remix-run/react';
import {
  Box,
  Heading,
  Text,
  Container,
  VStack,
  HStack,
  SimpleGrid,
  Card,
  CardHeader,
  CardBody,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Divider,
  Badge,
  Code,
  UnorderedList,
  ListItem,
  Button,
  Icon,
  useColorModeValue,
  Tabs,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
} from '@chakra-ui/react';
import { useState } from 'react';
import { InfoIcon, ChevronLeftIcon } from '@chakra-ui/icons';
import MainLayout from '~/components/layout/MainLayout';
import MetricRelationshipChart from '~/components/charts/MetricRelationshipChart';
import MetricInsightsPanel from '~/components/dashboard/MetricInsightsPanel';
import MetricCheatSheet from '~/components/dashboard/MetricCheatSheet';

// Import the metrics documentation from our component to maintain a single source of truth
import MetricInfoTooltip from '~/components/dashboard/MetricInfoTooltip';

// Access the metrics documentation directly from the component
// This is a bit of a hack since we're using TypeScript - in practice we would refactor 
// to have a shared metrics documentation module that both this component and MetricInfoTooltip use
declare global {
  interface Window {
    __metricsDocumentation: any;
  }
}

// Documentation page - comprehensive list of all metrics with detailed explanations
export default function MetricsDocumentation() {
  const [searchParams] = useSearchParams();
  const tabParam = searchParams.get('tab');
  let initialTab = 0;
  
  if (tabParam === 'relationships') initialTab = 1;
  else if (tabParam === 'insights') initialTab = 2;
  else if (tabParam === 'cheatsheet') initialTab = 3;
  
  const [primaryMetric, setPrimaryMetric] = useState('win_rate');
  const [secondaryMetric, setSecondaryMetric] = useState('profit_factor');
  
  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const accentColor = useColorModeValue('blue.500', 'blue.300');
  const codeBg = useColorModeValue('gray.50', 'gray.800');
  
  // List of all metrics to document (preserved from original code)
  const metricKeys = [
    'total_pnl',
    'win_rate',
    'profit_factor',
    'sharpe_ratio',
    'max_drawdown',
    'average_win_amount',
    'average_loss_amount',
    'expectancy',
    'risk_reward_ratio',
    'rank_score',
    'total_trades',
    'average_pnl_per_trade'
  ];
  
  // Handle metric relationship selection from insights panel
  const handleRelationshipSelect = (primary: string, secondary: string) => {
    setPrimaryMetric(primary);
    setSecondaryMetric(secondary);
    
    // Switch to the relationships tab
    window.history.replaceState(null, '', '?tab=relationships');
    setTabIndex(1);
  };
  
  // Control tab index directly
  const [tabIndex, setTabIndex] = useState(initialTab);
  const handleTabChange = (index: number) => {
    setTabIndex(index);
    
    // Update URL without navigating
    const tabNames = ['reference', 'relationships', 'insights', 'cheatsheet'];
    window.history.replaceState(null, '', `?tab=${tabNames[index]}`);
  };

  return (
    <MainLayout>
      <Box mb={8}>
        <Button 
          as={Link} 
          to="/metrics" 
          leftIcon={<ChevronLeftIcon />} 
          variant="outline" 
          size="sm"
          mb={4}
        >
          Back to Metrics
        </Button>
        <Heading size="xl">Trading Metrics Documentation</Heading>
        <Text mt={2} color="gray.500">
          Comprehensive guide to understanding the performance metrics used in the Know Defeat trading system.
        </Text>
      </Box>

      {/* Add tabs for different documentation sections */}
      <Tabs variant="enclosed" colorScheme="blue" mb={8} index={tabIndex} onChange={handleTabChange}>
        <TabList>
          <Tab>Metrics Reference</Tab>
          <Tab>Metric Relationships</Tab>
          <Tab>Insights</Tab>
          <Tab>Cheat Sheet</Tab>
        </TabList>
        
        <TabPanels>
          {/* Tab 1: Individual Metrics Documentation */}
          <TabPanel p={0} pt={4}>
            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6} mb={8}>
              <Card shadow="sm" bg={cardBg}>
                <CardHeader>
                  <Heading size="md">About This Documentation</Heading>
                </CardHeader>
                <CardBody>
                  <Text>
                    This page provides detailed explanations of all metrics used to evaluate trading bot performance in the 
                    Know Defeat system. Each metric includes information about:
                  </Text>
                  <UnorderedList mt={4} spacing={2}>
                    <ListItem>Definition and purpose</ListItem>
                    <ListItem>Mathematical formula for calculation</ListItem>
                    <ListItem>Data sources and dependencies</ListItem>
                    <ListItem>Normal value ranges and interpretation</ListItem>
                    <ListItem>Common issues and troubleshooting</ListItem>
                    <ListItem>Best practices for analysis</ListItem>
                  </UnorderedList>
                </CardBody>
              </Card>

              <Card shadow="sm" bg={cardBg}>
                <CardHeader>
                  <Heading size="md">How to Use This Guide</Heading>
                </CardHeader>
                <CardBody>
                  <Text>
                    This documentation serves several purposes:
                  </Text>
                  <UnorderedList mt={4} spacing={2}>
                    <ListItem>
                      <Text fontWeight="bold">Reference</Text>
                      <Text>Detailed explanations of each metric's meaning and calculation</Text>
                    </ListItem>
                    <ListItem>
                      <Text fontWeight="bold">Troubleshooting</Text>
                      <Text>Understanding unexpected values and common calculation issues</Text>
                    </ListItem>
                    <ListItem>
                      <Text fontWeight="bold">Analysis Guide</Text>
                      <Text>Best practices for interpreting and using each metric effectively</Text>
                    </ListItem>
                    <ListItem>
                      <Text fontWeight="bold">Learning Resource</Text>
                      <Text>Educational content for users less familiar with trading metrics</Text>
                    </ListItem>
                  </UnorderedList>
                </CardBody>
              </Card>
            </SimpleGrid>

            <Divider mb={8} />

            <Heading size="lg" mb={6}>Performance Metrics</Heading>

            <Accordion allowMultiple defaultIndex={[0]} mb={8}>
              {/* Total P&L */}
              <AccordionItem borderColor={borderColor}>
                <AccordionButton py={4}>
                  <Box flex="1" textAlign="left">
                    <HStack>
                      <Heading size="md">Total P&L (Profit and Loss)</Heading>
                      <Badge colorScheme="blue">Core Metric</Badge>
                    </HStack>
                  </Box>
                  <AccordionIcon />
                </AccordionButton>
                <AccordionPanel pb={4}>
                  <VStack align="stretch" spacing={4}>
                    <Box>
                      <Heading size="sm" mb={2}>Description</Heading>
                      <Text>
                        The net profit or loss generated by the bot over its entire trading history.
                        This metric represents the bottom-line performance in financial terms.
                      </Text>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={2}>Formula</Heading>
                      <Code p={3} borderRadius="md" bg={codeBg} display="block">
                        Total P&L = Sum(Exit Price - Entry Price) × Position Size - Commissions
                      </Code>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={2}>Data Source</Heading>
                      <Text>
                        Calculated from closed trades in the database. Requires complete trade records with 
                        entry prices, exit prices, position sizes, and commission data.
                      </Text>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={2}>Value Range</Heading>
                      <Text fontWeight="bold">Normal Range:</Text>
                      <Text>Any value (positive or negative)</Text>
                      
                      <Text fontWeight="bold" mt={2}>Interpretation:</Text>
                      <UnorderedList pl={4}>
                        <ListItem>
                          <Text fontWeight="bold">Low Values:</Text> Negative values indicate overall losses
                        </ListItem>
                        <ListItem>
                          <Text fontWeight="bold">High Values:</Text> Positive values indicate overall profits
                        </ListItem>
                        <ListItem>
                          <Text fontWeight="bold">Optimal Zone:</Text> Consistently positive and growing over time
                        </ListItem>
                      </UnorderedList>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={2}>Common Issues</Heading>
                      <VStack align="stretch" spacing={2}>
                        <Box p={3} borderRadius="md" borderWidth="1px" borderColor={borderColor}>
                          <Text fontWeight="bold">P&L calculation might not include commission or slippage costs</Text>
                          <Text>Ensure trade execution records accurately capture all costs associated with trades</Text>
                        </Box>
                        <Box p={3} borderRadius="md" borderWidth="1px" borderColor={borderColor}>
                          <Text fontWeight="bold">Does not account for risk taken to achieve the P&L</Text>
                          <Text>Analyze alongside risk metrics like Max Drawdown and Sharpe Ratio</Text>
                        </Box>
                      </VStack>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={2}>Best Practices</Heading>
                      <UnorderedList spacing={1}>
                        <ListItem>Track P&L over time to identify trends in profitability</ListItem>
                        <ListItem>Compare against initial capital to understand returns relative to investment</ListItem>
                        <ListItem>Break down P&L by time periods, market conditions, or trade types</ListItem>
                        <ListItem>Consider risk-adjusted P&L metrics for more balanced evaluation</ListItem>
                      </UnorderedList>
                    </Box>
                  </VStack>
                </AccordionPanel>
              </AccordionItem>

              {/* Win Rate */}
              <AccordionItem borderColor={borderColor}>
                <AccordionButton py={4}>
                  <Box flex="1" textAlign="left">
                    <HStack>
                      <Heading size="md">Win Rate</Heading>
                      <Badge colorScheme="blue">Core Metric</Badge>
                    </HStack>
                  </Box>
                  <AccordionIcon />
                </AccordionButton>
                <AccordionPanel pb={4}>
                  <VStack align="stretch" spacing={4}>
                    <Box>
                      <Heading size="sm" mb={2}>Description</Heading>
                      <Text>
                        The percentage of trades that resulted in a profit, representing the consistency 
                        of a trading strategy in generating winning trades.
                      </Text>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={2}>Formula</Heading>
                      <Code p={3} borderRadius="md" bg={codeBg} display="block">
                        Win Rate = (Number of Winning Trades / Total Number of Closed Trades) × 100
                      </Code>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={2}>Data Source</Heading>
                      <Text>
                        Calculated from trade history in the database, counting trades with positive and 
                        non-positive P&L values.
                      </Text>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={2}>Value Range</Heading>
                      <Text fontWeight="bold">Normal Range:</Text>
                      <Text>0% - 100%</Text>
                      
                      <Text fontWeight="bold" mt={2}>Interpretation:</Text>
                      <UnorderedList pl={4}>
                        <ListItem>
                          <Text fontWeight="bold">Low Values:</Text> Below 50% indicates more losing trades than winning trades
                        </ListItem>
                        <ListItem>
                          <Text fontWeight="bold">High Values:</Text> Above 50% indicates more winning trades than losing trades
                        </ListItem>
                        <ListItem>
                          <Text fontWeight="bold">Optimal Zone:</Text> Typically above 60%, but depends on strategy risk/reward profile
                        </ListItem>
                      </UnorderedList>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={2}>Common Issues</Heading>
                      <VStack align="stretch" spacing={2}>
                        <Box p={3} borderRadius="md" borderWidth="1px" borderColor={borderColor}>
                          <Text fontWeight="bold">A small sample size of trades might not be statistically significant</Text>
                          <Text>Wait for sufficient trades (50+) before drawing strong conclusions</Text>
                        </Box>
                        <Box p={3} borderRadius="md" borderWidth="1px" borderColor={borderColor}>
                          <Text fontWeight="bold">High win rate doesn't guarantee profitability if losses exceed wins</Text>
                          <Text>Evaluate with Profit Factor, Average Profit per Trade, and Risk/Reward Ratio</Text>
                        </Box>
                      </VStack>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={2}>Best Practices</Heading>
                      <UnorderedList spacing={1}>
                        <ListItem>Monitor how win rate changes across different market conditions</ListItem>
                        <ListItem>Use win rate as one component in evaluating strategy consistency</ListItem>
                        <ListItem>Compare with average win/loss amounts to understand overall profitability</ListItem>
                        <ListItem>Track win rate over time to identify strategy degradation</ListItem>
                      </UnorderedList>
                    </Box>
                  </VStack>
                </AccordionPanel>
              </AccordionItem>

              {/* Add similar AccordionItems for other metrics */}
              {/* Profit Factor */}
              <AccordionItem borderColor={borderColor}>
                <AccordionButton py={4}>
                  <Box flex="1" textAlign="left">
                    <HStack>
                      <Heading size="md">Profit Factor</Heading>
                      <Badge colorScheme="blue">Core Metric</Badge>
                    </HStack>
                  </Box>
                  <AccordionIcon />
                </AccordionButton>
                <AccordionPanel pb={4}>
                  <VStack align="stretch" spacing={4}>
                    <Box>
                      <Heading size="sm" mb={2}>Description</Heading>
                      <Text>
                        Measures the ratio of gross profits to gross losses. Indicates how many dollars 
                        are won for every dollar lost.
                      </Text>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={2}>Formula</Heading>
                      <Code p={3} borderRadius="md" bg={codeBg} display="block">
                        Profit Factor = Sum of Profits from Winning Trades / |Sum of Losses from Losing Trades|
                      </Code>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={2}>Data Source</Heading>
                      <Text>
                        Calculated from P&L of winning and losing trades.
                      </Text>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={2}>Value Range</Heading>
                      <Text fontWeight="bold">Normal Range:</Text>
                      <Text>Greater than 0, with values &gt; 1 indicating profitability</Text>
                      
                      <Text fontWeight="bold" mt={2}>Interpretation:</Text>
                      <UnorderedList pl={4}>
                        <ListItem>
                          <Text fontWeight="bold">Low Values:</Text> Values below 1 indicate the bot loses more money than it makes
                        </ListItem>
                        <ListItem>
                          <Text fontWeight="bold">High Values:</Text> Values above 1.5 or 2.0 suggest a robustly profitable strategy
                        </ListItem>
                        <ListItem>
                          <Text fontWeight="bold">Optimal Zone:</Text> Above 1.5 is good, above 2.0 is excellent
                        </ListItem>
                      </UnorderedList>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={2}>Common Issues</Heading>
                      <VStack align="stretch" spacing={2}>
                        <Box p={3} borderRadius="md" borderWidth="1px" borderColor={borderColor}>
                          <Text fontWeight="bold">Can be skewed by a single large winning or losing trade</Text>
                          <Text>Analyze the distribution of trade P&L and ensure sufficient sample size</Text>
                        </Box>
                        <Box p={3} borderRadius="md" borderWidth="1px" borderColor={borderColor}>
                          <Text fontWeight="bold">Undefined if there are no losing trades</Text>
                          <Text>Handle division by zero in calculations; interpret as highly profitable</Text>
                        </Box>
                      </VStack>
                    </Box>

                    <Box>
                      <Heading size="sm" mb={2}>Best Practices</Heading>
                      <UnorderedList spacing={1}>
                        <ListItem>Use as a primary indicator of overall strategy profitability efficiency</ListItem>
                        <ListItem>Combine with Win Rate to understand if profitability comes from frequent small wins or infrequent large wins</ListItem>
                        <ListItem>Calculate over different time periods to identify changes in strategy effectiveness</ListItem>
                      </UnorderedList>
                    </Box>
                  </VStack>
                </AccordionPanel>
              </AccordionItem>

              {/* Include additional key metrics (max_drawdown, sharpe_ratio, etc.) */}
            </Accordion>
          </TabPanel>
          
          {/* Tab 2: Metric Relationships */}
          <TabPanel p={0} pt={4}>
            <Box mb={6}>
              <Heading size="lg" mb={3}>Metric Relationships</Heading>
              <Text>
                While individual metrics provide valuable insights, understanding how metrics relate to each other 
                offers a more comprehensive view of trading performance. This interactive visualization allows you 
                to explore relationships between different performance indicators.
              </Text>
            </Box>
            
            {/* Interactive relationship chart */}
            <Card shadow="sm" mb={8} bg={cardBg}>
              <CardBody>
                <MetricRelationshipChart 
                  primaryMetric={primaryMetric}
                  secondaryMetric={secondaryMetric}
                />
              </CardBody>
            </Card>
            
            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6} mb={8}>
              <Card shadow="sm" bg={cardBg}>
                <CardHeader>
                  <Heading size="md">Understanding Metric Relationships</Heading>
                </CardHeader>
                <CardBody>
                  <VStack align="start" spacing={4}>
                    <Text>
                      Metrics should never be viewed in isolation. For example, a high win rate alone doesn't 
                      guarantee profitability - it must be paired with a favorable risk-reward ratio.
                    </Text>
                    
                    <Box p={3} borderRadius="md" borderWidth="1px" borderColor={borderColor} w="100%">
                      <Heading size="sm" mb={2}>Key Relationships</Heading>
                      <UnorderedList spacing={2}>
                        <ListItem>
                          <Text fontSize="sm" fontWeight="bold">Win Rate + Profit Factor</Text>
                          <Text fontSize="sm">
                            Together define overall profitability. A strategy with 40% win rate but 3.0 profit factor 
                            can outperform one with 70% win rate but 1.2 profit factor.
                          </Text>
                        </ListItem>
                        <ListItem>
                          <Text fontSize="sm" fontWeight="bold">Profit Factor + Max Drawdown</Text>
                          <Text fontSize="sm">
                            Indicates risk-adjusted returns. High profit factor with low drawdown represents 
                            the most desirable combination.
                          </Text>
                        </ListItem>
                        <ListItem>
                          <Text fontSize="sm" fontWeight="bold">Sharpe Ratio + Total P&L</Text>
                          <Text fontSize="sm">
                            Shows whether profits are achieved through skill or excessive risk-taking.
                          </Text>
                        </ListItem>
                      </UnorderedList>
                    </Box>
                  </VStack>
                </CardBody>
              </Card>
              
              <Card shadow="sm" bg={cardBg}>
                <CardHeader>
                  <Heading size="md">Applying Metric Relationships</Heading>
                </CardHeader>
                <CardBody>
                  <VStack align="start" spacing={4}>
                    <Text>
                      Understanding how metrics interact helps with both strategy development and bot evaluation:
                    </Text>
                    
                    <Box p={3} borderRadius="md" borderWidth="1px" borderColor={borderColor} w="100%">
                      <Heading size="sm" mb={2}>Strategy Development</Heading>
                      <Text fontSize="sm">
                        When designing trading strategies, focus on metrics that complement each other. For example, 
                        optimizing for profit factor rather than just win rate can lead to more robust performance.
                      </Text>
                    </Box>
                    
                    <Box p={3} borderRadius="md" borderWidth="1px" borderColor={borderColor} w="100%">
                      <Heading size="sm" mb={2}>Bot Evaluation</Heading>
                      <Text fontSize="sm">
                        Use the relationships between metrics to identify bots with sustainable performance versus those 
                        with metrics that suggest potential problems. For instance, a bot with high profit but also high 
                        drawdown might not be suitable for larger capital allocation.
                      </Text>
                    </Box>
                    
                    <Box p={3} borderRadius="md" borderWidth="1px" borderColor={borderColor} w="100%">
                      <Heading size="sm" mb={2}>Risk Management</Heading>
                      <Text fontSize="sm">
                        The relationship between profit metrics and risk metrics helps determine appropriate position 
                        sizing and capital allocation decisions across different bots.
                      </Text>
                    </Box>
                  </VStack>
                </CardBody>
              </Card>
            </SimpleGrid>
          </TabPanel>
          
          {/* Tab 3: Metric Insights */}
          <TabPanel p={0} pt={4}>
            <MetricInsightsPanel onSelectRelationship={handleRelationshipSelect} />
          </TabPanel>
          
          {/* Tab 4: Metrics Cheat Sheet */}
          <TabPanel p={0} pt={4}>
            <MetricCheatSheet />
          </TabPanel>
        </TabPanels>
      </Tabs>

      <Box textAlign="center" mb={8}>
        <Button 
          as={Link}
          to="/metrics"
          leftIcon={<ChevronLeftIcon />}
          colorScheme="blue"
        >
          Return to Metrics Dashboard
        </Button>
      </Box>
    </MainLayout>
  );
} 