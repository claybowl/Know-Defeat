import React from 'react';
import {
  Box,
  Text,
  Heading,
  SimpleGrid,
  Card,
  CardHeader,
  CardBody,
  Divider,
  Badge,
  Flex,
  Button,
  HStack,
  Icon,
  useColorModeValue,
} from '@chakra-ui/react';
import { InfoIcon, ArrowForwardIcon } from '@chakra-ui/icons';
import { getMetricDocumentation } from './MetricInfoTooltip';

// Define related metric pairs with their relationship insights
const metricRelationships = [
  {
    primary: 'win_rate',
    secondary: 'profit_factor',
    name: 'Profitability Insight',
    description: 'Win Rate and Profit Factor together determine overall profitability',
    insight: 'A high win rate combined with a high profit factor indicates a robust strategy, but a low win rate can still be profitable if profit factor is high enough.',
    recommendation: 'Focus on improving both, but prioritize profit factor if you must choose.',
    example: 'A strategy with 40% win rate but 3.0 profit factor often outperforms one with 70% win rate but 1.2 profit factor.',
    idealState: 'Win Rate > 50% and Profit Factor > 1.5',
  },
  {
    primary: 'profit_factor',
    secondary: 'max_drawdown',
    name: 'Risk-Adjusted Return',
    description: 'Profit Factor and Max Drawdown indicate risk-adjusted performance',
    insight: 'A high profit factor with low drawdown represents an ideal risk-reward balance.',
    recommendation: 'Avoid strategies with both high drawdown and low profit factor.',
    example: 'Two bots with identical 1.8 profit factors can have vastly different risk profiles if one has 8% max drawdown vs. another with 25%.',
    idealState: 'Profit Factor > 1.5 and Max Drawdown < 15%',
  },
  {
    primary: 'average_win_amount',
    secondary: 'average_loss_amount',
    name: 'Trade Size Management',
    description: 'Average win and loss amounts determine trade sizing efficiency',
    insight: 'The ratio between these metrics forms the foundation of position sizing strategy.',
    recommendation: 'Adjust risk per trade based on the historical win/loss ratio.',
    example: 'If average wins are $100 and average losses are $50, each trade risks $1 to potentially make $2.',
    idealState: 'Average Win > Average Loss (ideally at least 1.5-2x larger)',
  },
  {
    primary: 'sharpe_ratio',
    secondary: 'total_pnl',
    name: 'Return Quality Assessment',
    description: 'Sharpe Ratio contextualizes Total P&L by accounting for risk',
    insight: 'High total P&L with low Sharpe suggests returns come from excessive risk-taking.',
    recommendation: 'Prioritize strategies with both good absolute returns and strong risk-adjusted metrics.',
    example: 'A bot with $5,000 profit and 2.1 Sharpe is preferable to one with $6,000 profit but 0.9 Sharpe.',
    idealState: 'Positive Total P&L with Sharpe Ratio > 1.5',
  },
  {
    primary: 'win_rate',
    secondary: 'risk_reward_ratio',
    name: 'Strategy Viability',
    description: 'Win Rate and Risk-Reward Ratio together determine if a strategy is viable',
    insight: 'Lower win rates require higher risk-reward ratios to remain profitable.',
    recommendation: 'Ensure your win rate × risk-reward ratio creates positive expectancy.',
    example: 'A strategy with 30% win rate needs at least 2.4:1 risk-reward ratio to break even.',
    idealState: 'Win Rate × Risk-Reward Ratio > 1',
  },
];

interface MetricInsightsPanelProps {
  onSelectRelationship?: (primary: string, secondary: string) => void;
}

export default function MetricInsightsPanel({ onSelectRelationship }: MetricInsightsPanelProps) {
  const cardBg = useColorModeValue('white', 'gray.700');
  const accentBg = useColorModeValue('blue.50', 'blue.900');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  return (
    <Box>
      <Heading size="md" mb={4}>Metric Relationship Insights</Heading>
      <Text mb={6}>
        Metrics become most valuable when analyzed together. These key relationships provide deeper understanding of trading performance.
      </Text>
      
      <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={4} mb={6}>
        {metricRelationships.map((relationship, index) => {
          const primaryMetric = getMetricDocumentation(relationship.primary);
          const secondaryMetric = getMetricDocumentation(relationship.secondary);
          
          return (
            <Card key={index} shadow="sm" bg={cardBg}>
              <CardHeader pb={2}>
                <Heading size="sm">{relationship.name}</Heading>
                <Text fontSize="xs" color="gray.500" mt={1}>{relationship.description}</Text>
              </CardHeader>
              <CardBody>
                <HStack mb={3}>
                  <Badge colorScheme="blue" p={1} borderRadius="md">
                    {primaryMetric.name}
                  </Badge>
                  <Icon as={ArrowForwardIcon} color="gray.400" boxSize={3} />
                  <Badge colorScheme="purple" p={1} borderRadius="md">
                    {secondaryMetric.name}
                  </Badge>
                </HStack>
                
                <Text fontSize="sm" mb={3}>
                  {relationship.insight}
                </Text>
                
                <Divider mb={3} />
                
                <Box bg={accentBg} p={2} borderRadius="md" mb={3}>
                  <Text fontSize="xs" fontWeight="bold">Ideal State:</Text>
                  <Text fontSize="xs">{relationship.idealState}</Text>
                </Box>
                
                <Text fontSize="xs" color="gray.500" fontStyle="italic">
                  {relationship.example}
                </Text>
                
                {onSelectRelationship && (
                  <Button 
                    size="xs" 
                    mt={3} 
                    width="100%" 
                    colorScheme="blue" 
                    variant="outline"
                    onClick={() => onSelectRelationship(relationship.primary, relationship.secondary)}
                  >
                    Visualize Relationship
                  </Button>
                )}
              </CardBody>
            </Card>
          );
        })}
      </SimpleGrid>
    </Box>
  );
} 