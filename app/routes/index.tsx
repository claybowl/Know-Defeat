import {
  Box,
  Heading,
  Text,
  Button,
  Container,
  VStack,
  SimpleGrid,
  Flex,
  Icon,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  useColorModeValue
} from '@chakra-ui/react';
import { FiTrendingUp, FiUsers, FiBarChart2, FiActivity } from 'react-icons/fi';
import type { IconType } from 'react-icons';
import { Link } from '@remix-run/react';
import Header from '~/components/layout/Header';
import Footer from '~/components/layout/Footer';

interface FeatureCardProps {
  title: string;
  icon: IconType;
  description: string;
}

const FeatureCard = ({ title, icon, description }: FeatureCardProps) => {
  return (
    <Box
      p={6}
      borderWidth="1px"
      borderRadius="lg"
      shadow="md"
      bg={useColorModeValue('white', 'gray.700')}
      _hover={{ transform: 'translateY(-5px)', transition: 'all 0.3s ease' }}
    >
      <Flex
        w={16}
        h={16}
        align={'center'}
        justify={'center'}
        color={'white'}
        rounded={'full'}
        bg={useColorModeValue('brand.500', 'brand.300')}
        mb={4}
      >
        <Icon as={icon} w={8} h={8} />
      </Flex>
      <Heading fontSize="xl" mb={2}>
        {title}
      </Heading>
      <Text color={useColorModeValue('gray.600', 'gray.300')}>
        {description}
      </Text>
    </Box>
  );
};

interface StatCardProps {
  label: string;
  value: string;
  change: string;
  isIncrease: boolean;
}

const StatCard = ({ label, value, change, isIncrease }: StatCardProps) => {
  return (
    <Box
      p={5}
      shadow="md"
      borderWidth="1px"
      borderRadius="lg"
      bg={useColorModeValue('white', 'gray.700')}
    >
      <Stat>
        <StatLabel>{label}</StatLabel>
        <StatNumber fontSize="2xl">{value}</StatNumber>
        <StatHelpText>
          <StatArrow type={isIncrease ? 'increase' : 'decrease'} />
          {change}
        </StatHelpText>
      </Stat>
    </Box>
  );
};

export default function Index() {
  return (
    <Box>
      <Header />
      
      <Box bg={useColorModeValue('gray.50', 'gray.900')}>
        {/* Hero Section */}
        <Box 
          backgroundImage="url('/assets/know_defeat_gpt.png')"
          backgroundSize="cover"
          backgroundPosition="center"
          color="white" 
          py={20}
          px={4}
          position="relative"
        >
          {/* Dark overlay for better text readability */}
          <Box 
            position="absolute" 
            top="0" 
            left="0" 
            width="100%" 
            height="100%" 
            bg="rgba(0, 0, 0, 0.5)" 
          />
          
          <Container maxW="container.xl" position="relative" zIndex="1">
            <VStack spacing={8} textAlign="center">
              <Heading as="h1" size="2xl" textShadow="0 0 10px rgba(0,0,0,0.5)">
                Know Defeat Trading System
              </Heading>
              <Text fontSize="xl" maxW="container.md" textShadow="0 0 8px rgba(0,0,0,0.7)">
                Advanced algorithmic trading platform with real-time analytics, 
                bot management, and intelligent fund allocation strategies
              </Text>
              <Box pt={4}>
                <Button 
                  as={Link}
                  to="/dashboard"
                  colorScheme="whiteAlpha" 
                  size="lg" 
                  mr={4}
                  _hover={{ bg: "gray.700" }}
                >
                  Dashboard
                </Button>
                <Button 
                  variant="outline" 
                  colorScheme="whiteAlpha" 
                  size="lg"
                  _hover={{ bg: "whiteAlpha.200" }}
                >
                  Learn More
                </Button>
              </Box>
            </VStack>
          </Container>
        </Box>

        {/* Stats Overview */}
        <Container maxW="container.xl" py={16}>
          <Heading mb={10} textAlign="center" size="xl">
            Performance At a Glance
          </Heading>
          <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={10}>
            <StatCard 
              label="Total Bots" 
              value="126" 
              change="23% since last month" 
              isIncrease={true} 
            />
            <StatCard 
              label="Open Trades" 
              value="38" 
              change="12% since yesterday" 
              isIncrease={true} 
            />
            <StatCard 
              label="Total P&L" 
              value="$83,426" 
              change="8.3% increase" 
              isIncrease={true} 
            />
            <StatCard 
              label="Win Rate" 
              value="78.2%" 
              change="5.4% since last week" 
              isIncrease={true} 
            />
          </SimpleGrid>
        </Container>

        {/* Features */}
        <Box bg={useColorModeValue('gray.100', 'gray.800')} py={16}>
          <Container maxW="container.xl">
            <Heading mb={10} textAlign="center" size="xl">
              Key Features
            </Heading>
            <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={10}>
              <FeatureCard 
                title="Bot Management" 
                icon={FiUsers}
                description="Control and configure trading bots with advanced parameter settings and real-time monitoring."
              />
              <FeatureCard 
                title="Performance Metrics" 
                icon={FiBarChart2}
                description="Track comprehensive metrics including win rate, profit factor, drawdown, and risk-adjusted returns."
              />
              <FeatureCard 
                title="Fund Allocation" 
                icon={FiTrendingUp}
                description="Intelligent fund allocation strategies with automated weight adjustments based on performance."
              />
              <FeatureCard 
                title="Real-time Trading" 
                icon={FiActivity}
                description="Automated execution with Interactive Brokers integration and sub-second market data processing."
              />
            </SimpleGrid>
          </Container>
        </Box>

        {/* Call To Action */}
        <Container maxW="container.xl" py={16} textAlign="center">
          <VStack spacing={6}>
            <Heading size="lg">
              Ready to optimize your trading system?
            </Heading>
            <Text fontSize="lg" maxW="container.md">
              Access the full trading dashboard to manage your bots, monitor performance, 
              and make data-driven decisions.
            </Text>
            <Button 
              as={Link}
              to="/dashboard"
              mt={4}
              colorScheme="brand" 
              size="lg" 
              px={8}
            >
              Launch Dashboard
            </Button>
          </VStack>
        </Container>
      </Box>
      
      <Footer />
    </Box>
  );
}