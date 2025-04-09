import React, { ReactNode } from 'react';
import {
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Box,
  Flex,
  Icon,
  useColorModeValue,
} from '@chakra-ui/react';
import { IconType } from 'react-icons';

interface StatCardProps {
  title: string;
  value: string | number;
  helpText?: ReactNode;
  icon?: IconType;
  iconColor?: string;
  trend?: 'increase' | 'decrease' | 'neutral';
  isLoading?: boolean;
}

const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  helpText,
  icon,
  iconColor = 'blue.500',
  trend = 'neutral',
  isLoading = false,
}) => {
  // Color mode based styling
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  
  // Trend-based colors
  const trendColor = {
    increase: 'green.500',
    decrease: 'red.500',
    neutral: 'gray.500',
  };
  
  return (
    <Stat
      px={4}
      py={5}
      shadow="sm"
      borderWidth="1px"
      borderRadius="lg"
      borderColor={borderColor}
      bg={bgColor}
      position="relative"
      transition="all 0.3s"
      _hover={{
        shadow: "md",
        transform: "translateY(-2px)",
      }}
    >
      {icon && (
        <Box
          position="absolute"
          top={4}
          right={4}
          borderRadius="full"
          bg={`${iconColor}10`}
          p={2}
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <Icon as={icon} color={iconColor} boxSize={5} />
        </Box>
      )}
      
      <StatLabel fontSize="sm" fontWeight="medium" color="gray.500">
        {title}
      </StatLabel>
      
      <StatNumber 
        fontSize="3xl" 
        fontWeight="bold" 
        my={2}
        color={trend !== 'neutral' ? trendColor[trend] : undefined}
      >
        {isLoading ? '—' : value}
      </StatNumber>
      
      {helpText && (
        <StatHelpText fontSize="sm" color="gray.500" m={0}>
          <Flex align="center">
            {helpText}
          </Flex>
        </StatHelpText>
      )}
    </Stat>
  );
};

export default StatCard;