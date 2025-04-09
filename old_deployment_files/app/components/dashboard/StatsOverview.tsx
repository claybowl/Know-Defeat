import React from 'react';
import { SimpleGrid, Text, Flex, StatArrow } from '@chakra-ui/react';
import { FiUsers, FiBarChart, FiDollarSign, FiPercent } from 'react-icons/fi';
import StatCard from './StatCard';

interface StatsOverviewProps {
  stats: {
    totalBots: number;
    activeBots: number;
    totalOpenTrades: number;
    totalPnl: number;
    avgWinRate: number;
    botCount?: number;
    pnlChange?: number;
  };
  isLoading?: boolean;
}

const StatsOverview: React.FC<StatsOverviewProps> = ({ 
  stats, 
  isLoading = false 
}) => {
  // Format values
  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatPercent = (value: number): string => {
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6} mb={8}>
      {/* Total Bots Card */}
      <StatCard
        title="Trading Bots"
        value={stats.totalBots}
        helpText={<Text>{stats.activeBots} active</Text>}
        icon={FiUsers}
        iconColor="blue.500"
        isLoading={isLoading}
      />
      
      {/* Open Trades Card */}
      <StatCard
        title="Open Trades"
        value={stats.totalOpenTrades}
        helpText={<Text>Across {stats.botCount || '—'} bots</Text>}
        icon={FiBarChart}
        iconColor="purple.500"
        isLoading={isLoading}
      />
      
      {/* Total P&L Card */}
      <StatCard
        title="Total P&L"
        value={formatCurrency(stats.totalPnl)}
        helpText={
          <Flex align="center">
            {stats.pnlChange && (
              <>
                <StatArrow type={stats.pnlChange >= 0 ? 'increase' : 'decrease'} />
                {formatPercent(Math.abs(stats.pnlChange))}
              </>
            )}
            <Text ml={stats.pnlChange ? 2 : 0}>All time</Text>
          </Flex>
        }
        icon={FiDollarSign}
        iconColor={stats.totalPnl >= 0 ? "green.500" : "red.500"}
        trend={stats.totalPnl >= 0 ? "increase" : "decrease"}
        isLoading={isLoading}
      />
      
      {/* Win Rate Card */}
      <StatCard
        title="Win Rate"
        value={formatPercent(stats.avgWinRate)}
        helpText={<Text>System average</Text>}
        icon={FiPercent}
        iconColor="orange.500"
        isLoading={isLoading}
      />
    </SimpleGrid>
  );
};

export default StatsOverview;