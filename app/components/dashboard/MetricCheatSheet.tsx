import React, { useRef } from 'react';
import {
  Box,
  Heading,
  Text,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Button,
  VStack,
  Divider,
  useColorModeValue,
  Grid,
  GridItem,
} from '@chakra-ui/react';
import { getMetricDocumentation } from './MetricInfoTooltip';
import { DownloadIcon, CopyIcon } from '@chakra-ui/icons';

// List of metrics to include in the cheat sheet
const cheatSheetMetrics = [
  'win_rate',
  'profit_factor',
  'sharpe_ratio',
  'max_drawdown',
  'average_win_amount',
  'average_loss_amount', 
  'risk_reward_ratio',
  'expectancy',
  'total_pnl',
  'total_trades',
];

export default function MetricCheatSheet() {
  const printRef = useRef<HTMLDivElement>(null);
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  
  // Handle print button click
  const handlePrint = () => {
    const content = printRef.current;
    if (!content) return;
    
    const printWindow = window.open('', '_blank');
    if (!printWindow) {
      alert('Please allow pop-ups to print the cheat sheet');
      return;
    }
    
    // Add print-specific styles
    printWindow.document.write(`
      <html>
        <head>
          <title>Trading Metrics Cheat Sheet</title>
          <style>
            body {
              font-family: Arial, sans-serif;
              line-height: 1.4;
              padding: 20px;
            }
            table {
              width: 100%;
              border-collapse: collapse;
              margin-bottom: 15px;
            }
            th, td {
              border: 1px solid #ddd;
              padding: 8px 12px;
              text-align: left;
            }
            th {
              background-color: #f8f9fa;
              font-weight: bold;
            }
            h1, h2 {
              color: #2c5282;
            }
            .metric-card {
              break-inside: avoid;
              margin-bottom: 15px;
              border: 1px solid #ddd;
              padding: 12px;
              border-radius: 4px;
            }
            .ideal-values {
              background-color: #ebf8ff;
              padding: 8px;
              border-radius: 4px;
              margin-top: 8px;
              font-size: 0.9em;
            }
            .section {
              margin-bottom: 20px;
            }
            .page-break {
              page-break-after: always;
            }
            @media print {
              .no-print {
                display: none;
              }
            }
          </style>
        </head>
        <body>
          ${content.innerHTML}
        </body>
      </html>
    `);
    
    printWindow.document.close();
    printWindow.focus();
    setTimeout(() => {
      printWindow.print();
    }, 250);
  };
  
  // Create readable ideal values
  const getIdealValue = (metric: string) => {
    switch (metric) {
      case 'win_rate': return '> 50%, ideally > 60%';
      case 'profit_factor': return '> 1.5, ideally > 2.0';
      case 'sharpe_ratio': return '> 1.0, ideally > 1.5';
      case 'max_drawdown': return '< 20%, ideally < 10%';
      case 'average_win_amount': return '> Average Loss Amount';
      case 'average_loss_amount': return 'As small as possible';
      case 'risk_reward_ratio': return '> 1.5:1, ideally > 2:1';
      case 'expectancy': return 'Positive, higher is better';
      case 'total_pnl': return 'Positive, consistently growing';
      case 'total_trades': return 'Enough for statistical significance (50+)';
      default: return 'Varies by metric';
    }
  };
  
  // Create quick observations based on metric combinations
  const keyRelationships = [
    "Win Rate + Profit Factor = Overall profitability. Low win rate can be offset by high profit factor.",
    "Profit Factor + Max Drawdown = Risk-adjusted return quality.",
    "Average Win + Average Loss = Trade sizing efficiency and risk-reward balance.",
    "Win Rate + Risk-Reward = Strategy viability (Win Rate × R:R Ratio > 1 for profitability).",
    "Expectancy = Win Rate × Average Win - (1 - Win Rate) × Average Loss",
  ];

  return (
    <Box>
      <Box mb={6}>
        <Heading size="lg" mb={2}>Metrics Cheat Sheet</Heading>
        <Text>
          A printable reference guide to all trading metrics used in the Know Defeat system.
        </Text>
        <Button 
          leftIcon={<CopyIcon />} 
          colorScheme="blue" 
          onClick={handlePrint}
          mt={4}
        >
          Print / Save as PDF
        </Button>
      </Box>
      
      <Box 
        ref={printRef} 
        p={6} 
        bg={bgColor} 
        border="1px" 
        borderColor={borderColor} 
        borderRadius="md"
        className="print-container"
      >
        <VStack align="stretch" spacing={8}>
          {/* Header for printed version */}
          <Box className="section">
            <Heading as="h1" size="xl" mb={2}>Trading Metrics Cheat Sheet</Heading>
            <Text>Know Defeat Trading System - Quick Reference Guide</Text>
            <Divider my={4} />
          </Box>
          
          {/* Quick reference table */}
          <Box className="section">
            <Heading as="h2" size="md" mb={3}>Quick Reference Table</Heading>
            <Table variant="simple" size="sm">
              <Thead>
                <Tr>
                  <Th>Metric</Th>
                  <Th>Description</Th>
                  <Th>Formula</Th>
                  <Th>Ideal Values</Th>
                </Tr>
              </Thead>
              <Tbody>
                {cheatSheetMetrics.map(metricKey => {
                  const metric = getMetricDocumentation(metricKey);
                  return (
                    <Tr key={metricKey}>
                      <Td fontWeight="bold">{metric.name}</Td>
                      <Td>{metric.description}</Td>
                      <Td fontFamily="monospace">{metric.formula}</Td>
                      <Td>{getIdealValue(metricKey)}</Td>
                    </Tr>
                  );
                })}
              </Tbody>
            </Table>
          </Box>
          
          <Divider className="page-break" />
          
          {/* Detailed metric cards */}
          <Box className="section">
            <Heading as="h2" size="md" mb={4}>Detailed Metrics Reference</Heading>
            <Grid templateColumns={{ base: "1fr", md: "repeat(2, 1fr)" }} gap={4}>
              {cheatSheetMetrics.map(metricKey => {
                const metric = getMetricDocumentation(metricKey);
                return (
                  <GridItem key={metricKey} className="metric-card">
                    <Heading as="h3" size="sm" mb={2}>{metric.name}</Heading>
                    <Text fontSize="sm" mb={2}>{metric.description}</Text>
                    <Text fontSize="sm" fontWeight="bold" mb={1}>Formula:</Text>
                    <Text fontSize="sm" fontFamily="monospace" mb={2}>{metric.formula}</Text>
                    <Text fontSize="sm" fontWeight="bold" mb={1}>Interpretation:</Text>
                    <Text fontSize="xs">• Low: {metric.valueRange.interpretation.lowValue}</Text>
                    <Text fontSize="xs">• High: {metric.valueRange.interpretation.highValue}</Text>
                    <Box className="ideal-values">
                      <Text fontSize="sm" fontWeight="bold">Ideal: {getIdealValue(metricKey)}</Text>
                    </Box>
                  </GridItem>
                );
              })}
            </Grid>
          </Box>
          
          {/* Key relationships */}
          <Box className="section">
            <Heading as="h2" size="md" mb={3}>Key Metric Relationships</Heading>
            <VStack align="stretch" spacing={2}>
              {keyRelationships.map((relationship, index) => (
                <Text key={index} fontSize="sm">• {relationship}</Text>
              ))}
            </VStack>
          </Box>
          
          {/* Footer */}
          <Box textAlign="center" mt={4} pt={4} borderTop="1px" borderColor={borderColor}>
            <Text fontSize="sm">© Curve AI Solutions - Know Defeat Trading System</Text>
            <Text fontSize="xs">Metrics documentation generated on {new Date().toLocaleDateString()}</Text>
          </Box>
        </VStack>
      </Box>
    </Box>
  );
} 