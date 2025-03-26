import { json } from '@remix-run/node';
import { useLoaderData, Link } from '@remix-run/react';
import {
  Box,
  Heading,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Button,
  Text,
  Badge,
  Switch,
  Flex,
  HStack,
  Select,
  Input,
  InputGroup,
  InputLeftElement,
  Stack,
  Card,
  CardHeader,
  CardBody,
} from '@chakra-ui/react';
import { SearchIcon } from '@chakra-ui/icons';
import MainLayout from '~/components/layout/MainLayout';
import { getAllBots } from '~/lib/api.server';

export async function loader() {
  const bots = await getAllBots();
  return json({ bots });
}

export default function BotsIndex() {
  const { bots } = useLoaderData<typeof loader>();
  
  return (
    <MainLayout>
      <Flex justify="space-between" align="center" mb={8}>
        <Heading size="lg">Trading Bots</Heading>
        <Button colorScheme="brand">Register New Bot</Button>
      </Flex>
      
      {/* Filters */}
      <Stack 
        direction={{ base: 'column', md: 'row' }} 
        mb={6} 
        spacing={4}
        align={{ base: 'stretch', md: 'center' }}
      >
        <InputGroup maxW={{ md: '300px' }}>
          <InputLeftElement pointerEvents="none">
            <SearchIcon color="gray.300" />
          </InputLeftElement>
          <Input placeholder="Search bots..." />
        </InputGroup>
        
        <Select placeholder="Algorithm Type" maxW={{ md: '200px' }}>
          <option value="breakout">Breakout</option>
          <option value="mean_reversion">Mean Reversion</option>
          <option value="momentum">Momentum</option>
          <option value="volatility_breakout">Volatility Breakout</option>
          <option value="support_resistance">Support/Resistance</option>
        </Select>
        
        <Select placeholder="Symbol" maxW={{ md: '150px' }}>
          <option value="TSLA">TSLA</option>
          <option value="AAPL">AAPL</option>
          <option value="COIN">COIN</option>
          <option value="NVDA">NVDA</option>
          <option value="AMD">AMD</option>
        </Select>
        
        <HStack spacing={2}>
          <Text fontSize="sm">Active Only</Text>
          <Switch colorScheme="brand" />
        </HStack>
      </Stack>
      
      <Card shadow="base">
        <CardHeader pb={0}>
          <Heading size="md">Bot List</Heading>
        </CardHeader>
        <CardBody>
          <Box overflowX="auto">
            <Table variant="simple">
              <Thead>
                <Tr>
                  <Th>ID</Th>
                  <Th>Name</Th>
                  <Th>Symbol</Th>
                  <Th>Algorithm</Th>
                  <Th>Direction</Th>
                  <Th>Status</Th>
                  <Th>Actions</Th>
                </Tr>
              </Thead>
              <Tbody>
                {bots.map((bot) => (
                  <Tr key={bot.bot_id}>
                    <Td>{bot.bot_id}</Td>
                    <Td>
                      <Link to={`/bots/${bot.bot_id}`}>
                        {bot.name}
                      </Link>
                    </Td>
                    <Td>{bot.ticker}</Td>
                    <Td>{bot.algorithm_type}</Td>
                    <Td>{bot.trade_direction}</Td>
                    <Td>
                      <Badge 
                        colorScheme={bot.is_active ? 'green' : 'gray'}
                      >
                        {bot.is_active ? 'Active' : 'Inactive'}
                      </Badge>
                    </Td>
                    <Td>
                      <HStack spacing={2}>
                        <Button 
                          as={Link} 
                          to={`/bots/${bot.bot_id}`}
                          size="sm" 
                          colorScheme="blue"
                        >
                          View
                        </Button>
                        <Button 
                          size="sm" 
                          colorScheme={bot.is_active ? 'red' : 'green'}
                        >
                          {bot.is_active ? 'Disable' : 'Enable'}
                        </Button>
                      </HStack>
                    </Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          </Box>
        </CardBody>
      </Card>
    </MainLayout>
  );
}