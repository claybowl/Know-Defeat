# Know Defeat JavaScript UI - Sample Code Structure

This document provides sample code snippets to help kickstart development for the new JavaScript UI using Remix, Vite, Chakra UI, and ReCharts.

## Project Configuration

### `vite.config.js`
```javascript
import { defineConfig } from 'vite';
import { vitePlugin as remix } from '@remix-run/dev';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig({
  plugins: [
    remix({
      serverModuleFormat: 'esm',
    }),
    tsconfigPaths(),
  ],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
});
```

### `app/root.tsx`
```typescript
import { Links, LiveReload, Meta, Outlet, Scripts, ScrollRestoration } from '@remix-run/react';
import { ChakraProvider, ColorModeScript } from '@chakra-ui/react';
import { withEmotionCache } from '@emotion/react';
import theme from '~/lib/theme';

export const meta = () => {
  return [
    { title: 'Know Defeat Trading System' },
    { name: 'description', content: 'Advanced algorithmic trading platform' },
  ];
};

const Document = withEmotionCache(
  ({ children }: { children: React.ReactNode }, emotionCache) => {
    return (
      <html lang="en">
        <head>
          <meta charSet="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <Meta />
          <Links />
        </head>
        <body>
          <ColorModeScript initialColorMode={theme.config.initialColorMode} />
          {children}
          <ScrollRestoration />
          <Scripts />
          <LiveReload />
        </body>
      </html>
    );
  }
);

export default function App() {
  return (
    <Document>
      <ChakraProvider theme={theme}>
        <Outlet />
      </ChakraProvider>
    </Document>
  );
}
```

## Database Connection

### `app/lib/db.server.ts`
```typescript
import { Pool, QueryResult } from 'pg';

// Database connection pool
let pool: Pool;

// Create a singleton connection pool
function getPool(): Pool {
  if (!pool) {
    pool = new Pool({
      user: 'clayb',
      password: 'musicman',
      host: 'localhost',
      port: 5432,
      database: 'tick_data',
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    });

    // Log connection errors
    pool.on('error', (err) => {
      console.error('Unexpected error on idle client', err);
      process.exit(-1);
    });
  }
  return pool;
}

// Generic query function
export async function query<T>(
  text: string, 
  params: any[] = []
): Promise<QueryResult<T>> {
  const pool = getPool();
  const start = Date.now();
  const res = await pool.query(text, params);
  const duration = Date.now() - start;
  
  // Log slow queries
  if (duration > 500) {
    console.warn(`Slow query (${duration}ms): ${text}`);
  }
  
  return res;
}

// Close pool connection (for cleanup)
export async function closePool(): Promise<void> {
  if (pool) {
    await pool.end();
  }
}
```

## Authentication

### `app/lib/auth.server.ts`
```typescript
import { createCookieSessionStorage, redirect } from '@remix-run/node';
import { Authenticator, FormStrategy } from 'remix-auth';
import { query } from './db.server';
import bcrypt from 'bcryptjs';

// Define User type
export type User = {
  id: number;
  username: string;
  role: string;
};

// Session storage configuration
const sessionStorage = createCookieSessionStorage({
  cookie: {
    name: 'know_defeat_session',
    httpOnly: true,
    path: '/',
    sameSite: 'lax',
    secrets: ['YOUR_SECRET_KEY'], // Use env variable in production
    secure: process.env.NODE_ENV === 'production',
  },
});

// Create authenticator
export const authenticator = new Authenticator<User>(sessionStorage);

// Add form-based authentication
authenticator.use(
  new FormStrategy(async ({ form }) => {
    const username = form.get('username');
    const password = form.get('password');

    if (!username || typeof username !== 'string') {
      throw new Error('Username is required');
    }

    if (!password || typeof password !== 'string') {
      throw new Error('Password is required');
    }

    // Query the database for the user
    const result = await query<User>(
      'SELECT * FROM users WHERE username = $1',
      [username]
    );

    const user = result.rows[0];
    if (!user) {
      throw new Error('Invalid username or password');
    }

    // Verify password
    const isValidPassword = await bcrypt.compare(password, user.password_hash);
    if (!isValidPassword) {
      throw new Error('Invalid username or password');
    }

    // Return user without password
    const { password_hash, ...userWithoutPassword } = user;
    return userWithoutPassword as User;
  }),
  'form'
);

// Helper to get session
export async function getSession(request: Request) {
  const cookie = request.headers.get('Cookie');
  return sessionStorage.getSession(cookie);
}

// Require authentication
export async function requireUser(request: Request) {
  const user = await authenticator.isAuthenticated(request);
  if (!user) {
    throw redirect('/login');
  }
  return user;
}
```

## Layout Components

### `app/components/layout/DashboardLayout.tsx`
```typescript
import React, { ReactNode } from 'react';
import {
  Box,
  Flex,
  Icon,
  useColorModeValue,
  Link,
  BoxProps,
  Text,
  CloseButton,
  Drawer,
  DrawerContent,
  useDisclosure,
  IconButton,
  HStack,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  MenuDivider,
  Avatar,
  VStack,
  FlexProps,
} from '@chakra-ui/react';
import {
  FiHome,
  FiTrendingUp,
  FiCompass,
  FiStar,
  FiSettings,
  FiMenu,
  FiBell,
  FiChevronDown,
} from 'react-icons/fi';
import { IconType } from 'react-icons';
import { NavLink, useLocation } from '@remix-run/react';

interface LinkItemProps {
  name: string;
  icon: IconType;
  path: string;
}

const LinkItems: Array<LinkItemProps> = [
  { name: 'Dashboard', icon: FiHome, path: '/dashboard' },
  { name: 'Bot Management', icon: FiTrendingUp, path: '/bots' },
  { name: 'Trades', icon: FiCompass, path: '/trades' },
  { name: 'Metrics', icon: FiStar, path: '/metrics' },
  { name: 'Fund Allocation', icon: FiStar, path: '/allocation' },
  { name: 'Settings', icon: FiSettings, path: '/settings' },
];

export default function DashboardLayout({ children }: { children: ReactNode }) {
  const { isOpen, onOpen, onClose } = useDisclosure();
  
  return (
    <Box minH="100vh" bg={useColorModeValue('gray.50', 'gray.900')}>
      <SidebarContent
        onClose={() => onClose}
        display={{ base: 'none', md: 'block' }}
      />
      <Drawer
        autoFocus={false}
        isOpen={isOpen}
        placement="left"
        onClose={onClose}
        returnFocusOnClose={false}
        onOverlayClick={onClose}
        size="full">
        <DrawerContent>
          <SidebarContent onClose={onClose} />
        </DrawerContent>
      </Drawer>
      {/* Mobile nav */}
      <MobileNav onOpen={onOpen} />
      <Box ml={{ base: 0, md: 60 }} p="4">
        {children}
      </Box>
    </Box>
  );
}

interface SidebarProps extends BoxProps {
  onClose: () => void;
}

const SidebarContent = ({ onClose, ...rest }: SidebarProps) => {
  const location = useLocation();
  
  return (
    <Box
      transition="3s ease"
      bg={useColorModeValue('white', 'gray.900')}
      borderRight="1px"
      borderRightColor={useColorModeValue('gray.200', 'gray.700')}
      w={{ base: 'full', md: 60 }}
      pos="fixed"
      h="full"
      {...rest}>
      <Flex h="20" alignItems="center" mx="8" justifyContent="space-between">
        <Text fontSize="2xl" fontWeight="bold">
          Know Defeat
        </Text>
        <CloseButton display={{ base: 'flex', md: 'none' }} onClick={onClose} />
      </Flex>
      {LinkItems.map((link) => (
        <NavItem 
          key={link.name} 
          icon={link.icon} 
          path={link.path}
          isActive={location.pathname.startsWith(link.path)}
        >
          {link.name}
        </NavItem>
      ))}
    </Box>
  );
};

interface NavItemProps extends FlexProps {
  icon: IconType;
  path: string;
  isActive?: boolean;
  children: ReactNode;
}

const NavItem = ({ icon, path, isActive, children, ...rest }: NavItemProps) => {
  return (
    <Link 
      as={NavLink} 
      to={path} 
      style={{ textDecoration: 'none' }}
      _focus={{ boxShadow: 'none' }}
    >
      <Flex
        align="center"
        p="4"
        mx="4"
        borderRadius="lg"
        role="group"
        cursor="pointer"
        bg={isActive ? 'cyan.400' : 'transparent'}
        color={isActive ? 'white' : 'inherit'}
        _hover={{
          bg: 'cyan.400',
          color: 'white',
        }}
        {...rest}>
        {icon && (
          <Icon
            mr="4"
            fontSize="16"
            _groupHover={{
              color: 'white',
            }}
            as={icon}
          />
        )}
        {children}
      </Flex>
    </Link>
  );
};

interface MobileProps extends FlexProps {
  onOpen: () => void;
}

const MobileNav = ({ onOpen, ...rest }: MobileProps) => {
  return (
    <Flex
      ml={{ base: 0, md: 60 }}
      px={{ base: 4, md: 4 }}
      height="20"
      alignItems="center"
      bg={useColorModeValue('white', 'gray.900')}
      borderBottomWidth="1px"
      borderBottomColor={useColorModeValue('gray.200', 'gray.700')}
      justifyContent={{ base: 'space-between', md: 'flex-end' }}
      {...rest}>
      <IconButton
        display={{ base: 'flex', md: 'none' }}
        onClick={onOpen}
        variant="outline"
        aria-label="open menu"
        icon={<FiMenu />}
      />

      <Text
        display={{ base: 'flex', md: 'none' }}
        fontSize="2xl"
        fontWeight="bold">
        Know Defeat
      </Text>

      <HStack spacing={{ base: '0', md: '6' }}>
        <IconButton
          size="lg"
          variant="ghost"
          aria-label="open notifications"
          icon={<FiBell />}
        />
        <Flex alignItems={'center'}>
          <Menu>
            <MenuButton
              py={2}
              transition="all 0.3s"
              _focus={{ boxShadow: 'none' }}>
              <HStack>
                <Avatar
                  size={'sm'}
                  src={
                    'https://images.unsplash.com/photo-1619946794135-5bc917a27793?ixlib=rb-0.3.5&q=80&fm=jpg&crop=faces&fit=crop&h=200&w=200&s=b616b2c5b373a80ffc9636ba24f7a4a9'
                  }
                />
                <VStack
                  display={{ base: 'none', md: 'flex' }}
                  alignItems="flex-start"
                  spacing="1px"
                  ml="2">
                  <Text fontSize="sm">Admin User</Text>
                  <Text fontSize="xs" color="gray.600">
                    Administrator
                  </Text>
                </VStack>
                <Box display={{ base: 'none', md: 'flex' }}>
                  <FiChevronDown />
                </Box>
              </HStack>
            </MenuButton>
            <MenuList
              bg={useColorModeValue('white', 'gray.900')}
              borderColor={useColorModeValue('gray.200', 'gray.700')}>
              <MenuItem>Profile</MenuItem>
              <MenuItem>Settings</MenuItem>
              <MenuDivider />
              <MenuItem>Sign out</MenuItem>
            </MenuList>
          </Menu>
        </Flex>
      </HStack>
    </Flex>
  );
};
```

## Dashboard Components

### `app/routes/dashboard.tsx`
```typescript
import { Box, SimpleGrid, Text, Stat, StatLabel, StatNumber, StatHelpText, StatArrow, StatGroup } from '@chakra-ui/react';
import { LoaderFunction, json } from '@remix-run/node';
import { useLoaderData } from '@remix-run/react';
import DashboardLayout from '~/components/layout/DashboardLayout';
import PerformanceCard from '~/components/dashboard/PerformanceCard';
import TradeActivityChart from '~/components/charts/TradeActivityChart';
import MetricsTable from '~/components/tables/MetricsTable';
import { query } from '~/lib/db.server';
import { requireUser } from '~/lib/auth.server';

// Dashboard data structure
type DashboardData = {
  totalBots: number;
  activeBots: number;
  openTrades: number;
  todaysTrades: number;
  totalPnL: number;
  dailyPnL: number;
  topPerformers: any[];
  recentTrades: any[];
  metrics: any[];
};

// Loader function to get dashboard data
export const loader: LoaderFunction = async ({ request }) => {
  // Check authentication
  await requireUser(request);
  
  // Get dashboard data from database
  const totalBotsResult = await query('SELECT COUNT(*) FROM sim_bots');
  const activeBotsResult = await query('SELECT COUNT(*) FROM sim_bots WHERE is_active = true');
  const openTradesResult = await query('SELECT COUNT(*) FROM sim_bot_trades WHERE trade_status = $1', ['open']);
  const todaysTradesResult = await query(
    'SELECT COUNT(*) FROM sim_bot_trades WHERE entry_time >= CURRENT_DATE'
  );
  const totalPnLResult = await query('SELECT SUM(pnl) FROM sim_bot_trades WHERE trade_status = $1', ['closed']);
  const dailyPnLResult = await query(
    'SELECT SUM(pnl) FROM sim_bot_trades WHERE exit_time >= CURRENT_DATE AND trade_status = $1',
    ['closed']
  );
  
  // Get top performers
  const topPerformers = await query(
    `SELECT b.bot_id, b.name, b.ticker, m.total_pnl, m.win_rate, m.avg_profit_per_trade
     FROM bot_metrics m
     JOIN sim_bots b ON m.bot_id = b.bot_id
     ORDER BY m.rank_score DESC
     LIMIT 5`
  );
  
  // Get recent trades
  const recentTrades = await query(
    `SELECT t.trade_id, t.bot_id, b.name AS bot_name, t.ticker, t.entry_price, 
            t.exit_price, t.trade_direction, t.entry_time, t.exit_time, t.pnl, t.trade_status
     FROM sim_bot_trades t
     JOIN sim_bots b ON t.bot_id = b.bot_id
     ORDER BY COALESCE(t.exit_time, t.entry_time) DESC
     LIMIT 10`
  );
  
  // Return all data
  return json({
    totalBots: parseInt(totalBotsResult.rows[0].count),
    activeBots: parseInt(activeBotsResult.rows[0].count),
    openTrades: parseInt(openTradesResult.rows[0].count),
    todaysTrades: parseInt(todaysTradesResult.rows[0].count),
    totalPnL: parseFloat(totalPnLResult.rows[0].sum || '0'),
    dailyPnL: parseFloat(dailyPnLResult.rows[0].sum || '0'),
    topPerformers: topPerformers.rows,
    recentTrades: recentTrades.rows,
    metrics: []  // Additional metrics would be fetched here
  });
};

export default function Dashboard() {
  const data = useLoaderData<DashboardData>();
  
  return (
    <DashboardLayout>
      <Box>
        <Text fontSize="2xl" fontWeight="bold" mb={5}>
          System Dashboard
        </Text>
        
        {/* Performance overview */}
        <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={5} mb={8}>
          <PerformanceCard
            title="Total Bots"
            value={data.totalBots}
            subtitle={`${data.activeBots} active`}
            colorScheme="blue"
          />
          <PerformanceCard
            title="Open Trades"
            value={data.openTrades}
            subtitle={`${data.todaysTrades} today`}
            colorScheme="green"
          />
          <PerformanceCard
            title="Total P&L"
            value={`$${data.totalPnL.toFixed(2)}`}
            subtitle="All time"
            colorScheme={data.totalPnL >= 0 ? "green" : "red"}
          />
          <PerformanceCard
            title="Today's P&L"
            value={`$${data.dailyPnL.toFixed(2)}`}
            subtitle="Since midnight"
            colorScheme={data.dailyPnL >= 0 ? "green" : "red"}
          />
        </SimpleGrid>
        
        {/* Main content area */}
        <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={8}>
          {/* Trade activity chart */}
          <Box p={5} shadow="md" borderWidth="1px" borderRadius="md" bg="white">
            <Text fontSize="lg" fontWeight="medium" mb={4}>
              Trade Activity
            </Text>
            <TradeActivityChart />
          </Box>
          
          {/* Top performers table */}
          <Box p={5} shadow="md" borderWidth="1px" borderRadius="md" bg="white">
            <Text fontSize="lg" fontWeight="medium" mb={4}>
              Top Performing Bots
            </Text>
            <MetricsTable data={data.topPerformers} />
          </Box>
        </SimpleGrid>
      </Box>
    </DashboardLayout>
  );
}
```

## Chart Components

### `app/components/charts/TradeActivityChart.tsx`
```typescript
import React from 'react';
import { Box, useColorModeValue } from '@chakra-ui/react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

// Sample data - this would come from your API in a real implementation
const sampleData = [
  { date: '2025-03-20', trades: 12, pnl: 1540 },
  { date: '2025-03-21', trades: 19, pnl: 980 },
  { date: '2025-03-22', trades: 15, pnl: 1240 },
  { date: '2025-03-23', trades: 21, pnl: -590 },
  { date: '2025-03-24', trades: 28, pnl: 2100 },
  { date: '2025-03-25', trades: 24, pnl: 1600 },
  { date: '2025-03-26', trades: 18, pnl: 850 },
];

export default function TradeActivityChart() {
  const areaColor = useColorModeValue('blue.500', 'blue.200');
  const profitColor = useColorModeValue('green.500', 'green.200');
  const lossColor = useColorModeValue('red.500', 'red.200');
  
  return (
    <Box h="300px" w="100%">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={sampleData}
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id="colorTrades" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={areaColor} stopOpacity={0.8} />
              <stop offset="95%" stopColor={areaColor} stopOpacity={0.1} />
            </linearGradient>
            <linearGradient id="colorPnl" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={profitColor} stopOpacity={0.8} />
              <stop offset="95%" stopColor={profitColor} stopOpacity={0.1} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <Tooltip />
          <Legend />
          <Area
            yAxisId="left"
            type="monotone"
            dataKey="trades"
            stroke={areaColor}
            fillOpacity={1}
            fill="url(#colorTrades)"
          />
          <Area
            yAxisId="right"
            type="monotone"
            dataKey="pnl"
            stroke={profitColor}
            fillOpacity={1}
            fill="url(#colorPnl)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </Box>
  );
}
```

### `app/components/charts/FundAllocationChart.tsx`
```typescript
import React from 'react';
import { Box, useColorModeValue } from '@chakra-ui/react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';

// This would be populated with real data from your API
const sampleData = [
  { name: 'Bot 1 - TSLA', value: 2000 },
  { name: 'Bot 5 - NVDA', value: 2000 },
  { name: 'Bot 7 - AAPL', value: 2000 },
  { name: 'Bot 12 - COIN', value: 2000 },
  { name: 'Bot 23 - NVDA', value: 2000 },
];

// Color palette for the pie chart
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82ca9d', '#8dd1e1', '#a4de6c', '#d0ed57', '#ffc658'];

// Custom tooltip
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <Box 
        bg="white" 
        p={2} 
        shadow="md" 
        borderRadius="md" 
        borderWidth="1px"
      >
        <p>{`${payload[0].name}: $${payload[0].value}`}</p>
        <p>{`${(payload[0].payload.percent * 100).toFixed(2)}% of total`}</p>
      </Box>
    );
  }
  return null;
};

export default function FundAllocationChart() {
  // Calculate percentages
  const total = sampleData.reduce((sum, item) => sum + item.value, 0);
  const dataWithPercentage = sampleData.map(item => ({
    ...item,
    percent: item.value / total
  }));
  
  return (
    <Box h="300px" w="100%">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={dataWithPercentage}
            cx="50%"
            cy="50%"
            labelLine={false}
            outerRadius={100}
            fill="#8884d8"
            dataKey="value"
            label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
          >
            {dataWithPercentage.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </Box>
  );
}
```

## Table Components

### `app/components/tables/MetricsTable.tsx`
```typescript
import React from 'react';
import {
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Box,
  Badge,
  Text,
  chakra,
  Flex,
  IconButton,
  useColorModeValue,
} from '@chakra-ui/react';
import { TriangleDownIcon, TriangleUpIcon } from '@chakra-ui/icons';
import { FiEye, FiBarChart2 } from 'react-icons/fi';
import { useTable, useSortBy } from 'react-table';
import { Link } from '@remix-run/react';

export default function MetricsTable({ data }: { data: any[] }) {
  const columns = React.useMemo(
    () => [
      {
        Header: 'Bot ID',
        accessor: 'bot_id',
      },
      {
        Header: 'Name',
        accessor: 'name',
      },
      {
        Header: 'Ticker',
        accessor: 'ticker',
        Cell: ({ value }: { value: string }) => (
          <Badge colorScheme="blue" py={1} px={2} borderRadius="md">
            {value}
          </Badge>
        ),
      },
      {
        Header: 'Total P&L',
        accessor: 'total_pnl',
        Cell: ({ value }: { value: number }) => (
          <Text
            color={value >= 0 ? 'green.500' : 'red.500'}
            fontWeight="medium"
          >
            ${typeof value === 'number' ? value.toFixed(2) : value}
          </Text>
        ),
      },
      {
        Header: 'Win Rate',
        accessor: 'win_rate',
        Cell: ({ value }: { value: number }) => (
          <Text>{typeof value === 'number' ? (value * 100).toFixed(2) : value}%</Text>
        ),
      },
      {
        Header: 'Avg Profit/Trade',
        accessor: 'avg_profit_per_trade',
        Cell: ({ value }: { value: number }) => (
          <Text
            color={value >= 0 ? 'green.500' : 'red.500'}
            fontWeight="medium"
          >
            ${typeof value === 'number' ? value.toFixed(2) : value}
          </Text>
        ),
      },
      {
        Header: 'Actions',
        id: 'actions',
        Cell: ({ row }: { row: any }) => (
          <Flex>
            <IconButton
              as={Link}
              to={`/bots/${row.original.bot_id}`}
              aria-label="View bot"
              icon={<FiEye />}
              size="sm"
              mr={2}
              variant="ghost"
            />
            <IconButton
              as={Link}
              to={`/metrics/${row.original.bot_id}`}
              aria-label="View metrics"
              icon={<FiBarChart2 />}
              size="sm"
              variant="ghost"
            />
          </Flex>
        ),
      },
    ],
    []
  );

  const tableData = React.useMemo(() => data, [data]);

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    rows,
    prepareRow,
  } = useTable({ columns, data: tableData }, useSortBy);

  const bg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  return (
    <Box shadow="sm" borderWidth="1px" borderRadius="md" overflow="hidden" bg={bg}>
      <Box overflowX="auto">
        <Table {...getTableProps()} size="sm">
          <Thead bg={useColorModeValue('gray.50', 'gray.900')}>
            {headerGroups.map((headerGroup) => (
              <Tr {...headerGroup.getHeaderGroupProps()}>
                {headerGroup.headers.map((column: any) => (
                  <Th
                    {...column.getHeaderProps(column.getSortByToggleProps())}
                    px={4}
                    py={3}
                  >
                    <Flex align="center">
                      {column.render('Header')}
                      <chakra.span pl={2}>
                        {column.isSorted ? (
                          column.isSortedDesc ? (
                            <TriangleDownIcon aria-label="sorted descending" />
                          ) : (
                            <TriangleUpIcon aria-label="sorted ascending" />
                          )
                        ) : null}
                      </chakra.span>
                    </Flex>
                  </Th>
                ))}
              </Tr>
            ))}
          </Thead>
          <Tbody {...getTableBodyProps()}>
            {rows.map((row) => {
              prepareRow(row);
              return (
                <Tr
                  {...row.getRowProps()}
                  _hover={{ bg: useColorModeValue('gray.50', 'gray.700') }}
                >
                  {row.cells.map((cell) => (
                    <Td {...cell.getCellProps()} px={4} py={3} borderColor={borderColor}>
                      {cell.render('Cell')}
                    </Td>
                  ))}
                </Tr>
              );
            })}
          </Tbody>
        </Table>
      </Box>
    </Box>
  );
}
```

### `app/components/tables/TradesTable.tsx`
```typescript
import React from 'react';
import {
  Box,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  IconButton,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Text,
  useColorModeValue,
  Flex,
  Tooltip,
} from '@chakra-ui/react';
import { FiMoreVertical, FiX, FiEdit, FiBarChart2 } from 'react-icons/fi';
import { formatDistanceToNow } from 'date-fns';

// Format date to relative time
const formatDate = (dateString: string) => {
  try {
    const date = new Date(dateString);
    return formatDistanceToNow(date, { addSuffix: true });
  } catch (e) {
    return dateString;
  }
};

interface TradesTableProps {
  trades: any[];
  onCloseTrade?: (tradeId: number) => void;
}

export default function TradesTable({ trades, onCloseTrade }: TradesTableProps) {
  const bg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  
  // Function to get badge color based on trade status
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'open':
        return 'green';
      case 'closed':
        return 'gray';
      case 'pending_exit':
        return 'orange';
      default:
        return 'blue';
    }
  };
  
  // Function to format P&L with color
  const formatPnL = (pnl: number | null) => {
    if (pnl === null) return '-';
    const color = pnl >= 0 ? 'green.500' : 'red.500';
    return <Text color={color}>${pnl.toFixed(2)}</Text>;
  };
  
  return (
    <Box shadow="sm" borderWidth="1px" borderRadius="md" overflow="hidden" bg={bg}>
      <Box overflowX="auto">
        <Table size="sm">
          <Thead bg={useColorModeValue('gray.50', 'gray.900')}>
            <Tr>
              <Th>ID</Th>
              <Th>Bot</Th>
              <Th>Ticker</Th>
              <Th>Direction</Th>
              <Th>Entry Price</Th>
              <Th>Exit Price</Th>
              <Th>Status</Th>
              <Th>Entry Time</Th>
              <Th>P&L</Th>
              <Th>Actions</Th>
            </Tr>
          </Thead>
          <Tbody>
            {trades.map((trade) => (
              <Tr key={trade.trade_id} _hover={{ bg: useColorModeValue('gray.50', 'gray.700') }}>
                <Td borderColor={borderColor}>{trade.trade_id}</Td>
                <Td borderColor={borderColor}>
                  <Tooltip label={trade.bot_name}>
                    <Text>Bot {trade.bot_id}</Text>
                  </Tooltip>
                </Td>
                <Td borderColor={borderColor}>
                  <Badge colorScheme="blue">{trade.ticker}</Badge>
                </Td>
                <Td borderColor={borderColor}>
                  <Badge 
                    colorScheme={trade.trade_direction === 'LONG' ? 'green' : 'red'}
                  >
                    {trade.trade_direction}
                  </Badge>
                </Td>
                <Td borderColor={borderColor}>${trade.entry_price}</Td>
                <Td borderColor={borderColor}>
                  {trade.exit_price ? `$${trade.exit_price}` : '-'}
                </Td>
                <Td borderColor={borderColor}>
                  <Badge colorScheme={getStatusBadge(trade.trade_status)}>
                    {trade.trade_status}
                  </Badge>
                </Td>
                <Td borderColor={borderColor}>
                  {formatDate(trade.entry_time)}
                </Td>
                <Td borderColor={borderColor}>
                  {formatPnL(trade.pnl)}
                </Td>
                <Td borderColor={borderColor}>
                  <Menu>
                    <MenuButton
                      as={IconButton}
                      aria-label="Options"
                      icon={<FiMoreVertical />}
                      variant="ghost"
                      size="sm"
                    />
                    <MenuList>
                      <MenuItem 
                        icon={<FiBarChart2 />}
                        as="a" 
                        href={`/trades/${trade.trade_id}`}
                      >
                        View Details
                      </MenuItem>
                      {trade.trade_status === 'open' && (
                        <MenuItem 
                          icon={<FiX />} 
                          onClick={() => onCloseTrade && onCloseTrade(trade.trade_id)}
                        >
                          Close Trade
                        </MenuItem>
                      )}
                      <MenuItem icon={<FiEdit />}>Edit Trade</MenuItem>
                    </MenuList>
                  </Menu>
                </Td>
              </Tr>
            ))}
          </Tbody>
        </Table>
      </Box>
    </Box>
  );
}
```

## Card Components

### `app/components/dashboard/PerformanceCard.tsx`
```typescript
import React from 'react';
import { Box, Stat, StatLabel, StatNumber, StatHelpText, useColorModeValue } from '@chakra-ui/react';

interface PerformanceCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  colorScheme?: string;
  icon?: React.ReactNode;
}

export default function PerformanceCard({
  title,
  value,
  subtitle,
  colorScheme = 'blue',
  icon,
}: PerformanceCardProps) {
  // Generate colors based on colorScheme
  const lightColor = `${colorScheme}.50`;
  const mainColor = `${colorScheme}.500`;
  const darkColor = `${colorScheme}.600`;
  
  const bgColor = useColorModeValue(lightColor, `${colorScheme}.900`);
  const textColor = useColorModeValue(mainColor, `${colorScheme}.200`);
  const borderColor = useColorModeValue(mainColor, `${colorScheme}.700`);
  
  return (
    <Box
      borderRadius="lg"
      borderLeftWidth="4px"
      borderColor={borderColor}
      bg={bgColor}
      p={4}
      shadow="md"
    >
      <Stat>
        <StatLabel color={useColorModeValue('gray.600', 'gray.300')}>{title}</StatLabel>
        <StatNumber fontSize="2xl" color={textColor}>
          {value}
        </StatNumber>
        {subtitle && (
          <StatHelpText mb={0} fontSize="sm">
            {subtitle}
          </StatHelpText>
        )}
      </Stat>
    </Box>
  );
}
```

## Getting Started

To get started with this project, you'll need to:

1. Install Node.js (v16 or higher) and npm/yarn
2. Create a new project folder for the UI
3. Initialize the project with the files above
4. Install dependencies:

```bash
npm install remix vite @remix-run/dev @chakra-ui/react @emotion/react @emotion/styled framer-motion recharts react-icons date-fns remix-auth pg
```

5. Connect to your existing PostgreSQL database using the connection details in the code samples
6. Start the development server:

```bash
npm run dev
```

Remember to adjust the file paths, database connection details, and other specifics to match your environment.