import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import CssBaseline from '@mui/material/CssBaseline';
import theme from './theme';

// Import pages
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import Bots from './pages/Bots';
import BotDetail from './pages/BotDetail';
import Trades from './pages/Trades';
import Metrics from './pages/Metrics';
import Allocation from './pages/Allocation';
import NotFound from './pages/NotFound';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/bots" element={<Bots />} />
            <Route path="/bots/:id" element={<BotDetail />} />
            <Route path="/trades" element={<Trades />} />
            <Route path="/metrics" element={<Metrics />} />
            <Route path="/allocation" element={<Allocation />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Router>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
