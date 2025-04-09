// Bot type
export interface Bot {
  bot_id: number;
  name: string;
  ticker: string;
  algorithm_module: string;
  algorithm_type: string;
  trade_direction: 'LONG' | 'SHORT' | 'BOTH';
  position_size: number;
  trailing_stop_pct: number;
  description: string;
  version: string;
  is_active: boolean;
  created_at: string;
  last_updated: string;
  parameters?: Record<string, any>;
}

// Trade type
export interface Trade {
  trade_id: number;
  bot_id: number;
  ticker: string;
  entry_price: number;
  exit_price?: number;
  trade_size: number;
  trade_direction: 'LONG' | 'SHORT';
  entry_time: string;
  exit_time?: string;
  trade_status: 'open' | 'closed';
  pnl?: number;
  pnl_percent?: number;
  trailing_stop_price?: number;
  exit_reason?: string;
  bot_name?: string;
  algorithm_type?: string;
}

// Bot Metrics type
export interface BotMetrics {
  id: number;
  bot_id: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  total_pnl: number;
  average_pnl_per_trade: number;
  win_rate: number;
  average_win_amount: number;
  average_loss_amount: number;
  profit_factor: number;
  max_drawdown: number;
  sharpe_ratio: number;
  risk_reward_ratio: number;
  expectancy: number;
  rank_score: number;
  last_updated: string;
}

// Bot Detail (combines Bot with its trades and metrics)
export interface BotDetail extends Bot {
  trades: Trade[];
  metrics: BotMetrics | null;
}

// Dashboard data structure
export interface DashboardData {
  summary: {
    totalBots: number;
    activeBots: number;
    totalOpenTrades: number;
    totalPnl: number;
    avgWinRate: number;
  };
  topBots: BotMetrics[];
  recentTrades: Trade[];
  openTrades: Trade[];
}

// Allocation data structure
export interface AllocationData {
  totalAllocation: number;
  allocations: {
    bot_id: number;
    name: string;
    ticker: string;
    algorithm_type: string;
    rank_score: number;
    allocation: number;
    allocation_percent: number;
  }[];
}