import { 
  Bot, 
  BotDetail, 
  Trade, 
  BotMetrics, 
  DashboardData, 
  AllocationData 
} from '../types';

// Mock Bots
export const mockBots: Bot[] = [
  {
    bot_id: 1,
    name: "TSLA_Breakout_Bot",
    ticker: "TSLA",
    algorithm_module: "algorithms.breakout_algorithm",
    algorithm_type: "breakout",
    trade_direction: "BOTH",
    position_size: 1000.0,
    trailing_stop_pct: 0.01,
    description: "TSLA breakout strategy using volatility-based entry",
    version: "1.0",
    is_active: true,
    created_at: "2024-01-15T08:00:00Z",
    last_updated: "2024-02-20T14:30:00Z",
    parameters: {
      lookback_period: 20,
      volatility_threshold: 2.0,
      profit_target_pct: 0.02
    }
  },
  {
    bot_id: 2,
    name: "COIN_Mean_Reversion",
    ticker: "COIN",
    algorithm_module: "algorithms.mean_reversion",
    algorithm_type: "mean_reversion",
    trade_direction: "BOTH",
    position_size: 1500.0,
    trailing_stop_pct: 0.015,
    description: "Mean reversion strategy for COIN with RSI indicator",
    version: "1.2",
    is_active: true,
    created_at: "2024-01-18T09:15:00Z",
    last_updated: "2024-02-22T10:45:00Z",
    parameters: {
      rsi_period: 14,
      upper_threshold: 70,
      lower_threshold: 30
    }
  },
  {
    bot_id: 3,
    name: "SPY_Trend_Follower",
    ticker: "SPY",
    algorithm_module: "algorithms.trend_following",
    algorithm_type: "trend_following",
    trade_direction: "LONG",
    position_size: 2000.0,
    trailing_stop_pct: 0.008,
    description: "SPY trend following strategy with moving average crossover",
    version: "2.1",
    is_active: true,
    created_at: "2024-01-05T11:30:00Z",
    last_updated: "2024-02-15T16:20:00Z",
    parameters: {
      fast_period: 10,
      slow_period: 50,
      atr_multiple: 1.5
    }
  },
  {
    bot_id: 4,
    name: "AAPL_Support_Resistance",
    ticker: "AAPL",
    algorithm_module: "algorithms.support_resistance",
    algorithm_type: "support_resistance",
    trade_direction: "BOTH",
    position_size: 1200.0,
    trailing_stop_pct: 0.012,
    description: "AAPL strategy trading off key support and resistance levels",
    version: "1.5",
    is_active: false,
    created_at: "2024-01-08T15:45:00Z",
    last_updated: "2024-02-12T09:10:00Z",
    parameters: {
      pivot_lookback: 30,
      level_tolerance: 0.005,
      confirmation_candles: 2
    }
  },
  {
    bot_id: 5,
    name: "QQQ_Momentum",
    ticker: "QQQ",
    algorithm_module: "algorithms.momentum",
    algorithm_type: "momentum",
    trade_direction: "BOTH",
    position_size: 1800.0,
    trailing_stop_pct: 0.01,
    description: "QQQ momentum strategy with MACD indicator",
    version: "1.0",
    is_active: true,
    created_at: "2024-01-22T14:00:00Z",
    last_updated: "2024-02-18T11:50:00Z",
    parameters: {
      macd_fast: 12,
      macd_slow: 26,
      macd_signal: 9,
      volume_factor: 1.2
    }
  }
];

// Mock Trades
export const mockTrades: Trade[] = [
  {
    trade_id: 1001,
    bot_id: 1,
    ticker: "TSLA",
    entry_price: 248.50,
    exit_price: 252.75,
    trade_size: 1000.0,
    trade_direction: "LONG",
    entry_time: "2024-03-01T09:35:00Z",
    exit_time: "2024-03-01T15:45:00Z",
    trade_status: "closed",
    pnl: 171.0,
    pnl_percent: 0.0172,
    exit_reason: "target_reached",
    bot_name: "TSLA_Breakout_Bot",
    algorithm_type: "breakout"
  },
  {
    trade_id: 1002,
    bot_id: 2,
    ticker: "COIN",
    entry_price: 135.20,
    exit_price: 132.40,
    trade_size: 1500.0,
    trade_direction: "SHORT",
    entry_time: "2024-03-02T10:15:00Z",
    exit_time: "2024-03-03T11:20:00Z",
    trade_status: "closed",
    pnl: 312.0,
    pnl_percent: 0.0207,
    exit_reason: "rsi_exit",
    bot_name: "COIN_Mean_Reversion",
    algorithm_type: "mean_reversion"
  },
  {
    trade_id: 1003,
    bot_id: 3,
    ticker: "SPY",
    entry_price: 498.75,
    exit_price: 504.30,
    trade_size: 2000.0,
    trade_direction: "LONG",
    entry_time: "2024-03-03T09:45:00Z",
    exit_time: "2024-03-05T16:00:00Z",
    trade_status: "closed",
    pnl: 222.0,
    pnl_percent: 0.0111,
    exit_reason: "trailing_stop",
    bot_name: "SPY_Trend_Follower",
    algorithm_type: "trend_following"
  },
  {
    trade_id: 1004,
    bot_id: 5,
    ticker: "QQQ",
    entry_price: 386.50,
    trade_size: 1800.0,
    trade_direction: "LONG",
    entry_time: "2024-03-06T09:30:00Z",
    trade_status: "open",
    trailing_stop_price: 384.20,
    bot_name: "QQQ_Momentum",
    algorithm_type: "momentum"
  },
  {
    trade_id: 1005,
    bot_id: 1,
    ticker: "TSLA",
    entry_price: 253.40,
    trade_size: 1000.0,
    trade_direction: "SHORT",
    entry_time: "2024-03-06T13:15:00Z",
    trade_status: "open",
    trailing_stop_price: 257.20,
    bot_name: "TSLA_Breakout_Bot",
    algorithm_type: "breakout"
  },
  {
    trade_id: 1006,
    bot_id: 2,
    ticker: "COIN",
    entry_price: 138.60,
    exit_price: 142.75,
    trade_size: 1500.0,
    trade_direction: "LONG",
    entry_time: "2024-03-04T10:45:00Z",
    exit_time: "2024-03-06T09:20:00Z",
    trade_status: "closed",
    pnl: -93.75,
    pnl_percent: -0.0297,
    exit_reason: "stop_loss",
    bot_name: "COIN_Mean_Reversion",
    algorithm_type: "mean_reversion"
  }
];

// Mock Bot Metrics
export const mockBotMetrics: BotMetrics[] = [
  {
    id: 1,
    bot_id: 1,
    total_trades: 15,
    winning_trades: 9,
    losing_trades: 6,
    total_pnl: 856.50,
    average_pnl_per_trade: 57.10,
    win_rate: 0.60,
    average_win_amount: 145.25,
    average_loss_amount: -75.30,
    profit_factor: 2.89,
    max_drawdown: 320.45,
    sharpe_ratio: 1.75,
    risk_reward_ratio: 1.93,
    expectancy: 0.12,
    rank_score: 0.78,
    last_updated: "2024-03-06T18:00:00Z"
  },
  {
    id: 2,
    bot_id: 2,
    total_trades: 22,
    winning_trades: 15,
    losing_trades: 7,
    total_pnl: 1245.80,
    average_pnl_per_trade: 56.63,
    win_rate: 0.68,
    average_win_amount: 128.35,
    average_loss_amount: -92.45,
    profit_factor: 3.15,
    max_drawdown: 410.25,
    sharpe_ratio: 1.92,
    risk_reward_ratio: 1.39,
    expectancy: 0.15,
    rank_score: 0.85,
    last_updated: "2024-03-06T18:00:00Z"
  },
  {
    id: 3,
    bot_id: 3,
    total_trades: 12,
    winning_trades: 7,
    losing_trades: 5,
    total_pnl: 568.20,
    average_pnl_per_trade: 47.35,
    win_rate: 0.58,
    average_win_amount: 135.60,
    average_loss_amount: -68.45,
    profit_factor: 2.77,
    max_drawdown: 245.30,
    sharpe_ratio: 1.45,
    risk_reward_ratio: 1.98,
    expectancy: 0.09,
    rank_score: 0.72,
    last_updated: "2024-03-06T18:00:00Z"
  },
  {
    id: 4,
    bot_id: 4,
    total_trades: 10,
    winning_trades: 4,
    losing_trades: 6,
    total_pnl: -85.40,
    average_pnl_per_trade: -8.54,
    win_rate: 0.40,
    average_win_amount: 110.75,
    average_loss_amount: -85.30,
    profit_factor: 0.86,
    max_drawdown: 320.10,
    sharpe_ratio: 0.45,
    risk_reward_ratio: 1.30,
    expectancy: -0.02,
    rank_score: 0.35,
    last_updated: "2024-03-06T18:00:00Z"
  },
  {
    id: 5,
    bot_id: 5,
    total_trades: 18,
    winning_trades: 11,
    losing_trades: 7,
    total_pnl: 972.35,
    average_pnl_per_trade: 54.02,
    win_rate: 0.61,
    average_win_amount: 142.15,
    average_loss_amount: -79.85,
    profit_factor: 2.82,
    max_drawdown: 380.65,
    sharpe_ratio: 1.65,
    risk_reward_ratio: 1.78,
    expectancy: 0.11,
    rank_score: 0.75,
    last_updated: "2024-03-06T18:00:00Z"
  }
];

// Mock Dashboard Data
export const mockDashboardData: DashboardData = {
  summary: {
    totalBots: 5,
    activeBots: 4,
    totalOpenTrades: 2,
    totalPnl: 3557.45,
    avgWinRate: 0.574
  },
  topBots: mockBotMetrics.sort((a, b) => b.rank_score - a.rank_score).slice(0, 3),
  recentTrades: mockTrades.sort((a, b) => new Date(b.entry_time).getTime() - new Date(a.entry_time).getTime()).slice(0, 5),
  openTrades: mockTrades.filter(trade => trade.trade_status === "open")
};

// Mock Allocation Data
export const mockAllocationData: AllocationData = {
  totalAllocation: 20000,
  allocations: [
    {
      bot_id: 2,
      name: "COIN_Mean_Reversion",
      ticker: "COIN",
      algorithm_type: "mean_reversion",
      rank_score: 0.85,
      allocation: 4500,
      allocation_percent: 22.5
    },
    {
      bot_id: 1,
      name: "TSLA_Breakout_Bot",
      ticker: "TSLA",
      algorithm_type: "breakout",
      rank_score: 0.78,
      allocation: 4000,
      allocation_percent: 20.0
    },
    {
      bot_id: 5,
      name: "QQQ_Momentum",
      ticker: "QQQ",
      algorithm_type: "momentum",
      rank_score: 0.75,
      allocation: 3800,
      allocation_percent: 19.0
    },
    {
      bot_id: 3,
      name: "SPY_Trend_Follower",
      ticker: "SPY",
      algorithm_type: "trend_following",
      rank_score: 0.72,
      allocation: 3700,
      allocation_percent: 18.5
    },
    {
      bot_id: 4,
      name: "AAPL_Support_Resistance",
      ticker: "AAPL",
      algorithm_type: "support_resistance",
      rank_score: 0.35,
      allocation: 4000,
      allocation_percent: 20.0
    }
  ]
};

// Mock Bot Detail helpers
export const getMockBotDetail = (id: string | number): BotDetail => {
  const botId = typeof id === 'string' ? parseInt(id, 10) : id;
  const bot = mockBots.find(b => b.bot_id === botId);
  
  if (!bot) {
    throw new Error(`Bot with ID ${id} not found`);
  }
  
  const trades = mockTrades.filter(t => t.bot_id === botId);
  const metrics = mockBotMetrics.find(m => m.bot_id === botId) || null;
  
  return {
    ...bot,
    trades,
    metrics
  };
};