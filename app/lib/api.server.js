import db from './db.server';

export async function getDashboardData() {
  try {
    console.log("Fetching dashboard data...");
    
    // Run queries in parallel for better performance
    const [bots, openTrades, metrics] = await Promise.all([
      db.getBots(),
      db.getOpenTrades(),
      db.getBotMetrics(),
    ]);

    console.log(`Retrieved data: ${bots.length} bots, ${openTrades.length} open trades, ${metrics.length} bot metrics`);

    // Calculate overall system metrics
    const totalBots = bots.length;
    const activeBots = bots.filter(bot => bot.is_active).length;
    const totalOpenTrades = openTrades.length;
    
    // Calculate total P&L across all bots
    const totalPnl = metrics.reduce((sum, bot) => sum + parseFloat(bot.total_pnl || 0), 0);
    
    // Calculate average win rate
    const botsWithTrades = metrics.filter(bot => bot.total_trades > 0);
    const avgWinRate = botsWithTrades.length > 0 
      ? botsWithTrades.reduce((sum, bot) => sum + parseFloat(bot.win_rate || 0), 0) / botsWithTrades.length
      : 0;

    // Get top performing bots
    const topBots = metrics
      .filter(bot => bot.total_trades > 0)
      .sort((a, b) => parseFloat(b.rank_score || 0) - parseFloat(a.rank_score || 0))
      .slice(0, 5);

    // Get recent trades
    const recentTrades = await db.getTrades(10);

    return {
      summary: {
        totalBots,
        activeBots,
        totalOpenTrades,
        totalPnl,
        avgWinRate,
      },
      topBots,
      recentTrades,
      openTrades: openTrades.slice(0, 10),
    };
  } catch (error) {
    console.error('Error fetching dashboard data:', error);
    
    // Return mock data as fallback in case of error
    const mockData = {
      summary: {
        totalBots: 8,
        activeBots: 7,
        totalOpenTrades: 5,
        totalPnl: 11553.8,
        avgWinRate: 0.59,
      },
      topBots: [
        { bot_id: 1, win_rate: 0.65, profit_factor: 2.10, total_pnl: 2450.75, rank_score: 0.92 },
        { bot_id: 3, win_rate: 0.64, profit_factor: 1.95, total_pnl: 2120.50, rank_score: 0.89 },
        { bot_id: 7, win_rate: 0.64, profit_factor: 1.92, total_pnl: 1870.30, rank_score: 0.87 },
        { bot_id: 2, win_rate: 0.57, profit_factor: 1.68, total_pnl: 1650.20, rank_score: 0.78 },
        { bot_id: 8, win_rate: 0.55, profit_factor: 1.62, total_pnl: 1320.60, rank_score: 0.75 },
      ],
      recentTrades: [
        { trade_id: 1, bot_id: 1, bot_name: 'TSLA_Breakout_Bot', ticker: 'TSLA', trade_direction: 'LONG', trade_status: 'closed', pnl: 290.83 },
        { trade_id: 2, bot_id: 2, bot_name: 'COIN_Momentum_Bot', ticker: 'COIN', trade_direction: 'LONG', trade_status: 'closed', pnl: -211.39 },
        { trade_id: 3, bot_id: 3, bot_name: 'NVDA_Breakout_Bot', ticker: 'NVDA', trade_direction: 'LONG', trade_status: 'closed', pnl: 234.21 },
        { trade_id: 4, bot_id: 4, bot_name: 'AMD_Momentum_Bot', ticker: 'AMD', trade_direction: 'LONG', trade_status: 'closed', pnl: 156.52 },
        { trade_id: 5, bot_id: 5, bot_name: 'AAPL_Support_Resistance_Bot', ticker: 'AAPL', trade_direction: 'LONG', trade_status: 'closed', pnl: -137.61 },
      ],
      openTrades: [
        { trade_id: 6, bot_id: 1, ticker: 'TSLA', trade_direction: 'LONG', entry_price: 182.40 },
        { trade_id: 7, bot_id: 3, ticker: 'NVDA', trade_direction: 'LONG', entry_price: 965.25 },
        { trade_id: 8, bot_id: 7, ticker: 'META', trade_direction: 'LONG', entry_price: 485.00 },
        { trade_id: 9, bot_id: 8, ticker: 'AMZN', trade_direction: 'LONG', entry_price: 180.50 },
        { trade_id: 10, bot_id: 2, ticker: 'COIN', trade_direction: 'LONG', entry_price: 208.25 },
      ],
    };
    
    return mockData;
  }
}

export async function getAllBots() {
  try {
    console.log("Fetching all bots...");
    const bots = await db.getBots();
    console.log(`Retrieved ${bots.length} bots`);
    return bots;
  } catch (error) {
    console.error('Error fetching bots:', error);
    // Return mock data as fallback
    console.log("Falling back to mock bots data");
    return db.mockData ? db.mockData.bots : [];
  }
}

export async function getBotById(botId) {
  try {
    console.log(`Fetching bot with ID ${botId}...`);
    const bot = await db.getBotById(botId);
    if (bot) {
      console.log(`Successfully retrieved bot with ID ${botId}`);
    } else {
      console.log(`No bot found with ID ${botId}`);
    }
    return bot;
  } catch (error) {
    console.error(`Error fetching bot ${botId}:`, error);
    // Return mock data as fallback
    if (db.mockData && db.mockData.bots) {
      const mockBot = db.mockData.bots.find(b => b.bot_id === parseInt(botId));
      if (mockBot) {
        console.log(`Falling back to mock data for bot ${botId}`);
        
        // Get bot trades from mock data
        const trades = db.mockData.trades.filter(t => t.bot_id === parseInt(botId));
        
        // Get bot metrics from mock data
        const metrics = db.mockData.metrics.find(m => m.bot_id === parseInt(botId));
        
        return {
          ...mockBot,
          trades,
          metrics,
        };
      }
    }
    throw error;
  }
}

export default {
  getDashboardData,
  getAllBots,
  getBotById,
};