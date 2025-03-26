import db from './db.server';

export async function getDashboardData() {
  try {
    // Run queries in parallel for better performance
    const [bots, openTrades, metrics] = await Promise.all([
      db.getBots(),
      db.getOpenTrades(),
      db.getBotMetrics(),
    ]);

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
      .sort((a, b) => parseFloat(b.rank_score) - parseFloat(a.rank_score))
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
    throw error;
  }
}

export async function getAllBots() {
  try {
    const bots = await db.getBots();
    return bots;
  } catch (error) {
    console.error('Error fetching bots:', error);
    throw error;
  }
}

export async function getBotById(botId) {
  try {
    return await db.getBotById(botId);
  } catch (error) {
    console.error(`Error fetching bot ${botId}:`, error);
    throw error;
  }
}

export default {
  getDashboardData,
  getAllBots,
  getBotById,
};