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
      // Include all bots, not just those with trades
      .sort((a, b) => {
        // First by rank score (if available)
        if (a.rank_score !== undefined && b.rank_score !== undefined) {
          return parseFloat(b.rank_score) - parseFloat(a.rank_score);
        }
        // Then by profit factor
        if (a.profit_factor !== undefined && b.profit_factor !== undefined) {
          return parseFloat(b.profit_factor) - parseFloat(a.profit_factor);
        }
        // Finally by total PNL
        return parseFloat(b.total_pnl || 0) - parseFloat(a.total_pnl || 0);
      })
      .slice(0, 10); // Show top 10 bots

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
    throw error; // Don't use mock data, throw the error
  }
}

export async function getAllBots() {
  try {
    console.log("Fetching all bots...");
    
    // Get both bots and metrics in parallel
    const [bots, metrics] = await Promise.all([
      db.getBots(),
      db.getBotMetrics()
    ]);
    
    console.log(`Retrieved ${bots.length} bots and ${metrics.length} metrics records`);
    
    // Create a lookup map of bot metrics by bot_id for faster access
    const metricsMap = {};
    metrics.forEach(metric => {
      metricsMap[metric.bot_id] = metric;
    });
    
    // Merge metrics with each bot
    const botsWithMetrics = bots.map(bot => {
      const botMetrics = metricsMap[bot.bot_id] || null;
      
      return {
        ...bot,
        metrics: botMetrics ? {
          win_rate: botMetrics.win_rate,
          profit_factor: botMetrics.profit_factor,
          total_pnl: botMetrics.total_pnl,
          sharpe_ratio: botMetrics.sharpe_ratio,
          max_drawdown: botMetrics.max_drawdown,
          rank_score: botMetrics.rank_score
        } : null
      };
    });
    
    console.log(`Merged metrics data with ${botsWithMetrics.length} bots`);
    return botsWithMetrics;
  } catch (error) {
    console.error('Error fetching bots:', error);
    throw error; // Don't use mock data, throw the error
  }
}

export async function getBotById(botId) {
  try {
    console.log(`Fetching bot with ID ${botId}...`);
    const bot = await db.getBotById(botId);
    
    if (!bot) {
      console.log(`No bot found with ID ${botId}`);
      return null;
    }
    
    // If the bot already has metrics from getBotById, use them
    if (bot.metrics) {
      console.log(`Bot ${botId} already has metrics data`);
      return bot;
    }
    
    // Otherwise, fetch metrics specifically for this bot
    const metrics = await db.getBotMetrics();
    const botMetrics = metrics.find(m => m.bot_id === bot.bot_id);
    
    if (botMetrics) {
      console.log(`Found metrics for bot ${botId}`);
      bot.metrics = {
        win_rate: botMetrics.win_rate,
        profit_factor: botMetrics.profit_factor,
        total_pnl: botMetrics.total_pnl,
        sharpe_ratio: botMetrics.sharpe_ratio,
        max_drawdown: botMetrics.max_drawdown,
        rank_score: botMetrics.rank_score,
        average_win_amount: botMetrics.average_win_amount,
        average_loss_amount: botMetrics.average_loss_amount
      };
    } else {
      console.log(`No metrics found for bot ${botId}`);
      bot.metrics = null;
    }
    
    console.log(`Successfully retrieved and enriched bot with ID ${botId}`);
    return bot;
  } catch (error) {
    console.error(`Error fetching bot ${botId}:`, error);
    throw error; // Don't use mock data, throw the error
  }
}

export default {
  getDashboardData,
  getAllBots,
  getBotById,
};