import { json } from "@remix-run/node";
import { useLoaderData, useParams, Link } from "@remix-run/react";

export const loader = async ({ params }) => {
  const { botId } = params;
  
  try {
    // Fetch bot details from API
    const response = await fetch(`${process.env.BASE_URL || ''}/api/bots/${botId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch bot details: ${response.status} ${response.statusText}`);
    }
    
    const botData = await response.json();
    return json({ bot: botData, error: null });
  } catch (error) {
    console.error(`Error fetching bot ${botId}:`, error);
    return json({ bot: null, error: error.message });
  }
};

export default function BotDetail() {
  const { bot, error } = useLoaderData();
  const params = useParams();
  
  if (error) {
    return (
      <div className="card">
        <h2>Error Loading Bot #{params.botId}</h2>
        <p style={{ color: "#e53e3e" }}>{error}</p>
        <p><Link to="/bots" style={{ color: "#3182ce" }}>Return to bots list</Link></p>
      </div>
    );
  }
  
  if (!bot) {
    return <div className="card"><h2>Loading bot data...</h2></div>;
  }
  
  return (
    <div>
      <div style={{ marginBottom: "20px" }}>
        <Link 
          to="/bots" 
          style={{ 
            textDecoration: "none", 
            color: "#3182ce"
          }}
        >
          ← Back to bots list
        </Link>
      </div>
      
      <h2>Bot Details: {bot.name}</h2>
      
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", marginBottom: "20px" }}>
        {/* Bot Information Card */}
        <div className="card">
          <h3>Bot Information</h3>
          <table style={{ width: "100%" }}>
            <tbody>
              <tr>
                <td style={{ padding: "8px 0", fontWeight: "bold" }}>ID</td>
                <td style={{ padding: "8px 0" }}>{bot.bot_id}</td>
              </tr>
              <tr>
                <td style={{ padding: "8px 0", fontWeight: "bold" }}>Name</td>
                <td style={{ padding: "8px 0" }}>{bot.name}</td>
              </tr>
              <tr>
                <td style={{ padding: "8px 0", fontWeight: "bold" }}>Ticker</td>
                <td style={{ padding: "8px 0" }}>{bot.ticker}</td>
              </tr>
              <tr>
                <td style={{ padding: "8px 0", fontWeight: "bold" }}>Algorithm Type</td>
                <td style={{ padding: "8px 0" }}>{bot.algorithm_type}</td>
              </tr>
              <tr>
                <td style={{ padding: "8px 0", fontWeight: "bold" }}>Algorithm Module</td>
                <td style={{ padding: "8px 0" }}>{bot.algorithm_module}</td>
              </tr>
              <tr>
                <td style={{ padding: "8px 0", fontWeight: "bold" }}>Trade Direction</td>
                <td style={{ padding: "8px 0" }}>{bot.trade_direction}</td>
              </tr>
              <tr>
                <td style={{ padding: "8px 0", fontWeight: "bold" }}>Position Size</td>
                <td style={{ padding: "8px 0" }}>${bot.position_size.toLocaleString(undefined, {maximumFractionDigits: 2})}</td>
              </tr>
              <tr>
                <td style={{ padding: "8px 0", fontWeight: "bold" }}>Trailing Stop %</td>
                <td style={{ padding: "8px 0" }}>{(bot.trailing_stop_pct * 100).toFixed(2)}%</td>
              </tr>
              <tr>
                <td style={{ padding: "8px 0", fontWeight: "bold" }}>Status</td>
                <td style={{ padding: "8px 0" }}>
                  <span 
                    style={{ 
                      display: "inline-block", 
                      padding: "4px 8px", 
                      borderRadius: "4px", 
                      fontSize: "12px", 
                      fontWeight: "bold",
                      backgroundColor: bot.is_active ? "#d1fae5" : "#fee2e2",
                      color: bot.is_active ? "#047857" : "#b91c1c" 
                    }}
                  >
                    {bot.is_active ? "ACTIVE" : "INACTIVE"}
                  </span>
                </td>
              </tr>
              <tr>
                <td style={{ padding: "8px 0", fontWeight: "bold" }}>Version</td>
                <td style={{ padding: "8px 0" }}>{bot.version}</td>
              </tr>
              <tr>
                <td style={{ padding: "8px 0", fontWeight: "bold" }}>Created</td>
                <td style={{ padding: "8px 0" }}>{new Date(bot.created_at).toLocaleString()}</td>
              </tr>
              <tr>
                <td style={{ padding: "8px 0", fontWeight: "bold" }}>Last Updated</td>
                <td style={{ padding: "8px 0" }}>{new Date(bot.last_updated).toLocaleString()}</td>
              </tr>
            </tbody>
          </table>
        </div>
        
        {/* Bot Metrics Card */}
        <div className="card">
          <h3>Performance Metrics</h3>
          {bot.metrics ? (
            <table style={{ width: "100%" }}>
              <tbody>
                <tr>
                  <td style={{ padding: "8px 0", fontWeight: "bold" }}>Total Trades</td>
                  <td style={{ padding: "8px 0" }}>{bot.metrics.total_trades}</td>
                </tr>
                <tr>
                  <td style={{ padding: "8px 0", fontWeight: "bold" }}>Winning Trades</td>
                  <td style={{ padding: "8px 0" }}>{bot.metrics.winning_trades}</td>
                </tr>
                <tr>
                  <td style={{ padding: "8px 0", fontWeight: "bold" }}>Losing Trades</td>
                  <td style={{ padding: "8px 0" }}>{bot.metrics.losing_trades}</td>
                </tr>
                <tr>
                  <td style={{ padding: "8px 0", fontWeight: "bold" }}>Win Rate</td>
                  <td style={{ padding: "8px 0" }}>{(bot.metrics.win_rate * 100).toFixed(2)}%</td>
                </tr>
                <tr>
                  <td style={{ padding: "8px 0", fontWeight: "bold" }}>Total P&L</td>
                  <td style={{ padding: "8px 0", color: bot.metrics.total_pnl >= 0 ? "#047857" : "#b91c1c" }}>
                    ${parseFloat(bot.metrics.total_pnl).toLocaleString(undefined, {maximumFractionDigits: 2})}
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: "8px 0", fontWeight: "bold" }}>Avg. P&L Per Trade</td>
                  <td style={{ padding: "8px 0", color: bot.metrics.average_pnl_per_trade >= 0 ? "#047857" : "#b91c1c" }}>
                    ${parseFloat(bot.metrics.average_pnl_per_trade).toLocaleString(undefined, {maximumFractionDigits: 2})}
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: "8px 0", fontWeight: "bold" }}>Avg. Win Amount</td>
                  <td style={{ padding: "8px 0", color: "#047857" }}>
                    ${parseFloat(bot.metrics.average_win_amount).toLocaleString(undefined, {maximumFractionDigits: 2})}
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: "8px 0", fontWeight: "bold" }}>Avg. Loss Amount</td>
                  <td style={{ padding: "8px 0", color: "#b91c1c" }}>
                    ${parseFloat(bot.metrics.average_loss_amount).toLocaleString(undefined, {maximumFractionDigits: 2})}
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: "8px 0", fontWeight: "bold" }}>Profit Factor</td>
                  <td style={{ padding: "8px 0" }}>{parseFloat(bot.metrics.profit_factor).toFixed(2)}</td>
                </tr>
                <tr>
                  <td style={{ padding: "8px 0", fontWeight: "bold" }}>Max Drawdown</td>
                  <td style={{ padding: "8px 0", color: "#b91c1c" }}>
                    ${Math.abs(parseFloat(bot.metrics.max_drawdown)).toLocaleString(undefined, {maximumFractionDigits: 2})}
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: "8px 0", fontWeight: "bold" }}>Sharpe Ratio</td>
                  <td style={{ padding: "8px 0" }}>{parseFloat(bot.metrics.sharpe_ratio).toFixed(2)}</td>
                </tr>
                <tr>
                  <td style={{ padding: "8px 0", fontWeight: "bold" }}>Risk-Reward Ratio</td>
                  <td style={{ padding: "8px 0" }}>{parseFloat(bot.metrics.risk_reward_ratio).toFixed(2)}</td>
                </tr>
                <tr>
                  <td style={{ padding: "8px 0", fontWeight: "bold" }}>Expectancy</td>
                  <td style={{ padding: "8px 0" }}>{parseFloat(bot.metrics.expectancy).toFixed(2)}</td>
                </tr>
                <tr>
                  <td style={{ padding: "8px 0", fontWeight: "bold" }}>Rank Score</td>
                  <td style={{ padding: "8px 0" }}>{parseFloat(bot.metrics.rank_score).toFixed(2)}</td>
                </tr>
              </tbody>
            </table>
          ) : (
            <p>No metrics available for this bot</p>
          )}
        </div>
      </div>
      
      {/* Bot Trades */}
      <div className="card">
        <h3>Bot Trades</h3>
        {bot.trades && bot.trades.length > 0 ? (
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left", padding: "12px 8px", borderBottom: "1px solid #ddd" }}>ID</th>
                <th style={{ textAlign: "left", padding: "12px 8px", borderBottom: "1px solid #ddd" }}>Ticker</th>
                <th style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd" }}>Entry Price</th>
                <th style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd" }}>Exit Price</th>
                <th style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd" }}>Direction</th>
                <th style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd" }}>Entry Time</th>
                <th style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd" }}>Exit Time</th>
                <th style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd" }}>P&L</th>
                <th style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd" }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {bot.trades.map((trade) => (
                <tr key={trade.trade_id}>
                  <td style={{ padding: "8px", borderBottom: "1px solid #eee" }}>{trade.trade_id}</td>
                  <td style={{ padding: "8px", borderBottom: "1px solid #eee" }}>{trade.ticker}</td>
                  <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>${parseFloat(trade.entry_price).toFixed(2)}</td>
                  <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>{trade.exit_price ? `$${parseFloat(trade.exit_price).toFixed(2)}` : '-'}</td>
                  <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>{trade.trade_direction}</td>
                  <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>{new Date(trade.entry_time).toLocaleString()}</td>
                  <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>{trade.exit_time ? new Date(trade.exit_time).toLocaleString() : '-'}</td>
                  <td 
                    style={{ 
                      textAlign: "right", 
                      padding: "8px", 
                      borderBottom: "1px solid #eee",
                      color: trade.pnl > 0 ? "#047857" : trade.pnl < 0 ? "#b91c1c" : "inherit"
                    }}
                  >
                    {trade.pnl ? `$${parseFloat(trade.pnl).toLocaleString(undefined, {maximumFractionDigits: 2})}` : '-'}
                  </td>
                  <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>
                    <span 
                      style={{ 
                        display: "inline-block",
                        padding: "4px 8px", 
                        borderRadius: "4px", 
                        fontSize: "12px",
                        fontWeight: "bold",
                        backgroundColor: trade.trade_status === "open" ? "#e6f6ff" : "#f3f4f6",
                        color: trade.trade_status === "open" ? "#0369a1" : "#4b5563"
                      }}
                    >
                      {trade.trade_status.toUpperCase()}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>No trades available for this bot</p>
        )}
      </div>
      
      {/* Bot Description */}
      {bot.description && (
        <div className="card">
          <h3>Description</h3>
          <p>{bot.description}</p>
        </div>
      )}
    </div>
  );
}