import { json } from "@remix-run/node";
import { useLoaderData } from "@remix-run/react";
import { useEffect, useState } from "react";

export const loader = async () => {
  try {
    // In a server-side environment, fetch directly
    const response = await fetch(`${process.env.BASE_URL || ''}/api/dashboard`);
    if (!response.ok) {
      throw new Error(`Failed to fetch dashboard data: ${response.status} ${response.statusText}`);
    }
    const data = await response.json();
    return json({ data, error: null });
  } catch (error) {
    console.error("Error fetching dashboard data:", error);
    return json({ data: null, error: error.message });
  }
};

export default function Index() {
  const { data, error } = useLoaderData();
  const [dashboardData, setDashboardData] = useState(data);
  
  // Client-side fetching fallback if data wasn't loaded server-side
  useEffect(() => {
    if (!dashboardData && !error) {
      fetch('/api/dashboard')
        .then(response => {
          if (!response.ok) throw new Error('Network response was not ok');
          return response.json();
        })
        .then(data => setDashboardData(data))
        .catch(error => console.error('Error fetching dashboard data:', error));
    }
  }, [dashboardData, error]);

  if (error) {
    return (
      <div className="card">
        <h2>Dashboard Error</h2>
        <p className="error">{error}</p>
        <p>The server might be experiencing issues. Please try again later.</p>
      </div>
    );
  }

  if (!dashboardData) {
    return <div className="card"><h2>Loading dashboard data...</h2></div>;
  }

  const { summary, topBots, recentTrades, openTrades } = dashboardData;

  return (
    <div>
      <h2>System Dashboard</h2>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '20px', margin: '20px 0' }}>
        <StatCard label="Total Bots" value={summary.totalBots} />
        <StatCard label="Active Bots" value={summary.activeBots} />
        <StatCard label="Open Trades" value={summary.totalOpenTrades} />
        <StatCard label="Total P&L" value={`$${summary.totalPnl.toLocaleString(undefined, {maximumFractionDigits: 2})}`} />
        <StatCard label="Avg Win Rate" value={`${(summary.avgWinRate * 100).toFixed(1)}%`} />
      </div>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        <div className="card">
          <h3>Top Performing Bots</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #ddd' }}>Bot</th>
                <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>Win Rate</th>
                <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>Total P&L</th>
              </tr>
            </thead>
            <tbody>
              {topBots.map((bot, index) => (
                <tr key={index}>
                  <td style={{ padding: '8px', borderBottom: '1px solid #eee' }}>{bot.bot_id}. {bot.name || `Bot #${bot.bot_id}`}</td>
                  <td style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee' }}>{(parseFloat(bot.win_rate) * 100).toFixed(1)}%</td>
                  <td style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee' }}>${parseFloat(bot.total_pnl).toLocaleString(undefined, {maximumFractionDigits: 2})}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        <div className="card">
          <h3>Recent Trades</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #ddd' }}>Bot</th>
                <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #ddd' }}>Ticker</th>
                <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>Direction</th>
                <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>P&L</th>
              </tr>
            </thead>
            <tbody>
              {recentTrades.slice(0, 5).map((trade, index) => (
                <tr key={index}>
                  <td style={{ padding: '8px', borderBottom: '1px solid #eee' }}>{trade.bot_name}</td>
                  <td style={{ padding: '8px', borderBottom: '1px solid #eee' }}>{trade.ticker}</td>
                  <td style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee' }}>{trade.trade_direction}</td>
                  <td style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee', color: trade.pnl > 0 ? 'green' : 'red' }}>
                    {trade.pnl ? `$${parseFloat(trade.pnl).toLocaleString(undefined, {maximumFractionDigits: 2})}` : 'Open'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      <div className="card">
        <h3>Open Trades ({openTrades.length})</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #ddd' }}>Bot</th>
              <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #ddd' }}>Ticker</th>
              <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>Entry Price</th>
              <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>Direction</th>
              <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>Entry Time</th>
            </tr>
          </thead>
          <tbody>
            {openTrades.map((trade, index) => (
              <tr key={index}>
                <td style={{ padding: '8px', borderBottom: '1px solid #eee' }}>{trade.bot_name}</td>
                <td style={{ padding: '8px', borderBottom: '1px solid #eee' }}>{trade.ticker}</td>
                <td style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee' }}>${parseFloat(trade.entry_price).toFixed(2)}</td>
                <td style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee' }}>{trade.trade_direction}</td>
                <td style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee' }}>
                  {new Date(trade.entry_time).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StatCard({ label, value }) {
  return (
    <div style={{ background: 'white', padding: '20px', borderRadius: '8px', textAlign: 'center', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
      <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#2c5282', margin: '10px 0' }}>{value}</div>
      <div style={{ fontSize: '14px', color: '#718096' }}>{label}</div>
    </div>
  );
}