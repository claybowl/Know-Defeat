import { json } from "@remix-run/node";
import { useLoaderData } from "@remix-run/react";
import { useState } from "react";

export const loader = async () => {
  try {
    // In a server-side environment, fetch directly
    const response = await fetch(`${process.env.BASE_URL || ''}/api/metrics`);
    if (!response.ok) {
      throw new Error(`Failed to fetch metrics: ${response.status} ${response.statusText}`);
    }
    const data = await response.json();
    return json({ metrics: data, error: null });
  } catch (error) {
    console.error("Error fetching metrics:", error);
    return json({ metrics: [], error: error.message });
  }
};

export default function Metrics() {
  const { metrics, error } = useLoaderData();
  const [searchTerm, setSearchTerm] = useState("");
  const [sortField, setSortField] = useState("rank_score");
  const [sortDirection, setSortDirection] = useState("desc");

  if (error) {
    return (
      <div className="card">
        <h2>Error Loading Metrics</h2>
        <p className="error">{error}</p>
      </div>
    );
  }

  // Handle sorting
  const handleSort = (field) => {
    if (field === sortField) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection(field === "bot_id" ? "asc" : "desc");
    }
  };

  // Filter and sort metrics
  const filteredMetrics = metrics
    .filter((botMetric) => {
      // If there's no search term, include all metrics
      if (!searchTerm) return true;
      
      // Otherwise, filter by bot ID
      return botMetric.bot_id.toString().includes(searchTerm);
    })
    .sort((a, b) => {
      let valA = a[sortField];
      let valB = b[sortField];

      // Handle numerical comparison
      if (typeof valA === "string" && !isNaN(valA)) {
        valA = parseFloat(valA);
        valB = parseFloat(valB);
      }

      if (valA < valB) return sortDirection === "asc" ? -1 : 1;
      if (valA > valB) return sortDirection === "asc" ? 1 : -1;
      return 0;
    });

  return (
    <div>
      <h2>Bot Performance Metrics</h2>
      
      <div style={{ marginBottom: "20px" }}>
        <input
          type="text"
          placeholder="Search by bot ID..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          style={{
            padding: "8px 12px",
            borderRadius: "4px",
            border: "1px solid #ddd",
            width: "300px",
            fontSize: "16px"
          }}
        />
      </div>
      
      <div className="card" style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "1000px" }}>
          <thead>
            <tr>
              <th 
                style={{ textAlign: "left", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer", position: "sticky", top: 0, background: "white" }}
                onClick={() => handleSort("bot_id")}
              >
                Bot ID {sortField === "bot_id" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer", position: "sticky", top: 0, background: "white" }}
                onClick={() => handleSort("rank_score")}
              >
                Rank Score {sortField === "rank_score" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer", position: "sticky", top: 0, background: "white" }}
                onClick={() => handleSort("total_trades")}
              >
                Total Trades {sortField === "total_trades" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer", position: "sticky", top: 0, background: "white" }}
                onClick={() => handleSort("win_rate")}
              >
                Win Rate {sortField === "win_rate" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer", position: "sticky", top: 0, background: "white" }}
                onClick={() => handleSort("total_pnl")}
              >
                Total P&L {sortField === "total_pnl" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer", position: "sticky", top: 0, background: "white" }}
                onClick={() => handleSort("average_pnl_per_trade")}
              >
                Avg P&L/Trade {sortField === "average_pnl_per_trade" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer", position: "sticky", top: 0, background: "white" }}
                onClick={() => handleSort("profit_factor")}
              >
                Profit Factor {sortField === "profit_factor" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer", position: "sticky", top: 0, background: "white" }}
                onClick={() => handleSort("sharpe_ratio")}
              >
                Sharpe Ratio {sortField === "sharpe_ratio" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer", position: "sticky", top: 0, background: "white" }}
                onClick={() => handleSort("max_drawdown")}
              >
                Max Drawdown {sortField === "max_drawdown" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer", position: "sticky", top: 0, background: "white" }}
                onClick={() => handleSort("last_updated")}
              >
                Last Updated {sortField === "last_updated" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
            </tr>
          </thead>
          <tbody>
            {filteredMetrics.map((metric) => (
              <tr key={metric.bot_id}>
                <td style={{ padding: "8px", borderBottom: "1px solid #eee" }}>
                  <a 
                    href={`/bots/${metric.bot_id}`} 
                    style={{ textDecoration: "none", color: "#3182ce" }}
                  >
                    Bot #{metric.bot_id}
                  </a>
                </td>
                <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>
                  {parseFloat(metric.rank_score).toFixed(2)}
                </td>
                <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>
                  {metric.total_trades}
                </td>
                <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>
                  {(parseFloat(metric.win_rate) * 100).toFixed(2)}%
                </td>
                <td style={{ 
                  textAlign: "right", 
                  padding: "8px", 
                  borderBottom: "1px solid #eee",
                  color: parseFloat(metric.total_pnl) >= 0 ? "#047857" : "#b91c1c"
                }}>
                  ${parseFloat(metric.total_pnl).toLocaleString(undefined, {maximumFractionDigits: 2})}
                </td>
                <td style={{ 
                  textAlign: "right", 
                  padding: "8px", 
                  borderBottom: "1px solid #eee",
                  color: parseFloat(metric.average_pnl_per_trade) >= 0 ? "#047857" : "#b91c1c"
                }}>
                  ${parseFloat(metric.average_pnl_per_trade).toLocaleString(undefined, {maximumFractionDigits: 2})}
                </td>
                <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>
                  {parseFloat(metric.profit_factor).toFixed(2)}
                </td>
                <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>
                  {parseFloat(metric.sharpe_ratio).toFixed(2)}
                </td>
                <td style={{ 
                  textAlign: "right", 
                  padding: "8px", 
                  borderBottom: "1px solid #eee",
                  color: "#b91c1c"
                }}>
                  ${Math.abs(parseFloat(metric.max_drawdown)).toLocaleString(undefined, {maximumFractionDigits: 2})}
                </td>
                <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>
                  {new Date(metric.last_updated).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}