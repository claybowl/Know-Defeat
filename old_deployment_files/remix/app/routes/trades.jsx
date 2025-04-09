import { json } from "@remix-run/node";
import { useLoaderData, Link } from "@remix-run/react";
import { useState } from "react";

export const loader = async () => {
  try {
    // Get both all trades and open trades
    const [tradesResponse, openTradesResponse] = await Promise.all([
      fetch(`${process.env.BASE_URL || ''}/api/trades`),
      fetch(`${process.env.BASE_URL || ''}/api/trades/open`)
    ]);
    
    if (!tradesResponse.ok) {
      throw new Error(`Failed to fetch trades: ${tradesResponse.status} ${tradesResponse.statusText}`);
    }
    
    if (!openTradesResponse.ok) {
      throw new Error(`Failed to fetch open trades: ${openTradesResponse.status} ${openTradesResponse.statusText}`);
    }
    
    const trades = await tradesResponse.json();
    const openTrades = await openTradesResponse.json();
    
    return json({ 
      trades, 
      openTrades,
      error: null 
    });
  } catch (error) {
    console.error("Error fetching trades:", error);
    return json({ 
      trades: [], 
      openTrades: [],
      error: error.message 
    });
  }
};

export default function Trades() {
  const { trades, openTrades, error } = useLoaderData();
  const [activeTab, setActiveTab] = useState("open");
  const [searchTerm, setSearchTerm] = useState("");
  const [sortField, setSortField] = useState("entry_time");
  const [sortDirection, setSortDirection] = useState("desc");
  
  if (error) {
    return (
      <div className="card">
        <h2>Error Loading Trades</h2>
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
      setSortDirection("desc");
    }
  };
  
  // Filter and sort trades
  const currentTrades = activeTab === "open" ? openTrades : trades;
  const filteredTrades = currentTrades
    .filter(
      (trade) =>
        trade.ticker.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (trade.bot_name && trade.bot_name.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (trade.algorithm_type && trade.algorithm_type.toLowerCase().includes(searchTerm.toLowerCase()))
    )
    .sort((a, b) => {
      let valA = a[sortField];
      let valB = b[sortField];
      
      // Handle string comparison
      if (typeof valA === "string") {
        valA = valA.toLowerCase();
        valB = valB.toLowerCase();
      }
      
      // Handle date fields
      if (sortField === "entry_time" || sortField === "exit_time") {
        valA = new Date(valA || 0).getTime();
        valB = new Date(valB || 0).getTime();
      }
      
      if (valA < valB) return sortDirection === "asc" ? -1 : 1;
      if (valA > valB) return sortDirection === "asc" ? 1 : -1;
      return 0;
    });
  
  return (
    <div>
      <h2>Trades</h2>
      
      {/* Tab Navigation */}
      <div style={{ marginBottom: "20px", borderBottom: "1px solid #e2e8f0" }}>
        <button
          onClick={() => setActiveTab("open")}
          style={{
            padding: "8px 16px",
            border: "none",
            background: "none",
            borderBottom: activeTab === "open" ? "2px solid #4299e1" : "none",
            marginRight: "20px",
            cursor: "pointer",
            fontWeight: activeTab === "open" ? "bold" : "normal",
            color: activeTab === "open" ? "#2c5282" : "#4a5568"
          }}
        >
          Open Trades ({openTrades.length})
        </button>
        <button
          onClick={() => setActiveTab("all")}
          style={{
            padding: "8px 16px",
            border: "none",
            background: "none",
            borderBottom: activeTab === "all" ? "2px solid #4299e1" : "none",
            cursor: "pointer",
            fontWeight: activeTab === "all" ? "bold" : "normal",
            color: activeTab === "all" ? "#2c5282" : "#4a5568"
          }}
        >
          All Trades ({trades.length})
        </button>
      </div>
      
      {/* Search Box */}
      <div style={{ marginBottom: "20px" }}>
        <input
          type="text"
          placeholder="Search trades..."
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
      
      {/* Trades Table */}
      <div className="card">
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th 
                style={{ textAlign: "left", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer" }}
                onClick={() => handleSort("trade_id")}
              >
                ID {sortField === "trade_id" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "left", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer" }}
                onClick={() => handleSort("bot_name")}
              >
                Bot {sortField === "bot_name" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "left", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer" }}
                onClick={() => handleSort("ticker")}
              >
                Ticker {sortField === "ticker" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer" }}
                onClick={() => handleSort("entry_price")}
              >
                Entry Price {sortField === "entry_price" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer" }}
                onClick={() => handleSort("exit_price")}
              >
                Exit Price {sortField === "exit_price" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer" }}
                onClick={() => handleSort("trade_direction")}
              >
                Direction {sortField === "trade_direction" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer" }}
                onClick={() => handleSort("entry_time")}
              >
                Entry Time {sortField === "entry_time" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              {activeTab === "all" && (
                <th 
                  style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer" }}
                  onClick={() => handleSort("exit_time")}
                >
                  Exit Time {sortField === "exit_time" && (sortDirection === "asc" ? "↑" : "↓")}
                </th>
              )}
              <th 
                style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer" }}
                onClick={() => handleSort("pnl")}
              >
                P&L {sortField === "pnl" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              {activeTab === "all" && (
                <th 
                  style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer" }}
                  onClick={() => handleSort("trade_status")}
                >
                  Status {sortField === "trade_status" && (sortDirection === "asc" ? "↑" : "↓")}
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {filteredTrades.length > 0 ? (
              filteredTrades.map((trade) => (
                <tr key={trade.trade_id}>
                  <td style={{ padding: "8px", borderBottom: "1px solid #eee" }}>{trade.trade_id}</td>
                  <td style={{ padding: "8px", borderBottom: "1px solid #eee" }}>
                    <Link 
                      to={`/bots/${trade.bot_id}`}
                      style={{ textDecoration: "none", color: "#3182ce" }}
                    >
                      {trade.bot_name || `Bot #${trade.bot_id}`}
                    </Link>
                  </td>
                  <td style={{ padding: "8px", borderBottom: "1px solid #eee" }}>{trade.ticker}</td>
                  <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>
                    ${parseFloat(trade.entry_price).toFixed(2)}
                  </td>
                  <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>
                    {trade.exit_price ? `$${parseFloat(trade.exit_price).toFixed(2)}` : '-'}
                  </td>
                  <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>
                    {trade.trade_direction}
                  </td>
                  <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>
                    {new Date(trade.entry_time).toLocaleString()}
                  </td>
                  {activeTab === "all" && (
                    <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>
                      {trade.exit_time ? new Date(trade.exit_time).toLocaleString() : '-'}
                    </td>
                  )}
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
                  {activeTab === "all" && (
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
                  )}
                </tr>
              ))
            ) : (
              <tr>
                <td 
                  colSpan={activeTab === "all" ? 9 : 7} 
                  style={{ padding: "20px", textAlign: "center", borderBottom: "1px solid #eee" }}
                >
                  No trades found matching your search criteria
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}