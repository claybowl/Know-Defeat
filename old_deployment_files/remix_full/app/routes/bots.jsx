import { json } from "@remix-run/node";
import { useLoaderData, Link } from "@remix-run/react";
import { useState } from "react";

export const loader = async () => {
  try {
    // In a server-side environment, fetch directly
    const response = await fetch(`${process.env.BASE_URL || ''}/api/bots`);
    if (!response.ok) {
      throw new Error(`Failed to fetch bots: ${response.status} ${response.statusText}`);
    }
    const data = await response.json();
    return json({ bots: data, error: null });
  } catch (error) {
    console.error("Error fetching bots:", error);
    return json({ bots: [], error: error.message });
  }
};

export default function Bots() {
  const { bots, error } = useLoaderData();
  const [searchTerm, setSearchTerm] = useState("");
  const [sortField, setSortField] = useState("bot_id");
  const [sortDirection, setSortDirection] = useState("asc");

  if (error) {
    return (
      <div className="card">
        <h2>Error Loading Bots</h2>
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
      setSortDirection("asc");
    }
  };

  // Filter and sort bots
  const filteredBots = bots
    .filter(
      (bot) =>
        bot.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        bot.ticker.toLowerCase().includes(searchTerm.toLowerCase()) ||
        bot.algorithm_type.toLowerCase().includes(searchTerm.toLowerCase())
    )
    .sort((a, b) => {
      let valA = a[sortField];
      let valB = b[sortField];

      // Handle string comparison
      if (typeof valA === "string") {
        valA = valA.toLowerCase();
        valB = valB.toLowerCase();
      }

      if (valA < valB) return sortDirection === "asc" ? -1 : 1;
      if (valA > valB) return sortDirection === "asc" ? 1 : -1;
      return 0;
    });

  return (
    <div>
      <h2>Trading Bots</h2>
      
      <div style={{ marginBottom: "20px" }}>
        <input
          type="text"
          placeholder="Search bots..."
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
      
      <div className="card">
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th 
                style={{ textAlign: "left", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer" }}
                onClick={() => handleSort("bot_id")}
              >
                ID {sortField === "bot_id" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "left", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer" }}
                onClick={() => handleSort("name")}
              >
                Name {sortField === "name" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "left", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer" }}
                onClick={() => handleSort("ticker")}
              >
                Ticker {sortField === "ticker" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "left", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer" }}
                onClick={() => handleSort("algorithm_type")}
              >
                Algorithm {sortField === "algorithm_type" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "left", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer" }}
                onClick={() => handleSort("trade_direction")}
              >
                Direction {sortField === "trade_direction" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th 
                style={{ textAlign: "right", padding: "12px 8px", borderBottom: "1px solid #ddd", cursor: "pointer" }}
                onClick={() => handleSort("is_active")}
              >
                Status {sortField === "is_active" && (sortDirection === "asc" ? "↑" : "↓")}
              </th>
              <th style={{ textAlign: "center", padding: "12px 8px", borderBottom: "1px solid #ddd" }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredBots.map((bot) => (
              <tr key={bot.bot_id}>
                <td style={{ padding: "8px", borderBottom: "1px solid #eee" }}>{bot.bot_id}</td>
                <td style={{ padding: "8px", borderBottom: "1px solid #eee" }}>{bot.name}</td>
                <td style={{ padding: "8px", borderBottom: "1px solid #eee" }}>{bot.ticker}</td>
                <td style={{ padding: "8px", borderBottom: "1px solid #eee" }}>{bot.algorithm_type}</td>
                <td style={{ padding: "8px", borderBottom: "1px solid #eee" }}>{bot.trade_direction}</td>
                <td style={{ textAlign: "right", padding: "8px", borderBottom: "1px solid #eee" }}>
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
                <td style={{ textAlign: "center", padding: "8px", borderBottom: "1px solid #eee" }}>
                  <a 
                    href={`/bots/${bot.bot_id}`} 
                    style={{
                      textDecoration: "none",
                      padding: "4px 8px",
                      backgroundColor: "#e2e8f0",
                      color: "#2d3748",
                      borderRadius: "4px",
                      fontSize: "14px"
                    }}
                  >
                    View Details
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}