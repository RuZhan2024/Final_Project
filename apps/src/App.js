// src/App.jsx
import React from "react";
import { Routes, Route, Link, Navigate } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Monitor from "./pages/Monitor";
import Events from "./pages/Events";
import Settings from "./pages/Settings"

function App() {
  return (
    <div style={{ fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}>
      {/* Simple top nav */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0.75rem 1.5rem",
          borderBottom: "1px solid #e5e7eb",
          marginBottom: "1rem",
        }}
      >
        <div style={{ fontWeight: 600 }}>Guardian Angel</div>
        <nav style={{ display: "flex", gap: "1rem" }}>
          
          <Link to="/dashboard">Dashboard</Link>
          <Link to="/monitor">Monitor</Link>
          <Link to="/events">Events</Link>
          <Link to="/settings">Settings</Link>
        </nav>
      </header>

      {/* Page content */}
      <Routes>
        {/* Redirect / → /monitor */}
        <Route path="/" element={<Navigate to="/monitor" replace />} />

        <Route path="/Dashboard" element={<Dashboard />} />
        <Route path="/monitor" element={<Monitor />} />
        <Route path="/events" element={<Events />} />
        <Route path="/settings" element={<Settings />} />

        {/* Fallback 404 */}
        <Route
          path="*"
          element={
            <div style={{ padding: "1.5rem" }}>
              <h1>404 – Page not found</h1>
            </div>
          }
        />
      </Routes>
    </div>
  );
}

export default App;
