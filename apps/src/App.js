
import React from "react";
import { Routes, Route, Link, Navigate, NavLink } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Monitor from "./pages/Monitor";
import Events from "./pages/Events";
import Settings from "./pages/Settings";
import Monitor_demo from "./pages/Monitor_demo";



import styles from "./App.module.css"; 

function App() {
  return (
    <div className={styles.container}>
      
      <aside className={styles.sideNav}>
        <div className={styles.logoContainer}>
          <img 
          src="/logo_dark.png" 
          alt="Guardian Angel Logo" 
          className={styles.logoImage}
      />
        </div>
       <nav className={styles.sideNavList}>
  <NavLink 
    to="/dashboard" 
    className={({ isActive }) => isActive ? styles.activeLink : undefined}
  >
    Dashboard
  </NavLink>

  <NavLink 
    to="/monitor" 
    className={({ isActive }) => isActive ? styles.activeLink : undefined}
  >
    Live Monitor
  </NavLink>

  <NavLink 
    to="/events" 
    className={({ isActive }) => isActive ? styles.activeLink : undefined}
  >
    Event History
  </NavLink>

  <NavLink 
    to="/settings" 
    className={({ isActive }) => isActive ? styles.activeLink : undefined}
  >
    Settings
  </NavLink>
</nav>
      </aside>

      <main className={styles.mainContent}>
        <Routes>
          <Route path="/" element={<Navigate to="/monitor" replace />} />
          <Route path="/Dashboard" element={<Dashboard />} />
          <Route path="/monitor" element={<Monitor />} />
          <Route path="/monitor-demo" element={<Monitor_demo />} />
          <Route path="/events" element={<Events />} />
          <Route path="/settings" element={<Settings />} />
          <Route
            path="*"
            element={
              <div style={{ padding: "1.5rem" }}>
                <h1>404 – Page not found</h1>
              </div>
            }
          />
        </Routes>
      </main>
      
    </div>
  );
}

export default App;