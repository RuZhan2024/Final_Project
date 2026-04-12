import React, { useEffect, useState } from "react";
import {Routes, Route, Navigate, NavLink, useLocation} from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Monitor from "./pages/Monitor";
import Events from "./pages/Events";
import Settings from "./pages/Settings";

import { MonitoringProvider } from "./monitoring/MonitoringContext";

import styles from "./App.module.css";

function App() {
  const location = useLocation();
  const showMonitor = location.pathname === "/monitor";
  const [navOpen, setNavOpen] = useState(false);

  useEffect(() => {
    setNavOpen(false);
  }, [location.pathname]);

  const renderNavLinks = () => (
    <>
      <NavLink
        to="/dashboard"
        className={({ isActive }) =>
          isActive ? styles.activeLink : undefined
        }
      >
        Dashboard
      </NavLink>

      <NavLink
        to="/monitor"
        className={({ isActive }) =>
          isActive ? styles.activeLink : undefined
        }
      >
        Live Monitor
      </NavLink>

      <NavLink
        to="/events"
        className={({ isActive }) =>
          isActive ? styles.activeLink : undefined
        }
      >
        Event History
      </NavLink>

      <NavLink
        to="/settings"
        className={({ isActive }) =>
          isActive ? styles.activeLink : undefined
        }
      >
        Settings
      </NavLink>
    </>
  );

  return (
    <MonitoringProvider>
      <div className={styles.container}>
        <header className={styles.mobileTopBar}>
          <button
            type="button"
            className={styles.menuButton}
            onClick={() => setNavOpen(true)}
            aria-label="Open navigation"
            aria-expanded={navOpen}
          >
            <span></span>
            <span></span>
            <span></span>
          </button>
          <img
            src="/logo_dark.png"
            alt="Guardian Angel Logo"
            className={styles.mobileLogo}
          />
        </header>

        <div
          className={`${styles.mobileBackdrop} ${navOpen ? styles.mobileBackdropVisible : ""}`}
          onClick={() => setNavOpen(false)}
          aria-hidden={navOpen ? "false" : "true"}
        />

        <aside className={`${styles.sideNav} ${navOpen ? styles.sideNavOpen : ""}`}>
          <div className={styles.logoContainer}>
            <img
              src="/logo_dark.png"
              alt="Guardian Angel Logo"
              className={styles.logoImage}
            />
          </div>
          <nav className={styles.sideNavList}>
            <button
              type="button"
              className={styles.closeNavButton}
              onClick={() => setNavOpen(false)}
              aria-label="Close navigation"
            >
              Close
            </button>
            {renderNavLinks()}
          </nav>
        </aside>

        <main className={styles.mainContent}>
          {/*
            IMPORTANT: Keep the live monitoring pipeline mounted at the App level.
            This prevents route changes from stopping the camera / pose loop.
          */}
          <div style={{ display: showMonitor ? "block" : "none" }}>
            <Monitor isActive={showMonitor} />
          </div>

          <Routes>
            <Route path="/" element={<Navigate to="/monitor" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/monitor" element={<></>} />
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
    </MonitoringProvider>
  );
}

export default App;
