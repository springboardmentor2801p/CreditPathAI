import React, { useState, useEffect, useContext } from "react";
import { Link, useLocation } from "react-router-dom";
import { AuthContext } from "../context/AuthContext";

function Navbar() {
  const { user, logout } = useContext(AuthContext);
  const location = useLocation();
  const [scrolled, setScrolled] = useState(false);
  const [predictOpen, setPredictOpen] = useState(false);
  const [isNotifOpen, setIsNotifOpen] = useState(false);
  const [notifCount, setNotifCount] = useState(3);
  const [notifications] = useState([
    { id: 1, text: "🎉 Welcome to CreditPath AI!", time: "Just now" },
    { id: 2, text: "📊 New prediction model deployed.", time: "2h ago" },
    { id: 3, text: "💡 Tip: Check out the Scenario Simulator.", time: "5h ago" }
  ]);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const isActive = (path) => location.pathname === path;
  const predictActive = location.pathname.startsWith("/predict");

  return (
    <nav className="navbar" style={scrolled ? { background: "rgba(255, 255, 255, 0.95)", boxShadow: "0 4px 20px rgba(0,0,0,0.08)" } : {}}>

      {/* Logo */}
      <Link to="/" className="navbar-logo">
        <div className="navbar-logo-icon">💳</div>
        <span className="gradient-text">CreditPath AI</span>
      </Link>

      {/* Links */}
      <div className="navbar-links">
        <Link to="/" className={`nav-link ${isActive("/") ? "active" : ""}`}>Home</Link>
        <Link to="/about" className={`nav-link ${isActive("/about") ? "active" : ""}`}>About</Link>

        {/* Predict Dropdown */}
        <div style={{ position: "relative" }}>
          <span
            className={`nav-link ${predictActive ? "active" : ""}`}
            onClick={() => setPredictOpen(!predictOpen)}
            style={{ cursor: "pointer" }}
          >
            Predict ▾
          </span>

          {predictOpen && (
            <div className="nav-dropdown">
              <Link to="/predict/user" className="nav-dropdown-item" onClick={() => setPredictOpen(false)}>
                👤 User Prediction
              </Link>
              <Link to="/predict/bank" className="nav-dropdown-item bank" onClick={() => setPredictOpen(false)}>
                🏦 Bank Prediction
              </Link>
            </div>
          )}
        </div>

        <Link to="/contact" className={`nav-link ${isActive("/contact") ? "active" : ""}`}>Contact</Link>

        {user ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            {/* Notifications */}
            <div style={{ position: 'relative' }}>
              <button
                onClick={() => {
                  setIsNotifOpen(!isNotifOpen);
                  if (!isNotifOpen) {
                    setNotifCount(0);
                    localStorage.setItem('creditpath_notif_cleared', 'true');
                  }
                }}
                style={{ background: 'transparent', border: 'none', cursor: 'pointer', fontSize: '1.2rem', position: 'relative', display: 'flex', alignItems: 'center' }}
              >
                🔔
                {notifCount > 0 && !localStorage.getItem('creditpath_notif_cleared') && (
                  <span style={{ position: 'absolute', top: -4, right: -4, background: '#ef4444', color: 'white', fontSize: '10px', width: 14, height: 14, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    {notifCount}
                  </span>
                )}
              </button>

              {isNotifOpen && (
                <div style={{
                  position: 'absolute', top: 40, right: 0, width: 260, background: 'var(--bg-card)', border: '1px solid var(--border)',
                  borderRadius: 12, boxShadow: 'var(--shadow-lg)', zIndex: 1001, padding: '10px 0', overflow: 'hidden'
                }}>
                  <div style={{ padding: '8px 16px', fontWeight: 700, borderBottom: '1px solid var(--border)', marginBottom: 8, fontSize: '0.9rem' }}>Notifications</div>
                  {notifications.map(n => (
                    <div key={n.id} style={{ padding: '10px 16px', fontSize: '0.8rem', borderBottom: '1px solid var(--bg-secondary)', cursor: 'default' }}>
                      <div style={{ marginBottom: 4, fontWeight: 500 }}>{n.text}</div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{n.time}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <Link to="/dashboard" className={`nav-link ${isActive("/dashboard") ? "active" : ""}`} style={{ color: 'var(--accent)', fontWeight: 600 }}>
              {user.role === 'Admin' ? '🛠️ Admin' : `👤 ${user.name.split(' ')[0]}`}
            </Link>
            <button
              onClick={logout}
              className="nav-link nav-cta"
              style={{ padding: '6px 12px', background: 'transparent', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
            >
              Logout
            </button>
          </div>
        ) : (
          <Link to="/auth" className={`nav-link nav-cta ${isActive("/auth") ? "active" : ""}`}>
            Get Started 🚀
          </Link>
        )}
      </div>
    </nav>
  );
}

export default Navbar;