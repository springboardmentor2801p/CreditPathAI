import { NavLink } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { checkHealth } from '../api/creditApi';

export default function Navbar() {
  const [online, setOnline] = useState(null);

  useEffect(() => {
    checkHealth()
      .then(() => setOnline(true))
      .catch(() => setOnline(false));
  }, []);

  return (
    <nav className="navbar">
      <NavLink to="/" className="navbar-brand">
        <div className="navbar-logo">⚡</div>
        CreditPath<span>AI</span>
      </NavLink>

      <div className="navbar-links">
        <NavLink
          to="/"
          end
          className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}
        >
          Dashboard
        </NavLink>
        <NavLink
          to="/predict"
          className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}
        >
          Predict Risk
        </NavLink>
      </div>

      <div className="navbar-status">
        <div
          className="status-dot"
          style={{ background: online === null ? '#94a3b8' : online ? '#22c55e' : '#ef4444' }}
        />
        <span className="status-text">
          {online === null ? 'Checking…' : online ? 'API Online' : 'API Offline'}
        </span>
      </div>
    </nav>
  );
}
