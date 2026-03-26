import { NavLink, useLocation } from 'react-router-dom'
import './Sidebar.css'

const NAV_ITEMS = [
  { to: '/dashboard', icon: '▦', label: 'Dashboard' },
  { to: '/recommend', icon: '⚡', label: 'Predict Risk' },
  { to: '/agent',     icon: '🤖', label: 'Agent Insights' },
]

export default function Sidebar() {
  const location = useLocation()

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-icon">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <path d="M12 2L2 7l10 5 10-5-10-5z" fill="currentColor" opacity="0.9"/>
            <path d="M2 17l10 5 10-5" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round"/>
            <path d="M2 12l10 5 10-5" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" opacity="0.6"/>
          </svg>
        </div>
        <div>
          <div className="brand-name">CreditPath</div>
          <div className="brand-sub">AI</div>
        </div>
      </div>

      <div className="sidebar-divider" />

      <nav className="sidebar-nav">
        <div className="nav-section-label">Navigation</div>
        {NAV_ITEMS.map(item => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => `nav-item ${isActive ? 'nav-active' : ''}`}
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
            {location.pathname === item.to && <span className="nav-indicator" />}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="api-status">
          <span className="status-dot" id="status-dot" />
          <span className="status-text">FastAPI Live</span>
        </div>
        <div className="sidebar-version">v1.0.0</div>
      </div>
    </aside>
  )
}
