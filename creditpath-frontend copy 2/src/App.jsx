import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import './index.css'
import PredictPage from './pages/PredictPage'
import Dashboard from './pages/Dashboard'
import AgentPage from './pages/AgentPage'

function Navbar({ theme, toggleTheme }) {
  const location = useLocation()
  const links = [
    { to: '/', label: 'Home' },
    { to: '/dashboard', label: 'Dashboard' },
    { to: '/predict', label: 'Predict Risk' },
    { to: '/agent', label: 'Agent Dashboard' },
  ]
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <div className="brand-icon">⬡</div>
        <span className="brand-name">CreditPath <span className="brand-ai">AI</span></span>
      </div>
      <div className="nav-links">
        {links.map(link => (
          <Link
            key={link.to}
            to={link.to}
            className={`nav-link ${location.pathname === link.to ? 'active' : ''}`}
          >
            {link.label}
          </Link>
        ))}
      </div>
      {/* Theme Toggle */}
      <button onClick={toggleTheme} style={{
        background: 'var(--bg3)',
        border: '1px solid var(--border)',
        borderRadius: '20px',
        padding: '0.35rem 1rem',
        color: 'var(--text)',
        cursor: 'pointer',
        fontSize: '0.8rem',
        display: 'flex',
        alignItems: 'center',
        gap: '0.4rem',
        transition: 'all 0.2s'
      }}>
        {theme === 'dark' ? '☀ Light' : '🌙 Dark'}
      </button>
      <div className="nav-badge">ML Powered</div>
    </nav>
  )
}

function HomePage() {
  return (
    <div className="home-page">
      <div className="hero-grid">
        <div className="hero-content">
          <div className="hero-tag">Loan Recovery Intelligence</div>
          <h1 className="hero-title">
            Predict.<br />
            <span className="hero-accent">Recover.</span><br />
            Optimize.
          </h1>
          <p className="hero-desc">
            ML-powered platform that predicts borrower default risk and recommends
            personalized recovery actions in real time.
          </p>
          <div className="hero-actions">
            <Link to="/predict" className="btn-primary">Start Predicting →</Link>
            <Link to="/dashboard" className="btn-secondary">View Dashboard</Link>
          </div>
        </div>
        <div className="hero-stats">
          {[
            { value: '94.2%', label: 'Model Accuracy', color: '#00ff88' },
            { value: '3x', label: 'Recovery Rate', color: '#00cfff' },
            { value: '< 1s', label: 'Prediction Time', color: '#ff6b6b' },
            { value: '5', label: 'Risk Features', color: '#ffd700' },
          ].map((stat, i) => (
            <div key={i} className="stat-card" style={{ '--accent': stat.color }}>
              <div className="stat-value" style={{ color: stat.color }}>{stat.value}</div>
              <div className="stat-label">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="features-section">
        <h2 className="section-title">How It Works</h2>
        <div className="features-grid">
          {[
            { icon: '⬡', title: 'Input Borrower Data', desc: 'Enter credit score, loan amount, income, LTV, and debt ratio.' },
            { icon: '◈', title: 'ML Risk Analysis', desc: 'XGBoost & Logistic Regression models compute default probability.' },
            { icon: '◎', title: 'Recovery Action', desc: 'Get personalized recovery strategy, team assignment, and priority level.' },
          ].map((f, i) => (
            <div key={i} className="feature-card">
              <div className="feature-icon">{f.icon}</div>
              <h3 className="feature-title">{f.title}</h3>
              <p className="feature-desc">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function App() {
  const [theme, setTheme] = useState('dark')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark')
  }

  return (
    <Router>
      <div className="app-shell">
        <Navbar theme={theme} toggleTheme={toggleTheme} />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/predict" element={<PredictPage />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/agent" element={<AgentPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App