import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import BorrowerPage from './pages/BorrowerPage'
import ProviderPage from './pages/ProviderPage'

export default function App() {
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'light')

  useEffect(() => {
    localStorage.setItem('theme', theme)
    if (theme === 'dark') {
      document.body.classList.add('dark-mode')
    } else {
      document.body.classList.remove('dark-mode')
    }
  }, [theme])

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light')
  }

  return (
    <BrowserRouter>
      {/* Navigation Bar uses index.css 'nav' tag */}
      <nav>
        <b>CreditPath AI</b>
        <Link to="/">Home</Link>
        <Link to="/borrower">Borrower View</Link>
        <Link to="/provider">Provider View</Link>
        <button 
          onClick={toggleTheme} 
          className="btn btn-secondary" 
          style={{ marginLeft: 'auto', padding: '6px 12px', borderRadius: '20px', background: 'transparent' }}
        >
          {theme === 'light' ? '🌙 Dark Mode' : '☀️ Light Mode'}
        </button>
      </nav>

      {/* Main Content Area */}
      <div className="container" style={{ marginTop: '20px' }}>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/borrower" element={<BorrowerPage />} />
          <Route path="/provider" element={<ProviderPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
