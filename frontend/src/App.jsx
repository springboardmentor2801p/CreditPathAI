import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import BorrowerPage from './pages/BorrowerPage'
import ProviderPage from './pages/ProviderPage'

export default function App() {
  return (
    <BrowserRouter>
      {/* Simple Navigation Bar */}
      <nav style={{ padding: '15px 20px', backgroundColor: '#333', color: '#fff', display: 'flex', gap: '20px' }}>
        <b style={{ marginRight: 'auto' }}>CreditPath AI</b>
        <Link to="/" style={{ color: '#fff', textDecoration: 'none' }}>Home</Link>
        <Link to="/borrower" style={{ color: '#fff', textDecoration: 'none' }}>Borrower View</Link>
        <Link to="/provider" style={{ color: '#fff', textDecoration: 'none' }}>Provider View</Link>
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
