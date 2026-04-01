import { Link } from 'react-router-dom'
import './LandingPage.css' // Just clear it out if needed, but we rely on index.css

export default function LandingPage() {
  return (
    <div style={{ textAlign: 'center', marginTop: '50px' }}>
      <h1>Welcome to CreditPath AI.</h1>
      <p style={{ fontSize: '18px', color: '#666', marginBottom: '40px' }}>
        A simple system to evaluate credit risk and provide recommendations.
      </p>

      <h3>Who are you? Please choose your view:</h3>
      
      <div style={{ display: 'flex', justifyContent: 'center', gap: '30px', marginTop: '30px' }}>
        
        {/* Borrower Option */}
        <div className="card" style={{ width: '300px', textAlign: 'center', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', padding: '30px' }}>
          <div>
            <h2 style={{ marginBottom: '15px' }}>👤 Borrower</h2>
            <p style={{ color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              Check your loan eligibility and get personalized advice.
            </p>
          </div>
          <div style={{ marginTop: '30px' }}>
            <Link to="/borrower" className="btn" style={{ width: '100%', padding: '12px' }}>Enter as Borrower</Link>
          </div>
        </div>
        
        {/* Provider Option */}
        <div className="card" style={{ width: '300px', textAlign: 'center', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', padding: '30px' }}>
          <div>
            <h2 style={{ marginBottom: '15px' }}>🏢 Loan Provider</h2>
            <p style={{ color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              Evaluate borrower applications and view risk metrics.
            </p>
          </div>
          <div style={{ marginTop: '30px' }}>
            <Link to="/provider" className="btn" style={{ width: '100%', padding: '12px' }}>Enter as Provider</Link>
          </div>
        </div>

      </div>
    </div>
  )
}
