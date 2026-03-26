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
        <div className="card" style={{ width: '300px', textAlign: 'left' }}>
          <h2>👤 Person (Borrower)</h2>
          <p>
            I want to know if I can take a loan. If not, I want to know what I need to do 
            (decrease amount, increase salary, pay off debts) to get approved.
          </p>
          <br/>
          <Link to="/borrower" className="btn" style={{ width: '100%' }}>Go to Borrower View</Link>
        </div>
        
        {/* Provider Option */}
        <div className="card" style={{ width: '300px', textAlign: 'left' }}>
          <h2>🏢 Loan Provider</h2>
          <p>
            I want to check a borrower's details to see if the loan can be approved. 
            View risk metrics, agent insights, default reasons, and recovery methods.
          </p>
          <br/>
          <Link to="/provider" className="btn" style={{ width: '100%' }}>Go to Provider View</Link>
        </div>

      </div>
    </div>
  )
}
