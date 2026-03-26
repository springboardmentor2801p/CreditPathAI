import { useState } from 'react'
import PlotlyChart from '../components/PlotlyChart'
import { postRecommend, getRandomBorrower } from '../utils/api'

const DEFAULTS = {
  isJointApplication: 0,
  loanAmount: 350000,
  interestRate: 15.0,
  monthlyPayment: 9800,
  term_months: 36,
  yearsEmployment: 5,
  annualIncome: 500000,
  incomeVerified: 1,
  dtiRatio: 0.35,
  revolvingBalance: 50000,
  revolvingUtilizationRate: 0.40,
  lengthCreditHistory: 6,
  numTotalCreditLines: 8,
  numOpenCreditLines: 5,
  numOpenCreditLines1Year: 1,
  numDerogatoryRec: 0,
  numDelinquency2Years: 0,
  numChargeoff1year: 0,
  numInquiries6Mon: 1,
  grade_score: 5,
  loan_to_income_ratio: 0.70,
  payment_to_income_ratio: 0.23,
  repayment_velocity: 0.02,
  loan_amortization_rate: 0.03,
  open_credit_ratio: 0.62,
  recent_credit_velocity: 1,
  inquiry_intensity: 0.20,
  delinquency_density: 0.0,
  derogatory_density: 0.0,
  estimated_credit_limit: 150000,
  credit_utilization_recomputed: 0.33,
  log_loanAmount: 12.76,
  log_annualIncome: 13.12,
  log_revolvingBalance: 10.81,
  purpose_business: 0,
  purpose_debtconsolidation: 1,
  purpose_education: 0,
  purpose_healthcare: 0,
  purpose_homeimprovement: 0,
  purpose_other: 0,
  homeOwnership_own: 0,
  homeOwnership_rent: 1,
  threshold: 0.50,
}

export default function BorrowerPage() {
  const [form, setForm] = useState(DEFAULTS)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (key, val) => {
    setForm(prev => ({ ...prev, [key]: Number(val) }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    
    try {
      const payload = Object.fromEntries(
        Object.entries(form).map(([k, v]) => [k, v === '' ? 0 : Number(v)])
      )
      ;['term_months', 'yearsEmployment', 'lengthCreditHistory', 'numTotalCreditLines', 'numDelinquency2Years', 'numDerogatoryRec', 'numChargeoff1year', 'numInquiries6Mon'].forEach(f => {
        if (payload[f] !== undefined) payload[f] = Math.round(payload[f])
      })
      const res = await postRecommend(payload)
      setResult(res)
    } catch (err) {
      if (Array.isArray(err.detail)) {
        setError(err.detail.map(d => d.msg).join(', '))
      } else {
        setError(err.message || 'API Error. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  const loadRandom = async () => {
    try {
      const data = await getRandomBorrower()
      setForm(prev => ({ ...prev, ...data }))
    } catch (e) {
      console.warn("Could not load random data")
    }
  }

  return (
    <div>
      <h2>Borrower Loan Checker</h2>
      <p>Fill in this simple form to understand if you qualify for a loan.</p>

      <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
        {/* Simple Form */}
        <div className="card" style={{ flex: '1', minWidth: '300px' }}>
          <h3>My Details</h3>
          <form onSubmit={handleSubmit} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            {Object.keys(DEFAULTS).map(key => {
              if (key === 'threshold') return null;
              return (
                <div key={key} style={{ display: 'flex', flexDirection: 'column' }}>
                  <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '4px', textTransform: 'capitalize' }}>
                    {key.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ')}
                  </label>
                  <input
                    type="number"
                    value={form[key]}
                    onChange={e => handleChange(key, e.target.value)}
                    required
                    step="any"
                    style={{
                      padding: '8px',
                      borderRadius: '4px',
                      border: '1px solid var(--border)',
                      background: 'var(--bg-base)',
                      color: 'var(--text-primary)'
                    }}
                  />
                </div>
              );
            })}

            <div style={{ gridColumn: '1 / -1', display: 'flex', gap: '10px', marginTop: '10px' }}>
              <button type="submit" className="btn" disabled={loading} style={{ flex: 1, padding: '12px' }}>
                {loading ? 'Checking...' : 'Check My Eligibility'}
              </button>
              <button type="button" className="btn btn-secondary" onClick={loadRandom} style={{ flex: 1, padding: '12px' }}>
                Autofill Test Data
              </button>
            </div>
          </form>
          {error && <p style={{ color: 'red', marginTop: '10px' }}>{error}</p>}
        </div>

        {/* Results */}
        <div style={{ flex: '1', minWidth: '300px' }}>
          <h3>My Assessment</h3>
          {!result && <p>Results will appear here...</p>}
          
          {result && (
            <div className={`result-box ${result.predicted_default ? 'rejected' : 'approved'}`}>
              <h2 style={{ color: result.predicted_default ? 'var(--danger)' : 'var(--success)' }}>
                {result.predicted_default ? '❌ High Risk - Likely to be Denied' : '✅ Low Risk - Likely to be Approved'}
              </h2>
              <p>
                <strong>Risk Level:</strong> {result.risk_band}
              </p>
              
              <hr />

              <div className="card" style={{ margin: '20px 0', border: '1px solid var(--border)', background: 'var(--bg-card2)' }}>
                <h4 style={{ margin: '0 0 10px 0', textAlign: 'center' }}>My Risk Dashboard</h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                  <PlotlyChart
                    data={[{
                      type: 'indicator',
                      mode: 'gauge+number',
                      value: parseFloat((result.default_probability * 100).toFixed(1)),
                      gauge: {
                        axis: { range: [0, 100] },
                        bar: { color: result.predicted_default ? '#ef4444' : '#10b981' },
                        steps: [
                          { range: [0, 50], color: 'rgba(16,185,129,0.15)' },
                          { range: [50, 100], color: 'rgba(239,68,68,0.15)' }
                        ]
                      },
                      title: { text: 'Risk Score Indicator (%)', font: { size: 12 } },
                    }]}
                    layout={{ height: 180, margin: { t: 30, b: 10, l: 20, r: 20 }, paper_bgcolor: 'transparent' }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                  />
                  <PlotlyChart
                    data={[{
                      type: 'pie',
                      labels: ['New Loan EMI', 'Other Debts EMI', 'Disposable Income'],
                      values: [
                        form.monthlyPayment,
                        (form.annualIncome / 12) * form.dtiRatio - form.monthlyPayment,
                        (form.annualIncome / 12) * (1 - form.dtiRatio)
                      ].map(v => Math.max(0, v)),
                      hole: 0.5,
                      marker: { colors: ['#f59e0b', '#ef4444', '#10b981'] },
                      textinfo: 'percent',
                    }]}
                    layout={{ 
                      height: 180, 
                      margin: { t: 10, b: 10, l: 10, r: 10 }, 
                      paper_bgcolor: 'transparent',
                      showlegend: false,
                      title: { text: 'Monthly Income Usage', font: { size: 12 } }
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                  />
                </div>
              </div>

              {/* Suggestions to Borrower */}
              <h4>How to improve my chances?</h4>
              {result.predicted_default ? (
                <>
                  <p>Our recommendation engine found certain risk factors in your profile. To improve your chances, consider the following:</p>
                  <ul style={{ paddingLeft: '20px', lineHeight: '1.8' }}>
                    {/* Based on common fields */}
                    {form.loanAmount > form.annualIncome && <li><strong>Decrease Loan Amount:</strong> You requested more than you earn in a year.</li>}
                    {form.revolvingBalance > 0.4 * form.annualIncome && <li><strong>Pay down debt:</strong> Your specific revolving debt is quite high compared to income.</li>}
                    {result.risk_flags && result.risk_flags.map((flag, i) => (
                      <li key={i}>{flag} - This is flagged as a direct factor against your approval.</li>
                    ))}
                    <li><strong>Action Plan:</strong> The lender's system suggests: <i>{result.recommended_action}</i></li>
                  </ul>
                </>
              ) : (
                <p>Your profile looks great! Maintain your income and keep paying debts on time to safeguard your credit status.</p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
