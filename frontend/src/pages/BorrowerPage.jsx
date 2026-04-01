import { useState } from 'react'
import PlotlyChart from '../components/PlotlyChart'
import { postRecommend, getRandomBorrower } from '../utils/api'
import { BLANK_FORM, computePayload } from '../utils/defaults'

export default function BorrowerPage() {
  const [form, setForm] = useState(BLANK_FORM)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const handleChange = (key, val) => {
    setForm(prev => ({ ...prev, [key]: val === '' ? '' : Number(val) }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    
    try {
      const payload = computePayload(form)
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
    } catch (err) {
      console.warn("Could not load random data", err)
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
            {Object.keys(BLANK_FORM).map(key => {
              if (key === 'threshold') return null;
              const isBasic = ['loanAmount', 'annualIncome', 'monthlyPayment', 'interestRate', 'term_months', 'yearsEmployment'].includes(key);
              if (!showAdvanced && !isBasic) return null;
              
              const dynamicFallback = computePayload(form)[key];

              return (
                <div key={key} style={{ display: 'flex', flexDirection: 'column' }}>
                  <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '4px', textTransform: 'capitalize' }}>
                    {key.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ')}
                  </label>
                  <input
                    type="number"
                    value={form[key]}
                    onChange={e => handleChange(key, e.target.value)}
                    required={isBasic}
                    placeholder={`e.g. ${dynamicFallback}`}
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

            <div style={{ gridColumn: '1 / -1', display: 'flex', justifyContent: 'center', marginTop: '10px' }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setShowAdvanced(!showAdvanced)}
                style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 16px', background: 'transparent', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              >
                {showAdvanced ? 'Hide Advanced Fields ▲' : 'Show Advanced Fields ▼'}
              </button>
            </div>

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
                        computePayload(form).monthlyPayment,
                        (computePayload(form).annualIncome / 12) * computePayload(form).dtiRatio - computePayload(form).monthlyPayment,
                        (computePayload(form).annualIncome / 12) * (1 - computePayload(form).dtiRatio)
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
                    {computePayload(form).loanAmount > computePayload(form).annualIncome && <li><strong>Decrease Loan Amount:</strong> You requested more than you earn in a year.</li>}
                    {computePayload(form).revolvingBalance > 0.4 * computePayload(form).annualIncome && <li><strong>Pay down debt:</strong> Your specific revolving debt is quite high compared to income.</li>}
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
