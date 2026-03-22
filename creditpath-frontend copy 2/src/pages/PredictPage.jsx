import { useState } from 'react'
import { predictRisk } from '../services/api'

export default function PredictPage() {
  const [form, setForm] = useState({
    Credit_Score: '',
    loan_amount: '',
    income: '',
    LTV: '',
    dtir1: ''
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [submitted, setSubmitted] = useState(null)

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: parseFloat(e.target.value) })
  }

  const handleSubmit = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await predictRisk(form)
      setResult(data)
      setSubmitted({ ...form })
    } catch (err) {
      setError('Connection failed. Make sure the backend is running on port 8000.')
    }
    setLoading(false)
  }

  const getPriorityClass = (p) => {
    if (p === 'High' || p === 'Critical') return 'priority-high'
    if (p === 'Medium') return 'priority-medium'
    return 'priority-low'
  }

  const getRiskColor = (prob) => {
    if (prob > 0.6) return '#ff4d6d'
    if (prob > 0.3) return '#ffd700'
    return '#00ff88'
  }

  const getRiskLevel = (prob) => {
    if (prob < 0.20) return { label: 'Very Low Risk', color: '#00ff88' }
    if (prob < 0.40) return { label: 'Low Risk', color: '#7CFF6B' }
    if (prob < 0.60) return { label: 'Moderate Risk', color: '#ffd700' }
    if (prob < 0.80) return { label: 'High Risk', color: '#FF8C42' }
    return { label: 'Critical Risk', color: '#ff4d6d' }
  }

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Borrower Risk Predictor</h1>
        <p className="page-subtitle">Enter borrower details to get ML-powered default risk assessment</p>
      </div>

      <div className="predict-page">
        {/* Form */}
        <div className="form-card">
          <div className="form-card-title">Borrower Details</div>
          {[
            { label: 'Credit Score', name: 'Credit_Score', placeholder: '300 – 850' },
            { label: 'Loan Amount (₹)', name: 'loan_amount', placeholder: 'e.g. 200000' },
            { label: 'Annual Income (₹)', name: 'income', placeholder: 'e.g. 60000' },
            { label: 'LTV — Loan to Value (%)', name: 'LTV', placeholder: 'e.g. 80' },
            { label: 'DTI — Debt to Income Ratio (%)', name: 'dtir1', placeholder: 'e.g. 35' },
          ].map((field) => (
            <div className="form-group" key={field.name}>
              <label className="form-label">{field.label}</label>
              <input
                type="number"
                name={field.name}
                placeholder={field.placeholder}
                onChange={handleChange}
                className="form-input"
              />
            </div>
          ))}
          <button onClick={handleSubmit} disabled={loading} className="submit-btn">
            {loading ? 'Analyzing...' : 'Run Risk Analysis →'}
          </button>
          {error && <div className="error-msg">{error}</div>}
        </div>

        {/* Result */}
        <div className="result-card">
          <div className="result-title">Risk Assessment Output</div>
          {!result ? (
            <div className="empty-result">
              <div className="empty-icon">◎</div>
              <div className="empty-text">Fill in borrower details and<br />click Run Risk Analysis to<br />see the prediction.</div>
            </div>
          ) : (
            <>
              {/* Risk Meter */}
              <div className="risk-meter">
                <div className="risk-percent" style={{ color: getRiskColor(result.default_probability) }}>
                  {(result.default_probability * 100).toFixed(1)}%
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--muted)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                  Default Probability
                </div>
                <div className="risk-bar-bg">
                  <div className="risk-bar-fill" style={{ width: `${result.default_probability * 100}%`, background: getRiskColor(result.default_probability) }} />
                </div>
              </div>

              {/* Risk Level Badge */}
              <div style={{ textAlign: 'center', marginBottom: '1.25rem' }}>
                <span style={{
                  padding: '0.35rem 1.2rem',
                  borderRadius: '20px',
                  background: `${getRiskLevel(result.default_probability).color}18`,
                  border: `1px solid ${getRiskLevel(result.default_probability).color}`,
                  color: getRiskLevel(result.default_probability).color,
                  fontSize: '0.82rem',
                  fontWeight: '700',
                  letterSpacing: '0.05em'
                }}>
                  ● {getRiskLevel(result.default_probability).label}
                </span>
              </div>

              {/* Result Rows */}
              <div className="result-row">
                <span className="result-key">Expected Loss</span>
                <span className="result-val" style={{ color: 'var(--red)' }}>₹{result.expected_loss.toLocaleString()}</span>
              </div>
              <div className="result-row">
                <span className="result-key">Priority Level</span>
                <span className={`result-val ${getPriorityClass(result.priority)}`}>● {result.priority}</span>
              </div>
              <div className="result-row">
                <span className="result-key">Recovery Channel</span>
                <span className="result-val" style={{ color: 'var(--blue)' }}>{result.recovery_channel}</span>
              </div>
              <div className="result-row">
                <span className="result-key">Assigned Team</span>
                <span className="result-val" style={{ color: 'var(--purple)' }}>{result.assigned_team}</span>
              </div>
              <div className="result-row">
                <span className="result-key">Follow-up Frequency</span>
                <span className="result-val" style={{ color: 'var(--gold)' }}>{result.follow_up_frequency}</span>
              </div>
              <div className="result-row">
                <span className="result-key">Legal Action</span>
                <span className="result-val" style={{ color: result.legal_action ? 'var(--red)' : 'var(--green)' }}>
                  {result.legal_action ? '⚠ Required' : '✓ Not Required'}
                </span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Borrower Summary Card */}
      {submitted && result && (
        <div style={{
          background: 'var(--bg2)',
          border: '1px solid var(--border)',
          borderRadius: '16px',
          padding: '2rem',
          marginTop: '2rem'
        }}>
          <div className="result-title">Borrower Summary</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '1rem', marginTop: '1rem' }}>
            {[
              { label: 'Credit Score', val: submitted.Credit_Score },
              { label: 'Loan Amount', val: `₹${submitted.loan_amount?.toLocaleString()}` },
              { label: 'Annual Income', val: `₹${submitted.income?.toLocaleString()}` },
              { label: 'LTV', val: `${submitted.LTV}%` },
              { label: 'DTI Ratio', val: `${submitted.dtir1}%` },
            ].map((item, i) => (
              <div key={i} style={{
                background: 'var(--bg3)',
                border: '1px solid var(--border)',
                borderRadius: '10px',
                padding: '1rem',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '1.2rem', fontWeight: '700', color: 'var(--blue)', fontFamily: 'Syne, sans-serif' }}>{item.val}</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--muted)', marginTop: '0.3rem', letterSpacing: '0.08em', textTransform: 'uppercase' }}>{item.label}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      {/* Agent Tips & Explanation */}
      {result && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginTop: '2rem' }}>

          {/* Why this risk level */}
          <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: '16px', padding: '2rem' }}>
            <div className="result-title">📊 Why This Risk Level?</div>
            <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {[
                {
                  label: 'Credit Score',
                  val: submitted?.Credit_Score,
                  good: submitted?.Credit_Score >= 700,
                  goodMsg: 'Good credit score — lower default risk',
                  badMsg: 'Below average credit score — increases default risk'
                },
                {
                  label: 'Loan to Income',
                  val: `${((submitted?.loan_amount / (submitted?.income + 1)) * 100).toFixed(1)}%`,
                  good: submitted?.loan_amount / (submitted?.income + 1) < 5,
                  goodMsg: 'Loan amount is manageable vs income',
                  badMsg: 'Loan amount is very high compared to income'
                },
                {
                  label: 'LTV Ratio',
                  val: `${submitted?.LTV}%`,
                  good: submitted?.LTV < 80,
                  goodMsg: 'LTV is within safe limits',
                  badMsg: 'High LTV — borrower has low equity in asset'
                },
                {
                  label: 'Debt to Income',
                  val: `${submitted?.dtir1}%`,
                  good: submitted?.dtir1 < 40,
                  goodMsg: 'DTI ratio is acceptable',
                  badMsg: 'High DTI — borrower is over-leveraged'
                },
              ].map((item, i) => (
                <div key={i} style={{
                  background: 'var(--bg3)',
                  border: `1px solid ${item.good ? 'rgba(0,255,136,0.2)' : 'rgba(255,77,109,0.2)'}`,
                  borderRadius: '10px',
                  padding: '0.75rem 1rem',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  gap: '1rem'
                }}>
                  <div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text)', fontWeight: '600' }}>{item.label}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--muted)', marginTop: '0.2rem' }}>{item.good ? item.goodMsg : item.badMsg}</div>
                  </div>
                  <div style={{ fontSize: '0.85rem', fontWeight: '700', color: item.good ? 'var(--green)' : 'var(--red)', flexShrink: 0 }}>
                    {item.good ? '✓' : '✗'} {item.val}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Agent Tips */}
          <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: '16px', padding: '2rem' }}>
            <div className="result-title">💡 Agent Action Tips</div>
            <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {(result.priority === 'Low' ? [
                { step: '01', tip: 'Send automated Email reminder with repayment link', color: 'var(--green)' },
                { step: '02', tip: 'Schedule SMS follow-up after 15 days if no response', color: 'var(--green)' },
                { step: '03', tip: 'Monitor borrower account for any payment activity', color: 'var(--green)' },
                { step: '04', tip: 'No escalation needed at this stage', color: 'var(--green)' },
                { step: '05', tip: 'Re-evaluate risk score after 30 days', color: 'var(--green)' },
              ] : result.priority === 'Medium' ? [
                { step: '01', tip: 'Call the borrower within 48 hours of assignment', color: 'var(--gold)' },
                { step: '02', tip: 'Discuss EMI restructuring options to reduce burden', color: 'var(--gold)' },
                { step: '03', tip: 'Offer a repayment plan extension if needed', color: 'var(--gold)' },
                { step: '04', tip: 'Document all calls and borrower responses', color: 'var(--gold)' },
                { step: '05', tip: 'Escalate to High if no response within 2 weeks', color: 'var(--gold)' },
              ] : [
                { step: '01', tip: 'Assign dedicated recovery officer immediately', color: 'var(--red)' },
                { step: '02', tip: 'Send official legal notice within 7 days', color: 'var(--red)' },
                { step: '03', tip: 'Schedule in-person field visit to borrower location', color: 'var(--red)' },
                { step: '04', tip: 'Collect and secure all loan documentation', color: 'var(--red)' },
                { step: '05', tip: 'Initiate legal proceedings if no response in 7 days', color: 'var(--red)' },
              ]).map((item, i) => (
                <div key={i} style={{
                  background: 'var(--bg3)',
                  border: '1px solid var(--border)',
                  borderRadius: '10px',
                  padding: '0.75rem 1rem',
                  display: 'flex',
                  gap: '1rem',
                  alignItems: 'flex-start'
                }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: '700', color: item.color, flexShrink: 0, marginTop: '0.1rem' }}>{item.step}</div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--muted)', lineHeight: '1.5' }}>{item.tip}</div>
                </div>
              ))}
            </div>
          </div>

        </div>
      )}
    </div>
    
  )
}