import { useState } from 'react'
import { predictRisk } from '../services/api'

const initialBorrowers = [
  { id: 'BRW001', name: 'Rahul Sharma',  credit: 580, loan: 450000, income: 35000, risk: 'Critical Risk', priority: 'Critical', prob: 0.85, loss: 382500, team: 'Senior Recovery Team',          channel: 'Legal Action',                 follow_up: 'Immediate', legal: true  },
  { id: 'BRW002', name: 'Priya Patel',   credit: 620, loan: 280000, income: 48000, risk: 'High Risk',     priority: 'High',     prob: 0.72, loss: 201600, team: 'Dedicated Recovery Officer',   channel: 'Legal Notice + Field Visit',   follow_up: 'Daily',     legal: true  },
  { id: 'BRW003', name: 'Amit Kumar',    credit: 670, loan: 180000, income: 62000, risk: 'Moderate Risk', priority: 'Medium',   prob: 0.51, loss: 91800,  team: 'Call Center Agent',            channel: 'Phone Call + EMI Restructure', follow_up: 'Weekly',    legal: false },
  { id: 'BRW004', name: 'Sneha Reddy',   credit: 710, loan: 120000, income: 75000, risk: 'Low Risk',      priority: 'Low',      prob: 0.32, loss: 38400,  team: 'Automated System',             channel: 'Email + SMS Reminder',         follow_up: '15 days',   legal: false },
  { id: 'BRW005', name: 'Vikram Singh',  credit: 550, loan: 520000, income: 40000, risk: 'Critical Risk', priority: 'Critical', prob: 0.91, loss: 473200, team: 'Senior Recovery Team',          channel: 'Legal Action',                 follow_up: 'Immediate', legal: true  },
]

const actions = {
  Critical: [
    'Assign senior recovery officer immediately',
    'Send official legal notice within 24 hours',
    'Schedule field visit to borrower location',
    'Collect and secure all loan documentation',
    'Initiate court proceedings if no response in 7 days',
  ],
  High: [
    'Assign dedicated recovery officer within 48 hours',
    'Send legal notice and follow up by phone',
    'Schedule in-person field visit this week',
    'Document all borrower communications',
    'Escalate to Critical if no response in 2 weeks',
  ],
  Medium: [
    'Call borrower within 48 hours of assignment',
    'Discuss EMI restructuring and repayment options',
    'Offer repayment plan extension if needed',
    'Send weekly follow-up SMS and email reminders',
    'Escalate to High if no response in 2 weeks',
  ],
  Low: [
    'Send automated Email and SMS reminder',
    'Monitor borrower account for payment activity',
    'Schedule soft follow-up call after 15 days',
    'Offer early repayment incentive if applicable',
    'Re-evaluate risk score after 30 days',
  ],
}

const priorityColor = {
  Critical: '#ff4d6d',
  High: '#FF8C42',
  Medium: '#ffd700',
  Low: '#00ff88',
}

const getRiskLevel = (prob) => {
  if (prob < 0.20) return 'Very Low Risk'
  if (prob < 0.40) return 'Low Risk'
  if (prob < 0.60) return 'Moderate Risk'
  if (prob < 0.80) return 'High Risk'
  return 'Critical Risk'
}

export default function AgentPage() {
  const [borrowers, setBorrowers] = useState(initialBorrowers)
  const [filter, setFilter] = useState('All')
  const [selected, setSelected] = useState(null)
  const [checked, setChecked] = useState({})
  const [showForm, setShowForm] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [form, setForm] = useState({
    name: '', Credit_Score: '', loan_amount: '', income: '', LTV: '', dtir1: ''
  })

  const filtered = filter === 'All' ? borrowers : borrowers.filter(b => b.priority === filter)

  const toggleCheck = (borrowerId, idx) => {
    const key = `${borrowerId}-${idx}`
    setChecked(prev => ({ ...prev, [key]: !prev[key] }))
  }

  const handleFormChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value })
  }

  const handleAddBorrower = async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await predictRisk({
        Credit_Score: parseFloat(form.Credit_Score),
        loan_amount: parseFloat(form.loan_amount),
        income: parseFloat(form.income),
        LTV: parseFloat(form.LTV),
        dtir1: parseFloat(form.dtir1),
      })

      const newBorrower = {
        id: `BRW${String(borrowers.length + 1).padStart(3, '0')}`,
        name: form.name || `Borrower ${borrowers.length + 1}`,
        credit: parseFloat(form.Credit_Score),
        loan: parseFloat(form.loan_amount),
        income: parseFloat(form.income),
        risk: result.risk_level || getRiskLevel(result.default_probability),
        priority: result.priority,
        prob: result.default_probability,
        loss: result.expected_loss,
        team: result.assigned_team,
        channel: result.recovery_channel,
        follow_up: result.follow_up_frequency || 'Weekly',
        legal: result.legal_action || false,
      }

      setBorrowers(prev => [newBorrower, ...prev])
      setShowForm(false)
      setForm({ name: '', Credit_Score: '', loan_amount: '', income: '', LTV: '', dtir1: '' })
    } catch (err) {
      setError('Failed to predict. Make sure backend is running!')
    }
    setLoading(false)
  }

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 className="page-title">Agent Recommendation System</h1>
          <p className="page-subtitle">Assigned borrowers with personalized recovery action plans</p>
        </div>
        <button onClick={() => setShowForm(!showForm)} style={{
          padding: '0.75rem 1.5rem',
          background: 'var(--blue)',
          color: '#000',
          border: 'none',
          borderRadius: '8px',
          fontWeight: '700',
          fontSize: '0.85rem',
          cursor: 'pointer',
          marginTop: '0.5rem'
        }}>
          {showForm ? '✕ Cancel' : '+ Add Borrower'}
        </button>
      </div>

      {/* Add Borrower Form */}
      {showForm && (
        <div style={{
          background: 'var(--bg2)',
          border: '1px solid var(--blue)',
          borderRadius: '16px',
          padding: '2rem',
          marginBottom: '2rem',
        }}>
          <div style={{ fontSize: '0.85rem', fontWeight: '700', color: 'var(--blue)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '1.5rem' }}>
            New Borrower Details
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
            {[
              { label: 'Borrower Name', name: 'name', placeholder: 'e.g. John Doe', type: 'text' },
              { label: 'Credit Score', name: 'Credit_Score', placeholder: '300 – 850', type: 'number' },
              { label: 'Loan Amount (₹)', name: 'loan_amount', placeholder: 'e.g. 200000', type: 'number' },
              { label: 'Annual Income (₹)', name: 'income', placeholder: 'e.g. 60000', type: 'number' },
              { label: 'LTV (%)', name: 'LTV', placeholder: 'e.g. 80', type: 'number' },
              { label: 'DTI Ratio (%)', name: 'dtir1', placeholder: 'e.g. 35', type: 'number' },
            ].map((field) => (
              <div key={field.name}>
                <label className="form-label">{field.label}</label>
                <input
                  type={field.type}
                  name={field.name}
                  placeholder={field.placeholder}
                  value={form[field.name]}
                  onChange={handleFormChange}
                  className="form-input"
                />
              </div>
            ))}
          </div>
          {error && <div className="error-msg" style={{ marginTop: '1rem' }}>{error}</div>}
          <button onClick={handleAddBorrower} disabled={loading} style={{
            marginTop: '1.5rem',
            padding: '0.75rem 2rem',
            background: 'var(--blue)',
            color: '#000',
            border: 'none',
            borderRadius: '8px',
            fontWeight: '700',
            fontSize: '0.9rem',
            cursor: 'pointer',
          }}>
            {loading ? 'Predicting...' : 'Add & Predict →'}
          </button>
        </div>
      )}

      {/* Summary Stats */}
      <div className="dashboard-grid" style={{ marginBottom: '2rem' }}>
        {[
          { label: 'Total Assigned', value: borrowers.length,                                                    color: '#00cfff' },
          { label: 'Critical',       value: borrowers.filter(b => b.priority === 'Critical').length,             color: '#ff4d6d' },
          { label: 'High',           value: borrowers.filter(b => b.priority === 'High').length,                 color: '#FF8C42' },
          { label: 'Medium / Low',   value: borrowers.filter(b => ['Medium','Low'].includes(b.priority)).length, color: '#00ff88' },
        ].map((s, i) => (
          <div key={i} className="dash-stat" style={{ '--accent': s.color }}>
            <div className="dash-stat-value" style={{ color: s.color }}>{s.value}</div>
            <div className="dash-stat-label">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Filter Buttons */}
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
        {['All', 'Critical', 'High', 'Medium', 'Low'].map(f => (
          <button key={f} onClick={() => setFilter(f)} style={{
            padding: '0.4rem 1.2rem',
            borderRadius: '20px',
            border: `1px solid ${filter === f ? (priorityColor[f] || 'var(--blue)') : 'var(--border)'}`,
            background: filter === f ? `${priorityColor[f] || 'var(--blue)'}18` : 'var(--bg2)',
            color: filter === f ? (priorityColor[f] || 'var(--blue)') : 'var(--muted)',
            fontSize: '0.8rem', fontWeight: '600', cursor: 'pointer', transition: 'all 0.2s'
          }}>
            {f}
          </button>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: selected ? '1fr 1fr' : '1fr', gap: '1.5rem' }}>
        {/* Borrower List */}
        <div>
          {filtered.map(b => (
            <div key={b.id} onClick={() => setSelected(b)} style={{
              background: selected?.id === b.id ? 'var(--bg3)' : 'var(--bg2)',
              border: `1px solid ${selected?.id === b.id ? priorityColor[b.priority] : 'var(--border)'}`,
              borderRadius: '12px', padding: '1.25rem', marginBottom: '0.75rem',
              cursor: 'pointer', transition: 'all 0.2s',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div style={{
                    width: '40px', height: '40px', borderRadius: '50%',
                    background: `${priorityColor[b.priority]}22`,
                    border: `1px solid ${priorityColor[b.priority]}`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '0.85rem', fontWeight: '700', color: priorityColor[b.priority], flexShrink: 0
                  }}>
                    {b.name.split(' ').map(n => n[0]).join('')}
                  </div>
                  <div>
                    <div style={{ fontSize: '0.95rem', fontWeight: '700', color: 'var(--text)' }}>{b.name}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--muted)', marginTop: '0.1rem' }}>{b.id} · {b.risk}</div>
                  </div>
                </div>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '0.85rem', color: '#ff4d6d', fontWeight: '700' }}>₹{b.loss.toLocaleString()}</div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--muted)' }}>Expected Loss</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '0.85rem', color: priorityColor[b.priority], fontWeight: '700' }}>{(b.prob * 100).toFixed(0)}%</div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--muted)' }}>Default Prob</div>
                  </div>
                  <span style={{
                    padding: '0.25rem 0.75rem', borderRadius: '20px',
                    background: `${priorityColor[b.priority]}18`,
                    border: `1px solid ${priorityColor[b.priority]}`,
                    color: priorityColor[b.priority], fontSize: '0.72rem', fontWeight: '700'
                  }}>{b.priority}</span>
                </div>
              </div>
              <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border)' }}>
                {[
                  { label: 'Team', val: b.team },
                  { label: 'Channel', val: b.channel },
                  { label: 'Follow-up', val: b.follow_up },
                  { label: 'Legal', val: b.legal ? '⚠ Required' : '✓ Not Required' },
                ].map((item, i) => (
                  <div key={i}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--muted)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>{item.label}</div>
                    <div style={{ fontSize: '0.78rem', color: item.label === 'Legal' ? (b.legal ? '#ff4d6d' : '#00ff88') : 'var(--text)', marginTop: '0.15rem' }}>{item.val}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Action Checklist */}
        {selected && (
          <div style={{ position: 'sticky', top: '80px', height: 'fit-content' }}>
            <div style={{ background: 'var(--bg2)', border: `1px solid ${priorityColor[selected.priority]}`, borderRadius: '16px', padding: '2rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                <div>
                  <div style={{ fontSize: '1.1rem', fontWeight: '700', color: 'var(--text)' }}>{selected.name}</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--muted)', marginTop: '0.2rem' }}>{selected.id} · Action Checklist</div>
                </div>
                <button onClick={() => setSelected(null)} style={{
                  background: 'var(--bg3)', border: '1px solid var(--border)',
                  borderRadius: '8px', padding: '0.3rem 0.75rem',
                  color: 'var(--muted)', cursor: 'pointer', fontSize: '0.8rem'
                }}>✕ Close</button>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1.5rem' }}>
                {[
                  { label: 'Credit Score', val: selected.credit, color: selected.credit >= 700 ? '#00ff88' : '#ff4d6d' },
                  { label: 'Loan Amount',  val: `₹${selected.loan.toLocaleString()}`, color: 'var(--blue)' },
                  { label: 'Income',       val: `₹${selected.income.toLocaleString()}`, color: 'var(--text)' },
                  { label: 'Default Prob', val: `${(selected.prob * 100).toFixed(0)}%`, color: priorityColor[selected.priority] },
                ].map((item, i) => (
                  <div key={i} style={{ background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: '8px', padding: '0.75rem' }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{item.label}</div>
                    <div style={{ fontSize: '1rem', fontWeight: '700', color: item.color, marginTop: '0.2rem' }}>{item.val}</div>
                  </div>
                ))}
              </div>

              <div style={{ fontSize: '0.75rem', color: 'var(--muted)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '0.75rem' }}>
                Action Checklist
              </div>

              {actions[selected.priority]?.map((action, i) => {
                const key = `${selected.id}-${i}`
                return (
                  <div key={i} onClick={() => toggleCheck(selected.id, i)} style={{
                    display: 'flex', alignItems: 'flex-start', gap: '0.75rem',
                    padding: '0.75rem', borderRadius: '8px', marginBottom: '0.5rem',
                    background: checked[key] ? `${priorityColor[selected.priority]}10` : 'var(--bg3)',
                    border: `1px solid ${checked[key] ? priorityColor[selected.priority] : 'var(--border)'}`,
                    cursor: 'pointer', transition: 'all 0.2s'
                  }}>
                    <div style={{
                      width: '18px', height: '18px', borderRadius: '4px', flexShrink: 0,
                      border: `2px solid ${checked[key] ? priorityColor[selected.priority] : 'var(--muted)'}`,
                      background: checked[key] ? priorityColor[selected.priority] : 'transparent',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: '0.7rem', color: '#000', fontWeight: '700'
                    }}>
                      {checked[key] ? '✓' : ''}
                    </div>
                    <div style={{
                      fontSize: '0.85rem',
                      color: checked[key] ? 'var(--muted)' : 'var(--text)',
                      textDecoration: checked[key] ? 'line-through' : 'none',
                      lineHeight: '1.5'
                    }}>
                      {action}
                    </div>
                  </div>
                )
              })}

              <div style={{ marginTop: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--muted)', marginBottom: '0.4rem' }}>
                  <span>Progress</span>
                  <span>{actions[selected.priority]?.filter((_, i) => checked[`${selected.id}-${i}`]).length} / {actions[selected.priority]?.length} completed</span>
                </div>
                <div style={{ height: '6px', background: 'var(--bg3)', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%', borderRadius: '3px',
                    background: priorityColor[selected.priority],
                    width: `${(actions[selected.priority]?.filter((_, i) => checked[`${selected.id}-${i}`]).length / actions[selected.priority]?.length) * 100}%`,
                    transition: 'width 0.3s ease'
                  }} />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}