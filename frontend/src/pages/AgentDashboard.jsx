import { useState, useEffect } from 'react'
import PlotlyChart from '../components/PlotlyChart'
import { getBatchCases } from '../utils/api'
import './AgentDashboard.css'

const AGENT_CASES = [
  {
    id: 'BRW-001',
    risk_band: 'Very High',
    default_probability: 0.88,
    predicted_default: true,
    priority_level: 'Critical',
    assigned_team: 'Legal & Recovery',
    recovery_channel: 'Legal Notice',
    follow_up_frequency: 'Daily',
    legal_action: true,
    recommended_action: 'Initiate legal proceedings immediately.',
    escalation_notes: 'Borrower has 3+ delinquencies, high DTI, and short credit history.',
    risk_flags: ['High DTI Ratio', '3 Delinquencies (2 yrs)', 'Short Credit History', 'High Inquiry Intensity'],
    expected_loss: 490000,
    loan_amount: 560000,
  },
  {
    id: 'BRW-002',
    risk_band: 'High',
    default_probability: 0.72,
    predicted_default: true,
    priority_level: 'High',
    assigned_team: 'Senior Relationship',
    recovery_channel: 'Phone & SMS',
    follow_up_frequency: 'Weekly',
    legal_action: false,
    recommended_action: 'Offer restructuring or settlement plan.',
    escalation_notes: 'High revolving utilization. Consider hardship program.',
    risk_flags: ['High Revolving Utilization', 'Derogatory Records Present'],
    expected_loss: 310000,
    loan_amount: 430000,
  },
  {
    id: 'BRW-003',
    risk_band: 'High',
    default_probability: 0.67,
    predicted_default: true,
    priority_level: 'High',
    assigned_team: 'Collections',
    recovery_channel: 'Email & Phone',
    follow_up_frequency: 'Bi-weekly',
    legal_action: false,
    recommended_action: 'Send formal payment demand notice.',
    escalation_notes: 'Monitor for 30 days before escalating.',
    risk_flags: ['High Interest Rate', 'Low Open Credit Ratio'],
    expected_loss: 265000,
    loan_amount: 395000,
  },
  {
    id: 'BRW-004',
    risk_band: 'Medium',
    default_probability: 0.45,
    predicted_default: false,
    priority_level: 'Medium',
    assigned_team: 'Monitoring Team',
    recovery_channel: 'Email',
    follow_up_frequency: 'Monthly',
    legal_action: false,
    recommended_action: 'Proactive outreach - review payment plan.',
    escalation_notes: 'Borderline - watch for missed payments.',
    risk_flags: ['Moderate DTI', 'Low Income Verified'],
    expected_loss: 175000,
    loan_amount: 390000,
  },
  {
    id: 'BRW-005',
    risk_band: 'Medium',
    default_probability: 0.38,
    predicted_default: false,
    priority_level: 'Medium',
    assigned_team: 'Monitoring Team',
    recovery_channel: 'Email',
    follow_up_frequency: 'Monthly',
    legal_action: false,
    recommended_action: 'Monitor - send payment reminder.',
    escalation_notes: 'No immediate action needed.',
    risk_flags: ['Moderate Revolving Balance'],
    expected_loss: 112000,
    loan_amount: 295000,
  },
  {
    id: 'BRW-006',
    risk_band: 'Low',
    default_probability: 0.12,
    predicted_default: false,
    priority_level: 'Low',
    assigned_team: 'Standard Operations',
    recovery_channel: 'Automated',
    follow_up_frequency: 'Quarterly',
    legal_action: false,
    recommended_action: 'Routine check - no action needed.',
    escalation_notes: 'Healthy borrower profile.',
    risk_flags: [],
    expected_loss: 35000,
    loan_amount: 290000,
  },
]

const RISK_COLOR = {
  'Very Low':  '#10b981',
  'Low':       '#34d399',
  'Medium':    '#f59e0b',
  'High':      '#f97316',
  'Very High': '#ef4444',
}

const PLT_LAYOUT = {
  paper_bgcolor: 'transparent',
  plot_bgcolor:  'transparent',
  font: { family: 'Inter, sans-serif', color: '#94a3b8' },
  margin: { t: 30, b: 40, l: 50, r: 20 },
  xaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.05)' },
  yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.05)' },
}

export default function AgentDashboard() {
  const [selected, setSelected] = useState(null)
  const [results, setResults]   = useState(AGENT_CASES)
  const [loading, setLoading]   = useState(true)

  useEffect(() => {
    getBatchCases(8) // Get fewer cases for the agent drill-down page
      .then(data => {
        if (data && data.cases) {
          // Sort cases by highest risk first
          data.cases.sort((a, b) => b.default_probability - a.default_probability)
          setResults(data.cases)
        }
      })
      .catch(err => console.warn('Using fallback data:', err))
      .finally(() => setLoading(false))
  }, [])

  const totalEl  = results.reduce((s, c) => s + c.expected_loss, 0)
  const defaults = results.filter(c => c.predicted_default).length

  return (
    <div className="agent-page">
      <div className="page-header">
        <h1>🤖 Agent Recommendations</h1>
        <p>AI-generated action plans for each borrower — sorted by risk priority.</p>
      </div>

      <div className="agent-summary">
        <div className="summary-chip">
          <span className="chip-val">{results.length}</span>
          <span className="chip-lbl">Cases</span>
        </div>
        <div className="summary-chip">
          <span className="chip-val" style={{ color: 'var(--risk-very-high)' }}>{defaults}</span>
          <span className="chip-lbl">High Risk</span>
        </div>
        <div className="summary-chip">
          <span className="chip-val" style={{ color: 'var(--risk-high)' }}>
            Rs.{(totalEl / 1000).toFixed(0)}k
          </span>
          <span className="chip-lbl">Expected Loss</span>
        </div>
        <div className="summary-chip">
          <span className="chip-val" style={{ color: 'var(--risk-medium)' }}>
            {results.filter(c => c.legal_action).length}
          </span>
          <span className="chip-lbl">Legal Actions</span>
        </div>
      </div>

      <div className="agent-layout mt-4">
        {/* Cases List */}
        <div className="cases-list">
          {results.map(c => (
            <div
              key={c.id}
              className={`case-card ${selected?.id === c.id ? 'case-active' : ''}`}
              onClick={() => setSelected(selected?.id === c.id ? null : c)}
              id={`case-${c.id}`}
            >
              <div className="case-header">
                <div className="case-id">{c.id}</div>
                <span
                  className="band-badge"
                  style={{ background: `${RISK_COLOR[c.risk_band]}22`, color: RISK_COLOR[c.risk_band] }}
                >
                  {c.risk_band}
                </span>
                <span className={`priority-badge priority-${c.priority_level.toLowerCase()}`}>
                  {c.priority_level}
                </span>
                {c.legal_action && <span className="legal-badge">Legal Action</span>}
              </div>

              <div className="case-prob-row">
                <span className="case-prob-label">Default Probability</span>
                <span className="case-prob-val" style={{ color: RISK_COLOR[c.risk_band] }}>
                  {(c.default_probability * 100).toFixed(1)}%
                </span>
              </div>
              <div className="gauge-bar" style={{ marginTop: 4 }}>
                <div
                  className="gauge-fill"
                  style={{
                    width: `${c.default_probability * 100}%`,
                    background: `linear-gradient(90deg, var(--risk-very-low), ${RISK_COLOR[c.risk_band]})`,
                  }}
                />
              </div>

              {selected?.id === c.id && (
                <div className="case-detail">
                  <div className="divider" />
                  <div className="detail-grid">
                    <div className="detail-item">
                      <div className="detail-key">Assigned Team</div>
                      <div className="detail-val">{c.assigned_team}</div>
                    </div>
                    <div className="detail-item">
                      <div className="detail-key">Recovery Channel</div>
                      <div className="detail-val">{c.recovery_channel}</div>
                    </div>
                    <div className="detail-item">
                      <div className="detail-key">Follow-up</div>
                      <div className="detail-val">{c.follow_up_frequency}</div>
                    </div>
                    <div className="detail-item">
                      <div className="detail-key">Expected Loss</div>
                      <div className="detail-val" style={{ color: 'var(--risk-high)' }}>
                        Rs.{c.expected_loss.toLocaleString('en-IN')}
                      </div>
                    </div>
                  </div>
                  <div className="detail-action">
                    <div className="detail-key">Recommended Action</div>
                    <div className="detail-val">{c.recommended_action}</div>
                  </div>
                  <div className="detail-action" style={{ marginTop: 8 }}>
                    <div className="detail-key">Escalation Notes</div>
                    <div className="detail-val muted">{c.escalation_notes}</div>
                  </div>
                  {c.risk_flags.length > 0 && (
                    <div style={{ marginTop: 10 }}>
                      <div className="detail-key" style={{ marginBottom: 6 }}>Risk Flags</div>
                      <div className="flags-row">
                        {c.risk_flags.map((f, i) => (
                          <span key={i} className="flag-tag">🚩 {f}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Charts column */}
        <div className="agent-charts">
          <div className="card">
            <PlotlyChart
              data={[{
                type: 'bar',
                x: results.map(c => c.id),
                y: results.map(c => parseFloat((c.default_probability * 100).toFixed(1))),
                marker: { color: results.map(c => RISK_COLOR[c.risk_band]) },
                hovertemplate: '<b>%{x}</b><br>Prob: %{y}%<extra></extra>',
              }]}
              layout={{
                ...PLT_LAYOUT,
                title: { text: 'Default Probability by Borrower', font: { color: '#f1f5f9', size: 13 } },
                yaxis: { ...PLT_LAYOUT.yaxis, title: { text: '%', font: { color: '#94a3b8' } }, range: [0, 100] },
                height: 220,
                margin: { t: 30, b: 30, l: 40, r: 10 },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
            />
          </div>

          <div className="card mt-4">
            <PlotlyChart
              data={[{
                type: 'bar',
                orientation: 'h',
                x: results.map(c => c.expected_loss),
                y: results.map(c => c.id),
                marker: { color: results.map(c => RISK_COLOR[c.risk_band]) },
                hovertemplate: '<b>%{y}</b><br>Rs.%{x:,.0f}<extra></extra>',
              }]}
              layout={{
                ...PLT_LAYOUT,
                title: { text: 'Expected Loss per Case', font: { color: '#f1f5f9', size: 13 } },
                height: 260,
                margin: { t: 30, b: 40, l: 80, r: 10 },
                xaxis: { ...PLT_LAYOUT.xaxis, tickformat: ',.0f' },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
            />
          </div>

          <div className="card mt-4">
            <PlotlyChart
              data={[{
                type: 'pie',
                labels: [...new Set(results.map(c => c.assigned_team))],
                values: [...new Set(results.map(c => c.assigned_team))].map(
                  team => results.filter(c => c.assigned_team === team).length
                ),
                hole: 0.45,
                marker: { colors: ['#ef4444', '#f97316', '#f59e0b', '#6366f1', '#10b981'] },
                textinfo: 'label+percent',
                textfont: { family: 'Inter', size: 11, color: '#f1f5f9' },
                hovertemplate: '<b>%{label}</b>: %{value} cases<extra></extra>',
              }]}
              layout={{
                ...PLT_LAYOUT,
                title: { text: 'Team Workload Distribution', font: { color: '#f1f5f9', size: 13 } },
                height: 260,
                legend: { font: { color: '#94a3b8', size: 11 }, bgcolor: 'transparent' },
                margin: { t: 30, b: 10, l: 10, r: 10 },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
