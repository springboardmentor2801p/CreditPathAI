import { useState } from 'react'
import PlotlyChart from '../components/PlotlyChart'
import { postRecommend, getRandomBorrower } from '../utils/api'
import './RecommendPage.css'

const DEFAULTS = {
  isJointApplication: 0,
  loanAmount: 350000,
  interestRate: 19.5,
  monthlyPayment: 9800,
  term_months: 36,
  yearsEmployment: 2,
  annualIncome: 480000,
  incomeVerified: 1,
  dtiRatio: 0.45,
  revolvingBalance: 120000,
  revolvingUtilizationRate: 0.82,
  lengthCreditHistory: 5,
  numTotalCreditLines: 8,
  numOpenCreditLines: 5,
  numOpenCreditLines1Year: 2,
  numDerogatoryRec: 1,
  numDelinquency2Years: 3,
  numChargeoff1year: 1,
  numInquiries6Mon: 4,
  grade_score: 5,
  loan_to_income_ratio: 0.73,
  payment_to_income_ratio: 0.245,
  repayment_velocity: 0.028,
  loan_amortization_rate: 0.033,
  open_credit_ratio: 0.625,
  recent_credit_velocity: 2,
  inquiry_intensity: 0.67,
  delinquency_density: 0.6,
  derogatory_density: 0.2,
  estimated_credit_limit: 146000,
  credit_utilization_recomputed: 0.82,
  log_loanAmount: 12.766,
  log_annualIncome: 13.082,
  log_revolvingBalance: 11.695,
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

const RISK_COLOR = {
  'Very Low':  '#10b981',
  'Low':       '#34d399',
  'Medium':    '#f59e0b',
  'High':      '#f97316',
  'Very High': '#ef4444',
}

function getRiskColor(riskBand) {
  return RISK_COLOR[riskBand] || '#6366f1'
}

function VerdictBanner({ result }) {
  const prob    = result.default_probability
  const isHigh  = prob >= 0.65
  const isMed   = prob >= 0.35
  const cls     = isHigh ? 'verdict-reject' : isMed ? 'verdict-caution' : 'verdict-approve'
  const icon    = isHigh ? 'REJECT' : isMed ? 'CAUTION' : 'APPROVE'
  const emoji   = isHigh ? '⛔' : isMed ? '⚠️' : '✅'
  const verdict = isHigh ? 'REJECT — HIGH DEFAULT RISK' : isMed ? 'CAUTION — REVIEW REQUIRED' : 'APPROVE — LOW DEFAULT RISK'
  const color   = getRiskColor(result.risk_band)

  return (
    <div className={`verdict-banner ${cls}`}>
      <div className="verdict-top">
        <span className="verdict-icon">{emoji}</span>
        <div>
          <div className="verdict-label">{verdict}</div>
          <div className="verdict-sub">Risk Band: <strong style={{ color }}>{result.risk_band}</strong></div>
        </div>
        <div className="verdict-prob" style={{ color }}>
          {(prob * 100).toFixed(1)}%
          <span>Default Prob.</span>
        </div>
      </div>
      <div className="gauge-bar" style={{ margin: '16px 0 0' }}>
        <div
          className="gauge-fill"
          style={{
            width: `${(prob * 100).toFixed(1)}%`,
            background: `linear-gradient(90deg, var(--risk-very-low), ${color})`,
          }}
        />
      </div>
      <div className="gauge-labels">
        <span>0% — Safe</span>
        <span>50%</span>
        <span>100% — Default</span>
      </div>
    </div>
  )
}

function ResultMetrics({ result }) {
  const color = getRiskColor(result.risk_band)
  return (
    <div className="result-metrics">
      <div className="metric-item">
        <div className="metric-val">{(result.default_probability * 100).toFixed(2)}%</div>
        <div className="metric-lbl">Default Probability</div>
      </div>
      <div className="metric-item">
        <div className="metric-val" style={{ color: result.predicted_default ? 'var(--danger)' : 'var(--success)' }}>
          {result.predicted_default ? 'YES' : 'NO'}
        </div>
        <div className="metric-lbl">Predicted Default</div>
      </div>
      <div className="metric-item">
        <div className="metric-val" style={{ color }}>{result.risk_band}</div>
        <div className="metric-lbl">Risk Band</div>
      </div>
      <div className="metric-item">
        <div className="metric-val">Rs.{Number(result.expected_loss).toLocaleString('en-IN', { maximumFractionDigits: 0 })}</div>
        <div className="metric-lbl">Expected Loss</div>
      </div>
    </div>
  )
}

function ProbGauge({ prob }) {
  const pct = prob * 100
  const barColor = pct >= 65 ? '#ef4444' : pct >= 35 ? '#f59e0b' : '#10b981'
  return (
    <PlotlyChart
      data={[{
        type: 'indicator',
        mode: 'gauge+number+delta',
        value: parseFloat(pct.toFixed(1)),
        delta: { reference: 50, increasing: { color: '#ef4444' }, decreasing: { color: '#10b981' } },
        gauge: {
          axis: { range: [0, 100], tickcolor: '#475569', tickfont: { color: '#94a3b8', size: 11 } },
          bar:  { color: barColor, thickness: 0.25 },
          bgcolor: 'transparent',
          borderwidth: 0,
          steps: [
            { range: [0, 35],   color: 'rgba(16,185,129,0.12)' },
            { range: [35, 65],  color: 'rgba(245,158,11,0.12)' },
            { range: [65, 100], color: 'rgba(239,68,68,0.12)'  },
          ],
          threshold: {
            line: { color: '#818cf8', width: 3 },
            thickness: 0.75,
            value: 50,
          },
        },
        number: { suffix: '%', font: { color: '#f1f5f9', size: 32 } },
        title:  { text: 'Default Probability', font: { color: '#94a3b8', size: 13 } },
      }]}
      layout={{
        paper_bgcolor: 'transparent',
        plot_bgcolor:  'transparent',
        margin: { t: 28, b: 10, l: 10, r: 10 },
        font:   { family: 'Inter, sans-serif' },
        height: 220,
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  )
}

function RiskFlagsPanel({ flags }) {
  if (!flags || flags.length === 0) {
    return (
      <div className="flags-empty">
        <span>✅</span> No risk flags detected — borrower profile looks clean.
      </div>
    )
  }
  return (
    <div className="flags-list">
      {flags.map((f, i) => (
        <div key={i} className="flag-item">
          <span>🚩</span> {f}
        </div>
      ))}
    </div>
  )
}

const FIELDS = [
  {
    section: 'Loan Details',
    fields: [
      { key: 'loanAmount',          label: 'Loan Amount (Rs.)',    type: 'number', min: 1 },
      { key: 'interestRate',        label: 'Interest Rate (%)',    type: 'number', min: 0 },
      { key: 'monthlyPayment',      label: 'Monthly Payment',      type: 'number', min: 0 },
      { key: 'term_months',         label: 'Term (months)',         type: 'number', min: 1 },
      { key: 'isJointApplication',  label: 'Joint Application',    type: 'select', options: [{ v: 0, l: 'No' }, { v: 1, l: 'Yes' }] },
    ],
  },
  {
    section: 'Borrower Profile',
    fields: [
      { key: 'annualIncome',    label: 'Annual Income (Rs.)',  type: 'number', min: 1 },
      { key: 'yearsEmployment', label: 'Years Employed',        type: 'number', min: 0 },
      { key: 'incomeVerified',  label: 'Income Verified',       type: 'select', options: [{ v: 0, l: 'No' }, { v: 1, l: 'Yes' }] },
      { key: 'dtiRatio',        label: 'DTI Ratio',             type: 'number', min: 0, step: 0.01 },
      { key: 'grade_score',     label: 'Grade Score',           type: 'number', min: 0 },
    ],
  },
  {
    section: 'Credit History',
    fields: [
      { key: 'lengthCreditHistory',    label: 'Credit History (yrs)',   type: 'number', min: 0 },
      { key: 'numTotalCreditLines',    label: 'Total Credit Lines',     type: 'number', min: 0 },
      { key: 'numOpenCreditLines',     label: 'Open Credit Lines',      type: 'number', min: 0 },
      { key: 'numOpenCreditLines1Year',label: 'Open Lines (1 yr)',      type: 'number', min: 0 },
      { key: 'numDerogatoryRec',       label: 'Derogatory Records',     type: 'number', min: 0 },
      { key: 'numDelinquency2Years',   label: 'Delinquencies (2 yrs)', type: 'number', min: 0 },
      { key: 'numChargeoff1year',      label: 'Charge-offs (1 yr)',     type: 'number', min: 0 },
      { key: 'numInquiries6Mon',       label: 'Inquiries (6 mos)',      type: 'number', min: 0 },
    ],
  },
  {
    section: 'Revolving Credit',
    fields: [
      { key: 'revolvingBalance',         label: 'Revolving Balance (Rs.)', type: 'number', min: 0 },
      { key: 'revolvingUtilizationRate', label: 'Utilization Rate',        type: 'number', min: 0, step: 0.01 },
    ],
  },
  {
    section: 'Engineered Features',
    fields: [
      { key: 'loan_to_income_ratio',         label: 'Loan-to-Income Ratio',    type: 'number', step: 0.001 },
      { key: 'payment_to_income_ratio',      label: 'Payment-to-Income Ratio', type: 'number', step: 0.001 },
      { key: 'repayment_velocity',           label: 'Repayment Velocity',      type: 'number', step: 0.001 },
      { key: 'loan_amortization_rate',       label: 'Amortization Rate',       type: 'number', step: 0.001 },
      { key: 'open_credit_ratio',            label: 'Open Credit Ratio',       type: 'number', step: 0.001 },
      { key: 'recent_credit_velocity',       label: 'Recent Credit Velocity',  type: 'number', step: 0.01  },
      { key: 'inquiry_intensity',            label: 'Inquiry Intensity',       type: 'number', step: 0.01  },
      { key: 'delinquency_density',          label: 'Delinquency Density',     type: 'number', step: 0.01  },
      { key: 'derogatory_density',           label: 'Derogatory Density',      type: 'number', step: 0.01  },
      { key: 'estimated_credit_limit',       label: 'Est. Credit Limit (Rs.)', type: 'number', min: 0      },
      { key: 'credit_utilization_recomputed',label: 'Credit Util. (Recomp)',   type: 'number', step: 0.001 },
    ],
  },
  {
    section: 'Log-Transformed Features',
    fields: [
      { key: 'log_loanAmount',       label: 'log(Loan Amount)',     type: 'number', step: 0.001 },
      { key: 'log_annualIncome',     label: 'log(Annual Income)',   type: 'number', step: 0.001 },
      { key: 'log_revolvingBalance', label: 'log(Revolving Bal.)', type: 'number', step: 0.001 },
    ],
  },
  {
    section: 'Loan Purpose',
    fields: [
      { key: 'purpose_business',         label: 'Business',         type: 'select', options: [{ v: 0, l: 'No' }, { v: 1, l: 'Yes' }] },
      { key: 'purpose_debtconsolidation',label: 'Debt Consolidation',type: 'select', options: [{ v: 0, l: 'No' }, { v: 1, l: 'Yes' }] },
      { key: 'purpose_education',        label: 'Education',        type: 'select', options: [{ v: 0, l: 'No' }, { v: 1, l: 'Yes' }] },
      { key: 'purpose_healthcare',       label: 'Healthcare',       type: 'select', options: [{ v: 0, l: 'No' }, { v: 1, l: 'Yes' }] },
      { key: 'purpose_homeimprovement',  label: 'Home Improvement', type: 'select', options: [{ v: 0, l: 'No' }, { v: 1, l: 'Yes' }] },
      { key: 'purpose_other',            label: 'Other',            type: 'select', options: [{ v: 0, l: 'No' }, { v: 1, l: 'Yes' }] },
    ],
  },
  {
    section: 'Home Ownership',
    fields: [
      { key: 'homeOwnership_own',  label: 'Owns Home',  type: 'select', options: [{ v: 0, l: 'No' }, { v: 1, l: 'Yes' }] },
      { key: 'homeOwnership_rent', label: 'Rents Home', type: 'select', options: [{ v: 0, l: 'No' }, { v: 1, l: 'Yes' }] },
    ],
  },
  {
    section: 'Settings',
    fields: [
      { key: 'threshold', label: 'Decision Threshold (0-1)', type: 'number', min: 0, max: 1, step: 0.01 },
    ],
  },
]

export default function RecommendPage() {
  const [form,    setForm]    = useState(DEFAULTS)
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const [filling, setFilling] = useState(false)

  const handleChange = (key, val) => {
    setForm(prev => ({ ...prev, [key]: val === '' ? '' : Number(val) }))
  }

  const handleSelectChange = (key, val) => {
    setForm(prev => ({ ...prev, [key]: Number(val) }))
  }

  const handleFillRandom = async () => {
    setFilling(true)
    setError(null)
    try {
      const data = await getRandomBorrower()
      setForm(prev => ({
        ...prev,
        ...Object.fromEntries(Object.entries(data).filter(([k]) => k in DEFAULTS)),
      }))
    } catch (e) {
      setError('Could not load random borrower. Is FastAPI running?')
    } finally {
      setFilling(false)
    }
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
      const res = await postRecommend(payload)
      setResult(res)
      setTimeout(() => {
        document.getElementById('result-section')?.scrollIntoView({ behavior: 'smooth' })
      }, 100)
    } catch (e) {
      setError(e.message || 'Prediction failed. Ensure FastAPI is running on port 8000.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="recommend-page">
      <div className="page-header">
        <h1>⚡ Risk Prediction</h1>
        <p>Enter borrower details and get an instant default risk assessment from the AI model.</p>
      </div>

      <div className="recommend-layout">
        {/* Input Form */}
        <div className="form-panel">
          <div className="form-panel-header">
            <span className="section-title" style={{ margin: 0 }}>Borrower Input</span>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={handleFillRandom}
              disabled={filling}
              id="btn-random"
            >
              {filling ? <span className="spinner" /> : '🎲'} Fill Random
            </button>
          </div>

          <form onSubmit={handleSubmit} id="recommend-form">
            {FIELDS.map(section => (
              <div key={section.section} className="form-section">
                <div className="form-section-title">{section.section}</div>
                <div className="form-grid">
                  {section.fields.map(f => (
                    <div key={f.key} className="form-group">
                      <label className="form-label" htmlFor={`field-${f.key}`}>{f.label}</label>
                      {f.type === 'select' ? (
                        <select
                          id={`field-${f.key}`}
                          className="form-select"
                          value={form[f.key]}
                          onChange={e => handleSelectChange(f.key, e.target.value)}
                        >
                          {f.options.map(o => (
                            <option key={o.v} value={o.v}>{o.l}</option>
                          ))}
                        </select>
                      ) : (
                        <input
                          id={`field-${f.key}`}
                          type="number"
                          className="form-input"
                          value={form[f.key]}
                          min={f.min}
                          max={f.max}
                          step={f.step || 'any'}
                          onChange={e => handleChange(f.key, e.target.value)}
                        />
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}

            {error && (
              <div className="error-box">⚠️ {error}</div>
            )}

            <button
              type="submit"
              className="btn btn-primary w-full submit-btn"
              disabled={loading}
              id="btn-submit"
            >
              {loading ? <><span className="spinner" /> Analyzing...</> : '⚡ Predict Default Risk'}
            </button>
          </form>
        </div>

        {/* Results Panel */}
        <div className="result-panel" id="result-section">
          {!result && !loading && (
            <div className="result-idle">
              <div className="idle-icon">🔍</div>
              <h3>No Prediction Yet</h3>
              <p>Fill in the borrower details on the left and click <strong>Predict</strong> to see the risk assessment here.</p>
            </div>
          )}

          {loading && (
            <div className="result-idle">
              <div className="spinner" style={{ width: 36, height: 36, borderWidth: 3 }} />
              <p style={{ marginTop: 16, color: 'var(--text-secondary)' }}>Scoring borrower...</p>
            </div>
          )}

          {result && (
            <div className="result-content">
              <VerdictBanner result={result} />

              <div className="card mt-4">
                <ProbGauge prob={result.default_probability} />
              </div>

              <div className="card mt-4">
                <div className="section-title">Key Metrics</div>
                <ResultMetrics result={result} />
              </div>

              <div className="card mt-4">
                <div className="section-title">Risk Flags</div>
                <RiskFlagsPanel flags={result.risk_flags} />
              </div>

              <div className="card mt-4">
                <div className="section-title">Action Plan</div>
                <div className="action-grid">
                  <div className="action-item">
                    <div className="action-key">Priority</div>
                    <div className={`action-val priority-${result.priority_level?.toLowerCase()}`}>{result.priority_level}</div>
                  </div>
                  <div className="action-item">
                    <div className="action-key">Assigned Team</div>
                    <div className="action-val">{result.assigned_team}</div>
                  </div>
                  <div className="action-item">
                    <div className="action-key">Recovery Channel</div>
                    <div className="action-val">{result.recovery_channel}</div>
                  </div>
                  <div className="action-item">
                    <div className="action-key">Follow-up</div>
                    <div className="action-val">{result.follow_up_frequency}</div>
                  </div>
                  <div className="action-item">
                    <div className="action-key">Legal Action</div>
                    <div className={`action-val ${result.legal_action ? 'legal-yes' : 'legal-no'}`}>
                      {result.legal_action ? 'YES - Legal' : 'No'}
                    </div>
                  </div>
                  <div className="action-item full-width">
                    <div className="action-key">Recommended Action</div>
                    <div className="action-val">{result.recommended_action}</div>
                  </div>
                  <div className="action-item full-width">
                    <div className="action-key">Escalation Notes</div>
                    <div className="action-val muted">{result.escalation_notes}</div>
                  </div>
                </div>
              </div>

              <div className="card mt-4">
                <div className="section-title">Visual Risk Analytics</div>
                <div className="chart-row">
                  <div className="chart-card">
                    <PlotlyChart
                      data={[{
                        type: 'scatterpolar',
                        r: [
                          Math.min(100, (form.dtiRatio / 0.5) * 100),
                          Math.min(100, (form.revolvingUtilizationRate) * 100),
                          Math.min(100, (form.loan_to_income_ratio) * 100),
                          Math.min(100, (form.numInquiries6Mon / 5) * 100),
                          Math.min(100, (form.delinquency_density) * 100),
                          Math.min(100, (form.dtiRatio / 0.5) * 100)
                        ],
                        theta: ['DTI Severity', 'Credit Util.', 'Loan/Income', 'Inquiries Risk', 'Delinquency', 'DTI Severity'],
                        fill: 'toself',
                        fillcolor: 'rgba(99, 102, 241, 0.2)',
                        line: { color: '#818cf8', width: 2 },
                        name: 'Borrower',
                        hovertemplate: '%{theta}: %{r:.1f}% Risk<extra></extra>',
                      }]}
                      layout={{
                        paper_bgcolor: 'transparent',
                        plot_bgcolor:  'transparent',
                        font: { family: 'Inter, sans-serif', color: '#94a3b8' },
                        polar: {
                          radialaxis: { visible: true, range: [0, 100], color: '#475569', gridcolor: 'rgba(255,255,255,0.05)' },
                          angularaxis: { color: '#94a3b8', gridcolor: 'rgba(255,255,255,0.1)' },
                          bgcolor: 'transparent'
                        },
                        title: { text: 'Risk Factors Benchmark', font: { color: '#f1f5f9', size: 14 } },
                        margin: { t: 40, b: 30, l: 30, r: 30 },
                        height: 300,
                      }}
                      config={{ displayModeBar: false, responsive: true }}
                      style={{ width: '100%' }}
                    />
                  </div>
                  <div className="chart-card">
                    <PlotlyChart
                      data={[{
                        type: 'pie',
                        labels: ['New Loan EMI', 'Existing Debt EMI', 'Disposable Income'],
                        values: [
                          form.monthlyPayment,
                          (form.annualIncome / 12) * form.dtiRatio - form.monthlyPayment,
                          (form.annualIncome / 12) * (1 - form.dtiRatio)
                        ].map(v => Math.max(0, v)),
                        hole: 0.6,
                        marker: { colors: ['#f59e0b', '#ef4444', '#10b981'] },
                        textinfo: 'percent',
                        textfont: { family: 'Inter', color: '#f1f5f9', size: 12 },
                        hovertemplate: '<b>%{label}</b><br>Rs.%{value:,.0f}<extra></extra>',
                      }]}
                      layout={{
                        paper_bgcolor: 'transparent',
                        plot_bgcolor:  'transparent',
                        font: { family: 'Inter, sans-serif', color: '#94a3b8' },
                        title: { text: 'Estimated Monthly Cashflow', font: { color: '#f1f5f9', size: 14 } },
                        margin: { t: 40, b: 20, l: 20, r: 20 },
                        height: 300,
                        legend: { orientation: 'h', y: -0.1, font: { color: '#94a3b8' } }
                      }}
                      config={{ displayModeBar: false, responsive: true }}
                      style={{ width: '100%' }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
