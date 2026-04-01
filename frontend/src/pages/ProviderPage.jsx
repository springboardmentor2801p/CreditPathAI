import { useState } from 'react'
import PlotlyChart from '../components/PlotlyChart'
import { postRecommend, getRandomBorrower } from '../utils/api'

import { BLANK_FORM, computePayload } from '../utils/defaults'

const FIELDS = [
  {
    section: 'Loan Details',
    fields: [
      { key: 'loanAmount',          label: 'Loan Amount (Rs.)',    type: 'number', min: 1 },
      { key: 'interestRate',        label: 'Interest Rate (%)',    type: 'number', min: 0 },
      { key: 'monthlyPayment',      label: 'Monthly Payment',      type: 'number', min: 0 },
      { key: 'term_months',         label: 'Term (months)',        type: 'number', min: 1 },
      { key: 'isJointApplication',  label: 'Joint Application',    type: 'select', options: [{ v: 0, l: 'No' }, { v: 1, l: 'Yes' }] },
    ],
  },
  {
    section: 'Borrower Profile',
    fields: [
      { key: 'annualIncome',    label: 'Annual Income (Rs.)',  type: 'number', min: 1 },
      { key: 'yearsEmployment', label: 'Years Employed',       type: 'number', min: 0 },
      { key: 'incomeVerified',  label: 'Income Verified',      type: 'select', options: [{ v: 0, l: 'No' }, { v: 1, l: 'Yes' }] },
      { key: 'dtiRatio',        label: 'DTI Ratio',            type: 'number', min: 0, step: 0.01 },
      { key: 'grade_score',     label: 'Grade Score',          type: 'number', min: 0 },
    ],
  },
  {
    section: 'Credit History',
    fields: [
      { key: 'lengthCreditHistory',    label: 'Credit History (yrs)',  type: 'number', min: 0 },
      { key: 'numTotalCreditLines',    label: 'Total Credit Lines',    type: 'number', min: 0 },
      { key: 'numOpenCreditLines',     label: 'Open Credit Lines',     type: 'number', min: 0 },
      { key: 'numOpenCreditLines1Year',label: 'Open Lines (1 yr)',     type: 'number', min: 0 },
      { key: 'numDerogatoryRec',       label: 'Derogatory Records',    type: 'number', min: 0 },
      { key: 'numDelinquency2Years',   label: 'Delinquencies (2 yrs)', type: 'number', min: 0 },
      { key: 'numChargeoff1year',      label: 'Charge-offs (1 yr)',    type: 'number', min: 0 },
      { key: 'numInquiries6Mon',       label: 'Inquiries (6 mos)',     type: 'number', min: 0 },
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
    section: 'Derived Features',
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

export default function ProviderPage() {
  const [form, setForm] = useState(BLANK_FORM)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const handleChange = (key, val) => {
    setForm(prev => ({ ...prev, [key]: val === '' ? '' : Number(val) }))
  }

  const handleFillRandom = async () => {
    try {
      const data = await getRandomBorrower()
      setForm(prev => ({ ...prev, ...data }))
    } catch (err) {
      console.warn('Could not load random borrower.', err)
    }
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
    } catch (e) {
      if (Array.isArray(e.detail)) {
        setError(JSON.stringify(e.detail.map(d => d.msg).join(', ')))
      } else {
        setError(e.message || 'Prediction failed. Check input data.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h2>Loan Provider Portal</h2>
      <p>Evaluate borrower profiles and view AI-driven agent recommendations.</p>
      
      <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
        
        {/* Input Form */}
        <div className="card" style={{ flex: '1', minWidth: '350px' }}>
          <h3>Borrower Application Data</h3>
          <button type="button" className="btn btn-secondary" onClick={handleFillRandom} style={{ marginBottom: '15px' }}>
            Autofill Random Record
          </button>
          
          <form onSubmit={handleSubmit}>
            {FIELDS.map(section => {
              const advancedSections = ['Credit History', 'Revolving Credit', 'Derived Features', 'Log-Transformed Features', 'Loan Purpose', 'Home Ownership'];
              if (!showAdvanced && advancedSections.includes(section.section)) return null;

              return (
                <div key={section.section} style={{ marginBottom: '20px' }}>
                  <h4 style={{ borderBottom: '1px solid var(--border)', paddingBottom: '5px' }}>{section.section}</h4>
                  <div className="form-grid">
                    {section.fields.map(f => (
                      <div key={f.key}>
                        <label>{f.label}</label>
                        {f.type === 'select' ? (
                          <select 
                            value={form[f.key] !== undefined ? form[f.key] : ''} 
                            onChange={e => handleChange(f.key, e.target.value)}
                          >
                            <option value="" disabled>Default {computePayload(form)[f.key] === 1 ? 'Yes' : 'No'}...</option>
                            {f.options.map(opt => (
                              <option key={opt.v} value={opt.v}>{opt.l}</option>
                            ))}
                          </select>
                        ) : (
                          <input 
                            type="number"
                            step={f.step || 'any'}
                            placeholder={`e.g. ${computePayload(form)[f.key]}`}
                            value={form[f.key] !== undefined ? form[f.key] : ''} 
                            onChange={e => handleChange(f.key, e.target.value)} 
                          />
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}

            <div style={{ display: 'flex', justifyContent: 'center', margin: '20px 0' }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setShowAdvanced(!showAdvanced)}
                style={{ background: 'transparent', border: '1px solid var(--border)', color: 'var(--text-secondary)', padding: '8px 16px' }}
              >
                {showAdvanced ? 'Hide Advanced Sections ▲' : 'Show Advanced Sections ▼'}
              </button>
            </div>
            {error && <p style={{ color: 'red' }}>{error}</p>}
            <button type="submit" className="btn" disabled={loading} style={{ width: '100%' }}>
              {loading ? 'Evaluating...' : 'Run Credit Check'}
            </button>
          </form>
        </div>

        {/* Results */}
        <div style={{ flex: '1', minWidth: '350px' }}>
          <h3>Agent Recommendation & Decision</h3>
          {!result && <p>Results will appear here...</p>}
          
          {result && (
            <div>
              {/* Verdict */}
              <div className={`result-box ${result.predicted_default ? 'rejected' : 'approved'}`}>
                <h3>Decision: {result.predicted_default ? 'REJECT LOAN' : 'APPROVE LOAN'}</h3>
                <p><strong>Risk Level:</strong> {result.risk_band} Priority</p>
                <p><strong>Default Probability:</strong> {(result.default_probability * 100).toFixed(1)}%</p>
                <p><strong>Expected Financial Loss:</strong> Rs. {Number(result.expected_loss).toLocaleString('en-IN', { maximumFractionDigits: 0 })}</p>
              </div>

              {/* Agent Breakdown */}
              <div className="card" style={{ marginTop: '20px' }}>
                <h4 style={{ marginTop: 0 }}>Agent Insights & Recovery Protocol</h4>
                <table style={{ marginTop: '10px' }}>
                  <tbody>
                    <tr>
                      <th width="35%">Assigned Team</th>
                      <td>{result.assigned_team}</td>
                    </tr>
                    <tr>
                      <th>Recommended Action</th>
                      <td>{result.recommended_action}</td>
                    </tr>
                    <tr>
                      <th>Recovery Channel</th>
                      <td>{result.recovery_channel} (Frequency: {result.follow_up_frequency})</td>
                    </tr>
                    <tr>
                      <th>Legal Action Advised?</th>
                      <td>{result.legal_action ? 'YES' : 'NO'}</td>
                    </tr>
                    <tr>
                      <th>Agent Notes</th>
                      <td>{result.escalation_notes}</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Red Flags */}
              {result.risk_flags && result.risk_flags.length > 0 && (
                <div className="card" style={{ marginTop: '20px', borderLeft: '5px solid var(--danger)' }}>
                  <h4 style={{ marginTop: 0, color: 'var(--danger)' }}>Detected Risk Flags (Reasons for Default Default)</h4>
                  <ul>
                    {result.risk_flags.map((flag, idx) => (
                      <li key={idx}>{flag}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Provider Analytics Charts */}
              <div className="card" style={{ marginTop: '20px' }}>
                <h4 style={{ marginTop: 0 }}>Advanced Risk Analytics</h4>
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
                          { range: [0, 35], color: 'rgba(16,185,129,0.2)' },
                          { range: [35, 65], color: 'rgba(245,158,11,0.2)' },
                          { range: [65, 100], color: 'rgba(239,68,68,0.2)' }
                        ]
                      },
                      title: { text: 'Predictive Default Probability', font: { size: 12 } },
                    }]}
                    layout={{ height: 220, margin: { t: 30, b: 10, l: 20, r: 20 }, paper_bgcolor: 'transparent' }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                  />
                  <PlotlyChart
                    data={[{
                      type: 'scatterpolar',
                      r: [
                        Math.min(100, (form.dtiRatio / 0.5) * 100),
                        Math.min(100, (form.revolvingUtilizationRate) * 100),
                        Math.min(100, (form.loan_to_income_ratio) * 100),
                        Math.min(100, (form.numInquiries6Mon / 5) * 100),
                        Math.min(100, (form.dtiRatio / 0.5) * 100)
                      ],
                      theta: ['DTI Severity', 'Credit Util.', 'Loan/Income', 'Inquiries Risk', 'DTI Severity'],
                      fill: 'toself',
                      fillcolor: 'rgba(99, 102, 241, 0.2)',
                      line: { color: '#818cf8', width: 2 },
                    }]}
                    layout={{
                      polar: {
                        radialaxis: { visible: true, range: [0, 100] }
                      },
                      title: { text: 'Borrower Risk Footprint', font: { size: 12 } },
                      margin: { t: 30, b: 20, l: 20, r: 20 },
                      height: 220,
                      paper_bgcolor: 'transparent'
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%' }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  )
}
