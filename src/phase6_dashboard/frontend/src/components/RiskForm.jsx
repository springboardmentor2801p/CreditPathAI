import { useState } from 'react';

const STATES = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','DC'];
const EMPLOYMENT = ['< 1 year','1 year','2-5 years','6-9 years','10+ years'];
const OWNERSHIP  = ['rent','own','mortgage'];
const PURPOSES   = ['debtconsolidation','homeimprovement','business','medical','majorpurchase','smallbusiness','car','vacation','wedding','movingandrelo','education','other'];
const TERMS      = ['36 months','48 months','60 months'];
const GRADES     = ['A1','A2','A3','B1','B2','B3','C1','C2','C3','D1','D2','D3','E1','E2','E3'];

const DEFAULTS = {
  borrower: {
    residentialState:'CA', yearsEmployment:'2-5 years', homeOwnership:'rent',
    annualIncome:60000, incomeVerified:1, dtiRatio:18.5,
    lengthCreditHistory:5, numTotalCreditLines:15, numOpenCreditLines:10,
    numOpenCreditLines1Year:6, revolvingBalance:12000,
    revolvingUtilizationRate:65, numDerogatoryRec:0,
    numDelinquency2Years:0, numChargeoff1year:0, numInquiries6Mon:1,
  },
  loan: {
    purpose:'debtconsolidation', isJointApplication:0, loanAmount:20000,
    term:'60 months', interestRate:9.5, monthlyPayment:420, grade:'C3',
  },
};

export default function RiskForm({ onSubmit, loading }) {
  const [form, setForm] = useState(DEFAULTS);

  const setB = (k, v) => setForm((f) => ({ ...f, borrower: { ...f.borrower, [k]: v } }));
  const setL = (k, v) => setForm((f) => ({ ...f, loan:     { ...f.loan,     [k]: v } }));

  const num = (setter, key) => (e) => setter(key, e.target.value === '' ? '' : Number(e.target.value));
  const str = (setter, key) => (e) => setter(key, e.target.value);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(form.borrower, form.loan);
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="form-two-col">
        {/* ── Borrower ─────────────────────────────────────── */}
        <div className="form-section">
          <div className="form-section-title">👤 Borrower Profile</div>
          <div className="form-grid">
            <div className="form-group">
              <label>Residential State</label>
              <select value={form.borrower.residentialState} onChange={str(setB,'residentialState')}>
                {STATES.map((s) => <option key={s}>{s}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label>Years of Employment</label>
              <select value={form.borrower.yearsEmployment} onChange={str(setB,'yearsEmployment')}>
                {EMPLOYMENT.map((e) => <option key={e}>{e}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label>Home Ownership</label>
              <select value={form.borrower.homeOwnership} onChange={str(setB,'homeOwnership')}>
                {OWNERSHIP.map((o) => <option key={o} style={{textTransform:'capitalize'}}>{o}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label>Annual Income ($)</label>
              <input type="number" min="1" value={form.borrower.annualIncome} onChange={num(setB,'annualIncome')} />
            </div>
            <div className="form-group">
              <label>Income Verified</label>
              <select value={form.borrower.incomeVerified} onChange={(e) => setB('incomeVerified', Number(e.target.value))}>
                <option value={1}>Yes</option>
                <option value={0}>No</option>
              </select>
            </div>
            <div className="form-group">
              <label>DTI Ratio (%)</label>
              <input type="number" step="0.01" min="0" value={form.borrower.dtiRatio} onChange={num(setB,'dtiRatio')} />
            </div>
            <div className="form-group">
              <label>Credit History (years)</label>
              <input type="number" min="0" value={form.borrower.lengthCreditHistory} onChange={num(setB,'lengthCreditHistory')} />
            </div>
            <div className="form-group">
              <label>Total Credit Lines</label>
              <input type="number" min="0" value={form.borrower.numTotalCreditLines} onChange={num(setB,'numTotalCreditLines')} />
            </div>
            <div className="form-group">
              <label>Open Credit Lines</label>
              <input type="number" min="0" value={form.borrower.numOpenCreditLines} onChange={num(setB,'numOpenCreditLines')} />
            </div>
            <div className="form-group">
              <label>Open Lines (1yr)</label>
              <input type="number" min="0" value={form.borrower.numOpenCreditLines1Year} onChange={num(setB,'numOpenCreditLines1Year')} />
            </div>
            <div className="form-group">
              <label>Revolving Balance ($)</label>
              <input type="number" min="0" value={form.borrower.revolvingBalance} onChange={num(setB,'revolvingBalance')} />
            </div>
            <div className="form-group">
              <label>Revolving Utilization (%)</label>
              <input type="number" step="0.01" min="0" max="150" value={form.borrower.revolvingUtilizationRate} onChange={num(setB,'revolvingUtilizationRate')} />
            </div>
            <div className="form-group">
              <label>Derogatory Records</label>
              <input type="number" min="0" value={form.borrower.numDerogatoryRec} onChange={num(setB,'numDerogatoryRec')} />
            </div>
            <div className="form-group">
              <label>Delinquencies (2yr)</label>
              <input type="number" min="0" value={form.borrower.numDelinquency2Years} onChange={num(setB,'numDelinquency2Years')} />
            </div>
            <div className="form-group">
              <label>Charge-offs (1yr)</label>
              <input type="number" min="0" value={form.borrower.numChargeoff1year} onChange={num(setB,'numChargeoff1year')} />
            </div>
            <div className="form-group">
              <label>Inquiries (6 months)</label>
              <input type="number" min="0" value={form.borrower.numInquiries6Mon} onChange={num(setB,'numInquiries6Mon')} />
            </div>
          </div>
        </div>

        {/* ── Loan ──────────────────────────────────────────── */}
        <div className="form-section">
          <div className="form-section-title">🏦 Loan Details</div>
          <div className="form-grid">
            <div className="form-group">
              <label>Purpose</label>
              <select value={form.loan.purpose} onChange={str(setL,'purpose')}>
                {PURPOSES.map((p) => <option key={p} style={{textTransform:'capitalize'}}>{p}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label>Joint Application</label>
              <select value={form.loan.isJointApplication} onChange={(e) => setL('isJointApplication', Number(e.target.value))}>
                <option value={0}>No</option>
                <option value={1}>Yes</option>
              </select>
            </div>
            <div className="form-group">
              <label>Loan Amount ($)</label>
              <input type="number" min="1" value={form.loan.loanAmount} onChange={num(setL,'loanAmount')} />
            </div>
            <div className="form-group">
              <label>Term</label>
              <select value={form.loan.term} onChange={str(setL,'term')}>
                {TERMS.map((t) => <option key={t}>{t}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label>Interest Rate (%)</label>
              <input type="number" step="0.01" min="0.1" value={form.loan.interestRate} onChange={num(setL,'interestRate')} />
            </div>
            <div className="form-group">
              <label>Monthly Payment ($)</label>
              <input type="number" min="1" value={form.loan.monthlyPayment} onChange={num(setL,'monthlyPayment')} />
            </div>
            <div className="form-group" style={{ gridColumn:'1 / -1' }}>
              <label>Loan Grade</label>
              <select value={form.loan.grade} onChange={str(setL,'grade')}>
                {GRADES.map((g) => <option key={g}>{g}</option>)}
              </select>
            </div>
          </div>

          {/* Submit pinned at bottom of loan section */}
          <div style={{ marginTop:24 }}>
            <button type="submit" className="btn btn-primary btn-lg btn-full" disabled={loading}>
              {loading ? (
                <><div className="spinner" />Analysing…</>
              ) : (
                <>⚡ Analyse Risk</>
              )}
            </button>
          </div>
        </div>
      </div>
    </form>
  );
}
