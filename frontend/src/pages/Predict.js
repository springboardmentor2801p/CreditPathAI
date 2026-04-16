import React, { useState, useEffect, useContext } from "react";
import axios from "axios";
import Plot from "react-plotly.js";
import { AuthContext } from "../context/AuthContext";

const PLOT_LAYOUT = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#0f172a", family: "Inter, sans-serif", size: 11 },
  margin: { t: 44, r: 12, b: 36, l: 40 },
  hovermode: "closest",
  hoverlabel: { bgcolor: "#1e293b", font: { color: "#ffffff", size: 12 }, bordercolor: "transparent" },
};

/* ── Tooltip component ── */
function Tooltip({ text }) {
  return (
    <span className="tooltip-container">
      <span className="tooltip-icon">ℹ️</span>
      <span className="tooltip-text">{text}</span>
    </span>
  );
}

/* ── Risk badge component ── */
function RiskBadge({ level }) {
  const cls = !level ? "" : level.toLowerCase() === "low" ? "low" : level.toLowerCase() === "medium" ? "medium" : "high";
  const emoji = cls === "low" ? "🟢" : cls === "medium" ? "🟡" : "🔴";
  return <span className={`risk-badge ${cls}`}>{emoji} {level}</span>;
}

/* ── Helper: which info-box class to use ── */
function infoClass(risk) {
  if (risk === "Low" || risk === "Approved") return "info-box success-box";
  if (risk === "Medium") return "info-box warning-box";
  return "info-box danger-box";
}



/* ════════════════════════════════════════════ */
function Predict({ type }) {
  const { user } = useContext(AuthContext);

  /* ── State ── */
  const [userData, setUserData] = useState({ loan_type: "", income: "", credit_score: "", loan_amount: "", missed_payments: "" });
  const [bankData, setBankData] = useState({ loan_amount: "", income: "", Credit_Score: "", LTV: "", dtir1: "" });

  // Draft loading removed as user prefers manual entry

  // Goal Matching Logic: Pre-select loan type based on user goals
  useEffect(() => {
    if (user) {
      const savedGoals = localStorage.getItem(`creditpath_goals_${user.email}`);
      if (savedGoals) {
        const goals = JSON.parse(savedGoals);
        if (goals.length > 0) {
          const latestGoal = goals[goals.length - 1].title.toLowerCase();
          // Map goal keywords to loan categories
          const mapping = {
            'home': 'home', 'house': 'home', 'flat': 'home', 'apartment': 'home',
            'car': 'car', 'vehicle': 'car', 'bike': 'car',
            'study': 'edu', 'education': 'edu', 'college': 'edu', 'university': 'edu',
            'gold': 'gold', 'jewel': 'gold', 'ornament': 'gold',
            'food': 'food', 'restaurant': 'restaurant', 'cafe': 'restaurant',
            'mall': 'mall', 'shop': 'mall', 'store': 'mall',
            'farm': 'agri', 'agriculture': 'agri', 'crop': 'agri',
            'medical': 'medical', 'hospital': 'medical', 'health': 'medical',
            'business': 'msme', 'startup': 'msme', 'msme': 'msme',
            'wedding': 'wedding', 'marriage': 'wedding',
            'travel': 'travel', 'vacation': 'travel', 'trip': 'travel',
            'solar': 'solar', 'energy': 'solar', 'green': 'solar'
          };
          for (const [kw, type] of Object.entries(mapping)) {
            if (latestGoal.includes(kw)) {
              setUserData(prev => ({ ...prev, loan_type: type }));
              break;
            }
          }
        }
      }
    }
  }, [user]);

  const [userResult, setUserResult] = useState(null);
  const [bankResult, setBankResult] = useState(null);
  const [loadingUser, setLoadingUser] = useState(false);
  const [loadingBank, setLoadingBank] = useState(false);
  const [userError, setUserError] = useState(null);
  const [bankError, setBankError] = useState(null);
  const [userValidationErrors, setUserValidationErrors] = useState({});
  const [bankValidationErrors, setBankValidationErrors] = useState({});

  /* ── Clear results and inputs on tab switch ── */
  useEffect(() => {
    setUserResult(null);
    setBankResult(null);
    setUserError(null);
    setBankError(null);
    setUserValidationErrors({});
    setBankValidationErrors({});
    
    // Blank the forms out as per manual entry request (keeping loan_type from goals)
    setUserData(prev => ({ loan_type: prev.loan_type, income: "", credit_score: "", loan_amount: "", missed_payments: "" }));
    setBankData({ loan_amount: "", income: "", Credit_Score: "", LTV: "", dtir1: "" });
  }, [type]);

  /* ── Draft Saving Removed ── */

  /* ── Handlers ── */
  const handleUserChange = (e) => setUserData({ ...userData, [e.target.name]: e.target.value });
  const handleBankChange = (e) => setBankData({ ...bankData, [e.target.name]: e.target.value });

  /* ── Validation ── */
  const validateUserForm = () => {
    let errors = {};
    if (!userData.loan_type) errors.loan_type = "Loan type is required";
    if (!userData.income || userData.income <= 0) errors.income = "Please enter a valid income";
    if (!userData.credit_score || userData.credit_score < 300 || userData.credit_score > 900) errors.credit_score = "Credit score must be between 300 - 900";
    if (!userData.loan_amount || userData.loan_amount <= 0) errors.loan_amount = "Please enter a valid amount";

    if (userData.loan_amount && userData.income && Number(userData.loan_amount) > Number(userData.income) * 5) {
      errors.loan_amount = "⚠️ Loan amount is >5x your annual income. This usually leads to high risk.";
    }

    if (userData.missed_payments === "" || userData.missed_payments < 0) errors.missed_payments = "Must be 0 or greater";
    setUserValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const validateBankForm = () => {
    let errors = {};
    if (!bankData.loan_amount || bankData.loan_amount <= 0) errors.loan_amount = "Please enter a valid amount";
    if (!bankData.income || bankData.income <= 0) errors.income = "Please enter a valid income";
    if (!bankData.Credit_Score || bankData.Credit_Score < 300 || bankData.Credit_Score > 900) errors.Credit_Score = "Must be between 300 - 900";
    if (!bankData.LTV || bankData.LTV < 0 || bankData.LTV > 100) errors.LTV = "LTV must be 0-100%";
    if (!bankData.dtir1 || bankData.dtir1 < 0 || bankData.dtir1 > 100) errors.dtir1 = "DTI must be 0-100%";
    setBankValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  /* ── Auto-Prediction Removed (Now relies purely on Button Click) ── */

  /* ── User predict ── */
  const predictUser = async () => {
    if (!user) {
      setUserError("🔒 Please log in to your account to perform predictions.");
      return;
    }
    if (!validateUserForm()) {
      setUserError("⚠️ Please fill out all required fields correctly.");
      return;
    }
    setLoadingUser(true); setUserError(null); setUserResult(null);
    try {
      const payload = {
        ...userData,
        income: Number(userData.income),
        credit_score: Number(userData.credit_score),
        loan_amount: Number(userData.loan_amount),
        missed_payments: Number(userData.missed_payments),
      };
      const res = await axios.post("http://127.0.0.1:8000/user-risk", payload);
      setUserResult(res.data);

      if (user) {
        const historyKey = `creditpath_history_${user.email}`;
        const prev = JSON.parse(localStorage.getItem(historyKey) || '[]');
        prev.push({
          type: 'user',
          loan_type: payload.loan_type,
          timestamp: new Date().toISOString(),
          amount: payload.loan_amount,
          risk: res.data.risk_level
        });
        localStorage.setItem(historyKey, JSON.stringify(prev));
      }

    } catch {
      setUserError("⚠️ Cannot reach backend. Make sure FastAPI is running on port 8000.");
    } finally {
      setLoadingUser(false);
    }
  };

  /* ── Bank predict ── */
  const predictBank = async () => {
    if (!user) {
      setBankError("🔒 Please log in to your account to perform predictions.");
      return;
    }
    if (!validateBankForm()) {
      setBankError("⚠️ Please fill out all required fields correctly.");
      return;
    }
    setLoadingBank(true); setBankError(null); setBankResult(null);
    try {
      const payload = {
        loan_amount: Number(bankData.loan_amount),
        income: Number(bankData.income),
        Credit_Score: Number(bankData.Credit_Score),
        LTV: Number(bankData.LTV),
        dtir1: Number(bankData.dtir1),
      };
      const res = await axios.post("http://127.0.0.1:8000/bank-risk", payload);
      setBankResult(res.data);

      if (user) {
        const historyKey = `creditpath_history_${user.email}`;
        const prev = JSON.parse(localStorage.getItem(historyKey) || '[]');
        prev.push({
          type: 'bank',
          loan_type: 'commercial', // Defaults for bank analysis
          timestamp: new Date().toISOString(),
          amount: payload.loan_amount,
          risk: res.data.loan_status
        });
        localStorage.setItem(historyKey, JSON.stringify(prev));
      }

    } catch {
      setBankError("⚠️ Cannot reach backend. Make sure FastAPI is running on port 8000.");
    } finally {
      setLoadingBank(false);
    }
  };

  /* ── Derived values ── */
  const approvalRate = userResult
    ? userResult.risk_level === "Low" ? 90 : userResult.risk_level === "Medium" ? 60 : 30
    : 0;

  const rateColor = approvalRate >= 70
    ? "var(--success)" : approvalRate >= 50
      ? "var(--warning)" : "var(--danger)";

  const getExpectedLoss = () => {
    if (!bankResult) return 0;
    return Math.round(Number(bankData.loan_amount) * bankResult.default_probability * 0.45);
  };

  const getAnalysisPriority = () => {
    if (!bankResult) return "Low";
    const prob = bankResult.default_probability * 100;
    return prob > 20 ? "High" : prob > 10 ? "Medium" : "Low";
  };



  /* ── XAI Logic ── */
  const getXAI = () => {
    if (!userData.credit_score || !userResult) return [];
    let reasons = [];
    const cs = Number(userData.credit_score);
    const inc = Number(userData.income);
    const amt = Number(userData.loan_amount);

    if (cs >= 750) reasons.push({ label: "High credit score", impact: "+25%", pos: true });
    else if (cs < 600) reasons.push({ label: "Low credit score", impact: "-30%", pos: false });
    else reasons.push({ label: "Average credit score", impact: "+5%", pos: true });

    if (inc > amt * 0.5) reasons.push({ label: "Strong income coverage", impact: "+15%", pos: true });
    else reasons.push({ label: "Income to loan ratio", impact: "-10%", pos: false });

    if (userData.missed_payments > 0) reasons.push({ label: "Missed payments history", impact: "-15%", pos: false });
    else reasons.push({ label: "Perfect repayment history", impact: "+10%", pos: true });

    return reasons;
  };





  /* ── Optimization Logic ── */
  const getDTIAdvice = () => {
    if (!userResult || !userData.income) return null;
    const incomeMonth = Number(userData.income) / 12;
    if (userResult.risk_level === 'Low') return null;

    const targetMonthlyDebt = incomeMonth * 0.3;
    const currentMonthlyDebt = incomeMonth * (userResult.risk_level === 'High' ? 0.6 : 0.45);
    const reductionNeeded = Math.max(0, currentMonthlyDebt - targetMonthlyDebt);

    return {
      reduction: reductionNeeded.toFixed(0),
      targetDTI: '30%',
      urgency: userResult.risk_level === 'High' ? 'Critical' : 'Moderate'
    };
  };

  /* ════════════════ RENDER ════════════════ */
  return (
    <div className="predict-page">
      <div className="predict-header">
        <h1 style={{ fontFamily: "'Outfit', sans-serif", fontWeight: 800 }}>{type === "user" ? "👤 User Risk Prediction" : "🏦 Bank Risk Analysis"}</h1>
        <p>
          {type === "user"
            ? "Enter your financial details to get your loan eligibility and personalised tips."
            : "Enter borrower details to estimate default probability."}
        </p>
      </div>


      {type === "user" && (
        <div className="predict-card user-predict">
          <div className="predict-card-title">
            <div className="title-icon">👤</div>
            <h2 style={{ fontFamily: "'Outfit', sans-serif", fontWeight: 700, letterSpacing: '-0.02em' }}>Borrower Details</h2>
          </div>
          <div className="form-grid">
            <div className="form-group">
              <label>Loan Type</label>
              <select className={`form-select ${userValidationErrors.loan_type ? 'input-error' : ''}`} name="loan_type" value={userData.loan_type} onChange={handleUserChange}>
                <option value="">Select...</option>
                <option value="home">🏠 Home Loan</option>
                <option value="car">🚗 Car Loan</option>
                <option value="personal">💳 Personal Loan</option>
                <option value="gold">🏆 Gold Loan</option>
                <option value="food">🍔 Food Business</option>
                <option value="mall">🏬 Shopping Mall</option>
                <option value="restaurant">🍱 Restaurant</option>
                <option value="agri">🚜 Agriculture Loan</option>
                <option value="edu">🎓 Education Loan</option>
                <option value="medical">🏥 Medical Loan</option>
                <option value="commercial">🏢 Commercial Real Estate</option>
                <option value="msme">💻 MSME Business Loan</option>
                <option value="wedding">💍 Wedding Loan</option>
                <option value="travel">🛫 Travel Loan</option>
                <option value="solar">🔋 Solar/Green Energy</option>
              </select>
              {userValidationErrors.loan_type && <span className="error-text">{userValidationErrors.loan_type}</span>}
            </div>
            <div className="form-group">
              <label>Annual Income (₹)</label>
              <input className={`form-input ${userValidationErrors.income ? 'input-error' : ''}`} type="number" name="income" value={userData.income} onChange={handleUserChange} />
              {userValidationErrors.income && <span className="error-text">{userValidationErrors.income}</span>}
            </div>
            <div className="form-group">
              <label>Credit Score</label>
              <input className={`form-input ${userValidationErrors.credit_score ? 'input-error' : ''}`} type="number" name="credit_score" value={userData.credit_score} onChange={handleUserChange} />
              {userValidationErrors.credit_score && <span className="error-text">{userValidationErrors.credit_score}</span>}
            </div>
            <div className="form-group">
              <label>Loan Amount (₹)</label>
              <input className={`form-input ${userValidationErrors.loan_amount ? 'input-error' : ''}`} type="number" name="loan_amount" value={userData.loan_amount} onChange={handleUserChange} />
              {userValidationErrors.loan_amount && <span className="error-text">{userValidationErrors.loan_amount}</span>}
            </div>
            <div className="form-group" style={{ gridColumn: "1 / -1" }}>
              <label>Missed Payments <Tooltip text="The number of times you've missed loan or card payments in the last 12 months." /></label>
              <input className={`form-input ${userValidationErrors.missed_payments ? 'input-error' : ''}`} type="number" name="missed_payments" value={userData.missed_payments} onChange={handleUserChange} />
              {userValidationErrors.missed_payments && <span className="error-text">{userValidationErrors.missed_payments}</span>}
            </div>
          </div>
          <button className="btn-primary" onClick={predictUser} disabled={loadingUser}>
            {loadingUser ? <><span className="spinner" /> Analysing…</> : "📊 Predict Risk"}
          </button>

          {userError && <div className="error-box">{userError}</div>}

          {userResult && (
            <div className="results-section">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 }}>
                <div className="results-title" style={{ margin: 0 }}>🎯 Risk Assessment Results</div>
                <button className="btn-primary" style={{ padding: '8px 16px', width: 'auto', background: 'var(--bg-card)', color: 'var(--accent)', border: '1px solid var(--border)' }} onClick={() => window.print()}>
                  📄 Download PDF
                </button>
              </div>

              <div className="metric-cards">
                <div className="metric-card"><div className="metric-label">RISK LEVEL</div><RiskBadge level={userResult.risk_level} /></div>
                <div className="metric-card"><div className="metric-label">APPROVAL CHANCE</div><div className="metric-value" style={{ color: rateColor }}>{approvalRate}%</div></div>
                <div className="metric-card"><div className="metric-label">CREDIT SCORE</div><div className="metric-value" style={{ color: "var(--accent)" }}>{userData.credit_score}</div></div>
              </div>

              {/* AI Insights & Tips Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px', marginTop: 16 }}>

                {/* Card 1: AI Agent Recommendation */}
                <div className="info-box" style={{ background: "var(--bg-secondary)", borderColor: "var(--border)", margin: 0 }}>
                  <h4 style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12 }}>🤖 AI Recommendations</h4>
                  {userResult.recommendation_summary && (
                    <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                      {userResult.recommendation_summary.map((line, index) => <li key={index} style={{ marginBottom: 8, fontSize: '0.9rem', color: 'var(--text-secondary)' }}>{line}</li>)}
                    </ul>
                  )}
                </div>

                {/* Card 2: Risk Factor Impact */}
                <div className="info-box" style={{ background: "var(--bg-secondary)", borderColor: "var(--border)", margin: 0 }}>
                  <h4 style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12 }}>⚡ Risk Factor Impact (XAI)</h4>
                  <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                    {getXAI().map((r, i) => (
                      <li key={i} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, fontSize: '0.9rem' }}>
                        <span>{r.pos ? "✔ " : "❌ "}{r.label}</span>
                        <span style={{ color: r.pos ? 'var(--success)' : 'var(--danger)', fontWeight: 600 }}>{r.impact}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Card 3: Actionable Financial Tips */}
                <div className="info-box" style={{ background: "var(--bg-secondary)", borderColor: "var(--border)", margin: 0 }}>
                  <h4 style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12 }}>💡 Actionable Financial Tips</h4>
                  {userResult.tips && userResult.tips.length > 0 ? (
                    <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                      {userResult.tips.map((tip, index) => (
                        <span key={index} style={{ background: "var(--bg-card)", padding: "6px 12px", borderRadius: "20px", fontSize: "0.85rem", border: "1px solid var(--border)", color: "var(--text-primary)", display: "flex", alignItems: "center", gap: 6 }}>
                          <span style={{ color: "var(--success)" }}>✓</span> {tip}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </div>

                {/* Card 4: General Guidance */}
                <div className="info-box" style={{ background: "var(--bg-secondary)", borderColor: "var(--border)", margin: 0 }}>
                  <h4 style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12 }}>📘 General Guidance</h4>
                  <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                    <li style={{ marginBottom: 8, fontSize: '0.9rem' }}>• Maintain timely EMI payments to build history.</li>
                    <li style={{ marginBottom: 8, fontSize: '0.9rem' }}>• Keep credit utilization below 30% of your limit.</li>
                    <li style={{ marginBottom: 8, fontSize: '0.9rem' }}>• Avoid multiple credit inquiries in a short duration.</li>
                  </ul>
                </div>
              </div>

              <div className={infoClass(userResult.risk_level)} style={{ marginTop: 16 }}>
                <h4 style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>✅ Final Suggestion</h4>
                <p style={{ fontSize: "0.95rem", margin: 0, lineHeight: 1.5 }}>
                  {userResult.risk_level === "Low"
                    ? "You can proceed with the loan application confidently. Your profile shows strong repayment capacity."
                    : userResult.risk_level === "Medium"
                      ? "Consider improving your credit score slightly or clearing small debts before applying to get better interest rates."
                      : "We recommend waiting for 6 months and improving your financial health before applying for this loan."}
                </p>
              </div>

              {/* ⚡ Optimization Strategy */}
              {getDTIAdvice() && (
                <div style={{ marginTop: 20 }}>
                  <div className="results-title" style={{ marginBottom: 16 }}>⚡ Optimization Strategy</div>
                  <div className="info-box danger-box" style={{ background: 'rgba(239, 68, 68, 0.05)', border: '1px solid rgba(239, 68, 68, 0.2)', marginBottom: 30 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                      <h4 style={{ margin: 0, color: 'var(--danger)' }}>Reduce Debt to Qualify</h4>
                      <span style={{ fontSize: '0.75rem', padding: '4px 10px', borderRadius: '20px', background: 'var(--danger)', color: 'white', fontWeight: 600 }}>
                        {getDTIAdvice().urgency} Priority
                      </span>
                    </div>
                    <p style={{ fontSize: '0.9rem', marginBottom: 16 }}>
                      We recommend a monthly reduction of <strong style={{ color: 'var(--danger)' }}>₹{Number(getDTIAdvice().reduction).toLocaleString()}</strong> to reach a healthy <strong style={{ color: 'var(--success)' }}>{getDTIAdvice().targetDTI}</strong> DTI ratio.
                    </p>
                  </div>
                </div>
              )}

              {/* 2x2 Charts Grid matching the reference picture */}
              <div className="charts-grid" style={{ marginTop: 24, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px' }}>
                {/* 1. Loan Approval Pie (Top Left) */}
                <div className="chart-card">
                  <Plot
                    data={[{
                      values: [approvalRate, 100 - approvalRate],
                      labels: ["Approved", "Remaining"],
                      type: "pie", hole: 0.6,
                      marker: { colors: ["#10b981", "#ef4444"] },
                      textinfo: "percent",
                      hoverinfo: "label+value",
                      insidetextorientation: "radial"
                    }]}
                    layout={{ ...PLOT_LAYOUT, title: { text: "Loan Approval", font: { size: 13, weight: 'bold' } }, showlegend: true, legend: { orientation: "v", x: 1, y: 0.5 } }}
                    style={{ width: "100%", height: "230px" }}
                    useResizeHandler config={{ displayModeBar: false }}
                  />
                </div>

                {/* 2. Credit Score Comparison Bar (Top Right) */}
                <div className="chart-card">
                  <Plot
                    data={[{
                      x: ["Target", "Your Score"],
                      y: [750, Number(userData.credit_score)],
                      type: "bar",
                      marker: { color: ["#f59e0b", "#3b82f6"] }
                    }]}
                    layout={{ ...PLOT_LAYOUT, title: { text: "Credit Score Comparison", font: { size: 13, weight: 'bold' } }, xaxis: { tickfont: { size: 11 } }, yaxis: { tickfont: { size: 10 } } }}
                    style={{ width: "100%", height: "230px" }}
                    useResizeHandler config={{ displayModeBar: false }}
                  />
                </div>

                {/* 3. User Risk Level Pie (Bottom Left) */}
                <div className="chart-card">
                  <Plot
                    data={[{
                      values: [Number(userData.credit_score), 900 - Number(userData.credit_score)],
                      labels: ["Your Score", "Improvement Area"],
                      type: "pie", hole: 0.6,
                      marker: { colors: ["#3b82f6", "#e2e8f0"] },
                      textinfo: "percent",
                      hoverinfo: "label+value",
                      insidetextorientation: "radial"
                    }]}
                    layout={{ ...PLOT_LAYOUT, title: { text: `User Risk Level: ${userResult.risk_level}`, font: { size: 13, weight: 'bold' } }, showlegend: true, legend: { orientation: "v", x: 1, y: 0.5 } }}
                    style={{ width: "100%", height: "230px" }}
                    useResizeHandler config={{ displayModeBar: false }}
                  />
                </div>

                {/* 4. User Financial Overview Bar (Bottom Right) */}
                <div className="chart-card">
                  <Plot
                    data={[{
                      x: ["Income", "Loan Amt"],
                      y: [Number(userData.income), Number(userData.loan_amount)],
                      type: "bar",
                      marker: { color: ["#10b981", "#3b82f6"] }
                    }]}
                    layout={{ ...PLOT_LAYOUT, title: { text: "User Financial Overview", font: { size: 13, weight: 'bold' } }, xaxis: { tickfont: { size: 11 } }, yaxis: { tickfont: { size: 10 } } }}
                    style={{ width: "100%", height: "230px" }}
                    useResizeHandler config={{ displayModeBar: false }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {type === "bank" && (
        <div className="predict-card bank-predict">
          <div className="predict-card-title">
            <div className="title-icon">🏦</div>
            <h2 style={{ fontFamily: "'Outfit', sans-serif", fontWeight: 700, letterSpacing: '-0.02em' }}>Bank Risk Analysis</h2>
          </div>
          <div className="form-grid">
            <div className="form-group">
              <label>Loan Amount (₹)</label>
              <input className={`form-input ${bankValidationErrors.loan_amount ? 'input-error' : ''}`} type="number" name="loan_amount" value={bankData.loan_amount} onChange={handleBankChange} />
              {bankValidationErrors.loan_amount && <span className="error-text">{bankValidationErrors.loan_amount}</span>}
            </div>
            <div className="form-group">
              <label>Borrower Income (₹)</label>
              <input className={`form-input ${bankValidationErrors.income ? 'input-error' : ''}`} type="number" name="income" value={bankData.income} onChange={handleBankChange} />
              {bankValidationErrors.income && <span className="error-text">{bankValidationErrors.income}</span>}
            </div>
            <div className="form-group">
              <label>Credit Score (300-900)</label>
              <input className={`form-input ${bankValidationErrors.Credit_Score ? 'input-error' : ''}`} type="number" name="Credit_Score" value={bankData.Credit_Score} onChange={handleBankChange} />
              {bankValidationErrors.Credit_Score && <span className="error-text">{bankValidationErrors.Credit_Score}</span>}
            </div>
            <div className="form-group">
              <label>Loan-to-Value Ratio (LTV %)</label>
              <input className={`form-input ${bankValidationErrors.LTV ? 'input-error' : ''}`} type="number" name="LTV" value={bankData.LTV} onChange={handleBankChange} />
              {bankValidationErrors.LTV && <span className="error-text">{bankValidationErrors.LTV}</span>}
            </div>
            <div className="form-group">
              <label>Debt-to-Income-Ratio (DTI %)</label>
              <input className={`form-input ${bankValidationErrors.dtir1 ? 'input-error' : ''}`} type="number" name="dtir1" value={bankData.dtir1} onChange={handleBankChange} />
              {bankValidationErrors.dtir1 && <span className="error-text">{bankValidationErrors.dtir1}</span>}
            </div>
          </div>
          <button className="btn-primary" onClick={predictBank} disabled={loadingBank}>
            {loadingBank ? <><span className="spinner" /> Analysing…</> : "📊 Analyse"}
          </button>

          {bankError && <div className="error-box">{bankError}</div>}

          {bankResult && (
            <div className="results-section">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
                <div className="results-title" style={{ margin: 0 }}>📊 Financial Risk Scorecard</div>
                <button className="btn-primary" style={{ padding: '8px 16px', width: 'auto', background: 'var(--bg-card)', color: 'var(--accent)', border: '1px solid var(--border)' }} onClick={() => window.print()}>
                  📄 Download Report
                </button>
              </div>

              <div className="metric-cards">
                <div className="metric-card"><div className="metric-label">LOAN STATUS</div><RiskBadge level={bankResult.loan_status === 'Approved' ? 'Low' : 'High'} /></div>
                <div className="metric-card"><div className="metric-label">DEFAULT PROB.</div><div className="metric-value">{(bankResult.default_probability * 100).toFixed(1)}%</div></div>
                <div className="metric-card"><div className="metric-label">EXPECTED LOSS</div><div className="metric-value" style={{ color: 'var(--danger)' }}>₹{getExpectedLoss().toLocaleString('en-IN')}</div></div>
                <div className="metric-card"><div className="metric-label">PRIORITY</div><div className="metric-value" style={{ color: getAnalysisPriority() === 'High' ? 'var(--danger)' : 'var(--accent)' }}>{getAnalysisPriority()}</div></div>
              </div>

              {/* AI Decision Plan & Banking Guidelines */}
              <div style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 1fr)', gap: '16px', marginTop: 16 }}>
                <div className="info-box" style={{ background: "var(--bg-secondary)", borderColor: "var(--border)", margin: 0 }}>
                  <h4 style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12 }}>🤖 AI Decision Plan</h4>
                  <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                    <li style={{ marginBottom: 6 }}>• Recovery Channel: {bankResult.loan_status === 'Approved' ? "Email + SMS" : "Collection + Legal Notice"}</li>
                    <li style={{ marginBottom: 6 }}>• Follow-up Schedule: {bankResult.loan_status === 'Approved' ? "15 days" : "Immediate"}</li>
                    <li style={{ marginBottom: 6 }}>• Final Suggestion: {bankResult.loan_status === 'Approved' ? "Approve with minimal monitoring." : "Reject immediately due to high default risk."}</li>
                  </ul>
                </div>

                <div className="info-box" style={{ background: "var(--bg-secondary)", borderColor: "var(--border)", margin: 0 }}>
                  <h4 style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12 }}>🏦 Standard Banking Guidelines</h4>
                  <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                    <li style={{ marginBottom: 6 }}>• Check the borrower's identity and income proofs one more time before giving the loan.</li>
                    <li style={{ marginBottom: 6 }}>• Make sure the loan amount makes sense compared to the value of their property ({bankData.LTV ? `LTV ${bankData.LTV}%` : 'LTV 20%'}).</li>
                    <li style={{ marginBottom: 6 }}>• Check if the borrower already has too many other loans to pay off.</li>
                  </ul>
                </div>
              </div>

              {/* 2x2 Bank Charts Grid matching the reference picture */}
              <div className="charts-grid" style={{ marginTop: 24, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px' }}>

                {/* 1. Risk Distribution Pie (Top Left) */}
                <div className="chart-card">
                  <Plot
                    data={[{
                      values: [100 - (bankResult.default_probability * 100), bankResult.default_probability * 100],
                      labels: ["Safe", "Default Risk"],
                      type: "pie", hole: 0.6,
                      marker: { colors: ["#3b82f6", "#ef4444"] },
                      textinfo: "percent",
                      hoverinfo: "label+value",
                      insidetextorientation: "radial"
                    }]}
                    layout={{ ...PLOT_LAYOUT, title: { text: "Risk Distribution", font: { size: 13, weight: 'bold' } }, showlegend: true, legend: { orientation: "v", x: 1, y: 0.5 } }}
                    style={{ width: "100%", height: "230px" }}
                    useResizeHandler config={{ displayModeBar: false }}
                  />
                </div>

                {/* 2. Default Probability Bar (Top Right) */}
                <div className="chart-card">
                  <Plot
                    data={[{
                      x: ["Safe", "Default Prob"],
                      y: [100 - (bankResult.default_probability * 100), bankResult.default_probability * 100],
                      type: "bar",
                      marker: { color: ["#3b82f6", "#f59e0b"] }
                    }]}
                    layout={{ ...PLOT_LAYOUT, title: { text: "Default Probability", font: { size: 13, weight: 'bold' } }, margin: { t: 40, r: 20, b: 30, l: 40 }, xaxis: { showticklabels: true, tickfont: { size: 10 } }, yaxis: { tickfont: { size: 10 } } }}
                    style={{ width: "100%", height: "230px" }}
                    useResizeHandler config={{ displayModeBar: false }}
                  />
                </div>

                {/* 3. Value at Risk Bar (Bottom Left) */}
                <div className="chart-card">
                  <Plot
                    data={[{
                      x: ["Expected Loss", "Loan Amount"],
                      y: [getExpectedLoss(), Number(bankData.loan_amount) || 0],
                      type: "bar",
                      marker: { color: ["#10b981", "#3b82f6"] }
                    }]}
                    layout={{ ...PLOT_LAYOUT, title: { text: "Value at Risk (₹)", font: { size: 13, weight: 'bold' } }, margin: { t: 40, r: 20, b: 30, l: 40 }, xaxis: { showticklabels: true, tickfont: { size: 10 } }, yaxis: { tickfont: { size: 10 } } }}
                    style={{ width: "100%", height: "230px" }}
                    useResizeHandler config={{ displayModeBar: false }}
                  />
                </div>

                {/* 4. DTI Comparison Bar (Bottom Right) */}
                <div className="chart-card">
                  <Plot
                    data={[{
                      x: ["DTI", "Max Threshold"],
                      y: [Number(bankData.dtir1) || 0, 40],
                      type: "bar",
                      marker: { color: ["#3b82f6", "#f59e0b"] }
                    }]}
                    layout={{ ...PLOT_LAYOUT, title: { text: "DTI Comparison (%)", font: { size: 13, weight: 'bold' } }, margin: { t: 40, r: 20, b: 30, l: 40 }, xaxis: { showticklabels: true, tickfont: { size: 10 } }, yaxis: { tickfont: { size: 10 } } }}
                    style={{ width: "100%", height: "230px" }}
                    useResizeHandler config={{ displayModeBar: false }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default Predict;