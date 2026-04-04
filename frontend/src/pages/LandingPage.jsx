export default function LandingPage({ onSelectApplicant, onSelectBank }) {
  const styles = `
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #070e1a; --surface: #0c1828;
      --border: rgba(80,140,255,0.1); --border-active: rgba(80,140,255,0.25);
      --blue: #3b82f6; --blue-dim: #1d4ed8;
      --gold: #c9903e; --gold-light: #e8b86d;
      --text: #dde5f5; --text-2: #6b84a8; --text-3: #2e4265;
      --code-bg: #0a1525;
    }
    body { font-family:'DM Sans',sans-serif; background:var(--bg); color:var(--text); min-height:100vh; overflow-x:hidden; }
    .cp-page { min-height:100vh; position:relative; overflow:hidden; }
    .cp-page::before {
      content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
      background:
        radial-gradient(ellipse 60% 50% at 10% 0%,rgba(29,78,216,0.18) 0%,transparent 60%),
        radial-gradient(ellipse 40% 35% at 90% 80%,rgba(20,55,130,0.13) 0%,transparent 55%);
    }
    .cp-grid {
      position:fixed; inset:0; z-index:0; pointer-events:none;
      background-image:
        linear-gradient(rgba(59,130,246,0.04) 1px,transparent 1px),
        linear-gradient(90deg,rgba(59,130,246,0.04) 1px,transparent 1px);
      background-size:64px 64px;
    }
    .cp-nav {
      position:fixed; top:0; left:0; right:0; z-index:100;
      height:60px; padding:0 40px;
      display:flex; align-items:center; justify-content:space-between;
      background:rgba(7,14,26,0.88); backdrop-filter:blur(20px);
      border-bottom:1px solid var(--border);
    }
    .cp-nav-logo {
      display:flex; align-items:center; gap:10px;
      font-family:'Georgia',serif; font-size:1.1rem; font-weight:700; color:var(--text);
    }
    .cp-nav-icon {
      width:30px; height:30px;
      background:linear-gradient(135deg,var(--blue),var(--blue-dim));
      border-radius:8px; display:flex; align-items:center; justify-content:center;
      font-size:0.85rem; box-shadow:0 0 18px rgba(59,130,246,0.4);
    }
    .cp-nav-tag {
      font-size:0.65rem; font-weight:600; letter-spacing:0.14em; text-transform:uppercase;
      color:var(--gold-light); background:rgba(201,144,62,0.1);
      border:1px solid rgba(201,144,62,0.22); padding:3px 10px; border-radius:20px;
    }
    .cp-content { position:relative; z-index:1; max-width:1100px; margin:0 auto; padding:110px 32px 80px; }
    .cp-hero { text-align:center; margin-bottom:60px; animation:fadeUp 0.6s ease both; }
    .cp-eyebrow {
      display:inline-flex; align-items:center; gap:10px;
      font-size:0.7rem; font-weight:600; letter-spacing:0.18em; text-transform:uppercase;
      color:var(--blue); margin-bottom:18px;
    }
    .cp-eyebrow::before,.cp-eyebrow::after { content:''; width:30px; height:1px; background:var(--blue); opacity:0.45; }
    .cp-title {
      font-family:'Georgia',serif; font-size:clamp(2.4rem,5.5vw,3.6rem);
      font-weight:700; line-height:1.1; letter-spacing:-0.02em; margin-bottom:16px;
      background:linear-gradient(150deg,#dde5f5 25%,#6b84a8 100%);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    }
    .cp-title span {
      background:linear-gradient(135deg,var(--gold-light),var(--gold));
      -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    }
    .cp-subtitle { font-size:1rem; font-weight:300; color:var(--text-2); line-height:1.75; max-width:460px; margin:0 auto; }
    .cp-cards { display:grid; grid-template-columns:1fr 1fr; gap:24px; margin-bottom:64px; animation:fadeUp 0.6s 0.1s ease both; }
    .cp-card {
      background:var(--surface); border:1px solid var(--border); border-radius:18px; padding:44px 36px;
      text-align:center; transition:border-color 0.3s,transform 0.3s,box-shadow 0.3s;
      position:relative; overflow:hidden;
    }
    .cp-card::before {
      content:''; position:absolute; top:0; left:0; right:0; height:1px;
      background:linear-gradient(90deg,transparent,rgba(59,130,246,0.35),transparent);
      opacity:0; transition:opacity 0.3s;
    }
    .cp-card:hover { border-color:var(--border-active); transform:translateY(-4px); box-shadow:0 20px 50px rgba(0,0,0,0.35),0 0 30px rgba(59,130,246,0.08); }
    .cp-card:hover::before { opacity:1; }
    .cp-card-icon { width:64px; height:64px; margin:0 auto 24px; border-radius:16px; display:flex; align-items:center; justify-content:center; font-size:1.8rem; }
    .cp-card-icon.bank { background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.2); box-shadow:0 0 24px rgba(59,130,246,0.1); }
    .cp-card-icon.applicant { background:rgba(201,144,62,0.1); border:1px solid rgba(201,144,62,0.2); box-shadow:0 0 24px rgba(201,144,62,0.1); }
    .cp-card-label { font-size:0.65rem; font-weight:600; letter-spacing:0.16em; text-transform:uppercase; margin-bottom:10px; }
    .cp-card-label.bank { color:var(--blue); }
    .cp-card-label.applicant { color:var(--gold-light); }
    .cp-card h2 { font-family:'Georgia',serif; font-size:1.65rem; font-weight:700; margin-bottom:12px; color:var(--text); }
    .cp-card p { font-size:0.92rem; font-weight:300; color:var(--text-2); line-height:1.7; margin-bottom:32px; }
    .cp-btn {
      display:inline-flex; align-items:center; gap:8px; padding:13px 28px; border-radius:10px;
      font-weight:600; font-size:0.9rem; transition:all 0.25s; cursor:pointer; border:none; letter-spacing:0.02em;
    }
    .cp-btn.blue { background:linear-gradient(135deg,var(--blue-dim),var(--blue)); color:#fff; box-shadow:0 4px 20px rgba(59,130,246,0.3); }
    .cp-btn.blue:hover { box-shadow:0 6px 28px rgba(59,130,246,0.5); transform:translateY(-1px); }
    .cp-btn.gold { background:linear-gradient(135deg,#a06820,var(--gold)); color:#fff; box-shadow:0 4px 20px rgba(201,144,62,0.3); }
    .cp-btn.gold:hover { box-shadow:0 6px 28px rgba(201,144,62,0.45); transform:translateY(-1px); }
    .cp-features { margin-bottom:48px; animation:fadeUp 0.6s 0.2s ease both; }
    .cp-section-title {
      font-family:'Georgia',serif; font-size:1.25rem; font-weight:600; color:var(--text);
      margin-bottom:20px; display:flex; align-items:center; gap:12px;
    }
    .cp-section-title::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,var(--border),transparent); }
    .cp-feat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }
    .cp-feat {
      background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:22px 18px;
      text-align:center; transition:border-color 0.25s,box-shadow 0.25s;
    }
    .cp-feat:hover { border-color:rgba(59,130,246,0.2); box-shadow:0 8px 24px rgba(0,0,0,0.2); }
    .cp-feat-icon { font-size:1.6rem; margin-bottom:10px; }
    .cp-feat-name { font-weight:600; font-size:0.88rem; color:var(--text); margin-bottom:4px; }
    .cp-feat-desc { font-size:0.78rem; color:var(--text-2); font-weight:300; }
    .cp-footer {
      margin-top:36px; text-align:center; font-size:0.75rem; color:var(--text-3);
      animation:fadeUp 0.6s 0.35s ease both;
      display:flex; align-items:center; justify-content:center; gap:8px;
    }
    .cp-footer a { color:var(--blue); text-decoration:none; opacity:0.7; }
    .cp-footer a:hover { opacity:1; }
    .cp-footer .dot { width:3px; height:3px; background:var(--text-3); border-radius:50%; }
    @keyframes fadeUp { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }
    @media(max-width:700px){
      .cp-cards{grid-template-columns:1fr}
      .cp-feat-grid{grid-template-columns:1fr 1fr}
      .cp-nav{padding:0 20px}
      .cp-content{padding:90px 20px 60px}
      .cp-card{padding:32px 24px}
    }
  `

  return (
    <>
      <style>{styles}</style>
      <div className="cp-page">
        <div className="cp-grid" />

        <nav className="cp-nav">
          <div className="cp-nav-logo">
            <div className="cp-nav-icon">💳</div>
            CreditPath AI
          </div>
          <span className="cp-nav-tag">ML Powered</span>
        </nav>

        <div className="cp-content">
          <div className="cp-hero">
            <div className="cp-eyebrow">Dual Recommendation Engine</div>
            <h1 className="cp-title">
              Smarter Lending,<br /><span>Stronger Decisions</span>
            </h1>
            <p className="cp-subtitle">
              AI-driven insights for banks and applicants — assess risk, check eligibility, and plan your next move.
            </p>
          </div>

          <div className="cp-cards">
            <div className="cp-card">
              <div className="cp-card-icon bank">🏦</div>
              <div className="cp-card-label bank">For Lenders</div>
              <h2>Bank Perspective</h2>
              <p>Risk assessment, recovery strategy, and expected loss analysis for lenders</p>
              <button className="cp-btn blue" onClick={onSelectBank}>Launch Agent →</button>
            </div>

            <div className="cp-card">
              <div className="cp-card-icon applicant">📋</div>
              <div className="cp-card-label applicant">For Applicants</div>
              <h2>Applicant Perspective</h2>
              <p>Eligibility check, improvement suggestions, and reapplication timeline</p>
              <button className="cp-btn gold" onClick={onSelectApplicant}>Check Eligibility →</button>
            </div>
          </div>

          <div className="cp-features">
            <div className="cp-section-title">Powered by ML</div>
            <div className="cp-feat-grid">
              {[
                { icon:'🤖', name:'ML Predictions', desc:'Real model forecasts' },
                { icon:'📊', name:'Risk Analysis',  desc:'Bank perspective' },
                { icon:'✅', name:'Eligibility',    desc:'Applicant perspective' },
                { icon:'⚡', name:'Real-time',      desc:'Instant results' },
              ].map((f, i) => (
                <div key={i} className="cp-feat">
                  <div className="cp-feat-icon">{f.icon}</div>
                  <div className="cp-feat-name">{f.name}</div>
                  <div className="cp-feat-desc">{f.desc}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="cp-footer">
            <span>API</span>
            <a href="http://127.0.0.1:8000" target="_blank" rel="noreferrer">127.0.0.1:8000</a>
            <div className="dot" />
            <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">Swagger Docs</a>
          </div>
        </div>
      </div>
    </>
  )
}