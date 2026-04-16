import React from "react";
import { useNavigate } from "react-router-dom";

const features = [
  {
    icon: "🤖",
    title: "AI-Powered Risk Analysis",
    desc: "LightGBM model trained on real loan data to predict default probability with high accuracy.",
  },
  {
    icon: "📊",
    title: "Interactive Dashboards",
    desc: "Visual charts and graphs to understand your financial risk at a glance.",
  },
  {
    icon: "🏦",
    title: "Dual Perspective",
    desc: "Separate interfaces for borrowers and banks — each tailored to your role.",
  },
  {
    icon: "⚡",
    title: "Instant Predictions",
    desc: "Get results in seconds. Real-time AI analysis with no delays.",
  },
];

function Home() {
  const navigate = useNavigate();

  return (
    <div>
      {/* ── Hero ── */}
      <section className="hero">
        <div className="hero-badge">✦ Powered by LightGBM &amp; FastAPI</div>

        <h1 className="hero-title" style={{ fontFamily: "'Inter', sans-serif" }}>
          Smarter Loan Decisions<br />
          with <span className="gradient-text">AI Intelligence</span>
        </h1>

        <p className="hero-subtitle">
          CreditPath AI analyzes financial data to predict loan risk, protect
          banks from defaults, and help borrowers understand their credit standing.
        </p>

        {/* Role Cards */}
        <div className="role-cards">
          <div className="role-card user-card" onClick={() => navigate("/predict/user")}>
            <div className="role-card-icon">👤</div>
            <h3 style={{ fontFamily: "'Inter', sans-serif" }}>For Borrowers</h3>
            <p>
              Check your loan eligibility, understand your risk level, and get
              personalized tips to improve your credit health.
            </p>
            <button className="role-card-btn">User Prediction →</button>
          </div>

          <div className="role-card bank-card" onClick={() => navigate("/predict/bank")}>
            <div className="role-card-icon">🏦</div>
            <h3 style={{ fontFamily: "'Inter', sans-serif" }}>For Banks</h3>
            <p>
              Analyze borrower default probability, estimate expected loss, and
              get smart recovery channel recommendations.
            </p>
            <button className="role-card-btn">Bank Prediction →</button>
          </div>
        </div>

        {/* Stats */}
        <div className="stats-bar">
          <div className="stat-item">
            <div className="stat-number">95%+</div>
            <div className="stat-label">Model Accuracy</div>
          </div>
          <div className="stat-item">
            <div className="stat-number">2</div>
            <div className="stat-label">Prediction Modes</div>
          </div>
          <div className="stat-item">
            <div className="stat-number">36</div>
            <div className="stat-label">Risk Features</div>
          </div>
          <div className="stat-item">
            <div className="stat-number">&lt;1s</div>
            <div className="stat-label">Prediction Time</div>
          </div>
        </div>
      </section>

      {/* ── Features ── */}
      <section className="features-section">
        <div className="section-header">
          <h2 style={{ fontFamily: "'Inter', sans-serif" }}>Everything you need for smarter credit decisions</h2>
          <p>
            Powered by real financial data and machine learning to deliver highly accurate and actionable credit insights.
          </p>
        </div>
        <div className="features-grid">
          {features.map((f, i) => (
            <div className="feature-card" key={i}
              style={{ animationDelay: `${i * 0.08}s` }}
            >
              <div className="feature-icon">{f.icon}</div>
              <h4>{f.title}</h4>
              <p>{f.desc}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

export default Home;