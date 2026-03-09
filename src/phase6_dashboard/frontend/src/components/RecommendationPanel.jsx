export default function RecommendationPanel({ rec }) {
  if (!rec) return null;
  const tierClass = rec.risk_tier.replace(' ', '-');

  return (
    <div style={{ animation: 'slideUp .4s ease both' }}>
      {/* Decision header */}
      <div style={{ display:'flex', alignItems:'center', gap:14, marginBottom:20, flexWrap:'wrap' }}>
        <span className={`decision-badge decision-${rec.decision}`}>
          {rec.decision.replace(/_/g, ' ')}
        </span>
        <span className={`tier-badge ${tierClass}`}>● {rec.risk_tier} Risk</span>
        <span style={{ color:'#64748b', fontSize:'.875rem' }}>
          {rec.p_default * 100 < 10 ? '<' : ''}{(rec.p_default * 100).toFixed(1)}% default probability
        </span>
      </div>

      {/* Key metrics */}
      <div className="rec-grid">
        <div className="rec-metric">
          <div className="rec-metric-label">Risk Score</div>
          <div className="rec-metric-value" style={{ color: rec.risk_tier === 'Low' ? '#16a34a' : rec.risk_tier === 'Very High' ? '#dc2626' : '#0f2744' }}>
            {rec.risk_score}<span style={{ fontSize:'.9rem', fontWeight:400, color:'#94a3b8' }}>/1000</span>
          </div>
        </div>
        <div className="rec-metric">
          <div className="rec-metric-label">Recommended Rate</div>
          <div className="rec-metric-value">{rec.interest_rates.recommended}%</div>
          <div className="rec-metric-sub">{rec.interest_rates.minimum}% – {rec.interest_rates.maximum}%</div>
        </div>
        <div className="rec-metric">
          <div className="rec-metric-label">Max Loan Amount</div>
          <div className="rec-metric-value">
            ${rec.max_loan_amount?.toLocaleString('en-US', { maximumFractionDigits:0 })}
          </div>
        </div>
      </div>

      {/* Explanation */}
      {rec.explanation && (
        <div className="rec-section">
          <div className="rec-explanation">{rec.explanation}</div>
        </div>
      )}

      {/* Conditions */}
      {rec.conditions?.length > 0 && (
        <div className="rec-section">
          <h4>Conditions</h4>
          <ul className="rec-list">
            {rec.conditions.map((c, i) => <li key={i}>{c}</li>)}
          </ul>
        </div>
      )}

      {/* Tips */}
      {rec.improvement_tips?.length > 0 && (
        <div className="rec-section">
          <h4>Improvement Tips</h4>
          <ul className="rec-list">
            {rec.improvement_tips.map((t, i) => <li key={i}>{t}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}
