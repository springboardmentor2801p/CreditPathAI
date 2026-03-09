import { useState } from 'react';
import { getRecommendation } from '../api/creditApi';
import { useHistory } from '../hooks/useHistory';
import RiskForm from '../components/RiskForm';
import RiskScoreGauge from '../components/RiskScoreGauge';
import RecommendationPanel from '../components/RecommendationPanel';

export default function PredictPage() {
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState('');
  const { addEntry } = useHistory();

  const handleSubmit = async (borrower, loan) => {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await getRecommendation(borrower, loan);
      setResult(res.data);
      addEntry({ ...res.data, borrower, loan });
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.response?.data?.error  ||
        'Could not reach the API. Ensure both services are running.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ animation:'fadeIn .35s ease' }}>
      <div className="page-header">
        <h1>Predict Risk</h1>
        <p>Complete the borrower profile and loan details to receive an instant risk assessment.</p>
      </div>

      <RiskForm onSubmit={handleSubmit} loading={loading} />

      {error && <div className="error-box">⚠ {error}</div>}

      {result && (
        <div className="result-panel">
          <div className="result-header">
            <h2>Risk Assessment Result</h2>
            <span style={{ color:'#94a3b8', fontSize:'.8rem' }}>
              ID: {result.request_id || '—'}
            </span>
          </div>
          <div className="result-body">
            <div className="result-grid">
              {/* Gauge */}
              <div>
                <RiskScoreGauge score={result.risk_score} tier={result.risk_tier} />
                <div className="score-info">
                  <div className="score-meta">
                    <span className="score-meta-label">Risk Tier</span>
                    <span className={`tier-badge ${result.risk_tier?.replace(' ','-')}`}>
                      {result.risk_tier}
                    </span>
                  </div>
                  <div className="score-meta">
                    <span className="score-meta-label">Confidence</span>
                    <span className="score-meta-value">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="score-meta">
                    <span className="score-meta-label">P(Default)</span>
                    <span className="score-meta-value">{(result.p_default * 100).toFixed(2)}%</span>
                  </div>
                </div>
              </div>
              {/* Recommendation */}
              <div>
                <RecommendationPanel rec={result} />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
