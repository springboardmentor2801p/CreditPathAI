import { useHistory } from '../hooks/useHistory';
import RiskDistributionChart from '../components/RiskDistributionChart';
import RiskHistoryTable from '../components/RiskHistoryTable';

export default function DashboardPage() {
  const { history, clearHistory } = useHistory();

  const total     = history.length;
  const avgScore  = total ? Math.round(history.reduce((a, h) => a + h.risk_score, 0) / total) : 0;
  const highRisk  = total ? ((history.filter((h) => ['High','Very High'].includes(h.risk_tier)).length / total) * 100).toFixed(1) : 0;
  const approved  = total ? ((history.filter((h) => ['APPROVE','CONDITIONAL_APPROVE'].includes(h.decision)).length / total) * 100).toFixed(1) : 0;

  const STATS = [
    { icon:'📊', label:'Total Predictions',  value: total,          cls:'blue' },
    { icon:'⭐', label:'Avg Risk Score',      value: avgScore || '—', cls:'blue' },
    { icon:'⚠️', label:'High Risk Rate',      value: total ? `${highRisk}%` : '—', cls: parseFloat(highRisk) > 20 ? 'red' : 'amber' },
    { icon:'✅', label:'Approval Rate',       value: total ? `${approved}%` : '—', cls:'green' },
  ];

  return (
    <div style={{ animation:'fadeIn .35s ease' }}>
      <div className="page-header page-header-row">
        <div>
          <h1>Risk Intelligence Dashboard</h1>
          <p>Portfolio overview from your prediction history.</p>
        </div>
        <a href="/predict" style={{ textDecoration:'none' }}>
          <button className="btn btn-primary">+ New Prediction</button>
        </a>
      </div>

      {/* Stat cards */}
      <div className="stats-grid">
        {STATS.map((s, i) => (
          <div key={s.label} className={`stat-card ${s.cls}`} style={{ animationDelay: `${i * 0.08}s` }}>
            <div className="stat-icon">{s.icon}</div>
            <div className="stat-value">{s.value}</div>
            <div className="stat-label">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="charts-grid">
        <div className="card">
          <div className="card-header">
            <span className="card-title">Risk Tier Distribution</span>
          </div>
          <div className="card-body">
            <RiskDistributionChart history={history} />
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">Score Breakdown by Tier</span>
          </div>
          <div className="card-body" style={{ paddingTop:8 }}>
            {['Low','Medium','High','Very High'].map((tier) => {
              const count   = history.filter((h) => h.risk_tier === tier).length;
              const pct     = total ? (count / total * 100) : 0;
              const colors  = { Low:'#16a34a', Medium:'#ca8a04', High:'#ea580c', 'Very High':'#dc2626' };
              return (
                <div key={tier} style={{ marginBottom:16 }}>
                  <div style={{ display:'flex', justifyContent:'space-between', fontSize:'.85rem', marginBottom:5 }}>
                    <span style={{ fontWeight:600 }}>{tier}</span>
                    <span style={{ color:'#64748b' }}>{count} ({pct.toFixed(1)}%)</span>
                  </div>
                  <div style={{ background:'#f1f5f9', borderRadius:30, height:10, overflow:'hidden' }}>
                    <div style={{
                      width:`${pct}%`, height:'100%',
                      background: colors[tier], borderRadius:30,
                      transition:'width .6s cubic-bezier(.4,0,.2,1)',
                    }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* History table */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">Recent Predictions</span>
          <span style={{ color:'#94a3b8', fontSize:'.8rem' }}>{total} total</span>
        </div>
        <div className="card-body">
          <RiskHistoryTable history={history} onClear={clearHistory} />
        </div>
      </div>
    </div>
  );
}
