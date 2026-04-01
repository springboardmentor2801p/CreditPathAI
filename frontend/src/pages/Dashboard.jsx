import { useState, useEffect } from 'react'
import PlotlyChart from '../components/PlotlyChart'
import { getHealth, getBatchCases } from '../utils/api'
import './Dashboard.css'

const SAMPLE_RESULTS = [
  { risk_band: 'Very Low',  default_probability: 0.04, predicted_default: false, priority_level: 'Low',      expected_loss: 12000  },
  { risk_band: 'Low',       default_probability: 0.12, predicted_default: false, priority_level: 'Low',      expected_loss: 35000  },
  { risk_band: 'Low',       default_probability: 0.18, predicted_default: false, priority_level: 'Medium',   expected_loss: 48000  },
  { risk_band: 'Medium',    default_probability: 0.38, predicted_default: false, priority_level: 'Medium',   expected_loss: 112000 },
  { risk_band: 'Medium',    default_probability: 0.45, predicted_default: false, priority_level: 'High',     expected_loss: 175000 },
  { risk_band: 'High',      default_probability: 0.67, predicted_default: true,  priority_level: 'High',     expected_loss: 265000 },
  { risk_band: 'High',      default_probability: 0.72, predicted_default: true,  priority_level: 'Critical', expected_loss: 310000 },
  { risk_band: 'Very High', default_probability: 0.88, predicted_default: true,  priority_level: 'Critical', expected_loss: 490000 },
]

const BAND_COLORS = {
  'Very Low':  '#10b981',
  'Low':       '#34d399',
  'Medium':    '#f59e0b',
  'High':      '#f97316',
  'Very High': '#ef4444',
}

const PLT = (title) => ({
  paper_bgcolor: 'transparent',
  plot_bgcolor:  'transparent',
  font:   { family: 'Inter, sans-serif', color: '#94a3b8' },
  margin: { t: 30, b: 40, l: 40, r: 20 },
  title:  { text: title, font: { color: '#f1f5f9', size: 14 }, x: 0.02 },
  xaxis:  { gridcolor: 'rgba(128,128,128,0.2)', zerolinecolor: 'rgba(128,128,128,0.2)' },
  yaxis:  { gridcolor: 'rgba(128,128,128,0.2)', zerolinecolor: 'rgba(128,128,128,0.2)' },
})

function StatCard({ icon, label, value, sub, color }) {
  return (
    <div className="stat-card">
      <div className="stat-card-icon" style={{ background: `${color}22`, color }}>{icon}</div>
      <div>
        <div className="stat-card-value" style={{ color }}>{value}</div>
        <div className="stat-card-label">{label}</div>
        {sub && <div className="stat-card-sub">{sub}</div>}
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [health,   setHealth]   = useState(null)
  const [apiError, setApiError] = useState(false)
  const [results,  setResults]  = useState(SAMPLE_RESULTS)

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setApiError(true))
      
    getBatchCases(40)
      .then(data => {
        if (data && data.cases && data.cases.length > 0) {
          setResults(data.cases)
        }
      })
      .catch(err => console.warn('Using fallback data:', err))
  }, [])

  const bandCounts = results.reduce((acc, r) => {
    acc[r.risk_band] = (acc[r.risk_band] || 0) + 1
    return acc
  }, {})
  const priorityCounts = results.reduce((acc, r) => {
    acc[r.priority_level] = (acc[r.priority_level] || 0) + 1
    return acc
  }, {})
  const defaults   = results.filter(r => r.predicted_default).length
  const totalEl    = results.reduce((s, r) => s + r.expected_loss, 0)
  const avgProb    = (results.reduce((s, r) => s + r.default_probability, 0) / results.length) * 100
  const totalLoans = results.length

  return (
    <div className="dashboard-page">
      <div className="page-header">
        <h1>▦ Dashboard</h1>
        <p>Portfolio overview and model health at a glance.</p>
      </div>

      <div className={`health-banner ${apiError ? 'health-error' : health?.model_loaded ? 'health-ok' : 'health-warn'}`}>
        <span className="health-dot" />
        {apiError
          ? 'FastAPI is not reachable. Start it with: uvicorn fast_api.main:app --reload'
          : health
          ? `FastAPI Online - Model loaded - ${health.n_features} features`
          : 'Checking API...'}
      </div>

      <div className="grid-4 mt-4">
        <StatCard icon="📂" label="Total Borrowers"  value={totalLoans}
          sub="sample portfolio" color="var(--primary-light)" />
        <StatCard icon="⚠️" label="Predicted Defaults" value={defaults}
          sub={`${((defaults/totalLoans)*100).toFixed(0)}% of portfolio`} color="var(--risk-very-high)" />
        <StatCard icon="📉" label="Avg Default Prob." value={`${avgProb.toFixed(1)}%`}
          sub="portfolio-wide" color="var(--risk-medium)" />
        <StatCard icon="💸" label="Expected Loss"
          value={`Rs.${(totalEl/1000).toFixed(0)}k`}
          sub="across sample" color="var(--risk-high)" />
      </div>

      <div className="chart-row mt-6">
        <div className="card chart-card">
          <PlotlyChart
            data={[{
              type: 'pie',
              labels: Object.keys(bandCounts),
              values: Object.values(bandCounts),
              marker: { colors: Object.keys(bandCounts).map(b => BAND_COLORS[b]) },
              hole: 0.5,
              textinfo: 'label+percent',
              textfont: { family: 'Inter', color: '#f1f5f9', size: 12 },
              hovertemplate: '<b>%{label}</b><br>Count: %{value}<extra></extra>',
            }]}
            layout={{
              ...PLT('Risk Band Distribution'),
              legend: { font: { color: '#94a3b8' }, bgcolor: 'transparent' },
              height: 280,
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>

        <div className="card chart-card">
          <PlotlyChart
            data={[{
              type: 'bar',
              x: ['Low', 'Medium', 'High', 'Critical'],
              y: ['Low', 'Medium', 'High', 'Critical'].map(p => priorityCounts[p] || 0),
              marker: { color: ['#10b981', '#f59e0b', '#f97316', '#ef4444'] },
              hovertemplate: '<b>%{x}</b>: %{y} borrowers<extra></extra>',
            }]}
            layout={{
              ...PLT('Priority Level Distribution'),
              height: 280,
              bargap: 0.4,
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>
      </div>

      <div className="chart-row mt-4">
        <div className="card chart-card">
          <PlotlyChart
            data={[{
              type: 'scatter',
              mode: 'markers',
              x: results.map(r => r.default_probability * 100),
              y: results.map(r => r.expected_loss),
              marker: {
                size: 12,
                color: results.map(r => r.default_probability * 100),
                colorscale: [[0,'#10b981'],[0.35,'#f59e0b'],[0.65,'#f97316'],[1,'#ef4444']],
                showscale: true,
                colorbar: {
                  title: 'P(%)',
                  titlefont: { color: '#94a3b8' },
                  tickfont:  { color: '#94a3b8' },
                  bgcolor: 'transparent',
                  bordercolor: 'transparent',
                },
                line: { width: 1, color: 'rgba(255,255,255,0.3)' },
              },
              hovertemplate: 'Prob: %{x:.1f}%<br>Expected Loss: Rs.%{y:,.0f}<extra></extra>',
            }]}
            layout={{
              ...PLT('Default Prob. vs Expected Loss'),
              height: 280,
              xaxis: { gridcolor: 'rgba(128,128,128,0.2)', title: { text: 'Default Prob. (%)', font: { color: '#94a3b8' } } },
              yaxis: { gridcolor: 'rgba(128,128,128,0.2)', title: { text: 'Expected Loss', font: { color: '#94a3b8' } } },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>

        <div className="card chart-card">
          <PlotlyChart
            data={[{
              type: 'bar',
              orientation: 'h',
              x: results.map(r => r.expected_loss),
              y: results.map((_, i) => `Borrower ${i + 1}`),
              marker: { color: results.map(r => BAND_COLORS[r.risk_band]) },
              hovertemplate: '%{y}: Rs.%{x:,.0f}<extra></extra>',
            }]}
            layout={{
              ...PLT('Expected Loss per Borrower'),
              height: 280,
              margin: { t: 30, b: 40, l: 90, r: 20 },
              xaxis: { gridcolor: 'rgba(128,128,128,0.2)', tickformat: ',.0f' },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>
      </div>

      <div className="card mt-4">
        <div className="section-title">Sample Portfolio</div>
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Default Prob.</th>
                <th>Risk Band</th>
                <th>Predicted Default</th>
                <th>Priority</th>
                <th>Expected Loss</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>
                    <div className="prob-cell">
                      <div className="gauge-bar" style={{ width: 80, height: 6, display: 'inline-block' }}>
                        <div
                          className="gauge-fill"
                          style={{
                            width: `${r.default_probability * 100}%`,
                            background: `linear-gradient(90deg, var(--risk-very-low), ${BAND_COLORS[r.risk_band]})`,
                          }}
                        />
                      </div>
                      <span>{(r.default_probability * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td>
                    <span className="band-chip" style={{ background: `${BAND_COLORS[r.risk_band]}22`, color: BAND_COLORS[r.risk_band] }}>
                      {r.risk_band}
                    </span>
                  </td>
                  <td>
                    <span className={r.predicted_default ? 'badge badge-danger' : 'badge badge-success'}>
                      {r.predicted_default ? 'Yes' : 'No'}
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${
                      r.priority_level === 'Critical' ? 'badge-danger' :
                      r.priority_level === 'High'     ? 'badge-warning' :
                      'badge-info'
                    }`}>
                      {r.priority_level}
                    </span>
                  </td>
                  <td>Rs.{r.expected_loss.toLocaleString('en-IN')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
