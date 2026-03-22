import Plotly from 'react-plotly.js'
const Plot = Plotly.default || Plotly

export default function Dashboard() {
  const stats = [
    { value: '1,48,668', label: 'Total Borrowers', color: '#00cfff' },
    { value: '24.6%', label: 'Default Rate', color: '#ff4d6d' },
    { value: '88.3%', label: 'AUC-ROC Score', color: '#ffd700' },
    { value: '87.2%', label: 'Model Accuracy', color: '#00ff88' },
  ]

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Portfolio Dashboard</h1>
        <p className="page-subtitle">Overview of loan recovery performance and model metrics</p>
      </div>

      {/* Stats */}
      <div className="dashboard-grid">
        {stats.map((s, i) => (
          <div key={i} className="dash-stat" style={{ '--accent': s.color }}>
            <div className="dash-stat-value" style={{ color: s.color }}>{s.value}</div>
            <div className="dash-stat-label">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Charts Row 1 */}
      <div className="charts-grid">

        {/* Risk Distribution Pie */}
        <div className="chart-card">
          <div className="chart-title">Risk Distribution</div>
          <Plot
            data={[{
              type: 'pie',
              labels: ['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk'],
              values: [30, 24, 21, 15, 10],
              marker: { colors: ['#00ff88', '#7CFF6B', '#ffd700', '#FF8C42', '#ff4d6d'] },
              textinfo: 'label+percent',
              textfont: { color: '#ffffff', size: 11 },
              hole: 0.4,
            }]}
            layout={{
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              showlegend: false,
              margin: { t: 10, b: 10, l: 10, r: 10 },
              height: 260,
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>

        {/* Recovery Channels Bar */}
        <div className="chart-card">
          <div className="chart-title">Recovery Channels</div>
          <Plot
            data={[{
              type: 'bar',
              x: ['Email/SMS', 'Phone Call', 'EMI Restructure', 'Legal Notice', 'Court'],
              y: [42, 31, 18, 15, 5],
              marker: { color: ['#00cfff', '#b67cff', '#00ff88', '#ffd700', '#ff4d6d'] },
              text: ['42%', '31%', '18%', '15%', '5%'],
              textposition: 'outside',
              textfont: { color: '#ffffff', size: 11 },
            }]}
            layout={{
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              margin: { t: 20, b: 60, l: 30, r: 10 },
              height: 260,
              xaxis: { tickfont: { color: '#7A90A8', size: 10 }, gridcolor: 'transparent' },
              yaxis: { tickfont: { color: '#7A90A8' }, gridcolor: '#1A3A5C' },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>

        {/* Model Performance */}
        <div className="chart-card">
          <div className="chart-title">Model Performance</div>
          <Plot
            data={[{
              type: 'bar',
              y: [0.8836, 0.8725, 0.88, 0.86],
              y: ['AUC-ROC', 'Accuracy', 'Precision', 'Recall'],
              orientation: 'h',
              marker: { color: ['#00cfff', '#00ff88', '#b67cff', '#ffd700'] },
              text: ['88.3%', '87.2%', '88%', '86%'],
              textposition: 'outside',
              textfont: { color: '#ffffff', size: 11 },
            }]}
            layout={{
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              margin: { t: 10, b: 20, l: 80, r: 50 },
              height: 260,
              xaxis: { range: [0, 1.1], tickfont: { color: '#7A90A8' }, gridcolor: '#1A3A5C' },
              yaxis: { tickfont: { color: '#7A90A8', size: 11 }, gridcolor: 'transparent' },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>

        {/* Expected Loss by Priority */}
        <div className="chart-card">
          <div className="chart-title">Expected Loss by Priority</div>
          <Plot
            data={[{
              type: 'bar',
              x: ['Low', 'Medium', 'High', 'Critical'],
              y: [25000, 125000, 350000, 800000],
              marker: { color: ['#00ff88', '#ffd700', '#FF8C42', '#ff4d6d'] },
              text: ['₹25K', '₹1.25L', '₹3.5L', '₹8L'],
              textposition: 'outside',
              textfont: { color: '#ffffff', size: 11 },
            }]}
            layout={{
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              margin: { t: 20, b: 40, l: 50, r: 10 },
              height: 260,
              xaxis: { tickfont: { color: '#7A90A8' }, gridcolor: 'transparent' },
              yaxis: { tickfont: { color: '#7A90A8' }, gridcolor: '#1A3A5C' },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>

      </div>
    </div>
  )
}