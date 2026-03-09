import Plot from 'react-plotly.js';

const TIERS   = ['Low', 'Medium', 'High', 'Very High'];
const COLORS  = ['#16a34a', '#ca8a04', '#ea580c', '#dc2626'];
const BGCOLORS= ['#f0fdf4', '#fffbeb', '#fff7ed', '#fef2f2'];

export default function RiskDistributionChart({ history }) {
  const counts = TIERS.map((t) => history.filter((h) => h.risk_tier === t).length);
  const total  = counts.reduce((a, b) => a + b, 0);

  if (total === 0) {
    return (
      <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:260, color:'#94a3b8', fontSize:'.9rem' }}>
        No predictions yet — submit your first analysis
      </div>
    );
  }

  const donutData = [{
    type: 'pie', labels: TIERS, values: counts,
    marker: { colors: COLORS, line: { color: '#fff', width: 2 } },
    hole: 0.56,
    textinfo: 'percent',
    textfont: { size: 13, family: 'Inter' },
    hovertemplate: '<b>%{label}</b><br>%{value} applications<br>%{percent}<extra></extra>',
  }];

  const donutLayout = {
    showlegend: true,
    legend: { orientation: 'h', y: -0.08, font: { size: 12, family: 'Inter' } },
    margin: { t: 10, b: 50, l: 10, r: 10 },
    paper_bgcolor: 'transparent',
    height: 280,
    annotations: [{
      text: `<b>${total}</b><br><span style="font-size:11px">Total</span>`,
      x: 0.5, y: 0.5, showarrow: false,
      font: { size: 16, color: '#0f2744', family: 'Inter' },
    }],
  };

  const barData = [{
    type: 'bar',
    x: TIERS,
    y: counts,
    marker: { color: BGCOLORS, line: { color: COLORS, width: 2 } },
    text: counts.map(String),
    textposition: 'outside',
    hovertemplate: '<b>%{x}</b>: %{y}<extra></extra>',
  }];

  const barLayout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor:  'transparent',
    margin: { t: 20, b: 40, l: 30, r: 20 },
    height: 280,
    yaxis: { gridcolor: '#e2e8f0', tickfont: { size: 11 } },
    xaxis: { tickfont: { size: 12 } },
    font: { family: 'Inter' },
  };

  return (
    <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:0 }}>
      <Plot data={donutData} layout={donutLayout} config={{ displayModeBar:false, responsive:true }} style={{ width:'100%' }} />
      <Plot data={barData}   layout={barLayout}   config={{ displayModeBar:false, responsive:true }} style={{ width:'100%' }} />
    </div>
  );
}
