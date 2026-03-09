import Plot from 'react-plotly.js';

const TIER_COLOR = {
  Low: '#16a34a', Medium: '#ca8a04', High: '#ea580c', 'Very High': '#dc2626',
};

export default function RiskScoreGauge({ score = 0, tier = 'Low' }) {
  const color = TIER_COLOR[tier] || '#2563eb';

  const data = [{
    type: 'indicator',
    mode: 'gauge+number',
    value: score,
    number: { font: { size: 44, color: color, family: 'Inter' } },
    gauge: {
      axis: {
        range: [0, 1000],
        tickwidth: 1,
        tickcolor: '#cbd5e1',
        nticks: 6,
        tickfont: { size: 11, color: '#94a3b8' },
      },
      bar: { color, thickness: 0.26 },
      bgcolor: 'white',
      borderwidth: 0,
      steps: [
        { range: [0,   400],  color: '#fef2f2' },
        { range: [400, 650],  color: '#fff7ed' },
        { range: [650, 850],  color: '#fffbeb' },
        { range: [850, 1000], color: '#f0fdf4' },
      ],
    },
  }];

  const layout = {
    width: 300, height: 240,
    margin: { t: 20, b: 10, l: 20, r: 20 },
    paper_bgcolor: 'transparent',
    font: { family: 'Inter, sans-serif' },
  };

  return (
    <Plot
      data={data}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  );
}
