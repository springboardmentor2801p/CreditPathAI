export default function RiskHistoryTable({ history, onClear }) {
  if (history.length === 0) {
    return <p className="no-data">No predictions yet. Use the Predict page to get started.</p>;
  }

  const fmtDate = (iso) =>
    new Date(iso).toLocaleString('en-US', { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit' });

  return (
    <>
      <div style={{ display:'flex', justifyContent:'flex-end', marginBottom:10 }}>
        <button className="btn btn-outline" style={{ fontSize:'.8rem', padding:'5px 14px' }} onClick={onClear}>
          Clear History
        </button>
      </div>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Time</th>
              <th>State</th>
              <th>Income</th>
              <th>Loan Amt</th>
              <th>Purpose</th>
              <th>Risk Score</th>
              <th>P(Default)</th>
              <th>Tier</th>
              <th>Decision</th>
            </tr>
          </thead>
          <tbody>
            {history.map((row, i) => (
              <tr key={row.id} style={{ animationDelay: `${i * 0.04}s` }}>
                <td style={{ color:'#94a3b8', fontSize:'.8rem' }}>{history.length - i}</td>
                <td style={{ color:'#64748b', fontSize:'.8rem', whiteSpace:'nowrap' }}>{fmtDate(row.timestamp)}</td>
                <td><strong>{row.borrower?.residentialState}</strong></td>
                <td>${Number(row.borrower?.annualIncome).toLocaleString()}</td>
                <td>${Number(row.loan?.loanAmount).toLocaleString()}</td>
                <td style={{ textTransform:'capitalize' }}>{row.loan?.purpose}</td>
                <td><strong>{row.risk_score}</strong></td>
                <td>{(row.p_default * 100).toFixed(1)}%</td>
                <td>
                  <span className={`tier-badge ${row.risk_tier?.replace(' ', '-')}`}>
                    {row.risk_tier}
                  </span>
                </td>
                <td>
                  <span className={`decision-badge decision-${row.decision}`}
                    style={{ fontSize:'.75rem', padding:'3px 10px' }}>
                    {row.decision?.replace(/_/g, ' ')}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
