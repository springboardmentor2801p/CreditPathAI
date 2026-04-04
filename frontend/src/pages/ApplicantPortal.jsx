import { useState, useEffect } from 'react'
import axios from 'axios'
import Plotly from 'plotly.js-dist-min'

const D = {
  bg0:'#060d18', bg1:'#0a1525', bg2:'#0f1e35', bg3:'#162844', bg4:'#1e3354',
  blue:'#3b82f6', blueD:'#1d4ed8',
  text:'#f0f4ff', text2:'#94a3c4', text3:'#5a6f94',
  border:'rgba(255,255,255,0.07)', border2:'rgba(255,255,255,0.12)',
  green:'#22c55e', greenDim:'rgba(34,197,94,0.1)', greenD:'#86efac',
  red:'#ef4444', redDim:'rgba(239,68,68,0.1)', redD:'#fca5a5',
  amber:'#f59e0b', amberDim:'rgba(245,158,11,0.1)', amberD:'#fcd34d',
}

function Gauge({ pct, approved, conditional }) {
  const r = 80, cx = 110, cy = 110
  const circ = Math.PI * r
  const dash  = (pct / 100) * circ
  const color = approved ? D.green : conditional ? D.amber : D.red
  const angle = -180 + (pct / 100) * 180
  const rad = (angle * Math.PI) / 180
  const nx = cx + r * Math.cos(rad), ny = cy + r * Math.sin(rad)
  return (
    <svg viewBox="0 0 220 130" width="100%" style={{ maxWidth:260, display:'block', margin:'0 auto' }}>
      <path d={`M ${cx-r} ${cy} A ${r} ${r} 0 0 1 ${cx+r} ${cy}`} fill="none" stroke={D.bg3} strokeWidth="16" strokeLinecap="round"/>
      <path d={`M ${cx-r} ${cy} A ${r} ${r} 0 0 1 ${cx+r} ${cy}`} fill="none" stroke={color} strokeWidth="16" strokeLinecap="round"
        strokeDasharray={`${dash} ${circ}`} style={{ transition:'stroke-dasharray 1s ease', filter:`drop-shadow(0 0 8px ${color}88)` }}/>
      <circle cx={nx} cy={ny} r="7" fill={color} style={{ filter:`drop-shadow(0 0 6px ${color})` }}/>
      <text x={cx} y={cy-10} textAnchor="middle" style={{ fontFamily:"serif", fontSize:36, fontWeight:700, fill:color }}>{pct}</text>
      <text x={cx} y={cy+14} textAnchor="middle" style={{ fontSize:11, fill:D.text3 }}>Approval %</text>
      <text x={cx-r} y={cy+22} textAnchor="middle" style={{ fontSize:10, fill:D.text3 }}>0</text>
      <text x={cx+r} y={cy+22} textAnchor="middle" style={{ fontSize:10, fill:D.text3 }}>100</text>
    </svg>
  )
}

function CreditScoreBar({ score }) {
  const ranges = [
    { label:'Very Poor', min:300, max:579, color:'#ef4444' },
    { label:'Fair',      min:580, max:669, color:'#f97316' },
    { label:'Good',      min:670, max:739, color:'#eab308' },
    { label:'Very Good', min:740, max:799, color:'#22c55e' },
    { label:'Excellent', min:800, max:850, color:'#16a34a' },
  ]
  const pct    = Math.min(100, Math.max(0, ((score - 300) / 550) * 100))
  const active = ranges.find(r => score >= r.min && score <= r.max)
  return (
    <div>
      <div style={{ display:'flex', height:12, borderRadius:99, overflow:'hidden', marginBottom:10 }}>
        {ranges.map(r => (
          <div key={r.label} style={{ flex:r.max-r.min, background:r.color, opacity:score>=r.min&&score<=r.max?1:0.2, transition:'opacity 0.4s' }}/>
        ))}
      </div>
      <div style={{ position:'relative', height:16, marginBottom:6 }}>
        <div style={{ position:'absolute', left:`${pct}%`, transform:'translateX(-50%)', width:2, height:16, background:D.text, borderRadius:2, transition:'left 0.6s ease' }}/>
      </div>
      <div style={{ display:'flex', justifyContent:'space-between', marginBottom:16 }}>
        {ranges.map(r => (
          <span key={r.label} style={{ fontSize:'0.65rem', color:score>=r.min&&score<=r.max?r.color:D.text3, fontWeight:score>=r.min&&score<=r.max?700:400, flex:1, textAlign:'center' }}>{r.label}</span>
        ))}
      </div>
      <div style={{ textAlign:'center' }}>
        <span style={{ fontSize:'2rem', fontWeight:700, color:active?.color||D.text }}>{score}</span>
        <span style={{ fontSize:'0.85rem', color:D.text3, marginLeft:6 }}>/ 850</span>
        {active && <div style={{ marginTop:6, fontSize:'0.75rem', color:active.color, fontWeight:600 }}>{active.label}</div>}
      </div>
    </div>
  )
}

function ProbabilityChart({ approvalProb, defaultProb }) {
  useEffect(() => {
    Plotly.newPlot('probabilityChart', [{
      values:[Math.round(approvalProb*100), Math.round(defaultProb*100)],
      labels:['Approval','Default Risk'], type:'pie',
      marker:{ colors:[D.green,D.red], line:{color:D.bg3,width:2} },
      textinfo:'label+percent', textposition:'inside',
      hovertemplate:'<b>%{label}</b><br>%{value}%<extra></extra>',
    }], {
      title:{ text:'📊 Approval vs Default Risk', font:{color:D.text,size:16} },
      paper_bgcolor:D.bg2, plot_bgcolor:D.bg2,
      font:{color:D.text2}, margin:{l:0,r:0,t:40,b:0}, height:220, showlegend:true,
      legend:{ x:1.05, y:1, bgcolor:'rgba(0,0,0,0)' },
    }, { responsive:true, displayModeBar:false })
  }, [approvalProb, defaultProb])
  return <div id="probabilityChart" style={{ width:'100%' }} />
}

function MetricsBarChart({ creditScore, dti, ltv, income }) {
  useEffect(() => {
    const metrics = ['Credit Score','DTI Health','LTV Health','Income Level']
    const yours   = [
      Math.round((creditScore/850)*100),
      Math.max(0,100-dti),
      Math.max(0,100-ltv),
      Math.min(100,Math.round((income/500000)*100)),
    ]
    const ideal = [85,75,80,80]
    Plotly.newPlot('metricsChart', [
      { x:metrics, y:yours, name:'Your Score', type:'bar', marker:{color:D.blue,opacity:0.8}, hovertemplate:'<b>Your Score</b><br>%{x}<br>%{y}%<extra></extra>' },
      { x:metrics, y:ideal, name:'Ideal Score', type:'bar', marker:{color:D.green,opacity:0.6}, hovertemplate:'<b>Ideal Score</b><br>%{x}<br>%{y}%<extra></extra>' },
    ], {
      title:{ text:'📈 Your Metrics Performance', font:{color:D.text,size:16} },
      barmode:'group', paper_bgcolor:D.bg2, plot_bgcolor:D.bg3,
      font:{color:D.text2},
      xaxis:{ tickcolor:D.text3, linecolor:D.border, gridcolor:D.border },
      yaxis:{ tickcolor:D.text3, linecolor:D.border, gridcolor:D.border, range:[0,100] },
      margin:{l:50,r:20,t:50,b:50}, height:220, showlegend:true,
      legend:{x:0.02,y:0.98,bgcolor:'rgba(0,0,0,0.2)',bordercolor:D.border,borderwidth:1},
    }, { responsive:true, displayModeBar:false })
  }, [creditScore, dti, ltv, income])
  return <div id="metricsChart" style={{ width:'100%' }} />
}

function RiskProfileChart({ creditScore, dti, ltv, income, approvalProb }) {
  useEffect(() => {
    Plotly.newPlot('riskChart', [{
      type:'scatterpolar',
      r:[Math.round((creditScore/850)*100),Math.max(0,100-dti),Math.max(0,100-ltv),Math.min(100,Math.round((income/500000)*100)),Math.round(approvalProb*100)],
      theta:['Credit Score','DTI Health','LTV Health','Income Strength','Approval Score'],
      fill:'toself', fillcolor:D.blue+'40', line:{color:D.blue,width:2}, marker:{size:8,color:D.blue},
      hovertemplate:'<b>%{theta}</b><br>%{r}%<extra></extra>',
    }], {
      title:{ text:'🎯 Overall Risk Profile Analysis', font:{color:D.text,size:16} },
      polar:{ bgcolor:D.bg2, radialaxis:{visible:true,range:[0,100],tickcolor:D.text3,gridcolor:D.border,linecolor:D.border}, angularaxis:{tickcolor:D.text3,linecolor:D.border,gridcolor:D.border} },
      paper_bgcolor:D.bg2, font:{color:D.text2},
      margin:{l:80,r:80,t:50,b:80}, height:400, showlegend:false,
    }, { responsive:true, displayModeBar:false })
  }, [creditScore, dti, ltv, income, approvalProb])
  return <div id="riskChart" style={{ width:'100%' }} />
}

function RiskScoreGauge({ approvalProb }) {
  useEffect(() => {
    Plotly.newPlot('gaugeChart', [{
      type:'indicator', mode:'gauge+number+delta',
      value:Math.round(approvalProb*100),
      title:{ text:'Approval Score' },
      delta:{ reference:80, suffix:'%' },
      gauge:{
        axis:{range:[0,100]}, bar:{color:D.blue},
        steps:[{range:[0,33],color:D.red+'20'},{range:[33,66],color:D.amber+'20'},{range:[66,100],color:D.green+'20'}],
        threshold:{line:{color:D.red,width:2},thickness:0.75,value:90},
      },
      number:{font:{size:40,color:D.blue}}, domain:{x:[0,1],y:[0,1]},
    }], {
      paper_bgcolor:D.bg2, font:{color:D.text},
      margin:{l:0,r:0,t:60,b:0}, height:220,
      title:{text:'📊 Approval Score Gauge',font:{size:16,color:D.text}},
    }, { responsive:true, displayModeBar:false })
  }, [approvalProb])
  return <div id="gaugeChart" style={{ width:'100%' }} />
}

function ProbabilityDistributionChart({ approvalProb }) {
  useEffect(() => {
    const x = Array.from({length:100},(_,i)=>i)
    const center = Math.round(approvalProb*100), sigma = 15
    const y = x.map(i => (1/(sigma*Math.sqrt(2*Math.PI)))*Math.exp(-0.5*Math.pow((i-center)/sigma,2)))
    Plotly.newPlot('distributionChart', [{
      x, y, fill:'tozeroy', type:'scatter',
      marker:{color:D.blue}, fillcolor:D.blue+'30', line:{color:D.blue,width:3},
      hovertemplate:'Score: %{x}%<extra></extra>',
    }], {
      title:{text:'📉 Approval Probability Distribution',font:{color:D.text,size:16}},
      xaxis:{title:'Approval %',tickcolor:D.text3,linecolor:D.border,gridcolor:D.border},
      yaxis:{title:'Probability',tickcolor:D.text3,linecolor:D.border,gridcolor:D.border},
      paper_bgcolor:D.bg2, plot_bgcolor:D.bg3, font:{color:D.text2},
      margin:{l:60,r:20,t:50,b:50}, height:220, showlegend:false,
    }, { responsive:true, displayModeBar:false })
  }, [approvalProb])
  return <div id="distributionChart" style={{ width:'100%' }} />
}

export default function ApplicantPortal({ onBack }) {
  const [form, setForm]       = useState({ creditScore:'', loanAmount:'', annualIncome:'', ltv:'', dti:'' })
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [errors, setErrors]   = useState({})
  const [backendStatus, setBackendStatus] = useState('checking')
  const [backendError, setBackendError]   = useState(null)

  const set = (k, v) => setForm(p => ({ ...p, [k]: v }))

  useEffect(() => { checkBackendStatus() }, [])

  async function checkBackendStatus() {
    try {
      await axios.get('http://127.0.0.1:8000/health', { timeout:3000 })
      setBackendStatus('connected'); setBackendError(null)
    } catch (error) {
      setBackendStatus('disconnected'); setBackendError(error.message)
    }
  }

  function validate() {
    const e = {}
    if (!form.creditScore  || form.creditScore<300  || form.creditScore>850) e.creditScore  = 'Enter a score between 300–850'
    if (!form.loanAmount   || form.loanAmount<=0)                            e.loanAmount   = 'Enter a valid loan amount'
    if (!form.annualIncome || form.annualIncome<=0)                          e.annualIncome = 'Enter a valid annual income'
    if (!form.ltv || form.ltv<=0 || form.ltv>100)                           e.ltv          = 'Enter LTV between 1–100'
    if (!form.dti || form.dti<=0 || form.dti>100)                           e.dti          = 'Enter DTI between 1–100'
    setErrors(e); return Object.keys(e).length === 0
  }

  async function runAnalysis() {
    if (!validate()) return
    if (backendStatus !== 'connected') { alert('⚠️ Backend server is not connected.'); return }
    setLoading(true); setResult(null)
    try {
      const response = await axios.post('http://127.0.0.1:8000/applicant-recommendation', {
        Credit_Score: Number(form.creditScore),
        loan_amount:  Number(form.loanAmount),
        income:       Number(form.annualIncome),
        LTV:          Number(form.ltv),
        dtir1:        Number(form.dti),
      }, { timeout:10000, headers:{ 'Content-Type':'application/json' } })
      const rec = response.data.recommendation
      setResult({
        status:       rec.eligibility_status,
        approvalProb: rec.approval_probability,
        defaultProb:  rec.default_probability,
        headline:     rec.headline,
        summary:      rec.summary,
        improvements: rec.improvement_opportunities,
        timeline:     rec.reapplication_timeline,
        timelineMsg:  rec.timeline_message,
        nextSteps:    rec.next_steps,
      })
    } catch (error) {
      console.error('API Error:', error)
      setBackendError(error.message)
      alert(`❌ Error: ${error.response?.data?.detail || error.message}`)
    }
    setLoading(false)
  }

  const inp = (field) => ({
    width:'100%', padding:'12px 14px',
    background:D.bg3, border:`1px solid ${errors[field]?D.red+'88':D.border}`,
    borderRadius:10, color:D.text, fontSize:'0.9rem', outline:'none', transition:'border-color .2s,box-shadow .2s',
  })
  const lbl  = { display:'block', fontSize:'0.78rem', fontWeight:600, color:D.text3, marginBottom:6, letterSpacing:'0.02em' }
  const card = (extra={}) => ({ background:D.bg2, border:`1px solid ${D.border}`, borderRadius:16, padding:'26px 28px', ...extra })

  const verdict = result
    ? result.status==='APPROVED'
      ? { label:'✓ Approved',      color:D.green, dim:D.greenDim }
      : result.status==='CONDITIONAL'
        ? { label:'⚠ Conditional', color:D.amber, dim:D.amberDim }
        : { label:'✕ Not Approved', color:D.red,  dim:D.redDim }
    : null

  return (
    <div style={{ minHeight:'100vh', background:D.bg0, fontFamily:"sans-serif" }}>
      <style>{`@keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }`}</style>

      {/* Header */}
      <div style={{ borderBottom:`1px solid ${D.border}`, padding:'32px 5% 28px', background:D.bg1 }}>
        <div style={{ maxWidth:1500, margin:'0 auto' }}>
          <button onClick={onBack} style={{
            background:'none', border:`1px solid ${D.border}`, color:D.text2,
            borderRadius:8, padding:'6px 14px', fontSize:'0.8rem', cursor:'pointer',
            marginBottom:16, display:'flex', alignItems:'center', gap:6,
          }}>← Back</button>
          <div style={{ display:'inline-flex', alignItems:'center', gap:6, background:'rgba(59,130,246,0.1)', border:'1px solid rgba(59,130,246,0.2)', borderRadius:100, padding:'4px 14px', fontSize:'0.68rem', fontWeight:700, color:'#93c5fd', letterSpacing:'.07em', textTransform:'uppercase', marginBottom:12 }}>
            ML Powered Risk Engine
          </div>
          <h1 style={{ fontSize:'clamp(1.7rem,3.5vw,2.4rem)', color:D.text, marginBottom:6, letterSpacing:'-0.02em' }}>
            Borrower <em style={{ fontStyle:'italic', color:'#60a5fa' }}>Eligibility Predictor</em>
          </h1>
          <p style={{ fontSize:'0.88rem', color:D.text2 }}>📋 <strong>APPLICANT PERSPECTIVE:</strong> Check your loan eligibility & get improvement suggestions</p>
        </div>
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'minmax(340px,400px) 1fr', gap:24, padding:'32px 5%', maxWidth:1500, margin:'0 auto', alignItems:'start' }}>

        {/* Form */}
        <div style={{ ...card(), position:'sticky', top:28 }}>
          <h2 style={{ fontSize:'1.3rem', color:D.text, marginBottom:26, letterSpacing:'-0.01em' }}>Your Details</h2>
          <div style={{ display:'flex', flexDirection:'column', gap:18 }}>
            {[
              { k:'creditScore',  label:'Credit Score',      ph:'300 – 850' },
              { k:'loanAmount',   label:'Loan Amount (₹)',   ph:'e.g. 200000' },
              { k:'annualIncome', label:'Annual Income (₹)', ph:'e.g. 60000' },
            ].map(({ k, label, ph }) => (
              <div key={k}>
                <label style={lbl}>{label}</label>
                <input type="number" placeholder={ph} value={form[k]} onChange={e=>set(k,e.target.value)}
                  style={inp(k)}
                  onFocus={e=>{e.target.style.borderColor='rgba(59,130,246,0.5)';e.target.style.boxShadow='0 0 0 3px rgba(59,130,246,0.1)'}}
                  onBlur={e=>{e.target.style.borderColor=errors[k]?D.red+'88':D.border;e.target.style.boxShadow='none'}}/>
                {errors[k] && <p style={{ color:D.red, fontSize:'0.73rem', marginTop:4 }}>{errors[k]}</p>}
              </div>
            ))}
            <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:14 }}>
              {[{k:'ltv',label:'LTV (%)',ph:'e.g. 80'},{k:'dti',label:'DTI (%)',ph:'e.g. 35'}].map(({k,label,ph})=>(
                <div key={k}>
                  <label style={lbl}>{label}</label>
                  <input type="number" placeholder={ph} value={form[k]} onChange={e=>set(k,e.target.value)}
                    style={inp(k)}
                    onFocus={e=>{e.target.style.borderColor='rgba(59,130,246,0.5)';e.target.style.boxShadow='0 0 0 3px rgba(59,130,246,0.1)'}}
                    onBlur={e=>{e.target.style.borderColor=errors[k]?D.red+'88':D.border;e.target.style.boxShadow='none'}}/>
                  {errors[k] && <p style={{ color:D.red, fontSize:'0.73rem', marginTop:4 }}>{errors[k]}</p>}
                </div>
              ))}
            </div>
          </div>

          <button onClick={runAnalysis} disabled={loading||backendStatus!=='connected'} style={{
            width:'100%', marginTop:28, padding:'14px 0', fontWeight:700, fontSize:'0.92rem', color:'#fff',
            background:loading||backendStatus!=='connected'?D.bg4:'linear-gradient(135deg,#1d4ed8,#3b82f6)',
            border:'none', borderRadius:12, cursor:(loading||backendStatus!=='connected')?'not-allowed':'pointer',
            boxShadow:(loading||backendStatus!=='connected')?'none':'0 0 24px rgba(59,130,246,0.3)',
            display:'flex', alignItems:'center', justifyContent:'center', gap:10, transition:'all .2s',
          }}>
            {loading
              ? <><span style={{width:16,height:16,border:'2px solid rgba(255,255,255,.25)',borderTop:'2px solid #fff',borderRadius:'50%',animation:'spin .7s linear infinite',display:'inline-block'}}/> Checking...</>
              : backendStatus!=='connected' ? '⚠️ Backend Offline' : 'Check Eligibility →'}
          </button>

          <div style={{ marginTop:22, padding:'14px 16px', background:D.bg3, borderRadius:10, border:`1px solid ${D.border}` }}>
            <div style={{ fontSize:'0.68rem', fontWeight:700, color:D.text3, letterSpacing:'0.06em', textTransform:'uppercase', marginBottom:10 }}>Backend Status</div>
            <div style={{ fontSize:'0.75rem', color:backendStatus==='connected'?D.greenD:backendStatus==='checking'?D.amberD:D.redD, display:'flex', alignItems:'center', gap:6 }}>
              <span>{backendStatus==='connected'?'✓':backendStatus==='checking'?'⏳':'✕'}</span>
              {backendStatus==='connected'?'Connected to FastAPI':backendStatus==='checking'?'Checking...':'Disconnected'}
            </div>
            <div style={{ fontSize:'0.75rem', color:D.text3, marginTop:4 }}>http://127.0.0.1:8000</div>
            {backendError && <div style={{ fontSize:'0.7rem', color:D.red, marginTop:6 }}>{backendError}</div>}
          </div>
        </div>

        {/* Results */}
        <div style={{ display:'flex', flexDirection:'column', gap:20 }}>

          {!result && !loading && (
            <div style={{ ...card(), textAlign:'center', minHeight:380, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center' }}>
              <div style={{ width:70, height:70, borderRadius:'50%', background:D.bg3, display:'flex', alignItems:'center', justifyContent:'center', marginBottom:16, fontSize:28 }}>📊</div>
              <h3 style={{ fontSize:'1.3rem', color:D.text, marginBottom:8 }}>Check Your Eligibility</h3>
              <p style={{ fontSize:'0.88rem', color:D.text2, maxWidth:260, lineHeight:1.7 }}>Fill in your details and click <strong>Check Eligibility</strong> to get an ML-powered decision.</p>
            </div>
          )}

          {loading && (
            <div style={{ ...card(), textAlign:'center', minHeight:380, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', gap:18 }}>
              <div style={{ width:50, height:50, border:`2px solid ${D.border2}`, borderTop:`2px solid ${D.blue}`, borderRadius:'50%', animation:'spin .8s linear infinite' }}/>
              <p style={{ fontSize:'1.1rem', color:D.text }}>Getting model prediction...</p>
            </div>
          )}

          {result && verdict && (
            <>
              <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:20 }}>
                <div style={{ ...card(), border:`1px solid ${verdict.color}33`, boxShadow:`0 4px 32px ${verdict.color}18` }}>
                  <div style={{ display:'inline-flex', alignItems:'center', gap:9, background:verdict.dim, border:`1px solid ${verdict.color}44`, borderRadius:100, padding:'7px 16px', marginBottom:20 }}>
                    <span style={{ width:22, height:22, borderRadius:'50%', background:verdict.color, color:'#000', display:'flex', alignItems:'center', justifyContent:'center', fontWeight:800, fontSize:'0.8rem' }}>
                      {result.status==='APPROVED'?'✓':result.status==='CONDITIONAL'?'⚠':'✕'}
                    </span>
                    <span style={{ fontSize:'1rem', color:verdict.color }}>{verdict.label}</span>
                  </div>
                  <Gauge pct={Math.round(result.approvalProb*100)} approved={result.status==='APPROVED'} conditional={result.status==='CONDITIONAL'}/>
                </div>
                <div style={card()}>
                  <h3 style={{ fontSize:'1.1rem', color:D.text, marginBottom:22 }}>Credit Score</h3>
                  <CreditScoreBar score={Number(form.creditScore)}/>
                </div>
              </div>

              <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:20 }}>
                <div style={card()}><ProbabilityChart approvalProb={result.approvalProb} defaultProb={result.defaultProb}/></div>
                <div style={card()}><MetricsBarChart creditScore={Number(form.creditScore)} dti={Number(form.dti)} ltv={Number(form.ltv)} income={Number(form.annualIncome)}/></div>
              </div>

              <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:20 }}>
                <div style={card()}><RiskProfileChart creditScore={Number(form.creditScore)} dti={Number(form.dti)} ltv={Number(form.ltv)} income={Number(form.annualIncome)} approvalProb={result.approvalProb}/></div>
                <div style={card()}><ProbabilityDistributionChart approvalProb={result.approvalProb} defaultProb={result.defaultProb}/></div>
              </div>

              <div style={card()}>
                <h2 style={{ fontSize:'1.3rem', color:verdict.color, marginBottom:12 }}>{result.headline}</h2>
                <p style={{ fontSize:'0.9rem', color:D.text2, marginBottom:12, lineHeight:1.7 }}>{result.summary}</p>
                <div style={{ padding:'12px 14px', background:verdict.dim, border:`1px solid ${verdict.color}33`, borderRadius:10 }}>
                  <p style={{ margin:0, fontSize:'0.85rem', color:D.text }}><strong>Timeline:</strong> {result.timelineMsg}</p>
                </div>
              </div>

              {result.improvements?.length > 0 && (
                <div style={card()}>
                  <h3 style={{ fontSize:'1.1rem', color:D.text, marginBottom:18 }}>📈 How to Improve Your Chances</h3>
                  <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
                    {result.improvements.map((imp,i) => (
                      <div key={i} style={{ background:D.bg3, borderRadius:12, padding:'16px', border:`1px solid ${D.border}` }}>
                        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:10 }}>
                          <span style={{ fontWeight:600, fontSize:'0.9rem', color:D.text }}>{i+1}. {imp.area}</span>
                          <span style={{ fontSize:'0.7rem', fontWeight:700, color:imp.priority==='CRITICAL'?D.red:imp.priority==='HIGH'?D.amber:D.green, background:imp.priority==='CRITICAL'?D.redDim:imp.priority==='HIGH'?D.amberDim:D.greenDim, padding:'3px 10px', borderRadius:100 }}>{imp.priority}</span>
                        </div>
                        <div style={{ fontSize:'0.8rem', color:D.text3, marginBottom:8 }}>Current: <strong>{imp.current}</strong> → Target: <strong>{imp.target}</strong> (Gap: <strong>{imp.gap}</strong>)</div>
                        <div style={{ fontSize:'0.8rem', color:D.text3, marginBottom:10 }}>Timeline: {imp.timeline}</div>
                        <ul style={{ listStyle:'none', margin:0, padding:0, display:'flex', flexDirection:'column', gap:6 }}>
                          {imp.actions?.map((action,j) => (
                            <li key={j} style={{ fontSize:'0.78rem', color:D.text2, display:'flex', gap:8 }}><span>•</span>{action}</li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {result.nextSteps?.length > 0 && (
                <div style={card()}>
                  <h3 style={{ fontSize:'1.1rem', color:D.text, marginBottom:16 }}>Next Steps</h3>
                  <ol style={{ margin:0, padding:0, display:'flex', flexDirection:'column', gap:10 }}>
                    {result.nextSteps.map((step,i) => (
                      <li key={i} style={{ fontSize:'0.88rem', color:D.text2, paddingLeft:24 }}>{step}</li>
                    ))}
                  </ol>
                </div>
              )}

              <div style={{ background:D.blue+'11', border:`1px solid ${D.blue}33`, borderRadius:12, padding:'16px', fontSize:'0.8rem', color:D.text2 }}>
                <strong>📊 Prediction Details:</strong> Default probability: <strong>{(result.defaultProb*100).toFixed(1)}%</strong> | Approval probability: <strong>{(result.approvalProb*100).toFixed(1)}%</strong>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}