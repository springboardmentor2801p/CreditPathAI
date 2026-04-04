import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import Plotly from 'plotly.js-dist-min'

const D = {
  bg0:'#060d18', bg1:'#0a1525', bg2:'#0f1e35', bg3:'#162844', bg4:'#1e3354',
  text:'#f0f4ff', text2:'#94a3c4', text3:'#5a6f94',
  border:'rgba(255,255,255,0.07)', border2:'rgba(255,255,255,0.12)',
  blue:'#3b82f6', green:'#22c55e', greenText:'#86efac',
  red:'#ef4444', amber:'#f59e0b', orange:'#f97316',
}

const priorityColor = { CRITICAL:'#ef4444', HIGH:'#f97316', MEDIUM:'#f59e0b', LOW:'#22c55e' }
const priorityDim   = { CRITICAL:'rgba(239,68,68,0.1)', HIGH:'rgba(249,115,22,0.1)', MEDIUM:'rgba(245,158,11,0.1)', LOW:'rgba(34,197,94,0.1)' }

function GaugeChart({ prob, priority }) {
  const ref = useRef(null)
  useEffect(() => {
    if (!ref.current) return
    const color = priorityColor[priority] || D.blue
    Plotly.react(ref.current, [{
      type: 'indicator', mode: 'gauge+number',
      value: +(prob * 100).toFixed(1),
      number: { suffix: '%', font: { color, size: 36 } },
      gauge: {
        axis: { range:[0,100], tickwidth:1, tickcolor:D.text3, tickfont:{color:D.text3,size:11} },
        bar: { color, thickness:0.3 },
        bgcolor: D.bg3, borderwidth: 0,
        steps: [
          { range:[0,15],  color:'rgba(34,197,94,0.15)'  },
          { range:[15,40], color:'rgba(245,158,11,0.15)' },
          { range:[40,60], color:'rgba(249,115,22,0.15)' },
          { range:[60,100],color:'rgba(239,68,68,0.15)'  },
        ],
        threshold: { line:{color,width:3}, thickness:0.75, value:prob*100 },
      },
    }], {
      paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
      font:{color:D.text}, margin:{t:20,b:10,l:20,r:20}, height:220,
    }, { displayModeBar:false, responsive:true })
  }, [prob, priority])
  return <div ref={ref} style={{ width:'100%' }} />
}

function RiskBarChart({ creditScore, ltv, dti }) {
  const ref = useRef(null)
  useEffect(() => {
    if (!ref.current) return
    const factors = [
      { name:'Credit Score', actual:creditScore, safeMin:700, color:creditScore>=700?'#22c55e':creditScore>=600?'#f59e0b':'#ef4444', safeLabel:'Safe: ≥700' },
      { name:'LTV Ratio',    actual:ltv,         safeMin:80,  color:ltv<=80?'#22c55e':ltv<=90?'#f59e0b':'#ef4444',                 safeLabel:'Safe: ≤80%' },
      { name:'DTI Ratio',    actual:dti,         safeMin:36,  color:dti<=36?'#22c55e':dti<=50?'#f59e0b':'#ef4444',                 safeLabel:'Safe: ≤36%' },
    ]
    Plotly.react(ref.current, [
      {
        type:'bar', orientation:'h', name:'Your Value',
        y:factors.map(f=>f.name), x:factors.map(f=>f.actual),
        marker:{ color:factors.map(f=>f.color), opacity:0.9 },
        text:factors.map(f=>`${f.actual}`), textposition:'outside',
        textfont:{ color:D.text, size:13 },
        hovertemplate:'%{y}: %{x}<extra></extra>',
      },
      {
        type:'scatter', mode:'markers+text', name:'Safe Limit',
        y:factors.map(f=>f.name), x:factors.map(f=>f.safeMin),
        marker:{ symbol:'line-ns', size:28, color:'#ffffff', line:{color:'#ffffff',width:2} },
        text:factors.map(f=>f.safeLabel), textposition:'top center',
        textfont:{ color:D.text3, size:9 },
        hovertemplate:'Safe limit: %{x}<extra></extra>',
      },
    ], {
      paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
      font:{ color:D.text }, margin:{t:10,b:40,l:100,r:80}, height:240,
      xaxis:{ showgrid:true, gridcolor:'rgba(255,255,255,0.05)', zeroline:false, showticklabels:false },
      yaxis:{ tickfont:{color:D.text2,size:13}, gridcolor:'rgba(255,255,255,0.03)' },
      legend:{ font:{color:D.text2,size:11}, bgcolor:'rgba(0,0,0,0)', orientation:'h', x:0, y:-0.15 },
      bargap:0.45,
    }, { displayModeBar:false, responsive:true })
  }, [creditScore, ltv, dti])
  return <div ref={ref} style={{ width:'100%' }} />
}

const fields = [
  { key:'creditScore', label:'Credit Score',     ph:'300 – 850',    type:'number' },
  { key:'loanAmount',  label:'Loan Amount (₹)',   ph:'e.g. 2500000', type:'number' },
  { key:'income',      label:'Annual Income (₹)', ph:'e.g. 3500000', type:'number' },
  { key:'ltv',         label:'LTV (%)',           ph:'e.g. 70',      type:'number' },
  { key:'dti',         label:'DTI (%)',           ph:'e.g. 25',      type:'number' },
]

export default function BankPortal({ onBack }) {
  const [form, setForm]     = useState({ creditScore:'', loanAmount:'', income:'', ltv:'', dti:'' })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [errors, setErrors] = useState({})

  const set = (k, v) => setForm(p => ({ ...p, [k]: v }))

  function validate() {
    const e = {}
    const cs  = parseFloat(form.creditScore)
    const la  = parseFloat(form.loanAmount)
    const inc = parseFloat(form.income)
    const l   = parseFloat(form.ltv)
    const d   = parseFloat(form.dti)
    if (!form.creditScore || isNaN(cs) || cs<300 || cs>850) e.creditScore = 'Enter 300–850'
    if (!form.loanAmount  || isNaN(la)  || la<=0)           e.loanAmount  = 'Required'
    if (!form.income      || isNaN(inc) || inc<=0)          e.income      = 'Required'
    if (!form.ltv || isNaN(l) || l<=0 || l>100)            e.ltv         = 'Enter 1–100'
    if (!form.dti || isNaN(d) || d<=0 || d>100)            e.dti         = 'Enter 1–100'
    setErrors(e)
    return Object.keys(e).length === 0
  }

  async function runAnalysis() {
    if (!validate()) return
    setLoading(true); setResult(null)
    try {
      const payload = {
        Credit_Score: parseFloat(form.creditScore),
        loan_amount:  parseFloat(form.loanAmount),
        income:       parseFloat(form.income),
        LTV:          parseFloat(form.ltv),
        dtir1:        parseFloat(form.dti),
      }
      const response = await axios.post('http://127.0.0.1:8000/bank-recommendation', payload, {
        headers: { 'Content-Type': 'application/json' }
      })
      const rec = response.data.recommendation
      setResult({
        prob:           rec.default_probability,
        priority:       rec.risk_level,
        expectedLoss:   rec.expected_loss,
        team:           rec.assigned_team,
        channel:        rec.recovery_channel,
        followUp:       rec.follow_up_frequency,
        legalRequired:  rec.legal_action_required,
        insights:       rec.insights || [],
        approval:       rec.approval_status,
        rateAdjustment: rec.interest_rate_adjustment,
        creditScore:    parseFloat(form.creditScore),
        ltv:            parseFloat(form.ltv),
        dti:            parseFloat(form.dti),
      })
    } catch (err) {
      console.error('Error:', err)
      alert(`API Error: ${err.response?.status || err.message}`)
    }
    setLoading(false)
  }

  const inpStyle = (k) => ({
    width:'100%', padding:'11px 14px', boxSizing:'border-box',
    background:D.bg3, border:`1px solid ${errors[k] ? '#ef444466' : D.border}`,
    borderRadius:10, color:D.text, fontSize:'0.88rem', outline:'none',
  })
  const lbl = { display:'block', fontSize:'0.72rem', fontWeight:600, color:D.text3, marginBottom:5, letterSpacing:'0.04em', textTransform:'uppercase' }

  return (
    <div style={{ minHeight:'100vh', background:D.bg0, fontFamily:'sans-serif' }}>

      {/* Header */}
      <div style={{ background:D.bg1, borderBottom:`1px solid ${D.border}`, padding:'32px 5% 28px' }}>
        <div style={{ maxWidth:1400, margin:'0 auto' }}>
          <button onClick={onBack} style={{
            background:'none', border:`1px solid ${D.border}`, color:D.text2,
            borderRadius:8, padding:'6px 14px', fontSize:'0.8rem', cursor:'pointer',
            marginBottom:16, display:'flex', alignItems:'center', gap:6,
          }}>← Back</button>
          <div style={{ display:'inline-flex', alignItems:'center', gap:6, background:'rgba(167,139,250,0.1)', border:'1px solid rgba(167,139,250,0.2)', borderRadius:100, padding:'4px 14px', fontSize:'0.68rem', fontWeight:700, color:'#c4b5fd', letterSpacing:'.07em', textTransform:'uppercase', marginBottom:12 }}>
            Recovery Agent System
          </div>
          <h1 style={{ fontSize:'clamp(1.7rem,3.5vw,2.4rem)', color:D.text, marginBottom:6 }}>
            Agent <em style={{ fontStyle:'italic', color:'#60a5fa' }}>Recommendation Engine</em>
          </h1>
          <p style={{ fontSize:'0.88rem', color:D.text2 }}>🏦 <strong>BANK PERSPECTIVE:</strong> Risk assessment & recovery strategy (powered by ML model)</p>
        </div>
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'340px 1fr', gap:24, padding:'32px 5%', maxWidth:1400, margin:'0 auto', alignItems:'start' }}>

        {/* Form */}
        <div style={{ background:D.bg2, border:`1px solid ${D.border}`, borderRadius:16, padding:'28px', position:'sticky', top:28 }}>
          <h2 style={{ fontSize:'1.2rem', color:D.text, marginBottom:24 }}>Borrower Details</h2>
          <div style={{ display:'flex', flexDirection:'column', gap:16 }}>
            {fields.map(({ key, label, ph, type }) => (
              <div key={key}>
                <label style={lbl}>{label}</label>
                <input type={type} placeholder={ph} value={form[key]}
                  onChange={e => set(key, e.target.value)} style={inpStyle(key)} />
                {errors[key] && <p style={{ color:D.red, fontSize:'0.72rem', marginTop:4 }}>{errors[key]}</p>}
              </div>
            ))}
          </div>

          <button onClick={runAnalysis} disabled={loading} style={{
            width:'100%', marginTop:24, padding:'13px',
            background: loading ? D.bg4 : 'linear-gradient(135deg,#1d4ed8,#3b82f6)',
            color:'#fff', border:'none', borderRadius:10,
            fontWeight:700, fontSize:'0.88rem', cursor: loading ? 'not-allowed' : 'pointer',
          }}>
            {loading ? 'Analysing…' : '🔍 Get Recommendation →'}
          </button>

          <div style={{ marginTop:22, padding:'14px 16px', background:D.bg3, borderRadius:10, border:`1px solid ${D.border}` }}>
            <div style={{ fontSize:'0.68rem', fontWeight:700, color:D.text3, letterSpacing:'0.06em', textTransform:'uppercase', marginBottom:10 }}>Backend Status</div>
            <div style={{ fontSize:'0.75rem', color:D.greenText }}>✓ Connected to FastAPI Server</div>
            <div style={{ fontSize:'0.75rem', color:D.text3, marginTop:4 }}>http://127.0.0.1:8000</div>
          </div>
        </div>

        {/* Results */}
        <div style={{ display:'flex', flexDirection:'column', gap:20 }}>

          {!result && !loading && (
            <div style={{ background:D.bg2, border:`1px solid ${D.border}`, borderRadius:16, padding:'60px 28px', textAlign:'center' }}>
              <div style={{ fontSize:40, marginBottom:18 }}>🤖</div>
              <h3 style={{ fontSize:'1.4rem', color:D.text, marginBottom:10 }}>Ready to Analyse</h3>
              <p style={{ fontSize:'0.88rem', color:D.text2, maxWidth:300, margin:'0 auto', lineHeight:1.7 }}>
                Fill in borrower details and click <strong>Get Recommendation</strong> to generate a bank risk assessment.
              </p>
            </div>
          )}

          {loading && (
            <div style={{ background:D.bg2, border:`1px solid ${D.border}`, borderRadius:16, padding:'80px 28px', textAlign:'center' }}>
              <p style={{ fontSize:'1.1rem', color:D.text }}>Getting model prediction from backend...</p>
            </div>
          )}

          {result && (() => {
            const pc = priorityColor[result.priority] || D.blue
            const pd = priorityDim[result.priority]  || 'rgba(59,130,246,0.1)'
            return (
              <>
                {/* Row 1: Stats + Insights */}
                <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:20 }}>
                  <div style={{ background:D.bg2, border:`1px solid ${pc}44`, borderRadius:16, padding:'26px' }}>
                    <div style={{ display:'inline-flex', alignItems:'center', gap:10, background:pd, border:`1px solid ${pc}44`, borderRadius:100, padding:'7px 16px', marginBottom:16 }}>
                      <span style={{ fontWeight:700, color:pc }}>{result.priority} Priority</span>
                    </div>
                    <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:10, marginTop:8 }}>
                      {[
                        { label:'Default Probability', val:`${(result.prob*100).toFixed(1)}%`,                                            color:pc },
                        { label:'Approval',            val:result.approval,                                                               color:pc },
                        { label:'Expected Loss',       val:`₹${Math.round(result.expectedLoss).toLocaleString()}`,                        color:D.red },
                        { label:'Rate Adjust',         val:`${result.rateAdjustment>0?'+':''}${result.rateAdjustment.toFixed(2)}%`,        color:result.rateAdjustment>0?D.red:D.green },
                        { label:'Legal Action',        val:result.legalRequired?'Required':'Not Required',                                color:result.legalRequired?D.red:D.green },
                        { label:'Follow-up',           val:result.followUp,                                                               color:D.text2 },
                      ].map((s,i) => (
                        <div key={i} style={{ background:D.bg3, borderRadius:10, padding:'11px 13px', border:`1px solid ${D.border}` }}>
                          <div style={{ fontSize:'0.62rem', color:D.text3, marginBottom:4, textTransform:'uppercase' }}>{s.label}</div>
                          <div style={{ fontSize:'0.78rem', fontWeight:600, color:s.color }}>{s.val}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div style={{ background:D.bg2, border:`1px solid ${D.border}`, borderRadius:16, padding:'26px' }}>
                    <h3 style={{ fontSize:'1.1rem', color:D.text, marginBottom:18 }}>Risk Insights</h3>
                    <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
                      {result.insights.map((insight, i) => (
                        <div key={i} style={{ display:'flex', gap:12, padding:'12px 14px', background:D.bg3, borderRadius:10, border:`1px solid ${D.border}`, alignItems:'flex-start' }}>
                          <div style={{ width:22, height:22, borderRadius:'50%', background:pd, border:`1px solid ${pc}33`, display:'flex', alignItems:'center', justifyContent:'center', flexShrink:0, fontSize:'0.7rem', fontWeight:700, color:pc }}>{i+1}</div>
                          <span style={{ fontSize:'0.82rem', color:D.text2, lineHeight:1.6 }}>{insight}</span>
                        </div>
                      ))}
                    </div>
                    <div style={{ padding:'14px 16px', background:pd, border:`1px solid ${pc}33`, borderRadius:12, marginTop:16 }}>
                      <div style={{ fontSize:'0.65rem', fontWeight:700, color:pc, textTransform:'uppercase', marginBottom:6 }}>Recovery Channel</div>
                      <div style={{ fontSize:'0.9rem', fontWeight:600, color:D.text }}>{result.channel}</div>
                    </div>
                  </div>
                </div>

                {/* Row 2: Charts */}
                <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:20 }}>
                  <div style={{ background:D.bg2, border:`1px solid ${D.border}`, borderRadius:16, padding:'24px' }}>
                    <h3 style={{ fontSize:'1rem', color:D.text, marginBottom:2 }}>📊 Default Risk Gauge</h3>
                    <p style={{ fontSize:'0.75rem', color:D.text3, marginBottom:4 }}>ML model output — probability of loan default</p>
                    <GaugeChart prob={result.prob} priority={result.priority} />
                    <div style={{ display:'flex', justifyContent:'center', gap:12, flexWrap:'wrap', marginTop:4 }}>
                      {[['#22c55e','Low (0–15%)'],['#f59e0b','Medium (15–40%)'],['#f97316','High (40–60%)'],['#ef4444','Critical (60%+)']].map(([c,l]) => (
                        <div key={l} style={{ display:'flex', alignItems:'center', gap:4 }}>
                          <div style={{ width:9, height:9, borderRadius:'50%', background:c }}/>
                          <span style={{ fontSize:'0.65rem', color:D.text3 }}>{l}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div style={{ background:D.bg2, border:`1px solid ${D.border}`, borderRadius:16, padding:'24px' }}>
                    <h3 style={{ fontSize:'1rem', color:D.text, marginBottom:2 }}>📈 Risk Factor Analysis</h3>
                    <p style={{ fontSize:'0.75rem', color:D.text3, marginBottom:4 }}>Actual borrower values vs safe thresholds (white line)</p>
                    <div style={{ display:'flex', gap:14, marginBottom:8 }}>
                      {[['#22c55e','Safe'],['#f59e0b','Caution'],['#ef4444','Risky']].map(([c,l]) => (
                        <div key={l} style={{ display:'flex', alignItems:'center', gap:4 }}>
                          <div style={{ width:10, height:10, borderRadius:2, background:c }}/>
                          <span style={{ fontSize:'0.65rem', color:D.text3 }}>{l}</span>
                        </div>
                      ))}
                    </div>
                    <RiskBarChart creditScore={result.creditScore} ltv={result.ltv} dti={result.dti} />
                    <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:8, marginTop:8 }}>
                      {[
                        { label:'Credit Score', val:result.creditScore, safe:'≥700', good:result.creditScore>=700 },
                        { label:'LTV',          val:`${result.ltv}%`,   safe:'≤80%', good:result.ltv<=80 },
                        { label:'DTI',          val:`${result.dti}%`,   safe:'≤36%', good:result.dti<=36 },
                      ].map((f,i) => (
                        <div key={i} style={{ background:D.bg3, borderRadius:8, padding:'8px 10px', border:`1px solid ${D.border}`, textAlign:'center' }}>
                          <div style={{ fontSize:'0.6rem', color:D.text3, textTransform:'uppercase', marginBottom:3 }}>{f.label}</div>
                          <div style={{ fontSize:'0.9rem', fontWeight:700, color:f.good?'#22c55e':'#ef4444' }}>{f.val}</div>
                          <div style={{ fontSize:'0.6rem', color:D.text3, marginTop:2 }}>Safe: {f.safe}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Recovery details */}
                <div style={{ background:D.bg2, border:`1px solid ${D.border}`, borderRadius:16, padding:'26px 28px' }}>
                  <h3 style={{ fontSize:'1.1rem', color:D.text, marginBottom:16 }}>Recovery Details</h3>
                  <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:12 }}>
                    {[
                      { label:'Team',          val:result.team },
                      { label:'Channel',       val:result.channel },
                      { label:'Follow-up',     val:result.followUp },
                      { label:'Expected Loss', val:`₹${Math.round(result.expectedLoss).toLocaleString()}` },
                    ].map((s,i) => (
                      <div key={i} style={{ background:D.bg3, borderRadius:10, padding:'14px 12px', border:`1px solid ${D.border}` }}>
                        <div style={{ fontSize:'0.7rem', color:D.text3, marginBottom:6, fontWeight:600, textTransform:'uppercase' }}>{s.label}</div>
                        <div style={{ fontSize:'0.8rem', color:D.text }}>{s.val}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div style={{ background:D.blue+'11', border:`1px solid ${D.blue}33`, borderRadius:12, padding:'16px', fontSize:'0.8rem', color:D.text2 }}>
                  <strong>📊 Note:</strong> This recommendation is generated using your trained ML model via FastAPI backend. Default probability: <strong>{(result.prob*100).toFixed(1)}%</strong>
                </div>
              </>
            )
          })()}
        </div>
      </div>
    </div>
  )
}