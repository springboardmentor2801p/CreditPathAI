import { useState, useEffect, useRef } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Cell, Legend,
} from "recharts";

// ── Base API URL ──────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000";

// ── Global Styles ─────────────────────────────────────────────────────────────
const STYLE = `
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --navy:   #0a0f1e;
  --deep:   #0d1529;
  --card:   #111827;
  --border: #1e2d4a;
  --teal:   #00c2a8;
  --teal2:  #00e5c8;
  --amber:  #f59e0b;
  --rose:   #f43f5e;
  --muted:  #6b7fa3;
  --text:   #e2e8f0;
  --white:  #ffffff;
  --grad:   linear-gradient(135deg,#00c2a8 0%,#0ea5e9 100%);
  --grad2:  linear-gradient(135deg,#f59e0b 0%,#f43f5e 100%);
}

html { scroll-behavior: smooth; }
body {
  background: var(--navy);
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
  font-size: 15px;
  line-height: 1.6;
  min-height: 100vh;
  overflow-x: hidden;
}
h1,h2,h3,h4,h5 { font-family: 'Syne', sans-serif; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Nav */
.nav {
  position: fixed; top: 0; left: 0; right: 0; z-index: 100;
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 5%; height: 70px;
  background: rgba(10,15,30,0.92);
  backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--border);
}
.nav-logo { display:flex; align-items:center; gap:10px; cursor:pointer; }
.nav-logo-icon {
  width:36px; height:36px; border-radius:10px;
  background: var(--grad);
  display:flex; align-items:center; justify-content:center;
  font-family:'Syne',sans-serif; font-weight:800; font-size:14px; color:#fff;
}
.nav-logo-text { font-family:'Syne',sans-serif; font-weight:700; font-size:18px; }
.nav-logo-text span { color:var(--teal); }
.nav-links { display:flex; gap:32px; }
.nav-link {
  background:none; border:none; cursor:pointer;
  color:var(--muted); font-family:'DM Sans',sans-serif; font-size:14px;
  font-weight:500; transition:color .2s; padding:4px 0; position:relative;
}
.nav-link::after {
  content:''; position:absolute; bottom:-2px; left:0; width:0; height:2px;
  background:var(--teal); border-radius:2px; transition:width .25s;
}
.nav-link:hover, .nav-link.active { color:var(--white); }
.nav-link:hover::after, .nav-link.active::after { width:100%; }
.nav-cta { display:flex; gap:10px; }

/* Buttons */
.btn {
  display:inline-flex; align-items:center; justify-content:center; gap:8px;
  padding:10px 22px; border-radius:10px; font-family:'DM Sans',sans-serif;
  font-weight:500; font-size:14px; cursor:pointer; border:none;
  transition: transform .15s, opacity .2s;
}
.btn:hover:not(:disabled) { transform:translateY(-1px); opacity:.92; }
.btn:active { transform:translateY(0); }
.btn:disabled { opacity:.5; cursor:not-allowed; }
.btn-primary { background:var(--grad);  color:#fff; box-shadow:0 4px 18px rgba(0,194,168,.28); }
.btn-outline  { background:transparent; color:var(--teal); border:1.5px solid var(--teal); }
.btn-sm  { padding:7px 16px;  font-size:13px; border-radius:8px; }
.btn-lg  { padding:14px 32px; font-size:16px; border-radius:12px; }

/* Page */
.page { min-height:100vh; padding-top:70px; }

/* Hero */
.hero {
  min-height:calc(100vh - 70px);
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  text-align:center; padding:80px 5% 60px; position:relative; overflow:hidden;
}
.hero-glow { position:absolute; border-radius:50%; filter:blur(90px); pointer-events:none; animation:pulse 6s ease-in-out infinite; }
.hero-glow-1 { width:600px; height:600px; background:rgba(0,194,168,.10); top:-100px; left:-100px; }
.hero-glow-2 { width:500px; height:500px; background:rgba(14,165,233,.09); bottom:-100px; right:-100px; animation-delay:3s; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.7;transform:scale(1.08)} }
.hero-badge {
  display:inline-flex; align-items:center; gap:7px;
  background:rgba(0,194,168,.12); border:1px solid rgba(0,194,168,.3);
  color:var(--teal); padding:6px 16px; border-radius:99px;
  font-size:12px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; margin-bottom:28px;
}
.hero h1 { font-size:clamp(2.4rem,6vw,4.2rem); font-weight:800; line-height:1.1; margin-bottom:22px; }
.hero h1 .accent { background:var(--grad); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { max-width:600px; color:var(--muted); font-size:1.05rem; margin-bottom:42px; line-height:1.8; }
.hero-btns { display:flex; gap:14px; flex-wrap:wrap; justify-content:center; }
.hero-stats { display:flex; gap:48px; margin-top:70px; flex-wrap:wrap; justify-content:center; }
.hero-stat { text-align:center; }
.hero-stat-num { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:var(--teal); }
.hero-stat-label { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }

/* Section */
.section { padding:90px 5%; }
.section-header { text-align:center; margin-bottom:56px; }
.section-header h2 { font-size:clamp(1.8rem,4vw,2.8rem); font-weight:800; margin-bottom:14px; }
.section-header p { color:var(--muted); max-width:560px; margin:0 auto; }

/* Cards */
.cards-grid { display:grid; gap:24px; }
.grid-3 { grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); }
.grid-2 { grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }
.card {
  background:var(--card); border:1px solid var(--border); border-radius:18px; padding:30px;
  transition:border-color .2s, transform .2s, box-shadow .2s;
}
.card:hover { border-color:rgba(0,194,168,.35); transform:translateY(-3px); box-shadow:0 12px 40px rgba(0,0,0,.35); }
.card-icon { width:52px; height:52px; border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:22px; margin-bottom:18px; }
.card-icon-teal  { background:rgba(0,194,168,.15); }
.card-icon-amber { background:rgba(245,158,11,.15); }
.card-icon-rose  { background:rgba(244,63,94,.15);  }
.card-icon-blue  { background:rgba(14,165,233,.15); }
.card h3 { font-size:1.1rem; font-weight:700; margin-bottom:10px; }
.card p  { color:var(--muted); font-size:.9rem; line-height:1.7; }

/* Form */
.form-page { max-width:840px; margin:0 auto; padding:40px 5% 80px; }
.form-header { margin-bottom:36px; }
.form-header h2 { font-size:1.9rem; font-weight:800; margin-bottom:8px; }
.form-header p  { color:var(--muted); }
.form-card { background:var(--card); border:1px solid var(--border); border-radius:20px; padding:38px; }
.form-section-title {
  font-family:'Syne',sans-serif; font-weight:700; font-size:.78rem; letter-spacing:.1em;
  text-transform:uppercase; color:var(--teal); margin-bottom:20px; padding-bottom:10px;
  border-bottom:1px solid var(--border);
}
.form-grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:24px; }
.form-group { display:flex; flex-direction:column; gap:7px; }
.form-group.full { grid-column:1/-1; }
label { font-size:12px; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }

input, select {
  background:#0d1529; border:1.5px solid var(--border); border-radius:10px;
  color:var(--text); font-family:'DM Sans',sans-serif; font-size:14px;
  padding:11px 14px; width:100%; outline:none;
  transition:border-color .2s, box-shadow .2s; -webkit-appearance:none;
}
input:focus, select:focus { border-color:var(--teal); box-shadow:0 0 0 3px rgba(0,194,168,.12); }
input::placeholder { color:#3a4a6a; }
input.err { border-color:var(--rose); }
select { cursor:pointer; }
select option { background:var(--deep); }
.hint { font-size:11px; color:var(--muted); margin-top:3px; }
.hint-err { font-size:11px; color:var(--rose); margin-top:3px; }
.form-divider { height:1px; background:var(--border); margin:28px 0; }
.form-submit  { display:flex; justify-content:flex-end; gap:12px; margin-top:32px; }

/* Error banner */
.error-banner {
  background:rgba(244,63,94,.09); border:1.5px solid rgba(244,63,94,.3);
  border-radius:12px; padding:16px 18px; margin-top:22px;
  display:flex; align-items:flex-start; gap:12px; font-size:.88rem;
}
.err-icon { font-size:18px; flex-shrink:0; margin-top:1px; }
.err-text   { color:var(--rose); font-weight:600; }
.err-detail { color:var(--muted); font-size:.82rem; margin-top:5px; }
code { font-family:monospace; color:var(--teal); background:rgba(0,194,168,.08); padding:2px 6px; border-radius:4px; }

/* Results */
@keyframes slideUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
.result-wrap { margin-top:28px; display:flex; flex-direction:column; gap:20px; animation:slideUp .35s ease; }
.result-panel { border-radius:16px; padding:28px 30px; }
.panel-low      { background:rgba(0,194,168,.08); border:1.5px solid rgba(0,194,168,.3);  }
.panel-medium   { background:rgba(245,158,11,.07); border:1.5px solid rgba(245,158,11,.28);}
.panel-high     { background:rgba(244,63,94,.07);  border:1.5px solid rgba(244,63,94,.28); }
.panel-critical { background:rgba(244,63,94,.11);  border:1.5px solid rgba(244,63,94,.5);  }

.result-header { display:flex; align-items:center; gap:14px; margin-bottom:18px; }
.result-icon { font-size:34px; }
.result-title { font-family:'Syne',sans-serif; font-size:1.25rem; font-weight:800; }
.result-sub   { font-size:.84rem; color:var(--muted); margin-top:3px; }

.c-low      { color:var(--teal2); }
.c-medium   { color:var(--amber); }
.c-high     { color:var(--rose);  }
.c-critical { color:#ff2d55;      }

/* Score bar */
.bar-wrap { margin:18px 0; }
.bar-labels { display:flex; justify-content:space-between; font-size:13px; margin-bottom:8px; }
.bar-track { height:10px; background:var(--border); border-radius:99px; overflow:hidden; }
.bar-fill { height:100%; border-radius:99px; background:var(--grad); transition:width 1.3s cubic-bezier(.22,1,.36,1); }
.bar-amber { background:linear-gradient(90deg,#f59e0b,#fbbf24); }
.bar-red   { background:var(--grad2); }

/* Info grid */
.info-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:18px 0; }
.info-tile { background:rgba(255,255,255,.03); border:1px solid var(--border); border-radius:10px; padding:12px 14px; }
.info-tile-label { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); margin-bottom:5px; }
.info-tile-value { font-family:'Syne',sans-serif; font-weight:700; font-size:1rem; }

/* Plan */
.plan-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:14px; }
.plan-item { background:rgba(255,255,255,.03); border-radius:9px; padding:12px 14px; }
.plan-label { font-size:10px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); margin-bottom:5px; }
.plan-value { font-size:.88rem; font-weight:600; }
.legal-badge { display:inline-flex; align-items:center; gap:6px; padding:4px 12px; border-radius:99px; font-size:.78rem; font-weight:700; margin-top:10px; }
.legal-yes { background:rgba(244,63,94,.15); color:var(--rose); }
.legal-no  { background:rgba(0,194,168,.12); color:var(--teal2); }

/* List */
.list-block { margin-top:16px; }
.list-block h4 { font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); margin-bottom:10px; }
.list-item { display:flex; align-items:flex-start; gap:10px; padding:10px 14px; border-radius:9px; background:rgba(255,255,255,.03); margin-bottom:7px; font-size:.88rem; line-height:1.5; }
.ldot { margin-top:5px; width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.ldot-t { background:var(--teal); }
.ldot-a { background:var(--amber); }
.ldot-r { background:var(--rose); }

/* Approval badge */
.appr-badge { display:inline-flex; align-items:center; gap:8px; padding:8px 18px; border-radius:99px; font-weight:700; font-size:.88rem; margin-bottom:14px; }
.appr-high   { background:rgba(0,194,168,.15);  color:var(--teal2); border:1px solid rgba(0,194,168,.3); }
.appr-medium { background:rgba(245,158,11,.15); color:var(--amber);  border:1px solid rgba(245,158,11,.3); }
.appr-low    { background:rgba(244,63,94,.15);  color:var(--rose);   border:1px solid rgba(244,63,94,.3); }

/* Portal select */
.portal-select {
  min-height:calc(100vh - 70px);
  display:flex; flex-direction:column; align-items:center; justify-content:center; padding:40px 5%;
}
.portal-select h2 { font-size:clamp(1.6rem,4vw,2.4rem); font-weight:800; margin-bottom:10px; text-align:center; }
.portal-select > p { color:var(--muted); text-align:center; margin-bottom:44px; }
.portal-cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:24px; max-width:760px; width:100%; }
.portal-card { background:var(--card); border:1.5px solid var(--border); border-radius:20px; padding:36px 30px; cursor:pointer; transition:border-color .2s, transform .2s, box-shadow .2s; }
.portal-card:hover { transform:translateY(-4px); }
.pc-teal:hover  { border-color:var(--teal);  box-shadow:0 16px 48px rgba(0,194,168,.15); }
.pc-amber:hover { border-color:var(--amber); box-shadow:0 16px 48px rgba(245,158,11,.12); }
.portal-card-icon { width:62px; height:62px; border-radius:16px; display:flex; align-items:center; justify-content:center; font-size:26px; margin-bottom:20px; }
.portal-card h3 { font-size:1.2rem; font-weight:800; margin-bottom:8px; }
.portal-card p  { color:var(--muted); font-size:.87rem; line-height:1.7; }
.portal-card .arrow { display:inline-flex; align-items:center; gap:6px; font-size:.82rem; font-weight:600; margin-top:18px; }

/* Auth */
.auth-page { min-height:100vh; display:flex; align-items:center; justify-content:center; padding:80px 5% 40px; }
.auth-card { background:var(--card); border:1px solid var(--border); border-radius:24px; padding:48px 44px; width:100%; max-width:460px; }
.auth-logo { text-align:center; margin-bottom:32px; }
.logo-box { width:52px; height:52px; border-radius:14px; background:var(--grad); display:flex; align-items:center; justify-content:center; font-family:'Syne',sans-serif; font-weight:800; font-size:20px; color:#fff; margin:0 auto 12px; }
.auth-logo h2 { font-size:1.5rem; font-weight:800; }
.auth-logo p  { font-size:.85rem; color:var(--muted); margin-top:6px; }
.tab-row { display:flex; gap:4px; background:#0a0f1e; padding:5px; border-radius:12px; margin-bottom:28px; }
.tab-btn { flex:1; padding:9px; border-radius:9px; border:none; cursor:pointer; font-family:'DM Sans',sans-serif; font-weight:600; font-size:13px; transition:background .2s,color .2s; background:transparent; color:var(--muted); }
.tab-btn.active { background:var(--teal); color:#fff; }

/* Spinner */
.spinner { width:18px; height:18px; border:2px solid rgba(255,255,255,.3); border-top-color:#fff; border-radius:50%; animation:spin .7s linear infinite; display:inline-block; }
@keyframes spin { to{transform:rotate(360deg)} }

/* About hero */
.about-hero { padding:100px 5% 70px; background:radial-gradient(ellipse at 20% 50%,rgba(0,194,168,.06) 0%,transparent 60%),radial-gradient(ellipse at 80% 50%,rgba(14,165,233,.05) 0%,transparent 60%); }
.about-hero h1 { font-size:clamp(2rem,5vw,3.4rem); font-weight:800; margin-bottom:18px; }
.about-hero p  { max-width:620px; color:var(--muted); line-height:1.8; }

/* Footer */
footer { border-top:1px solid var(--border); padding:36px 5% 28px; display:flex; flex-wrap:wrap; gap:20px; justify-content:space-between; align-items:center; }
.foot-copy  { font-size:.82rem; color:var(--muted); }
.foot-links { display:flex; gap:24px; }
.foot-link  { font-size:.82rem; color:var(--muted); cursor:pointer; background:none; border:none; transition:color .2s; }
.foot-link:hover { color:var(--teal); }

@media(max-width:680px){
  .form-grid { grid-template-columns:1fr; }
  .nav-links { display:none; }
  .auth-card { padding:32px 22px; }
  .form-card { padding:22px; }
  .hero-stats { gap:28px; }
  .plan-grid { grid-template-columns:1fr; }
}

/* ── Charts ── */
.chart-section {
  margin-top:24px;
  background:var(--card);
  border:1px solid var(--border);
  border-radius:16px;
  padding:26px 28px;
}
.chart-section-title {
  font-family:'Syne',sans-serif; font-weight:700; font-size:.78rem;
  letter-spacing:.1em; text-transform:uppercase; color:var(--teal);
  margin-bottom:20px; padding-bottom:10px; border-bottom:1px solid var(--border);
  display:flex; align-items:center; gap:8px;
}
.charts-row { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:20px; }
@media(max-width:760px){ .charts-row { grid-template-columns:1fr; } }

/* Gauge */
.gauge-container { display:flex; flex-direction:column; align-items:center; }
.gauge-labels { display:flex; justify-content:space-between; width:200px; margin-top:4px; font-size:11px; color:var(--muted); }
.gauge-center-label { text-align:center; margin-top:-10px; }
.gauge-center-label .gauge-val { font-family:'Syne',sans-serif; font-size:1.7rem; font-weight:800; }
.gauge-center-label .gauge-sub { font-size:11px; color:var(--muted); }

/* Threshold bars */
.thresh-bars { display:flex; flex-direction:column; gap:14px; }
.thresh-row {}
.thresh-header { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:5px; }
.thresh-name { font-size:12px; font-weight:600; color:var(--text); }
.thresh-vals { font-size:11px; color:var(--muted); }
.thresh-track { position:relative; height:10px; background:var(--border); border-radius:99px; overflow:visible; }
.thresh-actual { height:100%; border-radius:99px; transition:width 1.3s cubic-bezier(.22,1,.36,1); }
.thresh-line {
  position:absolute; top:-4px; bottom:-4px; width:2px; border-radius:2px;
  background:rgba(255,255,255,.55);
}
.thresh-line-label {
  position:absolute; top:-18px; font-size:9px; color:rgba(255,255,255,.5);
  transform:translateX(-50%); white-space:nowrap;
}

/* Recharts overrides */
.recharts-text { fill: var(--muted) !important; font-size:11px !important; font-family:'DM Sans',sans-serif !important; }
.recharts-cartesian-grid-horizontal line,
.recharts-cartesian-grid-vertical line { stroke: var(--border) !important; }
.recharts-tooltip-wrapper .recharts-default-tooltip {
  background: var(--deep) !important; border:1px solid var(--border) !important;
  border-radius:10px !important; font-size:12px !important;
}
.recharts-polar-grid-angle line,
.recharts-polar-grid-concentric-polygon { stroke:var(--border) !important; }

.recharts-polar-grid-angle line,
.recharts-polar-grid-concentric-polygon { stroke:var(--border) !important; }

/* PASTE CONTACT CSS HERE */
.contact-page{
padding:100px 5%;
background:var(--navy);
color:var(--text);
}

.contact-header{
text-align:center;
margin-bottom:50px;
}

.contact-header h1{
font-size:42px;
font-weight:800;
}

.contact-header span{
color:var(--teal);
}

.contact-header p{
color:var(--muted);
margin-top:10px;
}

.contact-container{
display:grid;
grid-template-columns:1fr 1.2fr;
gap:30px;
}

.contact-info{
display:flex;
flex-direction:column;
gap:20px;
}

.contact-card{
display:flex;
gap:15px;
background:var(--card);
border:1px solid var(--border);
padding:20px;
border-radius:14px;
}

.contact-card .icon{
width:40px;
height:40px;
background:rgba(0,194,168,.15);
border-radius:10px;
display:flex;
align-items:center;
justify-content:center;
}

.contact-form{
background:var(--card);
border:1px solid var(--border);
padding:30px;
border-radius:16px;
}

.contact-form .form-grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:15px;
}

.contact-form textarea{
margin-top:15px;
height:120px;
resize:none;
}

.contact-btn{
margin-top:20px;
background:var(--grad);
border:none;
padding:12px;
border-radius:8px;
color:white;
width:100%;
cursor:pointer;
font-weight:600;
}
`;

// ── Helpers ───────────────────────────────────────────────────────────────────

function panelClass(level) {
  if (!level) return "panel-low";
  switch (level.toLowerCase()) {
    case "low":      return "panel-low";
    case "medium":   return "panel-medium";
    case "high":     return "panel-high";
    case "critical": return "panel-critical";
    default:         return "panel-low";
  }
}
function colorClass(level) {
  if (!level) return "c-low";
  switch (level.toLowerCase()) {
    case "low":      return "c-low";
    case "medium":   return "c-medium";
    case "high":     return "c-high";
    case "critical": return "c-critical";
    default:         return "c-low";
  }
}
function riskIcon(level) {
  switch ((level || "").toLowerCase()) {
    case "low":      return "✅";
    case "medium":   return "⚠️";
    case "high":     return "🚨";
    case "critical": return "🔴";
    default:         return "📋";
  }
}
function approvalPanelClass(chance) {
  switch ((chance || "").toLowerCase()) {
    case "high":   return "panel-low";
    case "medium": return "panel-medium";
    default:       return "panel-high";
  }
}
function approvalBadgeClass(chance) {
  switch ((chance || "").toLowerCase()) {
    case "high":   return "appr-high";
    case "medium": return "appr-medium";
    default:       return "appr-low";
  }
}
function barColorClass(prob) {
  if (prob <= 0.40) return "";
  if (prob <= 0.65) return "bar-amber";
  return "bar-red";
}

// ── Score bar ─────────────────────────────────────────────────────────────────
function ScoreBar({ value, max = 1, label, valueLabel }) {
  const [filled, setFilled] = useState(false);
  useEffect(() => { const t = setTimeout(() => setFilled(true), 100); return () => clearTimeout(t); }, []);
  const pct = Math.min(100, Math.round((value / max) * 100));
  const colorCls = max === 1 ? barColorClass(value) : "";
  return (
    <div className="bar-wrap">
      <div className="bar-labels">
        <span>{label}</span>
        <span style={{ fontWeight: 700 }}>{valueLabel || `${pct}%`}</span>
      </div>
      <div className="bar-track">
        <div className={`bar-fill ${colorCls}`} style={{ width: filled ? `${pct}%` : "0%" }} />
      </div>
    </div>
  );
}

// ── Derived metrics from form data ────────────────────────────────────────────
function deriveMetrics(fd) {
  const loan    = fd.loan_amount    || 0;
  const income  = fd.monthly_income || 1;
  const emi     = fd.existing_emi   || 0;
  const propVal = fd.property_value || 1;
  const score   = fd.credit_score   || 0;
  const age     = fd.age            || 0;
  const term    = fd.loan_term      || 1;

  const dti              = (emi / income) * 100;
  const ltv              = (loan / propVal) * 100;
  const loanIncomeMonths = loan / income;          // months-of-income
  const monthlyEmi       = loan / term;
  const totalEmi         = emi + monthlyEmi;
  const effectiveDti     = (totalEmi / income) * 100;
  return { dti, ltv, loanIncomeMonths, score, age, effectiveDti };
}

// ── SVG Gauge (default probability) ───────────────────────────────────────────
function DefaultGauge({ prob }) {
  const [animated, setAnimated] = useState(false);
  useEffect(() => { const t = setTimeout(() => setAnimated(true), 150); return () => clearTimeout(t); }, []);

  // Needle angle: prob=0 → π (left), prob=1 → 0 (right)
  const angle     = animated ? (1 - prob) * Math.PI : Math.PI;
  const tipX      = 100 + 65 * Math.cos(angle);
  const tipY      = 100 - 65 * Math.sin(angle);
  const pct       = Math.round(prob * 100);
  const valColor  = prob <= 0.40 ? "#00e5c8" : prob <= 0.65 ? "#f59e0b" : "#f43f5e";

  // Arc boundaries
  // 40% boundary: angle = 0.6π
  const b40x = 100 + 80 * Math.cos(0.6 * Math.PI);
  const b40y = 100 - 80 * Math.sin(0.6 * Math.PI);
  // 65% boundary: angle = 0.35π
  const b65x = 100 + 80 * Math.cos(0.35 * Math.PI);
  const b65y = 100 - 80 * Math.sin(0.35 * Math.PI);

  return (
    <div className="gauge-container">
      <svg viewBox="0 0 200 110" width="200" height="110" style={{ overflow:"visible" }}>
        {/* Background arc track */}
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#1e2d4a" strokeWidth="12" strokeLinecap="round" />

        {/* Green zone 0–40% */}
        <path d={`M 20 100 A 80 80 0 0 1 ${b40x.toFixed(1)} ${b40y.toFixed(1)}`}
          fill="none" stroke="#00c2a8" strokeWidth="12" strokeLinecap="butt" opacity="0.85" />
        {/* Amber zone 40–65% */}
        <path d={`M ${b40x.toFixed(1)} ${b40y.toFixed(1)} A 80 80 0 0 1 ${b65x.toFixed(1)} ${b65y.toFixed(1)}`}
          fill="none" stroke="#f59e0b" strokeWidth="12" strokeLinecap="butt" opacity="0.85" />
        {/* Red zone 65–100% */}
        <path d={`M ${b65x.toFixed(1)} ${b65y.toFixed(1)} A 80 80 0 0 1 180 100`}
          fill="none" stroke="#f43f5e" strokeWidth="12" strokeLinecap="butt" opacity="0.85" />

        {/* Needle */}
        <line x1="100" y1="100" x2={tipX.toFixed(1)} y2={tipY.toFixed(1)}
          stroke="white" strokeWidth="2.5" strokeLinecap="round"
          style={{ transition: animated ? "none" : undefined }} />
        {/* Needle pivot */}
        <circle cx="100" cy="100" r="6" fill="white" />
        <circle cx="100" cy="100" r="3" fill="#0a0f1e" />

        {/* Zone labels */}
        <text x="16" y="115" fontSize="9" fill="#00c2a8" textAnchor="middle">Low</text>
        <text x="100" y="12" fontSize="9" fill="#f59e0b" textAnchor="middle">Medium</text>
        <text x="184" y="115" fontSize="9" fill="#f43f5e" textAnchor="middle">High</text>
      </svg>

      <div className="gauge-center-label">
        <div className="gauge-val" style={{ color: valColor }}>{pct}%</div>
        <div className="gauge-sub">Default Probability</div>
      </div>
    </div>
  );
}

// ── Agent: metrics vs threshold bar chart ─────────────────────────────────────
const AGENT_TOOLTIP_STYLE = {
  backgroundColor:"#0d1529", border:"1px solid #1e2d4a",
  borderRadius:10, fontSize:12, color:"#e2e8f0",
};
function AgentMetricsChart({ fd, prob }) {
  const { dti, ltv, effectiveDti } = deriveMetrics(fd);
  const score = fd.credit_score || 0;

  // Normalise all to 0–100 scale for comparison
  // DTI: actual vs 40% threshold → show as % where 100 = threshold
  // LTV: actual vs 80% threshold
  // Credit Score: actual/900*100 vs 650/900*100
  const data = [
    {
      metric: "DTI Ratio",
      "Your Value": Math.min(parseFloat(effectiveDti.toFixed(1)), 120),
      "Safe Threshold": 40,
      unit: "%",
      overColor: "#f43f5e",
    },
    {
      metric: "LTV Ratio",
      "Your Value": Math.min(parseFloat(ltv.toFixed(1)), 120),
      "Safe Threshold": 80,
      unit: "%",
      overColor: "#f43f5e",
    },
    {
      metric: "Credit Score",
      "Your Value": Math.min(score, 900),
      "Safe Threshold": 650,
      unit: "",
      overColor: "#00c2a8", // for credit score higher is better
    },
    {
      metric: "Default Prob.",
      "Your Value": Math.min(parseFloat((prob * 100).toFixed(1)), 100),
      "Safe Threshold": 40,
      unit: "%",
      overColor: "#f43f5e",
    },
  ];

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    const unit = data.find(d => d.metric === label)?.unit || "";
    return (
      <div style={AGENT_TOOLTIP_STYLE}>
        <div style={{ fontWeight:700, marginBottom:6 }}>{label}</div>
        {payload.map(p => (
          <div key={p.name} style={{ color: p.color, marginBottom:3 }}>
            {p.name}: <strong>{p.value}{unit}</strong>
          </div>
        ))}
      </div>
    );
  };

  return (
    <ResponsiveContainer width="100%" height={230}>
      <BarChart data={data} margin={{ top:10, right:10, left:-10, bottom:20 }} barCategoryGap="30%">
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2d4a" vertical={false} />
        <XAxis dataKey="metric" tick={{ fill:"#6b7fa3", fontSize:11 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill:"#6b7fa3", fontSize:11 }} axisLine={false} tickLine={false} />
        <Tooltip content={<CustomTooltip />} cursor={{ fill:"rgba(255,255,255,.04)" }} />
        <Legend wrapperStyle={{ fontSize:11, color:"#6b7fa3", paddingTop:8 }} />
        <Bar dataKey="Your Value" radius={[4,4,0,0]} maxBarSize={36}>
          {data.map((d, i) => {
            // For credit score, green if above threshold; for others, red if above threshold
            const isOver = d.metric === "Credit Score"
              ? d["Your Value"] >= d["Safe Threshold"]
              : d["Your Value"] > d["Safe Threshold"];
            const fill = d.metric === "Credit Score"
              ? (isOver ? "#00c2a8" : "#f43f5e")
              : (isOver ? "#f43f5e" : "#00c2a8");
            return <Cell key={i} fill={fill} fillOpacity={0.85} />;
          })}
        </Bar>
        <Bar dataKey="Safe Threshold" fill="#1e2d4a" radius={[4,4,0,0]} maxBarSize={36} fillOpacity={0.9} />
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── User: financial health radar chart ────────────────────────────────────────
function HealthRadar({ fd }) {
  const { dti, ltv, loanIncomeMonths, score, age, effectiveDti } = deriveMetrics(fd);

  // Each axis scored 0–100 where 100 = best possible
  const creditHealth   = Math.max(0, Math.min(100, ((score - 300) / 600) * 100));
  const debtBurden     = Math.max(0, Math.min(100, 100 - (effectiveDti / 60) * 100));
  const collateralCvr  = Math.max(0, Math.min(100, 100 - (ltv / 100) * 100));
  const incomeAdequacy = Math.max(0, Math.min(100, 100 - (loanIncomeMonths / 30) * 100));
  // Age score peaks at 35–45, falls off toward 21 or 65
  const ageCentre      = 40;
  const ageScore       = Math.max(0, Math.min(100, 100 - Math.abs(age - ageCentre) * 2.8));

  const radarData = [
    { axis: "Credit Health",    score: Math.round(creditHealth) },
    { axis: "Debt Burden",      score: Math.round(debtBurden) },
    { axis: "Collateral Cover", score: Math.round(collateralCvr) },
    { axis: "Income Adequacy",  score: Math.round(incomeAdequacy) },
    { axis: "Age Suitability",  score: Math.round(ageScore) },
  ];

  const overallScore = Math.round(radarData.reduce((s, d) => s + d.score, 0) / radarData.length);
  const overallColor = overallScore >= 65 ? "#00e5c8" : overallScore >= 40 ? "#f59e0b" : "#f43f5e";

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    return (
      <div style={{ ...AGENT_TOOLTIP_STYLE, padding:"10px 14px" }}>
        <div style={{ fontWeight:700, marginBottom:4 }}>{payload[0]?.payload?.axis}</div>
        <div style={{ color:"#00e5c8" }}>Score: <strong>{payload[0]?.value}/100</strong></div>
      </div>
    );
  };

  return (
    <div>
      <div style={{ textAlign:"center", marginBottom:8 }}>
        <span style={{ fontFamily:"'Syne',sans-serif", fontSize:"1.5rem", fontWeight:800, color:overallColor }}>
          {overallScore}
        </span>
        <span style={{ fontSize:12, color:"#6b7fa3", marginLeft:6 }}>/ 100 Overall Financial Health</span>
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <RadarChart data={radarData} margin={{ top:10, right:30, bottom:10, left:30 }}>
          <PolarGrid stroke="#1e2d4a" />
          <PolarAngleAxis dataKey="axis" tick={{ fill:"#6b7fa3", fontSize:10 }} />
          <PolarRadiusAxis angle={90} domain={[0,100]} tick={{ fill:"#3a4a6a", fontSize:9 }} tickCount={4} />
          <Radar dataKey="score" stroke="#00c2a8" fill="#00c2a8" fillOpacity={0.18} strokeWidth={2} dot={{ r:3, fill:"#00c2a8" }} />
          <Tooltip content={<CustomTooltip />} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── User: threshold comparison bars ───────────────────────────────────────────
function ThresholdBars({ fd }) {
  const [ready, setReady] = useState(false);
  useEffect(() => { const t = setTimeout(() => setReady(true), 180); return () => clearTimeout(t); }, []);

  const { effectiveDti, ltv, score, loanIncomeMonths } = deriveMetrics(fd);

  const items = [
    {
      name: "Debt-to-Income (DTI)",
      actual: effectiveDti,
      threshold: 40,
      max: 100,
      higherIsBad: true,
      unit: "%",
      tip: "≤ 40% preferred",
    },
    {
      name: "Loan-to-Value (LTV)",
      actual: ltv,
      threshold: 80,
      max: 120,
      higherIsBad: true,
      unit: "%",
      tip: "≤ 80% preferred",
    },
    {
      name: "Credit Score",
      actual: score,
      threshold: 650,
      max: 900,
      higherIsBad: false,
      unit: "",
      tip: "≥ 650 preferred",
    },
    {
      name: "Loan / Income (months)",
      actual: loanIncomeMonths,
      threshold: 20,
      max: 40,
      higherIsBad: true,
      unit: "x",
      tip: "≤ 20× income preferred",
    },
  ];

  return (
    <div className="thresh-bars">
      {items.map(item => {
        const cappedActual  = Math.min(item.actual, item.max);
        const actualPct     = (cappedActual / item.max) * 100;
        const threshPct     = (item.threshold / item.max) * 100;
        const isGood = item.higherIsBad
          ? item.actual <= item.threshold
          : item.actual >= item.threshold;
        const fillColor = isGood ? "#00c2a8" : "#f43f5e";

        return (
          <div className="thresh-row" key={item.name}>
            <div className="thresh-header">
              <span className="thresh-name">{item.name}</span>
              <span className="thresh-vals">
                <span style={{ color: fillColor, fontWeight:700 }}>
                  {item.actual.toFixed(1)}{item.unit}
                </span>
                <span style={{ color:"#3a4a6a" }}> · {item.tip}</span>
              </span>
            </div>
            <div className="thresh-track">
              <div
                className="thresh-actual"
                style={{ width: ready ? `${actualPct}%` : "0%", background: fillColor, opacity: 0.8 }}
              />
              <div className="thresh-line" style={{ left:`${threshPct}%` }}>
                <span className="thresh-line-label">Safe</span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Agent result ──────────────────────────────────────────────────────────────
function AgentResult({ data, formData }) {
  const ar = data.agent_recommendation;
  const dp = ar.decision_plan;
  return (
    <>
    <div className={`result-panel ${panelClass(ar.risk_level)}`}>
      <div className="result-header">
        <span className="result-icon">{riskIcon(ar.risk_level)}</span>
        <div>
          <div className={`result-title ${colorClass(ar.risk_level)}`}>
            Risk Level: {ar.risk_level}
          </div>
          <div className="result-sub">Recommended Action — {ar.recommended_action}</div>
        </div>
      </div>

      <div className="info-grid">
        {[
          ["Default Probability", `${(ar.default_probability * 100).toFixed(1)}%`, colorClass(ar.risk_level)],
          ["Expected Loss",       `₹${ar.expected_loss.toLocaleString()}`,           ""],
          ["Assigned Team",       dp.assigned_team,                                  ""],
          ["Follow-up",           dp.follow_up_frequency,                            ""],
        ].map(([lbl, val, cls]) => (
          <div className="info-tile" key={lbl}>
            <div className="info-tile-label">{lbl}</div>
            <div className={`info-tile-value ${cls}`} style={{ fontSize: val.length > 12 ? ".82rem" : "1rem" }}>{val}</div>
          </div>
        ))}
      </div>

      <ScoreBar
        value={ar.default_probability}
        max={1}
        label="Default Probability"
        valueLabel={`${(ar.default_probability * 100).toFixed(1)}%`}
      />

      <div className="list-block">
        <h4>Recovery Decision Plan</h4>
        <div className="plan-grid">
          {[
            ["Recovery Channel",   dp.recovery_channel],
            ["Assigned Team",      dp.assigned_team],
            ["Follow-up Cadence",  dp.follow_up_frequency],
          ].map(([lbl, val]) => (
            <div className="plan-item" key={lbl}>
              <div className="plan-label">{lbl}</div>
              <div className="plan-value">{val}</div>
            </div>
          ))}
          <div className="plan-item">
            <div className="plan-label">Legal Action</div>
            <div className="plan-value">
              <span className={`legal-badge ${dp.legal_action ? "legal-yes" : "legal-no"}`}>
                {dp.legal_action ? "⚖️ Yes — Required" : "✅ Not Required"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    {/* ── Visualisation charts ── */}
    {formData && (
      <>
        {/* Gauge + bar chart side by side */}
        <div className="chart-section">
          <div className="chart-section-title">📊 Risk Visualisation</div>
          <div className="charts-row">
            {/* Gauge */}
            <div>
              <div style={{ fontSize:12, fontWeight:600, color:"#6b7fa3", textTransform:"uppercase", letterSpacing:".06em", marginBottom:14 }}>
                Default Probability Meter
              </div>
              <DefaultGauge prob={ar.default_probability} />
            </div>
            {/* Metrics chart */}
            <div>
              <div style={{ fontSize:12, fontWeight:600, color:"#6b7fa3", textTransform:"uppercase", letterSpacing:".06em", marginBottom:14 }}>
                Key Metrics vs Safe Thresholds
              </div>
              <AgentMetricsChart fd={formData} prob={ar.default_probability} />
            </div>
          </div>
        </div>
      </>
    )}
    </>
  );
}

// ── User result ───────────────────────────────────────────────────────────────
function UserResult({ data, formData }) {
  const br = data.borrower_recommendation;
  const ar = data.agent_recommendation;
  const chance = br.approval_chance;
  const isHigh = chance === "High";
  const dotCls = isHigh ? "ldot-t" : chance === "Medium" ? "ldot-a" : "ldot-r";

  return (
    <>
    <div className={`result-panel ${approvalPanelClass(chance)}`}>
      <div className="result-header">
        <span className="result-icon">{isHigh ? "✅" : chance === "Medium" ? "⚠️" : "❌"}</span>
        <div>
          <div className="result-title" style={{ color: isHigh ? "var(--teal2)" : chance === "Medium" ? "var(--amber)" : "var(--rose)" }}>
            {isHigh ? "You're likely eligible!" : chance === "Medium" ? "Moderate Eligibility" : "Eligibility Needs Improvement"}
          </div>
          <div className="result-sub">Based on your submitted financial profile</div>
        </div>
      </div>

      <span className={`appr-badge ${approvalBadgeClass(chance)}`}>
        {isHigh ? "✅" : chance === "Medium" ? "⚠️" : "❌"} Approval Chance: <strong>{chance}</strong>
      </span>

      <ScoreBar
        value={ar.default_probability}
        max={1}
        label="Default Risk (lower is better)"
        valueLabel={`${(ar.default_probability * 100).toFixed(1)}%`}
      />

      {br.suggestions && br.suggestions.length > 0 && (
        <div className="list-block">
          <h4>{isHigh ? "💡 Tips to Stay Eligible" : "⚠️ What to Improve"}</h4>
          {br.suggestions.map((s, i) => (
            <div key={i} className="list-item">
              <span className={`ldot ${dotCls}`} />
              <span>{s}</span>
            </div>
          ))}
        </div>
      )}
    </div>

    {/* ── Visualisation charts ── */}
    {formData && (
      <>
        {/* Radar + threshold bars */}
        <div className="chart-section">
          <div className="chart-section-title">📈 Your Financial Health Profile</div>
          <div className="charts-row">
            {/* Radar */}
            <div>
              <div style={{ fontSize:12, fontWeight:600, color:"#6b7fa3", textTransform:"uppercase", letterSpacing:".06em", marginBottom:8 }}>
                Overall Health Radar
              </div>
              <HealthRadar fd={formData} />
            </div>
            {/* Threshold bars */}
            <div>
              <div style={{ fontSize:12, fontWeight:600, color:"#6b7fa3", textTransform:"uppercase", letterSpacing:".06em", marginBottom:18 }}>
                Your Metrics vs Safe Thresholds
              </div>
              <ThresholdBars fd={formData} />
            </div>
          </div>
        </div>
      </>
    )}
    </>
  );
}

// ── Loan form (shared) ────────────────────────────────────────────────────────
const OCCUPANCY = [
  { value: 0, label: "Owner Occupied"      },
  { value: 1, label: "Rented / Investment" },
  { value: 2, label: "Second Home"         },
  { value: 3, label: "Commercial"          },
];

const REQUIRED = ["loan_amount","monthly_income","existing_emi","property_value","credit_score","age","loan_term"];

function LoanForm({ mode }) {
  const isAgent = mode === "agent";
  const empty = { loan_amount:"", monthly_income:"", existing_emi:"", property_value:"", credit_score:"", age:"", loan_term:"", occupancy_type:0, business_loan:0 };

  const [form,       setForm]       = useState(empty);
  const [result,     setResult]     = useState(null);
  const [formData,   setFormData]   = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [apiError,   setApiError]   = useState(null);
  const [touched,    setTouched]    = useState({});
  const resultRef = useRef(null);

  const set = (k, v) => setForm(p => ({ ...p, [k]: v }));
  const isValid = () => REQUIRED.every(f => String(form[f]).trim() !== "" && !isNaN(Number(form[f])));
  const fieldErr = (f) => touched[f] && (String(form[f]).trim() === "" || isNaN(Number(form[f])));

  const handleSubmit = async () => {
    const t = {};
    REQUIRED.forEach(f => (t[f] = true));
    setTouched(t);
    if (!isValid()) return;

    setLoading(true);
    setResult(null);
    setApiError(null);

    const payload = {
      loan_amount:    parseFloat(form.loan_amount),
      monthly_income: parseFloat(form.monthly_income),
      existing_emi:   parseFloat(form.existing_emi),
      property_value: parseFloat(form.property_value),
      credit_score:   parseFloat(form.credit_score),
      age:            parseInt(form.age, 10),
      loan_term:      parseInt(form.loan_term, 10),
      occupancy_type: parseInt(form.occupancy_type, 10),
      business_loan:  parseInt(form.business_loan, 10),
    };

    try {
      const res = await fetch(`${API_BASE}/risk-score`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server responded with status ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
      setFormData(payload);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior:"smooth", block:"start" }), 120);
    } catch (err) {
      setApiError(err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => { setForm(empty); setResult(null); setFormData(null); setApiError(null); setTouched({}); };

  return (
    <div className="form-page">
      <div className="form-header">
        <h2>{isAgent ? "🏦 Agent — Borrower Risk Assessment" : "📋 Check Your Loan Eligibility"}</h2>
        <p>
          {isAgent
            ? "Enter the borrower's details below. Our system will analyse the financial profile and generate a complete risk assessment and recovery plan."
            : "Fill in your financial profile. Our AI will assess your eligibility and give personalised improvement tips."}
        </p>
      </div>

      <div className="form-card">

        {/* Personal */}
        <div className="form-section-title">👤 {isAgent ? "Borrower" : "Personal"} Information</div>
        <div className="form-grid">
          <div className="form-group">
            <label>Age (years) *</label>
            <input type="number" placeholder="e.g. 35"
              className={fieldErr("age") ? "err" : ""}
              value={form.age} onChange={e => set("age", e.target.value)} />
            {fieldErr("age") && <span className="hint-err">Required</span>}
          </div>
          <div className="form-group">
            <label>Occupancy Type</label>
            <select value={form.occupancy_type} onChange={e => set("occupancy_type", e.target.value)}>
              {OCCUPANCY.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label>Loan Category</label>
            <select value={form.business_loan} onChange={e => set("business_loan", e.target.value)}>
              <option value={0}>Personal / Home Loan</option>
              <option value={1}>Business / Commercial Loan</option>
            </select>
          </div>
        </div>

        <div className="form-divider" />

        {/* Financial */}
        <div className="form-section-title">💰 Financial Profile</div>
        <div className="form-grid">
          <div className="form-group">
            <label>Monthly Income (₹) *</label>
            <input type="number" placeholder="e.g. 75000"
              className={fieldErr("monthly_income") ? "err" : ""}
              value={form.monthly_income} onChange={e => set("monthly_income", e.target.value)} />
            {fieldErr("monthly_income") && <span className="hint-err">Required</span>}
          </div>
          <div className="form-group">
            <label>Existing EMI / month (₹) *</label>
            <input type="number" placeholder="e.g. 12000 — enter 0 if none"
              className={fieldErr("existing_emi") ? "err" : ""}
              value={form.existing_emi} onChange={e => set("existing_emi", e.target.value)} />
            {fieldErr("existing_emi") && <span className="hint-err">Required (use 0 if none)</span>}
          </div>
          <div className="form-group">
            <label>Credit Score (300–900) *</label>
            <input type="number" placeholder="e.g. 720"
              className={fieldErr("credit_score") ? "err" : ""}
              value={form.credit_score} onChange={e => set("credit_score", e.target.value)} />
            {fieldErr("credit_score") && <span className="hint-err">Required</span>}
          </div>
          <div className="form-group">
            <label>Property / Collateral Value (₹) *</label>
            <input type="number" placeholder="e.g. 4500000"
              className={fieldErr("property_value") ? "err" : ""}
              value={form.property_value} onChange={e => set("property_value", e.target.value)} />
            {fieldErr("property_value") && <span className="hint-err">Required</span>}
          </div>
        </div>

        <div className="form-divider" />

        {/* Loan details */}
        <div className="form-section-title">📄 Loan Details</div>
        <div className="form-grid">
          <div className="form-group">
            <label>Requested Loan Amount (₹) *</label>
            <input type="number" placeholder="e.g. 2500000"
              className={fieldErr("loan_amount") ? "err" : ""}
              value={form.loan_amount} onChange={e => set("loan_amount", e.target.value)} />
            {fieldErr("loan_amount") && <span className="hint-err">Required</span>}
          </div>
          <div className="form-group">
            <label>Loan Term (months) *</label>
            <input type="number" placeholder="e.g. 120 = 10 yrs"
              className={fieldErr("loan_term") ? "err" : ""}
              value={form.loan_term} onChange={e => set("loan_term", e.target.value)} />
            <span className="hint">12 = 1 yr · 60 = 5 yrs · 120 = 10 yrs · 240 = 20 yrs</span>
            {fieldErr("loan_term") && <span className="hint-err">Required</span>}
          </div>
        </div>

        {apiError && (
          <div className="error-banner">
            <span className="err-icon">🔌</span>
            <div>
              <div className="err-text">Service Unavailable — Could not reach the assessment service</div>
              <div className="err-detail">{apiError}</div>
              <div className="err-detail" style={{ marginTop:8 }}>
                Please try again in a moment or contact support if the issue persists.
              </div>
            </div>
          </div>
        )}

        <div className="form-submit">
          <button className="btn btn-outline" onClick={handleReset}>Reset</button>
          <button className="btn btn-primary btn-lg" onClick={handleSubmit} disabled={loading}>
            {loading
              ? <><span className="spinner" /> Analysing…</>
              : isAgent ? "🔍 Assess Borrower" : "⚡ Check Eligibility"}
          </button>
        </div>
      </div>

      <div ref={resultRef}>
        {result && (
          <div className="result-wrap">
            {isAgent
              ? <AgentResult data={result} formData={formData} />
              : <UserResult  data={result} formData={formData} />}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Home page ─────────────────────────────────────────────────────────────────
function HomePage({ goTo }) {
  return (
    <div>
      <div className="hero">
        <div className="hero-glow hero-glow-1" />
        <div className="hero-glow hero-glow-2" />
        <div className="hero-badge"><span>🏦</span> AI-Powered Credit Intelligence</div>
        <h1>Smart Loan Decisions with<br /><span className="accent">CreditPath AI</span></h1>
        <p>
          Instantly evaluate borrower risk, predict approval chances, and receive personalised
          recovery guidance — all powered by advanced credit intelligence.
        </p>
        <div className="hero-btns">
          <button className="btn btn-primary btn-lg" onClick={() => goTo("portal")}>Open Portal →</button>
          <button className="btn btn-outline btn-lg" onClick={() => goTo("about")}>Learn More</button>
        </div>
        <div className="hero-stats">
          {[["< 2s","Decision Time"],["2 Portals","Agent & Applicant"],["Real-time","Risk Scoring"],["100%","Explainable Results"]].map(([n,l]) => (
            <div className="hero-stat" key={l}>
              <div className="hero-stat-num">{n}</div>
              <div className="hero-stat-label">{l}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="section">
        <div className="section-header">
          <h2>Everything you need for <span style={{ color:"var(--teal)" }}>smarter lending</span></h2>
          <p>CreditPathAI brings intelligent credit analysis to every loan officer and applicant.</p>
        </div>
        <div className="cards-grid grid-3">
          {[
            { icon:"🧠", cls:"card-icon-teal",  title:"Intelligent Risk Scoring",   desc:"Every application is evaluated across multiple financial dimensions — income, debt burden, credit history, and collateral — to produce a precise risk score." },
            { icon:"🏦", cls:"card-icon-blue",  title:"Agent Recovery Plans",       desc:"Bank agents receive a full recovery action plan including assigned team, communication channel, follow-up frequency, and legal action flags." },
            { icon:"👤", cls:"card-icon-amber", title:"Borrower Guidance",          desc:"Applicants get a clear approval chance rating along with personalised, actionable suggestions to strengthen their financial profile." },
            { icon:"📊", cls:"card-icon-rose",  title:"Transparent Decisions",      desc:"Every approval or rejection comes with clear, human-readable reasons — no black box. Borrowers always know exactly where they stand." },
            { icon:"⚡", cls:"card-icon-teal",  title:"Instant Results",            desc:"Submit your details and receive a full credit assessment in seconds — no waiting, no paperwork queues." },
            { icon:"🛡️", cls:"card-icon-blue",  title:"Secure & Reliable",          desc:"Built with data privacy in mind. All assessments are processed securely and results are never stored without consent." },
          ].map(f => (
            <div className="card" key={f.title}>
              <div className={`card-icon ${f.cls}`}>{f.icon}</div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="section" style={{ paddingTop:0 }}>
        <div style={{ background:"var(--card)", border:"1px solid var(--border)", borderRadius:24, padding:"60px 40px", maxWidth:720, margin:"0 auto", textAlign:"center" }}>
          <h2 style={{ marginBottom:14 }}>Ready to make <span style={{ color:"var(--teal)" }}>data-driven</span> loan decisions?</h2>
          <p style={{ color:"var(--muted)", marginBottom:32 }}>
            Whether you're a bank agent evaluating a borrower or an applicant checking your eligibility — CreditPathAI delivers clear, instant, and explainable credit decisions.
          </p>
          <div style={{ display:"flex", gap:12, justifyContent:"center", flexWrap:"wrap" }}>
            <button className="btn btn-primary btn-lg" onClick={() => goTo("portal")}>Open Portal →</button>
            <button className="btn btn-outline" onClick={() => goTo("about")}>Learn More</button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── About page ────────────────────────────────────────────────────────────────
function AboutPage() {
  return (
    <div>
      <div className="about-hero">
        <div className="hero-badge">🏦 About CreditPathAI</div>
        <h1>Lending decisions that are <br /><span style={{ color:"var(--teal)" }}>fair, fast, and explainable.</span></h1>
        <p style={{ marginTop:16 }}>
          CreditPathAI is an intelligent credit assessment platform designed to bring clarity and speed
          to the loan approval process — for both the institutions that lend and the people who borrow.
        </p>
      </div>

      {/* Mission */}
      <div className="section">
        <div className="section-header">
          <h2>Our Mission</h2>
          <p>Making credit decisions smarter, fairer, and more transparent for everyone involved.</p>
        </div>
        <div className="cards-grid grid-3">
          {[
            { icon:"⚖️", cls:"card-icon-teal",  title:"Fair Assessments",       desc:"Every applicant is evaluated on the same objective financial criteria — income, credit history, existing obligations, and collateral — with no room for bias or inconsistency." },
            { icon:"💡", cls:"card-icon-amber", title:"Explainable Outcomes",   desc:"We believe no one should be rejected without understanding why. Every decision comes with a clear breakdown of the factors that influenced it, along with steps to improve." },
            { icon:"🚀", cls:"card-icon-rose",  title:"Speed Without Compromise", desc:"What once took days of manual review now takes seconds. CreditPathAI delivers fully assessed loan decisions in real time, without sacrificing accuracy or rigour." },
          ].map(f => (
            <div className="card" key={f.title}>
              <div className={`card-icon ${f.cls}`}>{f.icon}</div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* How it works */}
      <div className="section" style={{ paddingTop:0 }}>
        <div className="section-header">
          <h2>How CreditPathAI Works</h2>
          <p>A structured, multi-factor approach to credit risk — from application to decision.</p>
        </div>
        <div className="cards-grid grid-2">
          {[
            { icon:"📋", step:"01", title:"Applicant Submits a Profile",       desc:"The borrower (or agent on their behalf) provides key financial details — monthly income, existing debt obligations, credit score, property value, and loan requirements." },
            { icon:"🔍", step:"02", title:"Multi-Factor Credit Analysis",      desc:"CreditPathAI evaluates the Debt-to-Income ratio, Loan-to-Value ratio, credit history, age eligibility, and income adequacy simultaneously to build a holistic risk picture." },
            { icon:"📊", step:"03", title:"Risk Classification",               desc:"Each application is classified into a risk tier — Low, Medium, High, or Critical — based on the combined weight of all financial indicators assessed." },
            { icon:"💬", step:"04", title:"Personalised Recommendations",      desc:"Borrowers receive a clear approval chance and specific, actionable suggestions. Bank agents receive a full recovery and follow-up action plan tailored to the risk level." },
          ].map(s => (
            <div className="card" key={s.step}>
              <div style={{ display:"flex", gap:14, alignItems:"flex-start" }}>
                <div className="card-icon card-icon-teal" style={{ flexShrink:0 }}>{s.icon}</div>
                <div>
                  <div style={{ fontSize:11, color:"var(--teal)", fontWeight:700, textTransform:"uppercase", letterSpacing:".07em", marginBottom:4 }}>Step {s.step}</div>
                  <h3>{s.title}</h3>
                  <p>{s.desc}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Key metrics evaluated */}
      <div className="section" style={{ paddingTop:0 }}>
        <div className="section-header">
          <h2>Key Credit Factors We Evaluate</h2>
          <p>CreditPathAI analyses six core financial dimensions for every application.</p>
        </div>
        <div className="cards-grid grid-3">
          {[
            { icon:"📉", cls:"card-icon-teal",  title:"Debt-to-Income Ratio (DTI)",    desc:"Measures how much of a borrower's monthly income is already committed to debt repayments. A DTI above 40% signals elevated financial stress." },
            { icon:"🏠", cls:"card-icon-blue",  title:"Loan-to-Value Ratio (LTV)",     desc:"Compares the loan amount against the value of the collateral or property. High LTV ratios indicate greater risk for the lender in case of default." },
            { icon:"⭐", cls:"card-icon-amber", title:"Credit Score",                  desc:"A borrower's credit score reflects their history of repaying past obligations. Scores below 650 are treated as a significant risk indicator." },
            { icon:"💰", cls:"card-icon-rose",  title:"Income Adequacy",               desc:"Assesses whether the borrower's income is sufficient to service the requested loan amount over the chosen repayment term without financial strain." },
            { icon:"🗓️", cls:"card-icon-teal",  title:"Age & Tenure Eligibility",     desc:"Loan eligibility is assessed in relation to age, ensuring the repayment period does not extend beyond standard working or retirement thresholds." },
            { icon:"📂", cls:"card-icon-blue",  title:"Loan Purpose & Type",           desc:"Whether a loan is for personal, residential, or commercial purposes affects the risk profile. Business loans are assessed under a separate set of criteria." },
          ].map(f => (
            <div className="card" key={f.title}>
              <div className={`card-icon ${f.cls}`}>{f.icon}</div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Who we serve */}
      <div className="section" style={{ paddingTop:0 }}>
        <div className="section-header"><h2>Who We Serve</h2></div>
        <div className="cards-grid grid-2">
          <div className="card">
            <div className="card-icon card-icon-teal">🏦</div>
            <h3>Banks & Lending Institutions</h3>
            <p style={{ marginBottom:14 }}>
              CreditPathAI gives loan officers and recovery agents the information they need to act
              decisively. For every borrower assessed, agents receive a complete picture:
            </p>
            <div className="list-block" style={{ marginTop:0 }}>
              {["Risk level classification (Low / Medium / High / Critical)","Expected financial loss exposure","Assigned recovery team and communication channel","Follow-up frequency and whether legal action is warranted"].map((item, i) => (
                <div key={i} className="list-item">
                  <span className="ldot ldot-t" />
                  <span>{item}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="card">
            <div className="card-icon card-icon-amber">👤</div>
            <h3>Loan Applicants</h3>
            <p style={{ marginBottom:14 }}>
              For borrowers, CreditPathAI removes the mystery from the loan approval process.
              Rather than a simple yes or no, applicants receive:
            </p>
            <div className="list-block" style={{ marginTop:0 }}>
              {["A clear approval chance rating — High, Medium, or Low","Specific reasons why their application may face challenges","Concrete steps they can take to improve their financial standing","An honest assessment based purely on financial facts, not paperwork delays"].map((item, i) => (
                <div key={i} className="list-item">
                  <span className="ldot ldot-a" />
                  <span>{item}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
/* ADD CONTACT PAGE HERE */
function ContactPage() {
  return (
    <div className="contact-page">
      
      <div className="contact-header">
        <h1>Get in <span>Touch</span></h1>
        <p>Have a question or need support? We're always here for you.</p>
      </div>

      <div className="contact-container">

        <div className="contact-info">
          <div className="contact-card">
            <div className="icon">📞</div>
            <div>
              <h4>PHONE</h4>
              <p>+91 98765 43210</p>
              <span>Mon – Sat, 9 AM – 6 PM IST</span>
            </div>
          </div>

          <div className="contact-card">
            <div className="icon">✉️</div>
            <div>
              <h4>EMAIL</h4>
              <p>support@creditpath.ai</p>
              <span>We reply within 24 hours</span>
            </div>
          </div>

          <div className="contact-card">
            <div className="icon">📍</div>
            <div>
              <h4>OFFICE</h4>
              <p>Hyderabad, Telangana</p>
              <span>HITEC City, 500081, India</span>
            </div>
          </div>

          <div className="contact-card">
            <div className="icon">⏰</div>
            <div>
              <h4>BUSINESS HOURS</h4>
              <p>Mon – Fri: 9 AM – 7 PM</p>
              <span>Sat: 10 AM – 4 PM</span>
            </div>
          </div>
        </div>

        <div className="contact-form">
          <h3>Send us a Message</h3>
          <p>Fill in the form and our team will get back to you.</p>

          <div className="form-grid">
            <input placeholder="Your Name" />
            <input placeholder="Email Address" />
            <input placeholder="Phone Number" />
            <input placeholder="Subject" />
          </div>

          <textarea placeholder="Tell us how we can help you..." />

          <button className="contact-btn">
            Send Message →
          </button>
        </div>

      </div>
    </div>
  );
}

// ── Portal select ─────────────────────────────────────────────────────────────
function PortalSelectPage({ goTo }) {
  return (
    <div className="portal-select">
      <h2>Welcome to CreditPathAI</h2>
      <p>Who are you? Choose your portal.</p>
      <div className="portal-cards">
        <div className="portal-card pc-teal" onClick={() => goTo("agent")}>
          <div className="portal-card-icon" style={{ background:"rgba(0,194,168,.12)" }}>🏦</div>
          <h3>Bank Agent Portal</h3>
          <p>Evaluate a borrower's application. Get ML-driven default probability, expected loss, recovery action plan, and team assignment.</p>
          <div className="arrow" style={{ color:"var(--teal)" }}>Enter as Agent →</div>
        </div>
        <div className="portal-card pc-amber" onClick={() => goTo("user")}>
          <div className="portal-card-icon" style={{ background:"rgba(245,158,11,.12)" }}>👤</div>
          <h3>Loan Applicant Portal</h3>
          <p>Check your loan eligibility instantly and get clear suggestions on how to improve your approval chances.</p>
          <div className="arrow" style={{ color:"var(--amber)" }}>Apply Now →</div>
        </div>
      </div>
    </div>
  );
}

// ── Login page ────────────────────────────────────────────────────────────────
function LoginPage({ goTo }) {
  const [tab,   setTab]   = useState("user");
  const [email, setEmail] = useState("");
  const [pass,  setPass]  = useState("");
  const handle = () => goTo(tab === "agent" ? "agent" : "user");
  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-logo">
          <div className="logo-box">CP</div>
          <h2>CreditPath<span style={{ color:"var(--teal)" }}>AI</span></h2>
          <p>Sign in to your portal</p>
        </div>
        <div className="tab-row">
          <button className={`tab-btn ${tab === "user"  ? "active" : ""}`} onClick={() => setTab("user")}>👤 Applicant</button>
          <button className={`tab-btn ${tab === "agent" ? "active" : ""}`} onClick={() => setTab("agent")}>🏦 Bank Agent</button>
        </div>
        <div className="form-group" style={{ marginBottom:16 }}>
          <label>Email Address</label>
          <input type="email" placeholder="you@example.com" value={email} onChange={e => setEmail(e.target.value)} />
        </div>
        <div className="form-group" style={{ marginBottom:28 }}>
          <label>Password</label>
          <input type="password" placeholder="••••••••" value={pass} onChange={e => setPass(e.target.value)} />
        </div>
        <button className="btn btn-primary" style={{ width:"100%", justifyContent:"center" }} onClick={handle}>
          Sign In →
        </button>
        <p style={{ textAlign:"center", fontSize:".82rem", color:"var(--muted)", marginTop:20 }}>
          No account?{" "}
          <span style={{ color:"var(--teal)", cursor:"pointer" }} onClick={() => goTo("portal")}>Go to Portal</span>
        </p>
      </div>
    </div>
  );
}

// ── Nav ───────────────────────────────────────────────────────────────────────
function Nav({ current, goTo }) {
  return (
    <nav className="nav">
      <div className="nav-logo" onClick={() => goTo("home")}>
        <div className="nav-logo-icon">CP</div>
        <div className="nav-logo-text">CreditPath<span>AI</span></div>
      </div>
      <div className="nav-links">
        {[["home","Home"],["about","About"],["portal","Portal"],["contact","Contact"]].map(([id,lbl]) => (
          <button key={id} className={`nav-link ${current===id?"active":""}`} onClick={() => goTo(id)}>{lbl}</button>
        ))}
      </div>
      <div className="nav-cta">
        <button className="btn btn-outline btn-sm" onClick={() => goTo("login")}>Login</button>
        <button className="btn btn-primary btn-sm" onClick={() => goTo("portal")}>Get Started</button>
      </div>
    </nav>
  );
}

// ── Footer ────────────────────────────────────────────────────────────────────
function Footer({ goTo }) {
  return (
    <footer>
      <div className="foot-copy">© 2025 CreditPathAI — Intelligent Credit Decisions</div>
      <div className="foot-links">
        {[["Home","home"],["About","about"],["Portal","portal"],["Login","login"]].map(([l,p]) => (
          <button key={p} className="foot-link" onClick={() => goTo(p)}>{l}</button>
        ))}
      </div>
    </footer>
  );
}

// ── App root ──────────────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage] = useState("home");
  const goTo = (p) => { setPage(p); window.scrollTo({ top:0, behavior:"smooth" }); };
  return (
    <>
      <style>{STYLE}</style>
      <Nav current={page} goTo={goTo} />
      <div className="page">
        {page === "home"   && <HomePage   goTo={goTo} />}
        {page === "about"  && <AboutPage  />}
        {page === "portal" && <PortalSelectPage goTo={goTo} />}
        {page === "contact" && <ContactPage />}
        {page === "login"  && <LoginPage  goTo={goTo} />}
        {page === "agent"  && <LoanForm   mode="agent" />}
        {page === "user"   && <LoanForm   mode="user"  />}
        <Footer goTo={goTo} />
      </div>
    </>
  );
}