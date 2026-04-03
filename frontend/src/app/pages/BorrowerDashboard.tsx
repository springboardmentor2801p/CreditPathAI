import { useState, useEffect } from "react";
import { useNavigate } from "react-router";
import { 
  TrendingUp, 
  AlertTriangle, 
  DollarSign, 
  Activity, 
  ArrowUpRight,
  ArrowDownRight,
  ShieldCheck,
  Target,
  Zap,
  ChevronRight
} from "lucide-react";
import Plot from "react-plotly.js";
import { apiFetch } from "../lib/api";

const iconMap: any = {
  TrendingUp,
  AlertTriangle,
  DollarSign,
  Activity
};

export function BorrowerDashboard() {
  const [data, setData] = useState<{stats: any[], creditTrajectory: any, debtStructure: any, recommendations: any[], liabilities: any[], profile?: any} | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const userId = localStorage.getItem('user_id');
    if (!userId) { navigate('/login'); return; }
    apiFetch('/api/borrower-dashboard')
      .then(res => {
        if (res.status === 401) { navigate('/login'); return null; }
        if (res.status === 404) throw new Error('no_profile');
        return res.json();
      })
      .then(d => { if (d) setData(d); })
      .catch(e => setError(e.message));
  }, []);

  if (error === 'no_profile') return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center space-y-4">
      <div className="w-16 h-16 rounded-full bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center">
        <Target className="w-7 h-7 text-emerald-400" />
      </div>
      <h2 className="text-xl font-black text-white uppercase tracking-tighter">Set Up Your Financial Profile</h2>
      <p className="text-zinc-500 text-sm max-w-sm">Your account is active but you haven't added your financial details yet. Go to your profile to add income, credit score and loans.</p>
      <button onClick={() => navigate('/borrower/profile')} className="px-5 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-black text-xs font-black tracking-widest rounded-md transition-colors">
        COMPLETE PROFILE →
      </button>
    </div>
  );

  if (!data && !error) return <div className="p-8 text-zinc-400 font-mono text-sm tracking-widest animate-pulse">SYNCING PROFILE...</div>;
  if (!data) return null;

  // Check for missing required fields in profile (top-level or nested profile)
  const requiredFields = [
    'full_name', 'credit_score', 'annual_income', 'employment_status'
  ];
  // Try top-level, then fallback to data.profile if present
  const profileObj = data.profile || data;
  const missingFields = requiredFields.filter(f => !profileObj[f] && profileObj[f] !== 0);

  if (missingFields.length > 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] text-center space-y-4">
        <div className="w-16 h-16 rounded-full bg-rose-500/10 border border-rose-500/30 flex items-center justify-center">
          <AlertTriangle className="w-7 h-7 text-rose-400" />
        </div>
        <h2 className="text-xl font-black text-white uppercase tracking-tighter">Complete Your Profile</h2>
        <p className="text-zinc-500 text-sm max-w-sm">Some required details are missing from your profile. Please update your profile to unlock your full dashboard and recommendations.</p>
        <button onClick={() => navigate('/borrower/profile')} className="px-5 py-2.5 bg-rose-500 hover:bg-rose-400 text-black text-xs font-black tracking-widest rounded-md transition-colors">
          COMPLETE PROFILE →
        </button>
      </div>
    );
  }

  const { stats, creditTrajectory, debtStructure, recommendations, liabilities } = data;

  // Map recommendation badge types to navigation targets
  const recActionMap: Record<string, { label: string; path: string }> = {
    "High Impact": { label: "RUN SIMULATION →", path: "/borrower/evaluator" },
    "Medium Impact": { label: "RUN SIMULATION →", path: "/borrower/evaluator" },
    "Action Needed": { label: "VIEW PROFILE →", path: "/borrower/profile" },
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-end mb-8 border-b border-zinc-800 pb-6">
        <div>
          <h1 className="text-3xl font-black tracking-tighter text-white uppercase mb-1">Financial Overview</h1>
          <p className="text-zinc-500 font-mono text-xs tracking-widest uppercase">Personal Health Dashboard // Live Data</p>
        </div>
        <div className="hidden sm:flex items-center gap-3">
          <button
            onClick={() => navigate('/borrower/evaluator')}
            className="flex items-center gap-2 px-3 py-1.5 bg-emerald-500 hover:bg-emerald-400 text-black rounded-md text-xs font-black tracking-wider transition-all shadow-[0_0_15px_rgba(16,185,129,0.3)]"
          >
            <Target className="h-3 w-3" />
            RUN SIMULATION
          </button>
          <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 rounded-md text-emerald-400 text-xs font-bold tracking-wider shadow-[0_0_10px_rgba(16,185,129,0.15)]">
            <ShieldCheck className="h-4 w-4" />
            STATUS: GOOD
          </div>
        </div>
      </div>

      {/* Top Stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((item) => {
          const IconComp = iconMap[item.iconType] || Activity;
          return (
            <div key={item.name} className="relative bg-[#131316] overflow-hidden rounded-xl border border-zinc-800/80 shadow-lg group hover:border-zinc-700 transition-colors">
              <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                <IconComp className="w-16 h-16 text-white" />
              </div>
              <div className="p-5 relative z-10">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-[10px] font-bold tracking-widest text-zinc-400 uppercase">{item.name.replace(/_/g, ' ')}</h3>
                  <div className={`p-2 rounded-md border ${item.bg}`}>
                    <IconComp className={`h-4 w-4 ${item.color}`} />
                  </div>
                </div>
                <div className="flex items-end justify-between">
                  <div className="text-3xl font-black tracking-tight text-white font-mono">
                    {item.value}
                  </div>
                  <div className={`flex items-center text-xs font-bold px-2 py-1 rounded bg-[#09090b] border ${
                    item.changeType === "positive" ? "text-emerald-400 border-emerald-500/20" :
                    item.changeType === "negative" ? "text-amber-400 border-amber-500/20" :
                    "text-cyan-400 border-cyan-500/20"
                  }`}>
                    {item.changeType === "positive" ? <ArrowUpRight className="h-3 w-3 mr-1" /> :
                     item.changeType === "negative" ? <ArrowDownRight className="h-3 w-3 mr-1" /> :
                     <Activity className="h-3 w-3 mr-1" />}
                    {item.change}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Charts */}
        <div className="lg:col-span-2 space-y-6">
          {/* Credit Trajectory */}
          <div className="bg-[#131316] rounded-xl border border-zinc-800/80 p-6 shadow-lg">
            <h3 className="text-xs font-bold tracking-widest text-zinc-400 uppercase mb-6 flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-emerald-400" />
              Credit Score Trajectory (12 Months)
            </h3>
            <div className="h-64 w-full">
              <Plot
                data={[{
                  x: creditTrajectory.labels,
                  y: creditTrajectory.values,
                  type: "scatter",
                  mode: "lines+markers",
                  line: { shape: "spline", color: "#10b981", width: 3 },
                  marker: { size: 6, color: "#09090b", line: { color: "#10b981", width: 2 } },
                  fill: "tozeroy",
                  fillcolor: "rgba(16,185,129,0.05)"
                }]}
                layout={{
                  autosize: true,
                  margin: { t: 10, r: 10, l: 30, b: 30 },
                  paper_bgcolor: "transparent",
                  plot_bgcolor: "transparent",
                  xaxis: { fixedrange: true, tickfont: { color: "#71717a" }, gridcolor: "transparent" },
                  yaxis: { fixedrange: true, gridcolor: "#27272a", range: [650, 780], tickfont: { color: "#71717a" } },
                  font: { family: "inherit" }
                }}
                useResizeHandler={true}
                style={{ width: "100%", height: "100%" }}
                config={{ displayModeBar: false }}
              />
            </div>
          </div>

          {/* Debt Structure */}
          <div className="bg-[#131316] rounded-xl border border-zinc-800/80 p-6 shadow-lg">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xs font-bold tracking-widest text-zinc-400 uppercase flex items-center gap-2">
                <DollarSign className="h-4 w-4 text-cyan-400" />
                Debt Structure Breakdown (₹)
              </h3>
              <button
                onClick={() => navigate('/borrower/evaluator')}
                className="text-[10px] font-bold tracking-widest text-emerald-400 hover:text-emerald-300 transition-colors flex items-center gap-1"
              >
                SIMULATE <ChevronRight className="h-3 w-3" />
              </button>
            </div>
            <div className="h-64 w-full">
              <Plot
                data={[{
                  values: debtStructure.values,
                  labels: debtStructure.labels,
                  type: "pie",
                  hole: 0.65,
                  marker: {
                    colors: ["#22d3ee", "#8b5cf6", "#10b981"],
                    line: { color: "#131316", width: 2 }
                  },
                  textinfo: "percent",
                  hoverinfo: "label+value",
                  textfont: { family: "inherit", color: "#ffffff", size: 12 }
                }]}
                layout={{
                  autosize: true,
                  margin: { t: 10, r: 10, l: 10, b: 10 },
                  paper_bgcolor: "transparent",
                  plot_bgcolor: "transparent",
                  font: { family: "inherit", color: "#a1a1aa" },
                  showlegend: true,
                  legend: { orientation: "h", y: -0.1, font: { color: "#a1a1aa" } }
                }}
                useResizeHandler={true}
                style={{ width: "100%", height: "100%" }}
                config={{ displayModeBar: false }}
              />
            </div>
          </div>
        </div>

        {/* Right sidebar */}
        <div className="space-y-6">
          {/* Smart Recommendations */}
          <div className="bg-gradient-to-br from-[#022c22] to-[#0f172a] border border-emerald-500/20 rounded-xl shadow-lg p-6 relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-10 pointer-events-none">
              <Target className="w-24 h-24 text-emerald-400" />
            </div>
            <h3 className="text-xs font-bold tracking-widest text-emerald-400 uppercase mb-6 flex items-center gap-2 relative z-10">
              <Zap className="h-4 w-4" />
              Smart Recommendations
            </h3>

            <div className="space-y-4 relative z-10">
              {recommendations.map((rec) => {
                const action = recActionMap[rec.type] || { label: "EXPLORE →", path: "/borrower/evaluator" };
                return (
                  <div key={rec.id} className="bg-[#09090b]/80 border border-emerald-500/10 p-4 rounded-lg hover:border-emerald-500/30 transition-colors">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="text-xs font-bold text-zinc-200 uppercase">{rec.title}</h4>
                      <span className={`text-[9px] font-black uppercase tracking-widest px-1.5 py-0.5 rounded border
                        ${rec.type === "High Impact" ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30" :
                          rec.type === "Action Needed" ? "bg-amber-500/10 text-amber-400 border-amber-500/30" :
                          "bg-cyan-500/10 text-cyan-400 border-cyan-500/30"}
                      `}>
                        {rec.impact}
                      </span>
                    </div>
                    <p className="text-[10px] font-mono text-zinc-400 leading-relaxed mb-3">
                      {rec.desc}
                    </p>
                    <button
                      onClick={() => navigate(action.path)}
                      className="text-[10px] font-black tracking-widest text-emerald-400 hover:text-emerald-300 transition-colors flex items-center gap-1"
                    >
                      {action.label}
                    </button>
                  </div>
                );
              })}
            </div>

            <button
              onClick={() => navigate('/borrower/evaluator')}
              className="mt-6 w-full bg-emerald-500/20 border border-emerald-500/50 text-emerald-300 px-4 py-3 rounded-md text-xs font-black tracking-widest uppercase hover:bg-emerald-500 hover:text-black transition-all relative z-10"
            >
              SIMULATE MY LOAN →
            </button>
          </div>

          {/* Active Liabilities */}
          <div className="bg-[#131316] rounded-xl border border-zinc-800/80 p-6 shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-bold tracking-widest text-zinc-400 uppercase">Active Liabilities (₹)</h3>
              <button
                onClick={() => navigate('/borrower/evaluator')}
                className="text-[10px] font-bold tracking-widest text-emerald-400 hover:text-emerald-300 transition-colors"
              >
                SIMULATE →
              </button>
            </div>
            <div className="space-y-4">
              {liabilities.map(liab => (
                <div key={liab.id} className="flex justify-between items-center border-b border-zinc-800/60 pb-3 last:border-0 last:pb-0">
                  <div>
                    <div className="text-sm font-bold text-zinc-200">{liab.name}</div>
                    <div className="text-[10px] font-mono text-zinc-500 mt-1">{liab.details}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-mono text-white">{liab.amount}</div>
                    <div className={`text-[10px] font-bold uppercase tracking-widest mt-1 ${
                      liab.statusType === 'good' ? 'text-emerald-400' :
                      liab.statusType === 'warning' ? 'text-amber-400' : 'text-rose-400'
                    }`}>
                      {liab.status}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
