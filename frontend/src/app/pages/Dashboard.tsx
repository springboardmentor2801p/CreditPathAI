import { useState, useEffect } from "react";
import { Link } from "react-router";
import { 
  TrendingUp, 
  AlertTriangle, 
  DollarSign, 
  Activity, 
  ArrowUpRight,
  ArrowDownRight,
  Zap
} from "lucide-react";
import Plot from 'react-plotly.js';
import { apiFetch } from "../lib/api";

const iconMap: any = {
  DollarSign: DollarSign,
  Activity: Activity,
  AlertTriangle: AlertTriangle,
  TrendingUp: TrendingUp
};

export function Dashboard() {
  const [data, setData] = useState<{stats: any[], recentAlerts: any[], riskMatrixData: any, recoveryVectorData: any} | null>(null);
  const [error, setError] = useState<string | null>(null);


  useEffect(() => {
    apiFetch('/api/dashboard')
      .then(async res => {
        if (!res.ok) {
          const msg = await res.text();
          throw new Error(`API Error: ${res.status} ${msg}`);
        }
        return res.json();
      })
      .then(setData)
      .catch((err) => {
        setError(err.message || 'Failed to sync dashboard data.');
      });
  }, []);

  if (error) {
    return (
      <div className="p-8 text-rose-400 font-mono text-sm tracking-widest animate-pulse">
        ERROR: {error}
      </div>
    );
  }
  if (!data) return <div className="p-8 text-zinc-400 font-mono text-sm tracking-widest animate-pulse">SYNCING DATA...</div>;

  const { stats, recentAlerts, riskMatrixData, recoveryVectorData } = data;

  return (
    <div className="space-y-6">
      {/* Header section */}
      <div className="flex justify-between items-end mb-8">
        <div>
          <h1 className="text-3xl font-black tracking-tighter text-white uppercase mb-1">Global Risk Overview</h1>
          <p className="text-zinc-500 font-mono text-xs tracking-widest uppercase">Global Portfolio Risk Metrics // RT-Active</p>
        </div>
        <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-cyan-500/10 border border-cyan-500/20 rounded-md text-cyan-400 text-xs font-bold tracking-wider">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          LIVE SYNC
        </div>
      </div>

      {/* Top Stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((item) => {
          const IconComponent = iconMap[item.iconType] || Activity;
          return (
          <div key={item.name} className="relative bg-[#131316] overflow-hidden rounded-xl border border-zinc-800/80 shadow-lg group hover:border-zinc-700 transition-colors">
            <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
              <IconComponent className="w-16 h-16" />
            </div>
            <div className="p-5 relative z-10">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-[10px] font-bold tracking-widest text-zinc-400 uppercase">{item.name}</h3>
                <div className={`p-2 rounded-md border ${item.bg}`}>
                  <IconComponent className={`h-4 w-4 ${item.color}`} />
                </div>
              </div>
              <div className="flex items-end justify-between">
                <div className="text-3xl font-black tracking-tight text-white font-mono">
                  {item.value}
                </div>
                <div className={`flex items-center text-xs font-bold px-2 py-1 rounded bg-[#09090b] border ${
                  item.changeType === 'positive' ? 'text-emerald-400 border-emerald-500/20' : 'text-rose-400 border-rose-500/20'
                }`}>
                  {item.changeType === 'positive' ? <ArrowUpRight className="h-3 w-3 mr-1" /> : <ArrowDownRight className="h-3 w-3 mr-1" />}
                  {item.change}
                </div>
              </div>
            </div>
          </div>
        )})}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Charts Section */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-[#131316] rounded-xl border border-zinc-800/80 p-6 shadow-lg">
            <h3 className="text-xs font-bold tracking-widest text-zinc-400 uppercase mb-6 flex items-center gap-2">
              <Activity className="h-4 w-4 text-cyan-400" />
              Portfolio Risk Factor Health
            </h3>
            <div className="h-72 w-full flex items-center justify-center rounded-lg relative">
              <Plot
                data={[
                  {
                    x: riskMatrixData.x,
                    y: riskMatrixData.y,
                    type: 'bar',
                    marker: { 
                      color: ['#10b981', '#fbbf24', '#fb7185', '#e11d48'],
                      line: { color: 'rgba(255,255,255,0.1)', width: 1 }
                    }
                  }
                ]}
                layout={{
                  autosize: true,
                  margin: { t: 10, r: 10, l: 30, b: 30 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  xaxis: { fixedrange: true, tickfont: { color: '#71717a' }, gridcolor: 'transparent' },
                  yaxis: { fixedrange: true, gridcolor: '#27272a', tickfont: { color: '#71717a' } },
                  font: { family: 'inherit' }
                }}
                useResizeHandler={true}
                className="w-full h-full"
                config={{ displayModeBar: false }}
              />
            </div>
          </div>

          <div className="bg-[#131316] rounded-xl border border-zinc-800/80 p-6 shadow-lg">
            <h3 className="text-xs font-bold tracking-widest text-zinc-400 uppercase mb-6 flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-violet-400" />
              Average Recovery Rate (6M)
            </h3>
            <div className="h-72 w-full flex items-center justify-center rounded-lg">
               <Plot
                data={[
                  {
                    x: recoveryVectorData.x,
                    y: recoveryVectorData.y,
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { shape: 'spline', color: '#22d3ee', width: 3 },
                    marker: { size: 8, color: '#09090b', line: { color: '#22d3ee', width: 2 } },
                    fill: 'tozeroy',
                    fillcolor: 'rgba(34, 211, 238, 0.05)'
                  }
                ]}
                layout={{
                  autosize: true,
                  margin: { t: 10, r: 10, l: 30, b: 30 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  xaxis: { fixedrange: true, gridcolor: '#27272a', tickfont: { color: '#71717a' } },
                  yaxis: { fixedrange: true, gridcolor: '#27272a', range: [50, 100], tickfont: { color: '#71717a' } },
                  font: { family: 'inherit' }
                }}
                useResizeHandler={true}
                className="w-full h-full"
                config={{ displayModeBar: false }}
              />
            </div>
          </div>
        </div>

        {/* Actionable Insights */}
        <div className="space-y-6">
          <div className="bg-[#131316] rounded-xl border border-zinc-800/80 p-6 shadow-lg">
            <h3 className="text-xs font-bold tracking-widest text-rose-400 uppercase mb-6 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              Recent Account Alerts
            </h3>
            <div className="flow-root">
              <ul className="-my-5 divide-y divide-zinc-800/60">
                {recentAlerts.map((alert) => (
                  <li key={alert.id} className="py-4">
                    <div className="flex items-center space-x-4">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-bold tracking-wide text-zinc-100 truncate">
                          {alert.borrower}
                        </p>
                        <p className="text-xs text-zinc-500 font-mono truncate mt-0.5">
                          ACT: {alert.action}
                        </p>
                      </div>
                      <div className="flex flex-col items-end">
                        <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-black uppercase tracking-wider border
                          ${alert.risk === 'Critical' ? 'bg-rose-500/10 text-rose-400 border-rose-500/20' : 
                            alert.risk === 'High' ? 'bg-orange-500/10 text-orange-400 border-orange-500/20' : 
                            'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'}
                        `}>
                          {alert.risk}
                        </span>
                        <span className="text-[10px] font-mono text-zinc-500 mt-1.5">{alert.time}</span>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
            <div className="mt-6">
              <Link
                to="/institution/recovery-actions"
                className="w-full flex justify-center items-center px-4 py-2 border border-zinc-700 rounded-md text-xs font-bold tracking-widest uppercase text-zinc-300 hover:bg-zinc-800 hover:text-white transition-all"
              >
                Access Full Log
              </Link>
            </div>
          </div>

          <div className="bg-gradient-to-br from-[#1a103c] to-[#0f172a] border border-violet-500/20 rounded-xl shadow-lg p-6 relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-4 opacity-10">
              <Zap className="w-24 h-24 text-violet-400" />
            </div>
            <h3 className="text-xs font-bold tracking-widest text-violet-300 uppercase mb-2 flex items-center gap-2 relative z-10">
              <Zap className="h-4 w-4" />
              Smart Routing Suggestion
            </h3>
            <p className="text-zinc-400 text-sm mb-6 leading-relaxed relative z-10">
              High probability of default detected on high-value case. Auto-routing to Legal Team recommended to mitigate loss.
            </p>
            <button className="relative z-10 bg-violet-500/20 border border-violet-500/50 text-violet-300 px-4 py-2.5 rounded-md text-xs font-black tracking-widest uppercase hover:bg-violet-500 hover:text-white transition-all shadow-[0_0_15px_-3px_rgba(139,92,246,0.3)] hover:shadow-[0_0_20px_rgba(139,92,246,0.5)] w-full">
              APPROVE_ROUTING
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
