import { useState, useEffect } from "react";
import { BarChart3, Download, RefreshCw, TrendingUp, AlertTriangle, CheckCircle, Users } from "lucide-react";
import Plot from "react-plotly.js";
import { apiFetch } from "../lib/api";

export function Analytics() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await apiFetch("/api/analytics");
      const json = await res.json();
      setData(json);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  };

  useEffect(() => { fetchData(); }, []);

  const isDark = document.documentElement.classList.contains("dark");
  const textColor = "#6b7280";
  const gridColor = isDark ? "#27272a" : "#e5e7eb";
  const paperBg = "transparent";

  if (loading) return (
    <div className="p-12 text-center">
      <RefreshCw className="h-8 w-8 text-blue-500 animate-spin mx-auto mb-3" />
      <p className="text-sm text-neutral-500 animate-pulse">Loading analytics…</p>
    </div>
  );

  if (!data) return <div className="p-8 text-red-500 text-sm">Failed to load analytics.</div>;

  const { recoveryEfficiency, defaultDistribution, teamExposure, evaluationTrend, kpis } = data;

  const totalOutstandingLakhs = kpis.totalOutstanding ? (kpis.totalOutstanding / 1e5).toFixed(1) : "0";

  const kpiCards = [
    { label: "Total Cases Evaluated", value: kpis.totalCases, icon: BarChart3, color: "text-blue-600 dark:text-blue-400", bg: "bg-blue-50 dark:bg-blue-900/20" },
    { label: "Active Cases",          value: kpis.activeCases, icon: AlertTriangle, color: "text-amber-600 dark:text-amber-400", bg: "bg-amber-50 dark:bg-amber-900/20" },
    { label: "Cases Resolved",        value: kpis.resolvedCases, icon: CheckCircle, color: "text-emerald-600 dark:text-emerald-400", bg: "bg-emerald-50 dark:bg-emerald-900/20" },
    { label: "Resolution Rate",       value: kpis.resolutionRate, icon: TrendingUp, color: "text-indigo-600 dark:text-indigo-400", bg: "bg-indigo-50 dark:bg-indigo-900/20" },
    { label: "Avg Default Probability", value: kpis.avgDefaultProb, icon: AlertTriangle, color: "text-rose-600 dark:text-rose-400", bg: "bg-rose-50 dark:bg-rose-900/20" },
    { label: "Total Outstanding (₹L)", value: `₹${totalOutstandingLakhs}L`, icon: Users, color: "text-purple-600 dark:text-purple-400", bg: "bg-purple-50 dark:bg-purple-900/20" },
  ];

  return (
    <>
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-neutral-900 dark:text-white flex items-center gap-2">
            <BarChart3 className="h-6 w-6 text-neutral-500" />
            Analytics & Reporting
          </h2>
          <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
            Live metrics from the risk engine and recovery database.
          </p>
        </div>
        <div className="flex gap-2">
          <button onClick={fetchData} className="flex items-center gap-2 px-3 py-2 border border-neutral-300 dark:border-neutral-700 rounded-lg text-sm font-medium text-neutral-700 dark:text-neutral-300 bg-white dark:bg-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-700 transition">
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </button>
          <button className="flex items-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition">
            <Download className="h-4 w-4" />
            Export Report
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {kpiCards.map(({ label, value, icon: Icon, color, bg }) => (
          <div key={label} className="bg-white dark:bg-neutral-900 rounded-xl border border-neutral-200 dark:border-neutral-800 p-4">
            <div className={`w-9 h-9 rounded-lg ${bg} flex items-center justify-center mb-3`}>
              <Icon className={`h-5 w-5 ${color}`} />
            </div>
            <div className={`text-2xl font-black ${color}`}>{value ?? "—"}</div>
            <div className="text-xs text-neutral-500 mt-1 leading-tight">{label}</div>
          </div>
        ))}
      </div>

      {/* Charts row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Default Distribution Donut */}
        <div className="bg-white dark:bg-neutral-900 shadow-sm rounded-xl border border-neutral-200 dark:border-neutral-800 p-6">
          <h3 className="text-base font-bold text-neutral-900 dark:text-white mb-1">Case Priority Distribution</h3>
          <p className="text-xs text-neutral-500 mb-4">Breakdown of all evaluated cases by risk priority</p>
          <div className="h-72">
            {defaultDistribution.values.every((v: number) => v === 0) ? (
              <div className="h-full flex flex-col items-center justify-center text-neutral-400">
                <AlertTriangle className="h-10 w-10 mb-2" />
                <p className="text-sm">No cases evaluated yet.</p>
              </div>
            ) : (
              <Plot
                data={[{
                  values: defaultDistribution.values,
                  labels: defaultDistribution.labels,
                  type: "pie",
                  hole: 0.5,
                  marker: { colors: ["#10b981", "#f59e0b", "#f97316", "#f43f5e"] },
                  textinfo: "label+percent",
                  hoverinfo: "label+value+percent",
                }]}
                layout={{
                  autosize: true,
                  margin: { t: 10, r: 10, l: 10, b: 10 },
                  paper_bgcolor: paperBg, plot_bgcolor: paperBg,
                  font: { family: "inherit", color: textColor, size: 11 },
                  showlegend: false,
                }}
                useResizeHandler
                style={{ width: "100%", height: "100%" }}
                config={{ displayModeBar: false }}
              />
            )}
          </div>
        </div>

        {/* Evaluation Trend */}
        <div className="bg-white dark:bg-neutral-900 shadow-sm rounded-xl border border-neutral-200 dark:border-neutral-800 p-6">
          <h3 className="text-base font-bold text-neutral-900 dark:text-white mb-1">Evaluation Activity (Last 7 Days)</h3>
          <p className="text-xs text-neutral-500 mb-4">Daily count of risk evaluations</p>
          <div className="h-72">
            {(!evaluationTrend || evaluationTrend.days.length === 0) ? (
              <div className="h-full flex flex-col items-center justify-center text-neutral-400">
                <TrendingUp className="h-10 w-10 mb-2" />
                <p className="text-sm">No trend data yet.</p>
              </div>
            ) : (
              <Plot
                data={[{
                  x: evaluationTrend.days,
                  y: evaluationTrend.counts,
                  type: "scatter",
                  mode: "lines+markers",
                  fill: "tozeroy",
                  line: { color: "#3b82f6", width: 2, shape: "spline" },
                  fillcolor: "rgba(59,130,246,0.1)",
                  marker: { size: 8, color: "#3b82f6" },
                }]}
                layout={{
                  autosize: true,
                  margin: { t: 10, r: 10, l: 40, b: 50 },
                  paper_bgcolor: paperBg, plot_bgcolor: paperBg,
                  xaxis: { fixedrange: true, gridcolor: gridColor, tickfont: { size: 10 } },
                  yaxis: { fixedrange: true, gridcolor: gridColor, tickformat: "d" },
                  font: { family: "inherit", color: textColor },
                }}
                useResizeHandler
                style={{ width: "100%", height: "100%" }}
                config={{ displayModeBar: false }}
              />
            )}
          </div>
        </div>
      </div>

      {/* Charts row 2 */}
      <div className="grid grid-cols-1 gap-6">
        {/* Team Exposure */}
        <div className="bg-white dark:bg-neutral-900 shadow-sm rounded-xl border border-neutral-200 dark:border-neutral-800 p-6">
          <h3 className="text-base font-bold text-neutral-900 dark:text-white mb-1">Portfolio Exposure by Team</h3>
          <p className="text-xs text-neutral-500 mb-4">Outstanding balance (₹ Lakhs) managed per recovery team</p>
          <div className="h-80">
            {(!teamExposure || teamExposure.teams.length === 0) ? (
              <div className="h-full flex flex-col items-center justify-center text-neutral-400">
                <Users className="h-10 w-10 mb-2" />
                <p className="text-sm">No data yet.</p>
              </div>
            ) : (
              <Plot
                data={[
                  {
                    x: teamExposure.teams,
                    y: teamExposure.outstanding_lakhs,
                    name: "Outstanding (₹L)",
                    type: "bar",
                    marker: { color: "#6366f1" },
                    text: teamExposure.outstanding_lakhs.map((v: number) => `₹${v}L`),
                    textposition: "outside",
                  },
                ]}
                layout={{
                  autosize: true,
                  margin: { t: 20, r: 10, l: 40, b: 80 },
                  paper_bgcolor: paperBg, plot_bgcolor: paperBg,
                  xaxis: { fixedrange: true, tickangle: -15, tickfont: { size: 10 }, gridcolor: gridColor },
                  yaxis: { title: "₹ Lakhs", fixedrange: true, gridcolor: gridColor },
                  font: { family: "inherit", color: textColor },
                  bargap: 0.4,
                }}
                useResizeHandler
                style={{ width: "100%", height: "100%" }}
                config={{ displayModeBar: false }}
              />
            )}
          </div>
        </div>
      </div>
    </div>
    </>
  );
}
