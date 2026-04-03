import { useState, useEffect, useCallback } from "react";
import { History, Search, RefreshCw, ChevronDown } from "lucide-react";
import { apiFetch } from "../lib/api";

const actionColor = (action: string) => {
  if (action.includes("Evaluated") || action.includes("Simulation")) return "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300";
  if (action.includes("Batch"))    return "bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300";
  if (action.includes("Updated") || action.includes("Resolved")) return "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300";
  if (action.includes("Login") || action.includes("Register")) return "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300";
  if (action.includes("Legal") || action.includes("Escalat")) return "bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300";
  return "bg-neutral-100 text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300";
};

export function HistoryTracking() {
  const [data, setData] = useState<{ logs: any[]; total: number; actionTypes: string[] } | null>(null);
  const [loading, setLoading]     = useState(true);
  const [search, setSearch]       = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [actionType, setActionType] = useState("all");

  // Debounce search input
  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search), 350);
    return () => clearTimeout(t);
  }, [search]);

  const fetchData = useCallback(async () => {
    setLoading(true);
    const params = new URLSearchParams();
    if (debouncedSearch) params.set("search", debouncedSearch);
    if (actionType !== "all") params.set("action_type", actionType);
    try {
      const res = await apiFetch(`/api/history?${params}`);
      const json = await res.json();
      setData(json);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, [debouncedSearch, actionType]);

  useEffect(() => { fetchData(); }, [fetchData]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-neutral-900 dark:text-white flex items-center gap-2">
            <History className="h-6 w-6 text-neutral-500" />
            Audit Log & History
          </h2>
          <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
            Full record of every system action and manual intervention.
          </p>
        </div>
        <button onClick={fetchData} className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold rounded-lg transition">
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {/* Stats strip */}
      {data && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: "Total Events", value: data.total, color: "text-blue-600 dark:text-blue-400" },
            { label: "Action Types", value: data.actionTypes.length, color: "text-indigo-600 dark:text-indigo-400" },
            { label: "Showing", value: data.logs.length, color: "text-neutral-600 dark:text-neutral-400" },
            { label: "Latest", value: data.logs[0]?.date?.split(" ")[0] ?? "—", color: "text-emerald-600 dark:text-emerald-400" },
          ].map(({ label, value, color }) => (
            <div key={label} className="bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-800 p-4">
              <div className="text-xs text-neutral-500 uppercase tracking-wide mb-1">{label}</div>
              <div className={`text-xl font-black ${color}`}>{value}</div>
            </div>
          ))}
        </div>
      )}

      {/* Table card */}
      <div className="bg-white dark:bg-neutral-900 shadow-sm rounded-xl border border-neutral-200 dark:border-neutral-800 overflow-hidden">
        {/* Toolbar */}
        <div className="p-4 border-b border-neutral-200 dark:border-neutral-800 flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-neutral-400 pointer-events-none" />
            <input
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search by action or details…"
              className="block w-full pl-9 pr-3 py-2 rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 text-sm text-neutral-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          <div className="relative">
            <select
              value={actionType}
              onChange={e => setActionType(e.target.value)}
              className="appearance-none pl-3 pr-8 py-2 rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 text-sm text-neutral-700 dark:text-neutral-300 focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Action Types</option>
              {data?.actionTypes.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
            <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-neutral-400 pointer-events-none" />
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          {loading ? (
            <div className="p-12 text-center text-neutral-400 animate-pulse text-sm">Loading audit logs…</div>
          ) : !data || data.logs.length === 0 ? (
            <div className="p-12 text-center text-neutral-400 text-sm">No logs match your search.</div>
          ) : (
            <table className="min-w-full divide-y divide-neutral-100 dark:divide-neutral-800">
              <thead className="bg-neutral-50 dark:bg-neutral-800/50">
                <tr>
                  {["Timestamp", "Action", "Details", "User"].map(h => (
                    <th key={h} className="px-5 py-3 text-left text-xs font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-100 dark:divide-neutral-800">
                {data.logs.map((log) => (
                  <tr key={log.id} className="hover:bg-neutral-50 dark:hover:bg-neutral-800/40 transition-colors">
                    <td className="px-5 py-3.5 whitespace-nowrap text-xs text-neutral-500 font-mono">
                      {log.date}
                    </td>
                    <td className="px-5 py-3.5 whitespace-nowrap">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold ${actionColor(log.action)}`}>
                        {log.action}
                      </span>
                    </td>
                    <td className="px-5 py-3.5 text-xs text-neutral-600 dark:text-neutral-300 max-w-xs truncate">
                      {log.details}
                    </td>
                    <td className="px-5 py-3.5 whitespace-nowrap">
                      <div className="flex items-center gap-2">
                        <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs text-white font-bold flex-shrink-0 ${log.user === "System" ? "bg-indigo-500" : "bg-blue-500"}`}>
                          {log.user === "System" ? "S" : log.user.charAt(0).toUpperCase()}
                        </div>
                        <span className="text-xs text-neutral-700 dark:text-neutral-300">{log.user}</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Footer */}
        {data && (
          <div className="px-5 py-3 border-t border-neutral-100 dark:border-neutral-800 flex items-center justify-between">
            <p className="text-xs text-neutral-500">
              Showing <span className="font-semibold text-neutral-700 dark:text-neutral-300">{data.logs.length}</span> results
            </p>
            <button
              onClick={() => { setSearch(""); setActionType("all"); }}
              className="text-xs text-blue-600 hover:underline"
            >
              Clear filters
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
