import { useState, useEffect, useCallback } from "react";
import {
  FileText, RefreshCw, CheckCircle, XCircle, Clock,
  AlertTriangle, ChevronRight, ArrowRight, Search, X
} from "lucide-react";
import { useNavigate } from "react-router";
import { apiFetch } from "../lib/api";

type Tab = "all" | "conditional" | "rejected" | "active";

const STATUS_META: Record<string, {
  label: string; color: string; bg: string; border: string;
  darkColor: string; darkBg: string; icon: any;
}> = {
  active: {
    label: "Approved",
    color: "text-emerald-700", bg: "bg-emerald-100", border: "border-emerald-300",
    darkColor: "dark:text-emerald-300", darkBg: "dark:bg-emerald-900/20",
    icon: CheckCircle,
  },
  conditional: {
    label: "Conditional",
    color: "text-amber-700", bg: "bg-amber-100", border: "border-amber-300",
    darkColor: "dark:text-amber-300", darkBg: "dark:bg-amber-900/20",
    icon: Clock,
  },
  rejected: {
    label: "Declined",
    color: "text-rose-700", bg: "bg-rose-100", border: "border-rose-300",
    darkColor: "dark:text-rose-300", darkBg: "dark:bg-rose-900/20",
    icon: XCircle,
  },
};

const PRIORITY_CLS: Record<string, string> = {
  Critical: "bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300",
  High:     "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300",
  Medium:   "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300",
  Low:      "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300",
};

// ── Approve Confirm Modal ───────────────────────────────────────────────────────
function ApproveModal({
  item, onClose, onDone,
}: { item: any; onClose: () => void; onDone: () => void }) {
  const [loading, setLoading] = useState(false);
  const [teams, setTeams] = useState<any[]>([]);
  const [selectedTeam, setSelectedTeam] = useState("");
  const [teamError, setTeamError] = useState("");

  useEffect(() => {
    // Fetch teams from API
    (async () => {
      try {
        const res = await apiFetch("/api/teams");
        const data = await res.json();
        setTeams(data.teams || []);
      } catch (e) {
        setTeams([
          { name: "Automated System" },
          { name: "Call Center" },
          { name: "Dedicated Field Officers" },
          { name: "Legal Team" },
        ]);
      }
    })();
  }, []);

  const handleApprove = async () => {
    setTeamError("");
    if (!selectedTeam) {
      setTeamError("Please select a team to assign.");
      return;
    }
    setLoading(true);
    try {
      await apiFetch(`/api/cases/${item.id}/approve`, {
        method: "PATCH",
        body: JSON.stringify({ status: "active", assigned_team: selectedTeam }),
      });
      onDone();
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="bg-white dark:bg-neutral-900 rounded-2xl shadow-2xl w-full max-w-md border border-neutral-200 dark:border-neutral-800">
        <div className="flex items-center justify-between p-6 border-b border-neutral-100 dark:border-neutral-800">
          <h3 className="font-bold text-neutral-900 dark:text-white flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-emerald-500" /> Approve Loan Application
          </h3>
          <button onClick={onClose} className="p-1.5 rounded-full hover:bg-neutral-100 dark:hover:bg-neutral-800 transition">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="p-6 space-y-4">
          <div className="bg-neutral-50 dark:bg-neutral-800 rounded-xl p-4 space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-neutral-500">Borrower</span>
              <span className="font-bold text-neutral-900 dark:text-white">{item.borrower_name}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-neutral-500">Loan Amount</span>
              <span className="font-bold text-neutral-900 dark:text-white">
                ₹{Number(item.loan_amount).toLocaleString("en-IN")}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-neutral-500">Risk Profile</span>
              <span className={`font-bold px-2 py-0.5 rounded-full text-xs ${PRIORITY_CLS[item.priority]}`}>
                {item.priority} · PD {((item.default_probability || 0) * 100).toFixed(1)}%
              </span>
            </div>
          </div>
          <div className="space-y-2">
            <label className="block text-xs font-semibold text-neutral-600 dark:text-neutral-400 uppercase tracking-wide mb-1">
              Assign Recovery Team
            </label>
            <select
              className="block w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 text-neutral-900 dark:text-white px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
              value={selectedTeam}
              onChange={e => setSelectedTeam(e.target.value)}
              disabled={loading}
            >
              <option value="">Select a team…</option>
              {teams.map((t: any) => (
                <option key={t.name} value={t.name}>{t.name}</option>
              ))}
            </select>
            {teamError && <div className="text-xs text-rose-600 mt-1">{teamError}</div>}
          </div>
          <div className="text-xs text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-3 leading-relaxed">
            By approving, this application will move to <strong>Active Portfolio</strong> and the assigned recovery team will begin monitoring the account.
          </div>
        </div>
        <div className="flex gap-3 px-6 pb-6">
          <button onClick={onClose} className="flex-1 py-2.5 rounded-xl border border-neutral-200 dark:border-neutral-700 text-sm font-semibold text-neutral-700 dark:text-neutral-300 hover:bg-neutral-50 dark:hover:bg-neutral-800 transition">
            Cancel
          </button>
          <button
            onClick={handleApprove}
            disabled={loading}
            className="flex-1 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-bold transition disabled:opacity-50"
          >
            {loading ? "Approving…" : "✅ Confirm Approval"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Case Detail Drawer ──────────────────────────────────────────────────────────
function DetailDrawer({ item, onClose, onApprove }: { item: any; onClose: () => void; onApprove: () => void }) {
  const meta = STATUS_META[item.status] || STATUS_META.conditional;
  const Icon = meta.icon;
  const navigate = useNavigate();

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-md bg-white dark:bg-neutral-900 shadow-2xl flex flex-col h-full border-l border-neutral-200 dark:border-neutral-800">
        <div className="flex items-center justify-between p-5 border-b border-neutral-100 dark:border-neutral-800">
          <div className="flex items-center gap-2">
            <span className={`p-2 rounded-lg ${meta.bg} ${meta.darkBg}`}>
              <Icon className={`h-5 w-5 ${meta.color} ${meta.darkColor}`} />
            </span>
            <div>
              <h3 className="font-bold text-neutral-900 dark:text-white text-sm">{item.borrower_name}</h3>
              <p className="text-xs text-neutral-500">{item.account_id}</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-full hover:bg-neutral-100 dark:hover:bg-neutral-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="overflow-y-auto flex-1 p-5 space-y-5">
          {/* Status */}
          <div className={`flex items-center gap-2 p-3 rounded-xl border ${meta.bg} ${meta.border} ${meta.darkBg}`}>
            <Icon className={`h-4 w-4 ${meta.color} ${meta.darkColor}`} />
            <span className={`text-sm font-bold ${meta.color} ${meta.darkColor}`}>
              {meta.label}
              {item.status === "conditional" && " — Awaiting Conditions to be Met"}
              {item.status === "rejected" && " — Declined & Archived"}
              {item.status === "active" && " — Monitored in Active Portfolio"}
            </span>
          </div>

          {/* Loan details */}
          <div className="bg-neutral-50 dark:bg-neutral-800 rounded-xl p-4 space-y-3">
            <div className="text-xs font-bold text-neutral-400 uppercase tracking-wider">Loan Details</div>
            {[
              { label: "Borrower", value: item.borrower_name },
              { label: "Account ID", value: item.account_id, mono: true },
              { label: "Loan Amount", value: `₹${Number(item.loan_amount).toLocaleString("en-IN")}` },
              { label: "Outstanding", value: `₹${Number(item.outstanding).toLocaleString("en-IN")}` },
              { label: "Credit Score", value: item.credit_score || "—" },
              { label: "Days Overdue", value: item.days_overdue > 0 ? `${item.days_overdue} days` : "None" },
            ].map(({ label, value, mono }) => (
              <div key={label} className="flex justify-between text-sm">
                <span className="text-neutral-500">{label}</span>
                <span className={`font-semibold text-neutral-900 dark:text-white ${mono ? "font-mono text-xs" : ""}`}>{value}</span>
              </div>
            ))}
          </div>

          {/* Risk metrics */}
          <div className="bg-neutral-50 dark:bg-neutral-800 rounded-xl p-4 space-y-3">
            <div className="text-xs font-bold text-neutral-400 uppercase tracking-wider">Risk Assessment</div>
            <div className="flex justify-between text-sm">
              <span className="text-neutral-500">Probability of Default</span>
              <span className={`font-bold ${(item.default_probability || 0) > 0.3 ? "text-rose-600" : "text-emerald-600"}`}>
                {((item.default_probability || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-neutral-500">Priority</span>
              <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${PRIORITY_CLS[item.priority]}`}>{item.priority}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-neutral-500">Assigned Team</span>
              <span className="font-semibold text-neutral-700 dark:text-neutral-300 text-xs">{item.assigned_team}</span>
            </div>
          </div>

          {/* Recommended action */}
          <div className="bg-neutral-50 dark:bg-neutral-800 rounded-xl p-4">
            <div className="text-xs font-bold text-neutral-400 uppercase tracking-wider mb-2">Recommended Action</div>
            <p className="text-sm text-neutral-700 dark:text-neutral-300">{item.recommended_action || "—"}</p>
          </div>

          {/* Timeline */}
          <div className="bg-neutral-50 dark:bg-neutral-800 rounded-xl p-4 space-y-2">
            <div className="text-xs font-bold text-neutral-400 uppercase tracking-wider">Timeline</div>
            <div className="flex justify-between text-xs text-neutral-500">
              <span>Created</span><span className="font-mono">{item.created_at?.slice(0, 16) || "—"}</span>
            </div>
            <div className="flex justify-between text-xs text-neutral-500">
              <span>Last Updated</span><span className="font-mono">{item.updated_at?.slice(0, 16) || "—"}</span>
            </div>
          </div>
        </div>

        {/* Footer actions */}
        {item.status === "conditional" && (
          <div className="p-5 border-t border-neutral-100 dark:border-neutral-800 space-y-2">
            <button
              onClick={onApprove}
              className="w-full py-3 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-sm transition flex items-center justify-center gap-2"
            >
              <CheckCircle className="h-4 w-4" /> Approve — Conditions Met
            </button>
            <button
              onClick={() => { onClose(); navigate("/institution/borrower-input"); }}
              className="w-full py-2.5 rounded-xl border border-neutral-200 dark:border-neutral-700 text-sm font-semibold text-neutral-600 dark:text-neutral-300 hover:bg-neutral-50 dark:hover:bg-neutral-800 transition flex items-center justify-center gap-2"
            >
              <ArrowRight className="h-4 w-4" /> Re-evaluate Borrower
            </button>
          </div>
        )}
        {item.status === "active" && (
          <div className="p-5 border-t border-neutral-100 dark:border-neutral-800">
            <div className="flex items-center justify-center gap-2 text-emerald-600 dark:text-emerald-400 text-sm font-semibold">
              <CheckCircle className="h-4 w-4" /> Active — Monitored by {item.assigned_team}
            </div>
          </div>
        )}
        {item.status === "rejected" && (
          <div className="p-5 border-t border-neutral-100 dark:border-neutral-800">
            <button
              onClick={() => { onClose(); navigate("/institution/borrower-input"); }}
              className="w-full py-2.5 rounded-xl border border-rose-200 dark:border-rose-800 text-sm font-semibold text-rose-600 dark:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-900/20 transition flex items-center justify-center gap-2"
            >
              <ArrowRight className="h-4 w-4" /> Re-evaluate with New Data
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main Page ───────────────────────────────────────────────────────────────────
export function LoanApplications() {
  const [cases, setCases]       = useState<any[]>([]);
  const [loading, setLoading]   = useState(true);
  const [tab, setTab]           = useState<Tab>("all");
  const [search, setSearch]     = useState("");
  const [detail, setDetail]     = useState<any>(null);
  const [approving, setApproving] = useState<any>(null);

  const fetchCases = useCallback(async () => {
    setLoading(true);
    try {
      // Fetch all three statuses separately and merge — avoids backend needing OR logic
      const [aRes, cRes, rRes] = await Promise.all([
        apiFetch(`/api/cases?status=active`),
        apiFetch(`/api/cases?status=conditional`),
        apiFetch(`/api/cases?status=rejected`),
      ]);
      const [a, c, r] = await Promise.all([aRes.json(), cRes.json(), rRes.json()]);
      setCases([...(c.cases || []), ...(a.cases || []), ...(r.cases || [])]);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { fetchCases(); }, [fetchCases]);

  const filtered = cases.filter(c => {
    const matchTab = tab === "all" || c.status === tab;
    const matchSearch = !search ||
      c.borrower_name?.toLowerCase().includes(search.toLowerCase()) ||
      c.account_id?.toLowerCase().includes(search.toLowerCase());
    return matchTab && matchSearch;
  });

  // Counts
  const counts = {
    all:         cases.length,
    conditional: cases.filter(c => c.status === "conditional").length,
    rejected:    cases.filter(c => c.status === "rejected").length,
    active:      cases.filter(c => c.status === "active").length,
  };

  const tabs: { key: Tab; label: string; count: number; color: string }[] = [
    { key: "all",         label: "All Applications",  count: counts.all,         color: "text-neutral-600 dark:text-neutral-300" },
    { key: "conditional", label: "Conditional",       count: counts.conditional, color: "text-amber-600 dark:text-amber-400" },
    { key: "rejected",    label: "Declined & Archived", count: counts.rejected,  color: "text-rose-600 dark:text-rose-400" },
    { key: "active",      label: "Approved / Active", count: counts.active,      color: "text-emerald-600 dark:text-emerald-400" },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-neutral-900 dark:text-white flex items-center gap-2">
            <FileText className="h-6 w-6 text-neutral-500" />
            Loan Applications
          </h2>
          <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
            All evaluated loan applications — conditional, declined, and approved.
          </p>
        </div>
        <button onClick={fetchCases} className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold rounded-lg transition">
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: "Total Evaluated", value: counts.all, Icon: FileText, cls: "text-neutral-900 dark:text-white" },
          { label: "Pending Conditions", value: counts.conditional, Icon: Clock, cls: "text-amber-600 dark:text-amber-400" },
          { label: "Declined / Archived", value: counts.rejected, Icon: XCircle, cls: "text-rose-600 dark:text-rose-400" },
          { label: "Approved & Active", value: counts.active, Icon: CheckCircle, cls: "text-emerald-600 dark:text-emerald-400" },
        ].map(({ label, value, Icon, cls }) => (
          <div key={label} className="bg-white dark:bg-neutral-900 rounded-xl border border-neutral-200 dark:border-neutral-800 p-4">
            <Icon className={`h-5 w-5 mb-2 ${cls}`} />
            <div className={`text-2xl font-black ${cls}`}>{value}</div>
            <div className="text-xs text-neutral-500 mt-0.5">{label}</div>
          </div>
        ))}
      </div>

      {/* Main card */}
      <div className="bg-white dark:bg-neutral-900 shadow-sm rounded-xl border border-neutral-200 dark:border-neutral-800 overflow-hidden">

        {/* Tabs + search */}
        <div className="border-b border-neutral-200 dark:border-neutral-800">
          <div className="flex flex-col sm:flex-row sm:items-center gap-0">
            <div className="flex overflow-x-auto">
              {tabs.map(t => (
                <button
                  key={t.key}
                  onClick={() => setTab(t.key)}
                  className={`flex items-center gap-1.5 px-4 py-3.5 text-sm font-semibold border-b-2 whitespace-nowrap transition-colors ${
                    tab === t.key
                      ? `border-blue-500 ${t.color}`
                      : "border-transparent text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300"
                  }`}
                >
                  {t.label}
                  <span className={`px-1.5 py-0.5 rounded-full text-xs font-bold ${
                    tab === t.key ? "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300" : "bg-neutral-100 dark:bg-neutral-800 text-neutral-500"
                  }`}>
                    {t.count}
                  </span>
                </button>
              ))}
            </div>
            <div className="flex-1 p-3 sm:px-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-neutral-400 pointer-events-none" />
                <input
                  type="text" value={search} onChange={e => setSearch(e.target.value)}
                  placeholder="Search borrower or account ID…"
                  className="w-full pl-8 pr-3 py-1.5 rounded-lg border border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800 text-sm text-neutral-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Table */}
        {loading ? (
          <div className="p-12 text-center text-neutral-400 animate-pulse text-sm">Loading applications…</div>
        ) : filtered.length === 0 ? (
          <div className="p-12 text-center">
            <AlertTriangle className="h-10 w-10 text-neutral-300 mx-auto mb-3" />
            <p className="text-neutral-500 text-sm">No applications match the selected filter.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-neutral-100 dark:divide-neutral-800">
              <thead className="bg-neutral-50 dark:bg-neutral-800/50">
                <tr>
                  {["Borrower / Account", "Loan Amount", "Risk (PD)", "Priority", "Status", "Assigned Team", "Saved On", ""].map(h => (
                    <th key={h} className="px-5 py-3 text-left text-xs font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-100 dark:divide-neutral-800">
                {filtered.map(c => {
                  const m = STATUS_META[c.status] || STATUS_META.conditional;
                  const Icon = m.icon;
                  return (
                    <tr key={c.id} className="hover:bg-neutral-50 dark:hover:bg-neutral-800/40 transition-colors">
                      <td className="px-5 py-3.5">
                        <div className="font-semibold text-sm text-neutral-900 dark:text-white">{c.borrower_name}</div>
                        <div className="text-xs text-neutral-400 font-mono">{c.account_id}</div>
                      </td>
                      <td className="px-5 py-3.5 whitespace-nowrap text-sm font-bold text-neutral-800 dark:text-neutral-200">
                        ₹{Number(c.loan_amount).toLocaleString("en-IN")}
                      </td>
                      <td className="px-5 py-3.5 whitespace-nowrap">
                        <span className={`text-sm font-bold ${(c.default_probability || 0) > 0.3 ? "text-rose-600 dark:text-rose-400" : "text-emerald-600 dark:text-emerald-400"}`}>
                          {((c.default_probability || 0) * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-5 py-3.5 whitespace-nowrap">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${PRIORITY_CLS[c.priority] || ""}`}>{c.priority}</span>
                      </td>
                      <td className="px-5 py-3.5 whitespace-nowrap">
                        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${m.bg} ${m.border} ${m.color} ${m.darkBg} ${m.darkColor}`}>
                          <Icon className="h-3 w-3" /> {m.label}
                        </span>
                      </td>
                      <td className="px-5 py-3.5 text-xs text-neutral-500 dark:text-neutral-400 max-w-[120px] truncate">
                        {c.assigned_team}
                      </td>
                      <td className="px-5 py-3.5 whitespace-nowrap text-xs text-neutral-400 font-mono">
                        {c.created_at?.slice(0, 10) || "—"}
                      </td>
                      <td className="px-5 py-3.5 whitespace-nowrap">
                        <button
                          onClick={() => setDetail(c)}
                          className="flex items-center gap-1 text-xs font-semibold text-blue-600 dark:text-blue-400 hover:underline"
                        >
                          View <ChevronRight className="h-3.5 w-3.5" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        <div className="px-5 py-3 border-t border-neutral-100 dark:border-neutral-800 flex items-center justify-between">
          <p className="text-xs text-neutral-500">
            Showing <span className="font-semibold text-neutral-700 dark:text-neutral-300">{filtered.length}</span> of {counts.all} applications
          </p>
          {search && (
            <button onClick={() => setSearch("")} className="text-xs text-blue-600 hover:underline">Clear search</button>
          )}
        </div>
      </div>

      {/* Detail drawer */}
      {detail && (
        <DetailDrawer
          item={detail}
          onClose={() => setDetail(null)}
          onApprove={() => { setApproving(detail); setDetail(null); }}
        />
      )}

      {/* Approve modal */}
      {approving && (
        <ApproveModal
          item={approving}
          onClose={() => setApproving(null)}
          onDone={async () => { setApproving(null); await fetchCases(); }}
        />
      )}
    </div>
  );
}
