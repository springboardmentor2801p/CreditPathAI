
import { useState, useEffect, useCallback } from "react";
import { Phone, Mail, Gavel, Users, Check, X, RefreshCw, AlertTriangle, Cpu, Scale } from "lucide-react";
import { apiFetch } from "../lib/api";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogClose,
} from "../components/ui/dialog";

const typeIcon = (type: string) => {
  if (type === "Legal") return <Scale className="h-4 w-4 text-rose-500" />;
  if (type === "Visit") return <Users className="h-4 w-4 text-emerald-500" />;
  if (type === "Call")  return <Phone className="h-4 w-4 text-blue-500" />;
  if (type === "Email") return <Mail className="h-4 w-4 text-indigo-500" />;
  return <Cpu className="h-4 w-4 text-neutral-400" />;
};

const typeBg = (type: string) => {
  if (type === "Legal") return "bg-rose-100 dark:bg-rose-900/30";
  if (type === "Visit") return "bg-emerald-100 dark:bg-emerald-900/30";
  if (type === "Call")  return "bg-blue-100 dark:bg-blue-900/30";
  return "bg-indigo-100 dark:bg-indigo-900/30";
};

const priorityCls = (p: string) => {
  if (p === "critical") return "bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300";
  if (p === "high")     return "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300";
  if (p === "medium")   return "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300";
  return "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300";
};


export function RecoveryActions() {
  const [data, setData]       = useState<{ actions: any[]; summary: any } | null>(null);
  const [loading, setLoading] = useState(true);
  const [priority, setPriority] = useState("all");
  const [team, setTeam]         = useState("all");
  const [status, setStatus]     = useState("all");
  const [updating, setUpdating] = useState<number | null>(null);

  // Modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [modalLoading, setModalLoading] = useState(false);
  const [modalError, setModalError] = useState<string | null>(null);
  const [caseDetails, setCaseDetails] = useState<any | null>(null);

  const openCaseModal = async (caseId: number) => {
    setModalOpen(true);
    setModalLoading(true);
    setModalError(null);
    setCaseDetails(null);
    try {
      const res = await apiFetch(`/api/cases/${caseId}`);
      if (!res.ok) throw new Error("Failed to fetch case details");
      const json = await res.json();
      setCaseDetails(json);
    } catch (e: any) {
      setModalError(e.message || "Error loading details");
    } finally {
      setModalLoading(false);
    }
  };

  const closeCaseModal = () => {
    setModalOpen(false);
    setCaseDetails(null);
    setModalError(null);
    setModalLoading(false);
  };

  const fetchData = useCallback(async () => {
    setLoading(true);
    const params = new URLSearchParams();
    if (priority !== "all") params.set("priority", priority);
    if (team !== "all")     params.set("team", team);
    if (status !== "all")   params.set("status", status);
    try {
      const res = await apiFetch(`/api/recovery?${params}`);
      const json = await res.json();
      setData(json);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, [priority, team, status]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const updateStatus = async (id: number, newStatus: string) => {
    setUpdating(id);
    try {
      await apiFetch(`/api/recovery/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ status: newStatus }),
      });
      await fetchData();
    } catch (e) { console.error(e); }
    finally { setUpdating(null); }
  };

  const selectCls = "bg-white border border-neutral-300 dark:border-neutral-700 text-neutral-700 dark:text-neutral-300 text-sm rounded-lg px-3 py-2 dark:bg-neutral-900 focus:ring-2 focus:ring-blue-500";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-neutral-900 dark:text-white">Active Recovery Panel</h2>
          <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
            Live NPA cases pulled from the risk engine. Mark resolved or dismiss below.
          </p>
        </div>
        <button onClick={fetchData} className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold rounded-lg transition">
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3">
        <select value={priority} onChange={e => setPriority(e.target.value)} className={selectCls}>
          <option value="all">All Priorities</option>
          <option value="critical">Critical</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
        </select>
        <select value={team} onChange={e => setTeam(e.target.value)} className={selectCls}>
          <option value="all">All Teams</option>
          <option value="Automated System">Automated System</option>
          <option value="Call Center">Call Center</option>
          <option value="Dedicated Field Officers">Field Officers</option>
          <option value="Legal Team">Legal Team</option>
        </select>
        <select value={status} onChange={e => setStatus(e.target.value)} className={selectCls}>
          <option value="all">Any Status</option>
          <option value="pending">Pending (Overdue)</option>
          <option value="completed">Resolved</option>
        </select>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Case list */}
        <div className="xl:col-span-2">
          <div className="bg-white dark:bg-neutral-900 shadow-sm rounded-xl border border-neutral-200 dark:border-neutral-800 overflow-hidden">
            {loading ? (
              <div className="p-12 text-center text-neutral-400 animate-pulse text-sm">Loading cases…</div>
            ) : !data || data.actions.length === 0 ? (
              <div className="p-12 text-center">
                <AlertTriangle className="h-10 w-10 text-neutral-300 mx-auto mb-3" />
                <p className="text-neutral-500 text-sm">No cases match the selected filters.</p>
              </div>
            ) : (
              <ul className="divide-y divide-neutral-100 dark:divide-neutral-800">
                {data.actions.map((action) => (
                  <li
                    key={action.id}
                    className={`p-4 sm:p-5 hover:bg-neutral-50 dark:hover:bg-neutral-800/40 transition-colors ${action.status === "resolved" ? "opacity-60" : ""}`}
                    style={{ cursor: "pointer" }}
                    onClick={() => openCaseModal(action.id)}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex items-start gap-3 min-w-0 flex-1">
                        <div className={`p-2.5 rounded-full flex-shrink-0 mt-0.5 ${typeBg(action.type)}`}>{typeIcon(action.type)}</div>
                        <div className="min-w-0">
                          <p className="text-sm font-semibold text-neutral-900 dark:text-white truncate">{action.borrower}</p>
                          <p className="text-xs text-neutral-500 mt-0.5 truncate">{action.action}</p>
                          <div className="flex flex-wrap items-center gap-2 mt-2">
                            <span className={`px-2 py-0.5 rounded-full text-xs font-bold border uppercase ${priorityCls(action.priority)}`}>{action.priorityLabel} Priority</span>
                            <span className="text-xs text-neutral-400">{action.team}</span>
                          </div>
                        </div>
                      </div>
                      <div className="flex flex-col items-end gap-2 flex-shrink-0">
                        <div className="text-right">
                          <div className="text-sm font-bold text-neutral-900 dark:text-white">{action.amount}</div>
                          <div className={`text-xs mt-0.5 ${action.due.includes("overdue") ? "text-rose-500 font-semibold" : "text-emerald-500"}`}>{action.due}</div>
                          <div className="text-xs text-neutral-400 mt-0.5">PD: {action.defaultProb}%</div>
                        </div>
                        {action.status === "resolved" ? (
                          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"><Check className="h-3 w-3" /> Resolved</span>
                        ) : (
                          <div className="flex items-center gap-1">
                            <button
                              onClick={e => { e.stopPropagation(); updateStatus(action.id, "resolved"); }}
                              disabled={updating === action.id}
                              title="Mark Resolved"
                              className="p-1.5 rounded-full text-emerald-600 hover:bg-emerald-50 dark:hover:bg-emerald-900/30 transition disabled:opacity-50"
                            >
                              {updating === action.id ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
                            </button>
                            <button
                              onClick={e => { e.stopPropagation(); updateStatus(action.id, "dismissed"); }}
                              disabled={updating === action.id}
                              title="Dismiss"
                              className="p-1.5 rounded-full text-rose-500 hover:bg-rose-50 dark:hover:bg-rose-900/30 transition disabled:opacity-50"
                            >
                              <X className="h-4 w-4" />
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  </li>
                ))}
                    {/* Modal for case details */}
                    <Dialog open={modalOpen} onOpenChange={setModalOpen}>
                      <DialogContent>
                        <DialogHeader>
                          <DialogTitle>Case Details</DialogTitle>
                        </DialogHeader>
                        {modalLoading ? (
                          <div className="py-8 text-center text-neutral-400">Loading…</div>
                        ) : modalError ? (
                          <div className="py-8 text-center text-rose-500">{modalError}</div>
                        ) : caseDetails ? (
                          <div className="space-y-2">
                            <div><b>Borrower:</b> {caseDetails.borrower_name}</div>
                            <div><b>Loan Amount:</b> ₹{caseDetails.loan_amount?.toLocaleString()}</div>
                            <div><b>Outstanding:</b> ₹{caseDetails.outstanding?.toLocaleString()}</div>
                            <div><b>Credit Score:</b> {caseDetails.credit_score}</div>
                            <div><b>Days Overdue:</b> {caseDetails.days_overdue}</div>
                            <div><b>Default Probability:</b> {(caseDetails.default_probability * 100).toFixed(1)}%</div>
                            <div><b>Priority:</b> {caseDetails.priority}</div>
                            <div><b>Recommended Action:</b> {caseDetails.recommended_action}</div>
                            <div><b>Assigned Team:</b> {caseDetails.assigned_team}</div>
                            <div><b>Status:</b> {caseDetails.status}</div>
                            {caseDetails.notes && <div><b>Notes:</b> {caseDetails.notes}</div>}
                          </div>
                        ) : (
                          <div className="py-8 text-center text-neutral-400">No details found.</div>
                        )}
                        <DialogClose asChild>
                          <button className="mt-6 w-full py-2 rounded bg-blue-600 text-white font-semibold hover:bg-blue-700 transition">Close</button>
                        </DialogClose>
                      </DialogContent>
                    </Dialog>
              </ul>
            )}
          </div>
        </div>

        {/* Summary sidebar */}
        {data && (
          <div className="space-y-4">
            <div className="bg-white dark:bg-neutral-900 rounded-xl border border-neutral-200 dark:border-neutral-800 p-5">
              <h3 className="text-sm font-bold text-neutral-900 dark:text-white mb-4">Action Summary</h3>
              <div className="space-y-4">
                {[
                  { label: "Cases Resolved", value: data.summary.completed, pct: data.summary.completedRatio, color: "bg-emerald-500" },
                  { label: "Automated (Low Risk)", value: data.summary.automated, pct: data.summary.automatedRatio, color: "bg-blue-500" },
                  { label: "Legal Escalations", value: data.summary.legalPending, pct: data.summary.legalRatio, color: "bg-rose-500" },
                ].map(({ label, value, pct, color }) => (
                  <div key={label}>
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-neutral-500">{label}</span>
                      <span className="font-bold text-neutral-800 dark:text-neutral-200">{value}</span>
                    </div>
                    <div className="w-full bg-neutral-100 dark:bg-neutral-800 rounded-full h-1.5">
                      <div className={`${color} h-1.5 rounded-full transition-all duration-500`} style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-gradient-to-br from-blue-600 to-indigo-700 rounded-xl p-5 text-white">
              <h3 className="text-sm font-bold mb-2">Automated Notifications</h3>
              <p className="text-xs text-blue-100 leading-relaxed mb-4">
                {data.summary.automated} low-risk borrower(s) handled automatically by the system, saving manual agent hours.
              </p>
              <div className="text-2xl font-black">{data.summary.automated}</div>
              <div className="text-xs text-blue-200 mt-0.5">Cases auto-managed today</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
