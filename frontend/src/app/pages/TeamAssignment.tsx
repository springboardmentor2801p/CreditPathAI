import { useState, useEffect, useCallback } from "react";
import { Users, Phone, ShieldCheck, Cpu, Scale, RefreshCw, X, ArrowRightLeft, ChevronRight, AlertTriangle } from "lucide-react";
import { apiFetch } from "../lib/api";

const TEAMS = [
  {
    name: "Automated System",
    iconType: "Cpu",
    description: "Handles low-risk borrowers (PD < 5%). Sends automated SMS/email reminders — no human intervention needed.",
    capacity: 10000,
    color: "text-indigo-600 dark:text-indigo-400",
    bg: "bg-indigo-100 dark:bg-indigo-900/30",
    ring: "ring-indigo-200 dark:ring-indigo-800",
    badge: "bg-indigo-50 text-indigo-700 border-indigo-200 dark:bg-indigo-900/20 dark:text-indigo-300",
  },
  {
    name: "Call Center",
    iconType: "Phone",
    description: "Handles medium-risk borrowers (PD 5–20%). Proactive calling, EMI restructure negotiation.",
    capacity: 200,
    color: "text-blue-600 dark:text-blue-400",
    bg: "bg-blue-100 dark:bg-blue-900/30",
    ring: "ring-blue-200 dark:ring-blue-800",
    badge: "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-900/20 dark:text-blue-300",
  },
  {
    name: "Dedicated Field Officers",
    iconType: "Users",
    description: "Handles high-risk borrowers (PD 20–50%). Field visits, asset verification, repayment negotiation.",
    capacity: 50,
    color: "text-emerald-600 dark:text-emerald-400",
    bg: "bg-emerald-100 dark:bg-emerald-900/30",
    ring: "ring-emerald-200 dark:ring-emerald-800",
    badge: "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-900/20 dark:text-emerald-300",
  },
  {
    name: "Legal Team",
    iconType: "Scale",
    description: "Handles critical borrowers (PD > 50%). Legal notices, court proceedings, asset seizure coordination.",
    capacity: 20,
    color: "text-rose-600 dark:text-rose-400",
    bg: "bg-rose-100 dark:bg-rose-900/30",
    ring: "ring-rose-200 dark:ring-rose-800",
    badge: "bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-900/20 dark:text-rose-300",
  },
];

const iconMap: Record<string, any> = { Cpu, Phone, Users, Scale, ShieldCheck };

// ── Reassign Modal ──────────────────────────────────────────────────────────────
function ReassignModal({ caseItem, onClose, onDone }: { caseItem: any; onClose: () => void; onDone: () => void }) {
  const [newTeam, setNewTeam]       = useState(caseItem.assigned_team);
  const [newPriority, setNewPriority] = useState(caseItem.priority);
  const [saving, setSaving]         = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      await apiFetch(`/api/cases/${caseItem.id}/reassign`, {
        method: "PATCH",
        body: JSON.stringify({ team: newTeam, priority: newPriority }),
      });
      onDone();
    } catch (e) { console.error(e); }
    finally { setSaving(false); }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="bg-white dark:bg-neutral-900 rounded-2xl shadow-2xl w-full max-w-md border border-neutral-200 dark:border-neutral-800">
        <div className="flex items-center justify-between p-6 border-b border-neutral-100 dark:border-neutral-800">
          <h3 className="font-bold text-neutral-900 dark:text-white flex items-center gap-2">
            <ArrowRightLeft className="h-5 w-5 text-blue-500" /> Reassign Case
          </h3>
          <button onClick={onClose} className="p-1.5 rounded-full hover:bg-neutral-100 dark:hover:bg-neutral-800 transition"><X className="h-4 w-4" /></button>
        </div>
        <div className="p-6 space-y-4">
          <div>
            <div className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-1">Borrower</div>
            <div className="font-bold text-neutral-900 dark:text-white">{caseItem.borrower_name}</div>
            <div className="text-xs text-neutral-400">₹{Number(caseItem.outstanding).toLocaleString("en-IN")} outstanding · PD {((caseItem.default_probability || 0) * 100).toFixed(1)}%</div>
          </div>
          <div>
            <label className="block text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-1">Assign to Team</label>
            <select
              value={newTeam}
              onChange={e => setNewTeam(e.target.value)}
              className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 text-sm text-neutral-900 dark:text-white px-3 py-2"
            >
              {TEAMS.map(t => <option key={t.name} value={t.name}>{t.name}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-1">Priority</label>
            <select
              value={newPriority}
              onChange={e => setNewPriority(e.target.value)}
              className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 text-sm text-neutral-900 dark:text-white px-3 py-2"
            >
              {["Low", "Medium", "High", "Critical"].map(p => <option key={p}>{p}</option>)}
            </select>
          </div>
        </div>
        <div className="flex gap-3 px-6 pb-6">
          <button onClick={onClose} className="flex-1 py-2.5 rounded-lg border border-neutral-200 dark:border-neutral-700 text-sm font-semibold text-neutral-700 dark:text-neutral-300 hover:bg-neutral-50 dark:hover:bg-neutral-800 transition">
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex-1 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-sm font-bold transition disabled:opacity-50"
          >
            {saving ? "Saving…" : "Confirm Reassignment"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Cases Drawer ────────────────────────────────────────────────────────────────
function CasesDrawer({ teamName, onClose, onReassign }: { teamName: string; onClose: () => void; onReassign: (c: any) => void }) {
  const [cases, setCases]   = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiFetch(`/api/cases?team=${encodeURIComponent(teamName)}`)
      .then(r => r.json())
      .then(d => setCases(d.cases || []))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [teamName]);

  const priorityCls = (p: string) =>
    p === "Critical" ? "bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300" :
    p === "High"     ? "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300" :
    p === "Medium"   ? "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300" :
                       "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300";

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-md bg-white dark:bg-neutral-900 shadow-2xl flex flex-col h-full border-l border-neutral-200 dark:border-neutral-800">
        <div className="flex items-center justify-between p-5 border-b border-neutral-100 dark:border-neutral-800 flex-shrink-0">
          <div>
            <h3 className="font-bold text-neutral-900 dark:text-white">{teamName}</h3>
            <p className="text-xs text-neutral-500">{cases.length} case(s) assigned</p>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-full hover:bg-neutral-100 dark:hover:bg-neutral-800"><X className="h-4 w-4" /></button>
        </div>
        <div className="overflow-y-auto flex-1 divide-y divide-neutral-100 dark:divide-neutral-800">
          {loading ? (
            <div className="p-8 text-center text-neutral-400 animate-pulse text-sm">Loading cases…</div>
          ) : cases.length === 0 ? (
            <div className="p-8 text-center">
              <AlertTriangle className="h-8 w-8 text-neutral-300 mx-auto mb-2" />
              <p className="text-neutral-400 text-sm">No active cases assigned to this team.</p>
            </div>
          ) : cases.map(c => (
            <div key={c.id} className="p-4 hover:bg-neutral-50 dark:hover:bg-neutral-800/40 transition-colors">
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <div className="font-semibold text-sm text-neutral-900 dark:text-white truncate">{c.borrower_name}</div>
                  <div className="text-xs text-neutral-500 mt-0.5">
                    ₹{Number(c.outstanding).toLocaleString("en-IN")} · PD {((c.default_probability || 0) * 100).toFixed(1)}%
                    {c.days_overdue > 0 && <span className="text-rose-500 ml-1">· {c.days_overdue}d overdue</span>}
                  </div>
                  <span className={`inline-block mt-1.5 px-2 py-0.5 rounded-full text-xs font-bold ${priorityCls(c.priority)}`}>
                    {c.priority}
                  </span>
                </div>
                <button
                  onClick={() => onReassign(c)}
                  className="flex-shrink-0 flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-neutral-200 dark:border-neutral-700 text-xs font-semibold text-neutral-600 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition"
                >
                  <ArrowRightLeft className="h-3 w-3" /> Reassign
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Main Component ──────────────────────────────────────────────────────────────
export function TeamAssignment() {
  const [teamCounts, setTeamCounts]   = useState<Record<string, number>>({});
  const [recentLog, setRecentLog]     = useState<any[]>([]);
  const [loading, setLoading]         = useState(true);
  const [drawer, setDrawer]           = useState<string | null>(null);     // team name shown in drawer
  const [reassignCase, setReassignCase] = useState<any>(null);             // case being reassigned

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      // Team counts from teams endpoint
      const tr = await apiFetch(`/api/teams`);
      const td = await tr.json();
      const counts: Record<string, number> = {};
      (td.teams || []).forEach((t: any) => { counts[t.name] = t.cases; });
      setTeamCounts(counts);
      setRecentLog(td.reassignments || []);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleReassignDone = async () => {
    setReassignCase(null);
    setDrawer(null);
    await fetchData();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-neutral-900 dark:text-white">Team Capacity Management</h2>
          <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
            Real-time case load per recovery team. Click "View Cases" to inspect or reassign.
          </p>
        </div>
        <button onClick={fetchData} className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold rounded-lg transition">
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {/* Team Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {TEAMS.map((team) => {
          const IconComp = iconMap[team.iconType] || Users;
          const caseCount = teamCounts[team.name] ?? 0;
          const pct = Math.min((caseCount / team.capacity) * 100, 100);
          const barColor = pct > 90 ? "bg-rose-500" : pct > 70 ? "bg-amber-500" : "bg-emerald-500";
          const isNearFull  = pct > 70;

          return (
            <div key={team.name} className={`bg-white dark:bg-neutral-900 rounded-xl border border-neutral-200 dark:border-neutral-800 shadow-sm flex flex-col ring-1 ${team.ring} transition-shadow hover:shadow-md`}>
              <div className="p-5 flex-1">
                <div className="flex items-start justify-between mb-3">
                  <div className={`p-2.5 rounded-lg ${team.bg}`}>
                    <IconComp className={`h-5 w-5 ${team.color}`} />
                  </div>
                  {isNearFull && (
                    <span className="text-xs font-bold text-amber-600 dark:text-amber-400 flex items-center gap-1">
                      <AlertTriangle className="h-3 w-3" /> {pct > 90 ? "At Capacity" : "High Load"}
                    </span>
                  )}
                </div>

                <h3 className="text-sm font-bold text-neutral-900 dark:text-white leading-tight mb-1">{team.name}</h3>
                <p className="text-xs text-neutral-400 dark:text-neutral-500 leading-relaxed mb-4">{team.description}</p>

                <div>
                  <div className="flex justify-between items-baseline mb-1.5">
                    <span className={`text-3xl font-black ${team.color}`}>{loading ? "—" : caseCount}</span>
                    <span className="text-xs text-neutral-400">of {team.capacity.toLocaleString()}</span>
                  </div>
                  <div className="w-full bg-neutral-100 dark:bg-neutral-800 rounded-full h-2 mb-1">
                    <div
                      className={`${barColor} h-2 rounded-full transition-all duration-700`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <div className="text-xs text-neutral-400 text-right">{pct.toFixed(0)}% capacity</div>
                </div>
              </div>

              <div className="px-5 py-3 border-t border-neutral-100 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-800/30 rounded-b-xl">
                <button
                  onClick={() => setDrawer(team.name)}
                  className={`text-sm font-semibold flex items-center gap-1 ${team.color} hover:underline`}
                >
                  View assigned cases <ChevronRight className="h-4 w-4" />
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Recent Activity */}
      <div className="bg-white dark:bg-neutral-900 shadow-sm rounded-xl border border-neutral-200 dark:border-neutral-800 p-6">
        <h3 className="text-base font-bold text-neutral-900 dark:text-white mb-4">Recent Activity</h3>
        {recentLog.length === 0 ? (
          <p className="text-sm text-neutral-400">No recent activity found.</p>
        ) : (
          <ul className="divide-y divide-neutral-100 dark:divide-neutral-800">
            {recentLog.map((item, idx) => (
              <li key={item.id || idx} className="py-3.5 flex items-start gap-4">
                <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <ArrowRightLeft className="h-4 w-4 text-blue-500" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-neutral-900 dark:text-white truncate">{item.title}</p>
                  <p className="text-xs text-neutral-500 truncate mt-0.5">{item.desc}</p>
                </div>
                <span className="text-xs text-neutral-400 flex-shrink-0">{item.time}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Cases Drawer */}
      {drawer && (
        <CasesDrawer
          teamName={drawer}
          onClose={() => setDrawer(null)}
          onReassign={(c) => setReassignCase(c)}
        />
      )}

      {/* Reassign Modal */}
      {reassignCase && (
        <ReassignModal
          caseItem={reassignCase}
          onClose={() => setReassignCase(null)}
          onDone={handleReassignDone}
        />
      )}
    </div>
  );
}
