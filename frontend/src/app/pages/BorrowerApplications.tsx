import { useState, useEffect } from "react";
import { apiFetch } from "../lib/api";
import { FileText, CheckCircle, Clock, XCircle, Activity, ExternalLink, RefreshCw } from "lucide-react";
import { Link } from "react-router";

const STATUS_META: Record<string, { label: string; icon: any; color: string; bg: string; border: string }> = {
  active: {
    label: "ACTIVE", icon: CheckCircle,
    color: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/30"
  },
  approved: {
    label: "APPROVED", icon: CheckCircle,
    color: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/30"
  },
  pending: {
    label: "PROCESSING", icon: Clock,
    color: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/30"
  },
  conditional: {
    label: "CONDITIONAL", icon: Activity,
    color: "text-violet-400", bg: "bg-violet-500/10", border: "border-violet-500/30"
  },
  rejected: {
    label: "DENIED", icon: XCircle,
    color: "text-rose-400", bg: "bg-rose-500/10", border: "border-rose-500/30"
  }
};

export function BorrowerApplications() {
  const [apps, setApps] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await apiFetch("/api/borrower-applications");
      const data = await res.json();
      setApps(data.applications || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const updateStatus = async (id: number, newStatus: string) => {
    try {
      const res = await apiFetch(`/api/borrower-applications/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ status: newStatus })
      });
      if (res.ok) {
        setApps(apps.map(app => app.id === id ? { ...app, status: newStatus } : app));
      }
    } catch (e) {
      console.error("Failed to update status", e);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (loading) return <div className="p-8 text-zinc-400 font-mono text-xs tracking-widest animate-pulse">FETCHING_APPLICATIONS...</div>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-end border-b border-zinc-800 pb-6">
        <div>
          <h1 className="text-3xl font-black tracking-tighter text-white uppercase mb-1">Application Registry</h1>
          <p className="text-zinc-500 font-mono text-xs tracking-widest uppercase">System Tracked Loan Submissions</p>
        </div>
        <div className="flex gap-3">
            <button 
              onClick={fetchData} 
              className="p-3 bg-zinc-900 border border-zinc-700 hover:border-zinc-500 rounded-md transition-colors"
              title="Refresh Registry"
            >
              <RefreshCw className="h-4 w-4 text-zinc-400" />
            </button>
            <Link 
              to="/borrower/evaluator"
              className="px-5 py-3 bg-fuchsia-600 hover:bg-fuchsia-500 text-black text-xs font-black tracking-widest rounded-md transition-colors shadow-[0_0_15px_rgba(192,38,211,0.3)] shadow-fuchsia-500/20 flex items-center gap-2"
            >
              <FileText className="h-4 w-4" />
              NEW_APPLICATION
            </Link>
        </div>
      </div>

      {apps.length === 0 ? (
        <div className="flex flex-col items-center justify-center p-12 bg-[#131316] border border-zinc-800 rounded-xl">
          <FileText className="h-12 w-12 text-zinc-700 mb-4" />
          <p className="text-zinc-500 text-sm font-mono tracking-widest uppercase mb-6">No applications found in system registry.</p>
          <Link to="/borrower/evaluator" className="text-emerald-400 text-xs font-bold tracking-widest uppercase hover:text-emerald-300">
            Initialize New Submission →
          </Link>
        </div>
      ) : (
        <div className="bg-[#131316] rounded-xl border border-zinc-800/80 shadow-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-zinc-900/50 border-b border-zinc-800">
                  <th className="px-6 py-4 text-[10px] font-black tracking-widest text-zinc-500 uppercase">Registry ID</th>
                  <th className="px-6 py-4 text-[10px] font-black tracking-widest text-zinc-500 uppercase">Initialization Date</th>
                  <th className="px-6 py-4 text-[10px] font-black tracking-widest text-zinc-500 uppercase">Loan Purpose</th>
                  <th className="px-6 py-4 text-[10px] font-black tracking-widest text-zinc-500 uppercase">Principal Volume</th>
                  <th className="px-6 py-4 text-[10px] font-black tracking-widest text-zinc-500 uppercase">Node Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/50">
                {apps.map(app => {
                  const s = STATUS_META[app.status.toLowerCase()] || STATUS_META.pending;
                  const Icon = s.icon;
                  return (
                    <tr key={app.id} className="hover:bg-zinc-900/30 transition-colors group">
                      <td className="px-6 py-4">
                        <div className="text-sm font-mono text-zinc-300 flex items-center gap-2">
                          #{app.id.toString().padStart(4, '0')}
                          <ExternalLink className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity text-zinc-600" />
                        </div>
                      </td>
                      <td className="px-6 py-4 text-zinc-400 font-mono text-xs">
                        {app.created_at.replace("T", " ").slice(0, 16)}
                      </td>
                      <td className="px-6 py-4">
                        <div className="text-sm font-bold text-zinc-200 capitalize">{app.loan_purpose.replace(/_/g, " ")}</div>
                        <div className="text-xs text-zinc-600 font-mono mt-1">{app.term_months} Months Term</div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="text-sm font-black text-emerald-400 tracking-tighter">
                          ₹{app.loan_amount.toLocaleString("en-IN")}
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <select
                          value={app.status.toLowerCase()}
                          onChange={(e) => updateStatus(app.id, e.target.value)}
                          className={`inline-flex items-center gap-1.5 pl-2 pr-6 py-1 rounded text-[10px] font-black uppercase tracking-wider border outline-none cursor-pointer appearance-none ${s.bg} ${s.color} ${s.border}`}
                          style={{ backgroundImage: 'linear-gradient(45deg, transparent 50%, currentColor 50%), linear-gradient(135deg, currentColor 50%, transparent 50%)', backgroundPosition: 'calc(100% - 10px) calc(1em + 2px), calc(100% - 5px) calc(1em + 2px)', backgroundSize: '5px 5px, 5px 5px', backgroundRepeat: 'no-repeat' }}
                        >
                          <option value="pending" className="bg-[#09090b] text-amber-400">PROCESSING</option>
                          <option value="conditional" className="bg-[#09090b] text-violet-400">CONDITIONAL</option>
                          <option value="active" className="bg-[#09090b] text-emerald-400">ACTIVE</option>
                          <option value="approved" className="bg-[#09090b] text-emerald-400">APPROVED</option>
                          <option value="rejected" className="bg-[#09090b] text-rose-400">DENIED</option>
                        </select>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
