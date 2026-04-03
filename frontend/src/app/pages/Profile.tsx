import { useState, useEffect } from "react";
import { useNavigate } from "react-router";
import {
  UserCircle, Mail, Briefcase, ShieldCheck, Save, Plus, Trash2,
  TrendingUp, CheckCircle2, AlertCircle, Edit2, X
} from "lucide-react";
import { apiFetch } from "../lib/api";

export function Profile() {
  const navigate = useNavigate();
  const isBorrower = localStorage.getItem("user_role") === "borrower";

  const [profile, setProfile] = useState<any>(null);
  const [isOnboarding, setIsOnboarding] = useState(false);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState("");
  const [form, setForm] = useState<any>({});
  const [showLoanForm, setShowLoanForm] = useState(false);
  const [loanForm, setLoanForm] = useState({
    loan_name: "", loan_amount: "", outstanding: "", rate: "", term_months: "", status: "current", status_type: "good"
  });

  const userId = localStorage.getItem("user_id");

  useEffect(() => {
    if (!userId) { navigate("/login"); return; }
    if (isBorrower) {
      apiFetch(`/api/borrower-profile`)
        .then(r => r.json())
        .then(d => {
          setProfile(d);
          setForm(d);
          // New user: no credit score set means onboarding needed
          const needsOnboarding = !d.credit_score || d.credit_score === 700 && !d.annual_income;
          setIsOnboarding(needsOnboarding);
          setEditing(needsOnboarding); // start in edit mode for new users
        })
        .catch(console.error);
    }
  }, []);

  const requiredFields = [
    'full_name', 'credit_score', 'annual_income', 'employment_status'
  ];
  const missingFields = requiredFields.filter(f => !form[f] && form[f] !== 0);

  const handleSave = async () => {
    setSaving(true);
    setSaveMsg("");
    if (missingFields.length > 0) {
      setSaveMsg("⚠ Please fill all required fields before saving.");
      setSaving(false);
      return;
    }
    try {
      const body = {
        full_name: form.full_name,
        credit_score: Number(form.credit_score),
        annual_income: Number(form.annual_income),
        employment_status: form.employment_status,
      };
      const res = await apiFetch(`/api/borrower-profile`, {
        method: "PUT",
        body: JSON.stringify(body)
      });
      if (!res.ok) throw new Error("Save failed");
      const updated = await apiFetch(`/api/borrower-profile`).then(r => r.json());
      setProfile(updated);
      setForm(updated);
      localStorage.setItem("user_name", updated.full_name);
      setEditing(false);
      if (isOnboarding) {
        // Onboarding complete — go to dashboard
        navigate("/borrower");
      } else {
        setSaveMsg("✓ Profile saved successfully");
      }
      setIsOnboarding(false);
    } catch (e) {
      setSaveMsg("⚠ Failed to save changes");
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteLoan = async (loanId: number) => {
    await apiFetch(`/api/borrower-loans/${loanId}`, { method: "DELETE" });
    const updated = await apiFetch(`/api/borrower-profile`).then(r => r.json());
    setProfile(updated);
  };

  const handleAddLoan = async (e: React.FormEvent) => {
    e.preventDefault();
    await apiFetch(`/api/borrower-loans`, {
      method: "POST",
      body: JSON.stringify({
        ...loanForm,
        loan_amount: Number(loanForm.loan_amount),
        outstanding: Number(loanForm.outstanding),
        rate: Number(loanForm.rate),
        term_months: Number(loanForm.term_months),
      })
    });
    const updated = await apiFetch(`/api/borrower-profile`).then(r => r.json());
    setProfile(updated);
    setShowLoanForm(false);
    setLoanForm({ loan_name: "", loan_amount: "", outstanding: "", rate: "", term_months: "", status: "current", status_type: "good" });
  };

  if (!profile) return (
    <div className="p-8 text-zinc-400 font-mono text-sm tracking-widest animate-pulse">LOADING PROFILE...</div>
  );

  const inputCls = "block w-full bg-[#09090b] border border-zinc-800 rounded-md text-zinc-100 text-sm px-3 py-2 focus:ring-1 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-colors disabled:opacity-50";
  const labelCls = "block text-[10px] font-bold tracking-widest text-zinc-500 uppercase mb-1.5";

  const dti = profile.debt_to_income ?? 0;
  const cs = profile.credit_score ?? 0;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {editing && missingFields.length > 0 && (
        <div className="bg-rose-900/20 border border-rose-500/30 text-rose-300 text-xs font-mono rounded-md px-4 py-3 mb-4">
          Please fill all required fields: {missingFields.map(f => f.replace(/_/g, ' ')).join(', ')}
        </div>
      )}

      {/* Onboarding Banner */}
      {isOnboarding && (
        <div className="relative overflow-hidden rounded-xl border border-emerald-500/30 bg-gradient-to-r from-emerald-900/30 to-[#0f172a] p-5 flex items-start gap-4">
          <div className="w-10 h-10 rounded-full bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center flex-shrink-0 mt-0.5">
            <ShieldCheck className="w-5 h-5 text-emerald-400" />
          </div>
          <div>
            <h2 className="text-sm font-black text-white uppercase tracking-wider mb-1">Welcome! Let's set up your financial profile</h2>
            <p className="text-xs text-zinc-400 font-mono leading-relaxed">
              You've just registered. Fill in your financial details below so we can generate accurate risk scores,
              smart loan recommendations, and a personalised dashboard — all based on your actual data.
            </p>
            <div className="flex gap-1.5 mt-3 flex-wrap">
              {["CIBIL Score", "Annual Income", "Employment Status", "Active Loans"].map(step => (
                <span key={step} className="text-[9px] font-black tracking-widest text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded bg-emerald-500/5">{step}</span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between mb-6 border-b border-zinc-800 pb-6">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 font-black text-xl">
            {profile.full_name?.split(" ").map((n: string) => n[0]).join("").slice(0, 2).toUpperCase()}
          </div>
          <div>
            <h1 className="text-2xl font-black tracking-tighter text-white uppercase">{profile.full_name}</h1>
            <p className="text-zinc-500 font-mono text-xs tracking-widest">{profile.email}</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {saveMsg && <span className="text-xs font-mono text-emerald-400">{saveMsg}</span>}
          {editing ? (
            <>
              {!isOnboarding && (
                <button onClick={() => { setEditing(false); setForm(profile); }} className="flex items-center gap-1.5 px-3 py-2 text-xs font-bold tracking-widest text-zinc-400 hover:text-zinc-200 border border-zinc-700 rounded-md transition-colors">
                  <X size={13} /> Cancel
                </button>
              )}
              <button onClick={handleSave} disabled={saving} className="flex items-center gap-1.5 px-3 py-2 text-xs font-black tracking-widest text-black bg-emerald-500 hover:bg-emerald-400 rounded-md disabled:opacity-50 transition-colors">
                <Save size={13} /> {saving ? "Saving..." : isOnboarding ? "Save & Go to Dashboard →" : "Save Changes"}
              </button>
            </>
          ) : (
            <button onClick={() => setEditing(true)} className="flex items-center gap-1.5 px-3 py-2 text-xs font-bold tracking-widest text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/10 rounded-md transition-colors">
              <Edit2 size={13} /> Edit Profile
            </button>
          )}
        </div>
      </div>

      {/* Score summary bar */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: "CIBIL Score", value: cs, suffix: "", color: cs >= 750 ? "emerald" : cs >= 650 ? "amber" : "rose",
            note: cs >= 750 ? "Excellent" : cs >= 650 ? "Good" : "Needs Work" },
          { label: "Debt-to-Income", value: dti.toFixed(1), suffix: "%", color: dti < 36 ? "emerald" : dti < 50 ? "amber" : "rose",
            note: dti < 36 ? "Healthy" : dti < 50 ? "Elevated" : "Critical" },
          { label: "Total Debt", value: `₹${(profile.total_debt || 0).toLocaleString("en-IN")}`, suffix: "",
            color: "cyan", note: `${(profile.loans || []).length} active loans` },
        ].map(stat => (
          <div key={stat.label} className={`bg-[#131316] border border-${stat.color}-500/20 rounded-xl p-4`}>
            <div className="text-[10px] font-bold tracking-widest text-zinc-500 uppercase mb-2">{stat.label}</div>
            <div className={`text-2xl font-black text-${stat.color}-400 font-mono`}>{stat.value}{stat.suffix}</div>
            <div className={`text-[10px] font-bold text-${stat.color}-600 mt-1`}>{stat.note}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Personal Info */}
        <div className="bg-[#131316] rounded-xl border border-zinc-800/80 p-6 shadow-lg relative overflow-hidden">
          <div className="absolute top-0 left-0 w-1 h-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]"></div>
          <h3 className="text-xs font-bold tracking-widest text-zinc-400 uppercase mb-5 flex items-center gap-2">
            <UserCircle size={14} /> Personal Details
          </h3>
          <div className="space-y-4">
            <div>
              <label className={labelCls}>Full Name</label>
              {editing ? <input className={inputCls} value={form.full_name || ""} onChange={e => setForm({...form, full_name: e.target.value})} />
                : <div className="text-sm font-mono text-zinc-200">{profile.full_name}</div>}
            </div>
            <div>
              <label className={labelCls}>Email</label>
              <div className="flex items-center gap-2 text-sm font-mono text-zinc-200">
                <Mail size={13} className="text-zinc-500" /> {profile.email}
              </div>
            </div>
            <div>
              <label className={labelCls}>Employment Status</label>
              {editing ? (
                <select className={inputCls} value={form.employment_status || "employed"} onChange={e => setForm({...form, employment_status: e.target.value})}>
                  <option value="employed">Salaried / Full-Time Employee</option>
                  <option value="self_employed">Self-Employed / Business Owner</option>
                  <option value="unemployed">Currently Unemployed</option>
                  <option value="retired">Retired</option>
                </select>
              ) : <div className="text-sm font-mono text-zinc-200 capitalize">{(profile.employment_status || "employed").replace("_", " ")}</div>}
            </div>
          </div>
        </div>

        {/* Financial Info */}
        <div className="bg-[#131316] rounded-xl border border-zinc-800/80 p-6 shadow-lg relative overflow-hidden">
          <div className="absolute top-0 left-0 w-1 h-full bg-violet-500 shadow-[0_0_10px_rgba(139,92,246,0.5)]"></div>
          <h3 className="text-xs font-bold tracking-widest text-zinc-400 uppercase mb-5 flex items-center gap-2">
            <TrendingUp size={14} /> Financial Profile
          </h3>
          <div className="space-y-4">
            <div>
              <label className={labelCls}>CIBIL / Credit Score (300–900)</label>
              {editing ? <input type="number" min={300} max={900} className={inputCls} value={form.credit_score || ""} onChange={e => setForm({...form, credit_score: e.target.value})} />
                : <div className="text-sm font-mono text-zinc-200">{profile.credit_score}</div>}
            </div>
            <div>
              <label className={labelCls}>Annual Income (₹)</label>
              {editing ? <input type="number" min={0} className={inputCls} value={form.annual_income || ""} onChange={e => setForm({...form, annual_income: e.target.value})} />
                : <div className="text-sm font-mono text-zinc-200">₹{(profile.annual_income || 0).toLocaleString("en-IN")}</div>}
            </div>

          </div>
        </div>
      </div>

      {/* Loans Manager */}
      <div className="bg-[#131316] rounded-xl border border-zinc-800/80 p-6 shadow-lg">
        <div className="flex items-center justify-between mb-5">
          <h3 className="text-xs font-bold tracking-widest text-zinc-400 uppercase flex items-center gap-2">
            <Briefcase size={14} /> Active Loans & Liabilities
          </h3>
          <button
            onClick={() => setShowLoanForm(!showLoanForm)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-black tracking-widest text-black bg-emerald-500 hover:bg-emerald-400 rounded-md transition-colors"
          >
            <Plus size={12} /> ADD LOAN
          </button>
        </div>

        {showLoanForm && (
          <form onSubmit={handleAddLoan} className="mb-6 p-4 bg-[#09090b] border border-emerald-500/20 rounded-lg">
            <h4 className="text-[10px] font-bold tracking-widest text-emerald-400 uppercase mb-4">New Loan / Liability</h4>
            <div className="grid grid-cols-2 gap-3 mb-3">
              <div className="col-span-2"><label className={labelCls}>Loan Name</label>
                <input required className={inputCls} placeholder="e.g. SBI Home Loan" value={loanForm.loan_name} onChange={e => setLoanForm({...loanForm, loan_name: e.target.value})} /></div>
              <div><label className={labelCls}>Original Amount (₹)</label>
                <input required type="number" min={0} className={inputCls} placeholder="5000000" value={loanForm.loan_amount} onChange={e => setLoanForm({...loanForm, loan_amount: e.target.value})} /></div>
              <div><label className={labelCls}>Outstanding (₹)</label>
                <input required type="number" min={0} className={inputCls} placeholder="4200000" value={loanForm.outstanding} onChange={e => setLoanForm({...loanForm, outstanding: e.target.value})} /></div>
              <div><label className={labelCls}>Interest Rate (%)</label>
                <input required type="number" step="0.01" min={0} className={inputCls} placeholder="8.5" value={loanForm.rate} onChange={e => setLoanForm({...loanForm, rate: e.target.value})} /></div>
              <div><label className={labelCls}>Term (Months)</label>
                <input required type="number" min={0} className={inputCls} placeholder="240" value={loanForm.term_months} onChange={e => setLoanForm({...loanForm, term_months: e.target.value})} /></div>
              <div><label className={labelCls}>Status</label>
                <select className={inputCls} value={loanForm.status} onChange={e => setLoanForm({...loanForm, status: e.target.value, status_type: e.target.value === "current" ? "good" : e.target.value === "high_util" ? "warning" : "bad"})}>
                  <option value="current">Current</option>
                  <option value="high_util">High Utilization</option>
                  <option value="overdue">Overdue</option>
                </select>
              </div>
            </div>
            <div className="flex gap-2">
              <button type="submit" className="px-4 py-2 text-xs font-black tracking-widest text-black bg-emerald-500 hover:bg-emerald-400 rounded-md transition-colors">SAVE LOAN</button>
              <button type="button" onClick={() => setShowLoanForm(false)} className="px-4 py-2 text-xs font-bold tracking-widest text-zinc-400 border border-zinc-700 rounded-md hover:text-zinc-200 transition-colors">Cancel</button>
            </div>
          </form>
        )}

        {(profile.loans || []).length === 0 ? (
          <div className="text-center py-8 text-zinc-500 font-mono text-xs">
            No loans added yet. Click "ADD LOAN" to track your liabilities.
          </div>
        ) : (
          <div className="space-y-3">
            {(profile.loans || []).map((loan: any) => (
              <div key={loan.id} className="flex items-center justify-between p-4 bg-[#09090b] border border-zinc-800 rounded-lg hover:border-zinc-700 transition-colors">
                <div className="flex-1">
                  <div className="text-sm font-bold text-zinc-200">{loan.loan_name}</div>
                  <div className="text-[10px] font-mono text-zinc-500 mt-0.5">
                    Rate: {loan.rate}% • Term: {loan.term_months}M • Original: ₹{loan.loan_amount.toLocaleString("en-IN")}
                  </div>
                </div>
                <div className="text-right mr-4">
                  <div className="text-sm font-mono font-bold text-white">₹{loan.outstanding.toLocaleString("en-IN")}</div>
                  <div className={`text-[10px] font-bold uppercase mt-0.5 ${
                    loan.status_type === "good" ? "text-emerald-400" :
                    loan.status_type === "warning" ? "text-amber-400" : "text-rose-400"
                  }`}>{loan.status.replace("_", " ")}</div>
                </div>
                <button onClick={() => handleDeleteLoan(loan.id)} className="text-zinc-600 hover:text-rose-400 transition-colors">
                  <Trash2 size={15} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Account status */}
      <div className="bg-[#131316] rounded-xl border border-zinc-800/80 p-5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <ShieldCheck className="w-5 h-5 text-emerald-400" />
          <div>
            <div className="text-xs font-bold text-zinc-200 uppercase tracking-widest">Account Verified</div>
            <div className="text-[10px] font-mono text-zinc-500">{profile.email} • Role: {profile.role}</div>
          </div>
        </div>
        {dti < 36 && cs >= 700
          ? <div className="flex items-center gap-1.5 text-emerald-400 text-xs font-bold"><CheckCircle2 size={14} /> Good Standing</div>
          : <div className="flex items-center gap-1.5 text-amber-400 text-xs font-bold"><AlertCircle size={14} /> Needs Attention</div>
        }
      </div>
    </div>
  );
}
