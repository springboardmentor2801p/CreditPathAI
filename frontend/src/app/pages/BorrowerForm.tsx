import { useState } from "react";
import { useForm } from "react-hook-form";
import { AlertCircle, TrendingDown, Activity, ShieldCheck, Users, Phone, Cpu, Scale } from "lucide-react";
import { apiFetch } from "../lib/api";

// ─── Types ────────────────────────────────────────────────────────────────────
type NewLoanData = {
  fullName: string;
  income: number;
  creditScore: number;
  loanAmount: number;
  propertyValue: number;
  term: number;
  employmentStatus: string;
  loanType: string;
  age: number;
};

type ExistingLoanData = {
  fullName: string;
  creditScore: number;
  loanAmount: number;        // original loan amount
  outstandingBalance: number; // current outstanding (EAD)
  propertyValue: number;
  daysOverdue: number;
  employmentStatus: string;
  monthlyIncome: number;
};

// ─── Priority colour helper ────────────────────────────────────────────────────
const priorityClass = (p: string) => {
  if (p === "Critical") return "bg-rose-100 text-rose-700 border-rose-300 dark:bg-rose-900/40 dark:text-rose-300 dark:border-rose-700";
  if (p === "High")     return "bg-orange-100 text-orange-700 border-orange-300 dark:bg-orange-900/40 dark:text-orange-300 dark:border-orange-700";
  if (p === "Medium")   return "bg-amber-100 text-amber-700 border-amber-300 dark:bg-amber-900/40 dark:text-amber-300 dark:border-amber-700";
  return "bg-emerald-100 text-emerald-700 border-emerald-300 dark:bg-emerald-900/40 dark:text-emerald-300 dark:border-emerald-700";
};

const teamIcon = (team: string) => {
  if (team.includes("Legal"))  return <Scale className="h-5 w-5 text-rose-400" />;
  if (team.includes("Field"))  return <Users className="h-5 w-5 text-orange-400" />;
  if (team.includes("Call"))   return <Phone className="h-5 w-5 text-amber-400" />;
  return <Cpu className="h-5 w-5 text-emerald-400" />;
};

const inputCls = "block w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 text-neutral-900 dark:text-white px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent transition";
const labelCls = "block text-xs font-semibold text-neutral-600 dark:text-neutral-400 uppercase tracking-wide mb-1";

// ─── Component ────────────────────────────────────────────────────────────────
export function BorrowerForm() {
  const [mode, setMode] = useState<"new" | "existing">("new");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving]   = useState(false);
  const [saved, setSaved]     = useState<string | null>(null); // account_id after save
  const [result, setResult]   = useState<any>(null);

  const newForm     = useForm<NewLoanData>({ defaultValues: { loanType: "secured", age: 35 } });
  const existForm   = useForm<ExistingLoanData>();
  const newErrors   = newForm.formState.errors;
  const existErrors = existForm.formState.errors;

  // ── New Loan Submit ──────────────────────────────────────────────────────────
  const onSubmitNew = async (data: NewLoanData) => {
    setLoading(true);
    setResult(null);
    try {
      const res = await apiFetch("/risk-score", {
        method: "POST",
        body: JSON.stringify({
          fullName: data.fullName,
          income: Number(data.income),
          credit_score: Number(data.creditScore),
          loan_amount: Number(data.loanAmount),
          property_value: Number(data.propertyValue),
          age: Number(data.age),
          term: Number(data.term),
          days_overdue: 0,
          loan_type: data.loanType,
          employment_status: data.employmentStatus,
        }),
      });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const r = await res.json();
      setResult({
        mode: "new",
        defaultProb:      (r.default_probability * 100).toFixed(1),
        lgd:              ((r.lgd ?? 0) * 100).toFixed(0),
        expLoss:          Number(r.expected_loss ?? 0),
        recoverableValue: Number(r.recoverable_value ?? 0),
        efficiencyScore:  r.recovery_efficiency_score ?? "N/A",
        priority:         r.priority_level || "Medium",
        team:             r.team_assignment || "Under Review",
        ltv:              (r.ltv_pct ?? ((Number(data.loanAmount) / Math.max(Number(data.propertyValue), 1)) * 100)).toFixed(1),
        loanDecision:     r.loan_decision || "CONDITIONAL",
        decisionLabel:    r.decision_label || "Conditional Approval",
        decisionColor:    r.decision_color || "amber",
        decisionReason:   r.decision_reason || "",
        urgency:          r.urgency_level || "Normal",
        action:           r.recommended_action || "Standard Review",
        // raw snapshot for save-to-portfolio — status derived from lending decision
        _save: {
          borrower_name:       data.fullName,
          loan_amount:         Number(data.loanAmount),
          outstanding:         Number(data.loanAmount),
          credit_score:        Number(data.creditScore),
          days_overdue:        0,
          default_probability: r.default_probability,
          priority:            r.priority_level || "Medium",
          recommended_action:  r.recommended_action || "Standard Review",
          assigned_team:       r.team_assignment || "Under Review",
          loan_decision:       r.loan_decision || "CONDITIONAL",
          // APPROVE → active (disbursement approved)
          // CONDITIONAL → conditional (pending conditions being met)
          // REJECT → rejected (declined, archived for record)
          status: r.loan_decision === "APPROVE"
            ? "active"
            : r.loan_decision === "REJECT"
            ? "rejected"
            : "conditional",
        },
      });
      setSaved(null);
    } catch (e: any) {
      alert(`Evaluation failed: ${e.message}`);
    } finally { setLoading(false); }
  };

  // ── Existing Loan Submit ─────────────────────────────────────────────────────
  const onSubmitExist = async (data: ExistingLoanData) => {
    setLoading(true);
    setResult(null);
    try {
      const res = await apiFetch("/risk-score", {
        method: "POST",
        body: JSON.stringify({
          fullName: data.fullName,
          income: Number(data.monthlyIncome) * 12,  // annualise monthly income
          credit_score: Number(data.creditScore),
          loan_amount: Number(data.outstandingBalance),   // EAD = outstanding balance
          property_value: Number(data.propertyValue),
          age: "35-44",
          term: 120,
          days_overdue: Number(data.daysOverdue),
        }),
      });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const r = await res.json();

      // Employment context enriches the action description shown to the lender
      const empNote: Record<string, string> = {
        employed:      "Borrower is currently employed — repayment plan or EMI restructuring likely feasible.",
        self_employed: "Self-employed borrower — income may be irregular; verify business cashflow.",
        unemployed:    "Borrower is unemployed — recovery via collateral or guarantor recommended.",
        retired:       "Retired borrower — assess pension income and fixed assets before escalation.",
      };
      const employmentContext = empNote[data.employmentStatus] || "";

      setResult({
        mode: "existing",
        defaultProb:      (r.default_probability * 100).toFixed(1),
        expLoss:          Number(r.expected_loss ?? 0),
        recoverableValue: Number(r.recoverable_value ?? 0),
        efficiencyScore:  r.recovery_efficiency_score ?? "N/A",
        priority:         r.priority_level || "High",
        team:             r.team_assignment || "Call Center",
        urgency:          r.urgency_level || "Normal",
        action:           r.recommended_action || "Assign recovery agent",
        ltv:              (r.ltv_pct ?? 0).toFixed(1),
        daysOverdue:      data.daysOverdue,
        outstanding:      Number(data.outstandingBalance),
        employmentStatus: data.employmentStatus,
        employmentContext,
        // raw snapshot for save-to-portfolio
        _save: {
          borrower_name:       data.fullName,
          loan_amount:         Number(data.loanAmount),
          outstanding:         Number(data.outstandingBalance),
          credit_score:        Number(data.creditScore),
          days_overdue:        Number(data.daysOverdue),
          default_probability: r.default_probability,
          priority:            r.priority_level || "High",
          recommended_action:  r.recommended_action || "Assign recovery agent",
          assigned_team:       r.team_assignment || "Call Center",
          loan_decision:       "EXISTING_LOAN",
        },
      });
      setSaved(null);
    } catch (e: any) {
      alert(`Evaluation failed: ${e.message}`);
    } finally { setLoading(false); }
  };

  // ── Save to Portfolio ────────────────────────────────────────────────────────
  const saveToPortfolio = async () => {
    if (!result?._save) return;
    setSaving(true);
    try {
      const payload = { ...result._save };
      const res = await apiFetch("/api/cases", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`Save failed: ${res.status}`);
      const d = await res.json();
      setSaved(d.account_id);
    } catch (e: any) {
      alert(`Could not save: ${e.message}`);
    } finally { setSaving(false); }
  };

  // ──────────────────────────────────────────────────────────────────────────────
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

      {/* ── LEFT: Form panel ── */}
      <div className="bg-white dark:bg-neutral-900 shadow-sm rounded-xl border border-neutral-200 dark:border-neutral-800 p-6 flex flex-col gap-6">

        {/* Mode toggle */}
        <div>
          <h2 className="text-xl font-bold text-neutral-900 dark:text-white mb-4">Institution Risk Evaluation Matrix</h2>
          <div className="flex rounded-lg overflow-hidden border border-neutral-200 dark:border-neutral-700 text-sm font-semibold">
            <button
              type="button"
              onClick={() => { setMode("new"); setResult(null); }}
              className={`flex-1 py-2.5 transition-colors ${mode === "new"
                ? "bg-blue-600 text-white"
                : "bg-white dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300 hover:bg-neutral-50 dark:hover:bg-neutral-700"}`}
            >
              📋 New Loan Application
            </button>
            <button
              type="button"
              onClick={() => { setMode("existing"); setResult(null); }}
              className={`flex-1 py-2.5 transition-colors ${mode === "existing"
                ? "bg-orange-600 text-white"
                : "bg-white dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300 hover:bg-neutral-50 dark:hover:bg-neutral-700"}`}
            >
              ⚠️ Existing Loan / NPA
            </button>
          </div>
          <p className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
            {mode === "new"
              ? "Evaluate a new loan application — get an APPROVE / CONDITIONAL / REJECT lending decision."
              : "Assess an existing disbursed loan in distress — get a priority level and recovery action plan."}
          </p>
        </div>

        {/* ── NEW LOAN FORM ── */}
        {mode === "new" && (
          <form onSubmit={newForm.handleSubmit(onSubmitNew)} className="space-y-4">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label className={labelCls}>Borrower Full Name</label>
                <input type="text" {...newForm.register("fullName", { required: "Required" })} className={inputCls} placeholder="e.g. Rahul Verma" />
                {newErrors.fullName && <p className="mt-1 text-xs text-red-500">{newErrors.fullName.message}</p>}
              </div>

              <div>
                <label className={labelCls}>Annual Income (₹)</label>
                <input type="number" {...newForm.register("income", { required: "Required", min: { value: 50000, message: "Min ₹50,000" } })} className={inputCls} placeholder="1200000" />
                {newErrors.income && <p className="mt-1 text-xs text-red-500">{newErrors.income.message}</p>}
              </div>

              <div>
                <label className={labelCls}>CIBIL / Credit Score</label>
                <input type="number" {...newForm.register("creditScore", { required: "Required", min: { value: 300, message: "Min 300" }, max: { value: 900, message: "Max 900" } })} className={inputCls} placeholder="300–900" />
                {newErrors.creditScore && <p className="mt-1 text-xs text-red-500">{newErrors.creditScore.message}</p>}
              </div>

              <div>
                <label className={labelCls}>Requested Loan Amount (₹)</label>
                <input type="number" {...newForm.register("loanAmount", { required: "Required", min: { value: 10000, message: "Min ₹10,000" } })} className={inputCls} placeholder="5000000" />
                {newErrors.loanAmount && <p className="mt-1 text-xs text-red-500">{newErrors.loanAmount.message}</p>}
              </div>

              <div>
                <label className={labelCls}>Property / Asset Value (₹)</label>
                <input type="number" {...newForm.register("propertyValue", { required: "Required", min: { value: 0, message: "Min 0" } })} className={inputCls} placeholder="7500000" />
                <p className="text-xs text-neutral-400 mt-1">Enter 0 for unsecured loans</p>
                {newErrors.propertyValue && <p className="mt-1 text-xs text-red-500">{newErrors.propertyValue.message}</p>}
              </div>

              <div>
                <label className={labelCls}>Loan Term (Months)</label>
                <input type="number" {...newForm.register("term", { required: "Required", min: { value: 12, message: "Min 12 mo" }, max: { value: 480, message: "Max 480 mo" } })} className={inputCls} placeholder="180" />
                {newErrors.term && <p className="mt-1 text-xs text-red-500">{newErrors.term.message}</p>}
              </div>

              <div>
                <label className={labelCls}>Employment Status</label>
                <select {...newForm.register("employmentStatus")} className={inputCls}>
                  <option value="employed">Salaried / Full-Time</option>
                  <option value="self_employed">Self-Employed / Business</option>
                  <option value="retired">Retired</option>
                  <option value="unemployed">Unemployed</option>
                </select>
              </div>

              <div>
                <label className={labelCls}>Loan Type</label>
                <select {...newForm.register("loanType", { required: "Required" })} className={inputCls}>
                  <option value="secured">Secured (Home/Auto/Gold)</option>
                  <option value="education">Education Loan</option>
                  <option value="unsecured">Unsecured (Personal/Education)</option>
                  <option value="other">Other</option>
                </select>
                {newErrors.loanType && <p className="mt-1 text-xs text-red-500">{newErrors.loanType.message}</p>}
              </div>

              <div>
                <label className={labelCls}>Borrower Age</label>
                <input type="number" {...newForm.register("age", { required: "Required", min: { value: 18, message: "Min 18" }, max: { value: 100, message: "Max 100" } })} className={inputCls} placeholder="35" />
                {newErrors.age && <p className="mt-1 text-xs text-red-500">{newErrors.age.message}</p>}
              </div>
            </div>

            <button type="submit" disabled={loading}
              className="w-full py-3 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-sm font-bold tracking-wide transition disabled:opacity-50">
              {loading ? "Running Model…" : "Evaluate Loan Application"}
            </button>
          </form>
        )}

        {/* ── EXISTING LOAN FORM ── */}
        {mode === "existing" && (
          <form onSubmit={existForm.handleSubmit(onSubmitExist)} className="space-y-4">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="sm:col-span-2">
                <label className={labelCls}>Borrower Full Name</label>
                <input type="text" {...existForm.register("fullName", { required: "Required" })} className={inputCls} placeholder="e.g. Neha Singh" />
                {existErrors.fullName && <p className="mt-1 text-xs text-red-500">{existErrors.fullName.message}</p>}
              </div>

              <div>
                <label className={labelCls}>CIBIL / Credit Score</label>
                <input type="number" {...existForm.register("creditScore", { required: "Required", min: { value: 300, message: "Min 300" }, max: { value: 900, message: "Max 900" } })} className={inputCls} placeholder="300–900" />
                {existErrors.creditScore && <p className="mt-1 text-xs text-red-500">{existErrors.creditScore.message}</p>}
              </div>

              <div>
                <label className={labelCls}>Days Overdue</label>
                <input type="number" {...existForm.register("daysOverdue", { required: "Required", min: { value: 0, message: "Min 0" } })} className={inputCls} placeholder="e.g. 90" />
                <p className="text-xs text-neutral-400 mt-1">0 = current, 1–29 = normal, 30–89 = urgent, 90+ = critical</p>
                {existErrors.daysOverdue && <p className="mt-1 text-xs text-red-500">{existErrors.daysOverdue.message}</p>}
              </div>

              <div>
                <label className={labelCls}>Original Loan Amount (₹)</label>
                <input type="number" {...existForm.register("loanAmount", { required: "Required", min: { value: 1000, message: "Min ₹1,000" } })} className={inputCls} placeholder="5000000" />
                {existErrors.loanAmount && <p className="mt-1 text-xs text-red-500">{existErrors.loanAmount.message}</p>}
              </div>

              <div>
                <label className={labelCls}>Outstanding Balance (₹)</label>
                <input type="number" {...existForm.register("outstandingBalance", { required: "Required", min: { value: 1000, message: "Min ₹1,000" } })} className={inputCls} placeholder="4200000" />
                <p className="text-xs text-neutral-400 mt-1">Current unpaid principal (EAD)</p>
                {existErrors.outstandingBalance && <p className="mt-1 text-xs text-red-500">{existErrors.outstandingBalance.message}</p>}
              </div>

              <div>
                <label className={labelCls}>Property / Collateral Value (₹)</label>
                <input type="number" {...existForm.register("propertyValue", { required: "Required", min: { value: 0, message: "Min 0" } })} className={inputCls} placeholder="7500000" />
                <p className="text-xs text-neutral-400 mt-1">Enter 0 if unsecured</p>
                {existErrors.propertyValue && <p className="mt-1 text-xs text-red-500">{existErrors.propertyValue.message}</p>}
              </div>
              <div className="sm:col-span-2">
                <label className={labelCls}>Employment Status</label>
                <select {...existForm.register("employmentStatus", { required: "Required" })} className={inputCls}>
                  <option value="employed">Salaried / Full-Time Employed</option>
                  <option value="self_employed">Self-Employed / Business Owner</option>
                  <option value="unemployed">Currently Unemployed</option>
                  <option value="retired">Retired</option>
                </select>
                <p className="text-xs text-neutral-400 mt-1">Affects recovery strategy — unemployed borrowers require a different approach than employed ones</p>
                {existErrors.employmentStatus && <p className="mt-1 text-xs text-red-500">{existErrors.employmentStatus.message}</p>}
              </div>

              <div className="sm:col-span-2">
                <label className={labelCls}>Current Monthly Income (₹)</label>
                <input type="number" {...existForm.register("monthlyIncome", { required: "Required", min: { value: 0, message: "Min 0" } })} className={inputCls} placeholder="e.g. 45000" />
                <p className="text-xs text-neutral-400 mt-1">Enter current income — 0 if unemployed or income has ceased</p>
                {existErrors.monthlyIncome && <p className="mt-1 text-xs text-red-500">{existErrors.monthlyIncome.message}</p>}
              </div>
            </div>

            <button type="submit" disabled={loading}
              className="w-full py-3 rounded-lg bg-orange-600 hover:bg-orange-700 text-white text-sm font-bold tracking-wide transition disabled:opacity-50">
              {loading ? "Running Model…" : "Assess Recovery Priority"}
            </button>
          </form>
        )}
      </div>

      {/* ── RIGHT: Results panel ── */}
      <div className="bg-neutral-50 dark:bg-neutral-800/50 rounded-xl border border-neutral-200 dark:border-neutral-800 p-6 flex flex-col">
        {result ? (

          <div className="space-y-5 flex-1 animate-in fade-in slide-in-from-bottom-4 duration-500">

            {/* ── NEW LOAN RESULTS ── */}
            {result.mode === "new" && (<>

              {/* Decision banner */}
              <div className={`rounded-xl border-2 p-5 flex items-start gap-4 ${
                result.loanDecision === "APPROVE"
                  ? "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-400"
                  : result.loanDecision === "CONDITIONAL"
                  ? "bg-amber-50 dark:bg-amber-900/20 border-amber-400"
                  : "bg-rose-50 dark:bg-rose-900/20 border-rose-400"
              }`}>
                <div className="text-3xl">{result.loanDecision === "APPROVE" ? "✅" : result.loanDecision === "CONDITIONAL" ? "⚠️" : "❌"}</div>
                <div className="flex-1">
                  <div className={`text-xs font-bold uppercase tracking-widest mb-0.5 ${
                    result.loanDecision === "APPROVE" ? "text-emerald-600 dark:text-emerald-400"
                    : result.loanDecision === "CONDITIONAL" ? "text-amber-600 dark:text-amber-400"
                    : "text-rose-600 dark:text-rose-400"
                  }`}>Lending Decision</div>
                  <div className={`text-xl font-black mb-1 ${
                    result.loanDecision === "APPROVE" ? "text-emerald-800 dark:text-emerald-200"
                    : result.loanDecision === "CONDITIONAL" ? "text-amber-800 dark:text-amber-200"
                    : "text-rose-800 dark:text-rose-200"
                  }`}>{result.decisionLabel}</div>
                  <p className="text-sm text-neutral-600 dark:text-neutral-300 leading-relaxed">{result.decisionReason}</p>
                </div>
              </div>

              {/* Basel III metrics */}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-white dark:bg-neutral-900 p-4 rounded-lg border border-neutral-100 dark:border-neutral-800">
                  <div className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-1">Prob. of Default (PD)</div>
                  <div className="text-2xl font-black text-neutral-900 dark:text-white">
                    {result.defaultProb}%
                    {parseFloat(result.defaultProb) > 30 && <AlertCircle className="inline ml-2 h-4 w-4 text-rose-500" />}
                  </div>
                </div>
                <div className="bg-white dark:bg-neutral-900 p-4 rounded-lg border border-neutral-100 dark:border-neutral-800">
                  <div className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-1">LGD (Collateral-adj.)</div>
                  <div className="text-2xl font-black text-neutral-900 dark:text-white">{result.lgd}%</div>
                </div>
                <div className="bg-white dark:bg-neutral-900 p-4 rounded-lg border border-neutral-100 dark:border-neutral-800">
                  <div className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-1">Expected Credit Loss</div>
                  <div className="text-xl font-black text-neutral-900 dark:text-white">
                    ₹{result.expLoss.toLocaleString("en-IN", { maximumFractionDigits: 0 })}
                  </div>
                  <div className="text-xs text-neutral-400 mt-0.5">EL = PD × LGD × EAD</div>
                </div>
                <div className="bg-white dark:bg-neutral-900 p-4 rounded-lg border border-neutral-100 dark:border-neutral-800">
                  <div className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-1">Recoverable Value</div>
                  <div className="text-xl font-black text-emerald-600 dark:text-emerald-400">
                    ₹{result.recoverableValue.toLocaleString("en-IN", { maximumFractionDigits: 0 })}
                  </div>
                </div>
              </div>

              {/* LTV / Net ROI row */}
              <div className="grid grid-cols-3 gap-3">
                <div className="bg-white dark:bg-neutral-900 p-3 rounded-lg border text-center">
                  <div className="text-xs font-semibold text-neutral-500 uppercase mb-1">LTV</div>
                  <div className={`text-lg font-bold ${parseFloat(result.ltv) > 90 ? "text-rose-500" : parseFloat(result.ltv) > 75 ? "text-amber-500" : "text-emerald-500"}`}>
                    {result.ltv}%
                  </div>
                </div>
                <div className="bg-white dark:bg-neutral-900 p-3 rounded-lg border text-center">
                  <div className="text-xs font-semibold text-neutral-500 uppercase mb-1">Net ROI</div>
                  <div className="text-lg font-bold text-neutral-900 dark:text-white">{result.efficiencyScore}x</div>
                </div>
                <div className="bg-white dark:bg-neutral-900 p-3 rounded-lg border text-center">
                  <div className="text-xs font-semibold text-neutral-500 uppercase mb-1">Urgency</div>
                  <div className={`text-lg font-bold ${result.urgency === "Critical" ? "text-rose-500" : result.urgency === "Urgent" ? "text-amber-500" : "text-emerald-500"}`}>
                    {result.urgency}
                  </div>
                </div>
              </div>

              {/* ── Lender Action: decision-aware save ── */}
              <div className="border-t border-neutral-100 dark:border-neutral-800 pt-4 space-y-2">
                <div className="text-xs font-bold text-neutral-400 uppercase tracking-wider mb-2">Lender Action</div>

                {saved ? (
                  // Confirmation after save
                  <div className={`w-full flex flex-col items-center justify-center gap-1 py-3 rounded-xl border-2 text-sm font-semibold ${
                    result.loanDecision === "APPROVE"
                      ? "bg-emerald-50 border-emerald-300 text-emerald-700 dark:bg-emerald-900/20 dark:border-emerald-700 dark:text-emerald-300"
                      : result.loanDecision === "CONDITIONAL"
                      ? "bg-amber-50 border-amber-300 text-amber-700 dark:bg-amber-900/20 dark:border-amber-700 dark:text-amber-300"
                      : "bg-neutral-50 border-neutral-300 text-neutral-600 dark:bg-neutral-800 dark:border-neutral-700 dark:text-neutral-400"
                  }`}>
                    <span>{result.loanDecision === "APPROVE" ? "✅ Approved & Added to Portfolio" : result.loanDecision === "CONDITIONAL" ? "⚠️ Saved as Conditional — Pending Review" : "❌ Declined & Archived"}</span>
                    <span className="font-mono font-black text-xs opacity-70">{saved}</span>
                  </div>
                ) : result.loanDecision === "APPROVE" ? (
                  <button
                    onClick={saveToPortfolio}
                    disabled={saving}
                    className="w-full py-3 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-sm transition disabled:opacity-50"
                  >
                    {saving ? "Saving…" : "✅ Approve & Add to Active Portfolio"}
                  </button>
                ) : result.loanDecision === "CONDITIONAL" ? (
                  <div className="space-y-2">
                    <div className="text-xs text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-2.5 leading-relaxed">
                      <strong>Conditions must be met before disbursal.</strong> Saving as pending — lender must manually approve once conditions are satisfied.
                    </div>
                    <button
                      onClick={saveToPortfolio}
                      disabled={saving}
                      className="w-full py-3 rounded-xl bg-amber-500 hover:bg-amber-600 text-white font-bold text-sm transition disabled:opacity-50"
                    >
                      {saving ? "Saving…" : "⚠️ Save as Pending Conditional Approval"}
                    </button>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <div className="text-xs text-rose-600 dark:text-rose-400 bg-rose-50 dark:bg-rose-900/20 border border-rose-200 dark:border-rose-800 rounded-lg p-2.5 leading-relaxed">
                      Application does not meet lending criteria. Archiving for regulatory record-keeping.
                    </div>
                    <button
                      onClick={saveToPortfolio}
                      disabled={saving}
                      className="w-full py-3 rounded-xl bg-neutral-700 hover:bg-neutral-800 text-white font-bold text-sm transition disabled:opacity-50"
                    >
                      {saving ? "Archiving…" : "❌ Decline & Archive Application"}
                    </button>
                  </div>
                )}
              </div>
            </>)}


            {/* ── EXISTING LOAN RESULTS ── */}
            {result.mode === "existing" && (<>

              {/* Priority badge — prominent */}
              <div className="flex flex-col items-center justify-center py-6 gap-4">
                <span className={`px-6 py-3 rounded-full text-lg font-black uppercase tracking-widest border-2 ${priorityClass(result.priority)}`}>
                  {result.priority} Priority
                </span>
                <div className="text-center">
                  <div className="text-xs uppercase tracking-widest text-neutral-500 dark:text-neutral-400 mb-1">NPA Risk Score</div>
                  <div className="text-5xl font-black text-neutral-900 dark:text-white">{result.defaultProb}%</div>
                  <div className="text-sm text-neutral-400 mt-1">Probability of Default</div>
                </div>
              </div>

              {/* Urgency strip */}
              <div className={`rounded-lg p-3 text-center text-sm font-bold uppercase tracking-wider ${
                result.urgency === "Critical" ? "bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300"
                : result.urgency === "Urgent"  ? "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300"
                : "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
              }`}>
                ⏱ Urgency: {result.urgency} — {result.daysOverdue} Days Overdue
              </div>

              {/* Key numbers */}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-white dark:bg-neutral-900 p-4 rounded-lg border border-neutral-100 dark:border-neutral-800">
                  <div className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-1">Outstanding EAD</div>
                  <div className="text-xl font-black text-neutral-900 dark:text-white">₹{result.outstanding.toLocaleString("en-IN", { maximumFractionDigits: 0 })}</div>
                </div>
                <div className="bg-white dark:bg-neutral-900 p-4 rounded-lg border border-neutral-100 dark:border-neutral-800">
                  <div className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-1">Expected Loss</div>
                  <div className="text-xl font-black text-rose-600 dark:text-rose-400">₹{result.expLoss.toLocaleString("en-IN", { maximumFractionDigits: 0 })}</div>
                </div>
                <div className="bg-white dark:bg-neutral-900 p-4 rounded-lg border border-neutral-100 dark:border-neutral-800">
                  <div className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-1">Recoverable Value</div>
                  <div className="text-xl font-black text-emerald-600 dark:text-emerald-400">₹{result.recoverableValue.toLocaleString("en-IN", { maximumFractionDigits: 0 })}</div>
                </div>
                <div className="bg-white dark:bg-neutral-900 p-4 rounded-lg border border-neutral-100 dark:border-neutral-800">
                  <div className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-1">Recovery Net ROI</div>
                  <div className="text-xl font-black text-neutral-900 dark:text-white">{result.efficiencyScore}x</div>
                </div>
              </div>

              {/* Recovery action card */}
              <div className="bg-white dark:bg-neutral-900 border border-neutral-100 dark:border-neutral-800 rounded-xl p-5 space-y-4">
                <div className="text-xs font-bold text-neutral-400 uppercase tracking-wider">Recovery Action Plan</div>

                {/* Employment status badge */}
                <div className="flex items-start gap-3">
                  <div className={`px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-wide border flex-shrink-0 ${
                    result.employmentStatus === "employed"      ? "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-900/20 dark:text-emerald-300"
                    : result.employmentStatus === "self_employed" ? "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-900/20 dark:text-blue-300"
                    : result.employmentStatus === "unemployed"    ? "bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-900/20 dark:text-rose-300"
                    : "bg-neutral-50 text-neutral-600 border-neutral-200 dark:bg-neutral-800 dark:text-neutral-300"
                  }`}>
                    {result.employmentStatus === "employed" ? "💼 Employed"
                      : result.employmentStatus === "self_employed" ? "🏢 Self-Employed"
                      : result.employmentStatus === "unemployed" ? "⚠️ Unemployed"
                      : "🧓 Retired"}
                  </div>
                  {result.employmentContext && (
                    <p className="text-xs text-neutral-500 dark:text-neutral-400 leading-relaxed">{result.employmentContext}</p>
                  )}
                </div>

                {/* Action */}
                <div className="flex items-center gap-3 pt-2 border-t border-neutral-100 dark:border-neutral-800">
                  {teamIcon(result.team)}
                  <div>
                    <div className="text-sm font-black text-neutral-900 dark:text-white">{result.action}</div>
                    <div className="text-xs text-neutral-500">Assigned to: <span className="font-semibold text-neutral-700 dark:text-neutral-300">{result.team}</span></div>
                  </div>
                </div>
              </div>


              {/* Save to Recovery Queue */}
              <div className="space-y-2">
                {saved ? (
                  <div className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-300 dark:border-emerald-700 text-emerald-700 dark:text-emerald-300 text-sm font-semibold">
                    ✅ Saved as <span className="font-mono font-black">{saved}</span> — tracked in Active Recovery
                  </div>
                ) : (
                  <button
                    onClick={saveToPortfolio}
                    disabled={saving}
                    className="w-full bg-orange-600 hover:bg-orange-700 text-white font-bold text-sm py-2.5 rounded-lg transition-colors disabled:opacity-50"
                  >
                    {saving ? "Saving…" : "📥 Save to Recovery Queue"}
                  </button>
                )}
              </div>
            </>)}

          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-center p-8">
            <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center mb-4">
              <TrendingDown className="h-8 w-8 text-blue-600 dark:text-blue-400" />
            </div>
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-white">
              {mode === "new" ? "Awaiting Loan Evaluation" : "Awaiting Recovery Assessment"}
            </h3>
            <p className="mt-2 text-sm text-neutral-500 dark:text-neutral-400 max-w-sm">
              {mode === "new"
                ? "Fill in the application details to get an APPROVE / CONDITIONAL / REJECT lending decision with Basel III metrics."
                : "Fill in the existing loan details to get a Priority level and Recovery Action plan for NPA management."}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
