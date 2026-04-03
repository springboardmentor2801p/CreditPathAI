import { useState, useEffect } from "react";
import { useNavigate } from "react-router";
import { useForm } from "react-hook-form";
import { Target, Activity, Zap, CheckCircle2 } from "lucide-react";
import { apiFetch } from "../lib/api";

type LoanEvaluationData = {
  loanPurpose: string;
  loanAmount: number;
  annualIncome: number;
  employmentStatus: string;
  creditScore: number;
  monthlyDebt: number;
  propertyValue: number;
  term: number;
};

export function BorrowerEvaluator() {
  const { register, handleSubmit, reset, formState: { errors } } = useForm<LoanEvaluationData>({
    defaultValues: { creditScore: 720, term: 180, loanPurpose: 'home_improvement', employmentStatus: 'employed' }
  });
  const [loading, setLoading] = useState(false);
  const [fetchingProfile, setFetchingProfile] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [currentFormData, setCurrentFormData] = useState<LoanEvaluationData | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    // Fetch profile data to pre-fill the form (identity matrix values)
    apiFetch('/api/borrower-profile')
      .then(res => {
        if (!res.ok) throw new Error("Failed to load profile");
        return res.json();
      })
      .then(d => {
        if (d) {
          // Update form defaults with actual profile data while keeping initial logic for other fields
          reset({
            loanPurpose: 'home_improvement',
            loanAmount: 1000000,
            annualIncome: Number(d.annual_income) || 0,
            employmentStatus: d.employment_status || 'employed',
            creditScore: Number(d.credit_score) || 720,
            monthlyDebt: Number(d.total_monthly_obligations) || 0,
            propertyValue: 0,
            term: 180
          });
        }
      })
      .catch(err => {
        console.error("Error fetching borrower profile for simulator:", err);
      })
      .finally(() => setFetchingProfile(false));
  }, [reset]);

  const onSubmit = async (data: LoanEvaluationData) => {
    setLoading(true);
    setApiError(null);
    try {
      const response = await apiFetch("/api/evaluate-loan", {
        method: "POST",
        body: JSON.stringify({
          loanPurpose: data.loanPurpose,
          loanAmount: Number(data.loanAmount),
          annualIncome: Number(data.annualIncome),
          monthlyDebt: Number(data.monthlyDebt),
          creditScore: Number(data.creditScore),
          employmentStatus: data.employmentStatus,
          propertyValue: Number(data.propertyValue),
          term: Number(data.term)
        })
      });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const resData = await response.json();
      setResult(resData);
      setCurrentFormData(data);
    } catch (error: any) {
      console.error("Error evaluating loan:", error);
      setApiError(error.message || 'Connection failed. Ensure the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleProceedApplication = async () => {
    if (!currentFormData) return;
    setSubmitting(true);
    setApiError(null);
    try {
      const response = await apiFetch("/api/borrower-applications", {
        method: "POST",
        body: JSON.stringify({
          loan_purpose: currentFormData.loanPurpose,
          loan_amount: Number(currentFormData.loanAmount),
          term_months: Number(currentFormData.term),
          property_value: Number(currentFormData.propertyValue)
        })
      });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      navigate('/borrower/applications');
    } catch (error: any) {
      console.error("Error submitting application:", error);
      setApiError(error.message || 'Submission failed.');
    } finally {
      setSubmitting(false);
    }
  };

  const inputClasses = "block w-full bg-[#050505] border border-zinc-800 rounded-md text-zinc-100 text-sm font-mono focus:ring-1 focus:ring-emerald-500/50 focus:border-emerald-500/50 px-3 py-2.5 transition-colors placeholder-zinc-700";
  const labelClasses = "block text-[10px] font-bold tracking-widest text-zinc-400 uppercase mb-1.5";

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
      {/* Input Form */}
      <div className="lg:col-span-7 bg-[#131316] rounded-xl border border-zinc-800/80 p-6 shadow-lg">
        <div className="flex items-center gap-3 mb-8 border-b border-zinc-800/80 pb-4">
          <Target className="text-emerald-400 h-5 w-5" />
          <div>
            <h2 className="text-lg font-black tracking-widest text-white uppercase">Pre-Approval Simulator</h2>
            <p className="text-[10px] font-mono text-zinc-500 uppercase mt-1">Test your metrics against lender algorithms</p>
          </div>
        </div>
        
        {fetchingProfile ? (
          <div className="flex flex-col items-center justify-center py-20 space-y-4">
            <div className="w-8 h-8 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest">Integrating Identity Matrix...</p>
          </div>
        ) : (
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-5">
            <div className="grid grid-cols-1 gap-y-5 gap-x-6 sm:grid-cols-2">
              <div className="sm:col-span-2">
                <label htmlFor="loanPurpose" className={labelClasses}>Loan Purpose</label>
                <select id="loanPurpose" {...register("loanPurpose", { required: "Please select a loan purpose" })} className={inputClasses}>
                  <option value="debt_consolidation">Debt Consolidation</option>
                  <option value="home_improvement">Home Improvement</option>
                  <option value="auto">Vehicle Purchase</option>
                  <option value="business">Business Capital</option>
                  <option value="other">Other / Personal</option>
                </select>
                {errors.loanPurpose && <p className="mt-1 text-[10px] text-rose-400 font-mono">⚠ {errors.loanPurpose.message}</p>}
              </div>

              <div>
                <label htmlFor="loanAmount" className={labelClasses}>Loan Amount (₹)</label>
                <input type="number" id="loanAmount" {...register("loanAmount", {
                  required: "Enter the loan amount you need (e.g. ₹50,00,000)",
                  min: { value: 1000, message: "Loan amount must be at least ₹1,000" }
                })} className={inputClasses} placeholder="5000000" />
                {errors.loanAmount && <p className="mt-1 text-[10px] text-rose-400 font-mono">⚠ {errors.loanAmount.message}</p>}
              </div>

              <div>
                <label htmlFor="propertyValue" className={labelClasses}>Asset / Property Value (₹)</label>
                <input type="number" id="propertyValue" {...register("propertyValue", {
                  required: "Enter the current market value of your asset or property (e.g. ₹75,00,000)",
                  min: { value: 0, message: "Asset value cannot be negative" }
                })} className={inputClasses} placeholder="7500000" />
                {errors.propertyValue && <p className="mt-1 text-[10px] text-rose-400 font-mono">⚠ {errors.propertyValue.message}</p>}
              </div>

              <div>
                <label htmlFor="annualIncome" className={labelClasses}>Annual Income (₹)</label>
                <input type="number" id="annualIncome" {...register("annualIncome", {
                  required: "Enter your total yearly income before tax (e.g. ₹15,00,000)",
                  min: { value: 0, message: "Income cannot be negative" }
                })} className={inputClasses} placeholder="1500000" />
                {errors.annualIncome && <p className="mt-1 text-[10px] text-rose-400 font-mono">⚠ {errors.annualIncome.message}</p>}
              </div>

              <div>
                <label htmlFor="monthlyDebt" className={labelClasses}>Existing Monthly EMIs / Obligations (₹)</label>
                <input type="number" id="monthlyDebt" {...register("monthlyDebt", {
                  required: "Enter your total monthly loan payments and obligations (e.g. ₹25,00,000). Enter 0 if none.",
                  min: { value: 0, message: "Monthly obligations cannot be negative" }
                })} className={inputClasses} placeholder="25000" />
                {errors.monthlyDebt && <p className="mt-1 text-[10px] text-rose-400 font-mono">⚠ {errors.monthlyDebt.message}</p>}
              </div>

              <div>
                <label htmlFor="term" className={labelClasses}>Loan Repayment Term (Months)</label>
                <input type="number" id="term" {...register("term", {
                  required: "Enter loan duration in months (e.g. 120 = 10 years, 180 = 15 years, 240 = 20 years)",
                  min: { value: 1, message: "Minimum repayment term is 1 month" },
                  max: { value: 480, message: "Maximum repayment term is 480 months (40 years)" }
                })} className={inputClasses} placeholder="180" />
                {errors.term && <p className="mt-1 text-[10px] text-rose-400 font-mono">⚠ {errors.term.message}</p>}
              </div>

              <div>
                <label htmlFor="creditScore" className={labelClasses}>CIBIL / Credit Score</label>
                <input type="number" id="creditScore" {...register("creditScore", {
                  required: "Enter your CIBIL score (ranges from 300 to 900; 750+ is considered excellent)",
                  min: { value: 300, message: "Minimum valid CIBIL score is 300" },
                  max: { value: 900, message: "Maximum valid CIBIL score is 900" }
                })} className={inputClasses} placeholder="300–900" />
                {errors.creditScore && <p className="mt-1 text-[10px] text-rose-400 font-mono">⚠ {errors.creditScore.message}</p>}
              </div>

              <div className="sm:col-span-2">
                <label htmlFor="employmentStatus" className={labelClasses}>Employment Status</label>
                <select id="employmentStatus" {...register("employmentStatus", { required: "Please select your current employment status" })} className={inputClasses}>
                  <option value="employed">Salaried / Full-Time Employee</option>
                  <option value="self_employed">Self-Employed / Freelancer / Business Owner</option>
                  <option value="unemployed">Currently Unemployed</option>
                  <option value="retired">Retired</option>
                </select>
                {errors.employmentStatus && <p className="mt-1 text-[10px] text-rose-400 font-mono">⚠ {errors.employmentStatus.message}</p>}
              </div>
            </div>


            <div className="pt-6">
              <button
                type="submit"
                disabled={loading}
                className="w-full flex justify-center py-3.5 px-4 rounded-md text-xs font-black tracking-widest text-black uppercase bg-emerald-500 hover:bg-emerald-400 focus:outline-none disabled:opacity-50 transition-all shadow-[0_0_20px_-5px_rgba(16,185,129,0.4)] hover:shadow-[0_0_25px_rgba(16,185,129,0.6)]"
              >
                {loading ? (
                  <span className="flex items-center gap-2">
                    <div className="w-3 h-3 border-2 border-black border-t-transparent rounded-full animate-spin"></div>
                    SIMULATING_ALGORITHMS...
                  </span>
                ) : "RUN_SIMULATION"}
              </button>
              {apiError && (
                <div className="mt-3 p-3 bg-rose-900/30 border border-rose-500/40 rounded-md text-xs font-mono text-rose-400">
                  ⚠ {apiError}
                </div>
              )}
              {Object.keys(errors).length > 0 && (
                <div className="mt-3 p-3 bg-amber-900/30 border border-amber-500/40 rounded-md text-xs font-mono text-amber-400">
                  ⚠ Fill in all required fields: {Object.keys(errors).join(', ')}
                </div>
              )}
            </div>
          </form>
        )}
      </div>

      {/* Results Panel */}
      <div className="lg:col-span-5 bg-[#09090b] rounded-xl border border-zinc-800/80 p-6 flex flex-col relative overflow-hidden shadow-lg">
        {/* Techy background */}
        <div className="absolute inset-0 opacity-5 pointer-events-none" style={{ backgroundImage: 'radial-gradient(circle at 2px 2px, #10b981 1px, transparent 0)', backgroundSize: '24px 24px' }}></div>

        {result ? (
          <div className="relative z-10 space-y-6 flex-1 animate-in fade-in zoom-in-95 duration-500">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-4">
              <h3 className="text-sm font-bold tracking-widest text-white uppercase flex items-center">
                <Zap className="mr-2 text-emerald-400 h-4 w-4" />
                Output Vectors
              </h3>
              <span className={`px-2 py-1 rounded text-[10px] font-black uppercase tracking-wider border bg-${result.tierColor}-500/10 text-${result.tierColor}-400 border-${result.tierColor}-500/20 shadow-[0_0_10px_rgba(0,0,0,0.5)]`}>
                {result.tier}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-[#131316] p-4 rounded-lg border border-zinc-800 shadow-inner">
                <div className="text-[10px] font-bold tracking-widest text-zinc-500 uppercase mb-2">Approval Prob.</div>
                <div className="text-3xl font-black text-white font-mono flex items-baseline gap-1 tracking-tighter">
                  {result.approvalProb}<span className="text-xl text-zinc-500">%</span>
                </div>
              </div>
              <div className="bg-[#131316] p-4 rounded-lg border border-zinc-800 shadow-inner">
                <div className="text-[10px] font-bold tracking-widest text-zinc-500 uppercase mb-2">Est. APR</div>
                <div className="text-xl font-black text-emerald-400 font-mono tracking-tighter mt-1">
                  {result.estRate}
                </div>
              </div>
            </div>

            <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-4 flex justify-between items-center">
              <div>
                <div className="text-[10px] font-bold tracking-widest text-zinc-500 uppercase mb-1">Debt-to-Income (DTI)</div>
                <div className="text-lg font-mono text-zinc-200">{result.dti}%</div>
              </div>
              <div className={`w-16 h-2 rounded-full ${parseFloat(result.dti) > 40 ? 'bg-rose-500' : 'bg-emerald-500'} shadow-[0_0_5px_currentColor]`}></div>
            </div>

            <div>
              <h4 className="text-[10px] font-bold tracking-widest text-zinc-500 uppercase mb-3">Lender Insights</h4>
              <ul className="space-y-3">
                {result.recommendations.map((rec: string, idx: number) => (
                  <li key={idx} className="flex items-start gap-3 p-3 bg-[#131316] rounded-md border border-zinc-800">
                    <CheckCircle2 className="h-4 w-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                    <span className="text-xs font-mono text-zinc-400 uppercase leading-relaxed">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
            
            <div className="mt-auto pt-4 border-t border-zinc-800">
              <button 
                onClick={handleProceedApplication}
                disabled={submitting}
                className="w-full bg-zinc-100 text-black text-xs font-black tracking-widest uppercase py-3 rounded-md hover:bg-white transition-colors shadow-[0_0_15px_rgba(255,255,255,0.2)] disabled:opacity-50"
              >
                {submitting ? "SECURING APPLICATION..." : "PROCEED_TO_APPLICATION"}
              </button>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-center p-8 relative z-10">
            <div className="w-20 h-20 border border-zinc-800 rounded-full flex items-center justify-center mb-6 relative">
              <div className="absolute inset-0 border border-emerald-500/20 rounded-full animate-ping"></div>
              <Target className="h-8 w-8 text-zinc-600" />
            </div>
            <h3 className="text-sm font-bold tracking-widest uppercase text-zinc-400">Awaiting Data Core</h3>
            <p className="mt-3 text-xs font-mono text-zinc-600 max-w-sm">
              Input metrics to reverse-engineer lender approval models and optimize your application.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
