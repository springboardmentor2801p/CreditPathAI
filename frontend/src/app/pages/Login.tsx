import { useState } from "react";
import { useNavigate } from "react-router";
import { ShieldAlert, Zap, Building2, User, Eye, EyeOff } from "lucide-react";
import { apiFetch } from "../lib/api";

export function Login() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<"select" | "login" | "register">("select");
  const [role, setRole] = useState<"institution" | "borrower">("borrower");
  const [form, setForm] = useState({
    email: "",
    password: "",
    full_name: "",
    credit_score: "",
    annual_income: "",
    employment_status: "employed"
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showPass, setShowPass] = useState(false);

  const handleRoleClick = (r: "institution" | "borrower") => {
    setRole(r);
    setMode("login");
    setError("");
  };

  const handleSubmit = async (e: React.FormEvent, action: "login" | "register") => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const endpoint = action === "login" ? "/api/auth/login" : "/api/auth/register";
      let body: any;
      if (action === "login") {
        body = { email: form.email, password: form.password };
      } else {
        body = { email: form.email, password: form.password, full_name: form.full_name, role };
        if (role === "borrower") {
          body.credit_score = Number(form.credit_score);
          body.annual_income = Number(form.annual_income);
          body.employment_status = form.employment_status;
        }
      }

      const res = await apiFetch(endpoint, {
        method: "POST",
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Authentication failed");
      }

      const data = await res.json();
      // Store auth state in localStorage
      localStorage.setItem("user_id", String(data.user_id));
      localStorage.setItem("user_role", data.role);
      localStorage.setItem("user_name", data.full_name);
      localStorage.setItem("user_email", data.email);

      // After login go to role home; after register go to profile setup for borrowers
      if (action === "register" && data.role === "borrower") {
        navigate("/borrower/profile");
      } else {
        navigate(data.role === "institution" ? "/institution" : "/borrower");
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const themeColor = role === "institution" ? "cyan" : "emerald";
  const themeShadow = role === "institution"
    ? "shadow-[0_0_20px_-5px_rgba(34,211,238,0.5)]"
    : "shadow-[0_0_20px_-5px_rgba(16,185,129,0.5)]";
  const btnClass = role === "institution"
    ? "bg-cyan-500 hover:bg-cyan-400 text-black"
    : "bg-emerald-500 hover:bg-emerald-400 text-black";

  return (
    <div className="min-h-screen bg-[#050505] flex flex-col justify-center py-12 sm:px-6 lg:px-8 relative overflow-hidden text-zinc-100 font-sans selection:bg-cyan-500/30">
      {/* Background elements */}
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mix-blend-overlay pointer-events-none"></div>
      <div className="absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-violet-600/20 blur-[150px] rounded-full pointer-events-none animate-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-cyan-500/10 blur-[120px] rounded-full pointer-events-none" />

      <div className="relative z-10 sm:mx-auto sm:w-full sm:max-w-2xl">
        <div className="flex justify-center items-center gap-3 text-cyan-400 drop-shadow-[0_0_15px_rgba(34,211,238,0.5)]">
          <Zap size={48} className="text-cyan-400" />
        </div>
        <h2 className="mt-6 text-center text-4xl font-black tracking-tighter uppercase bg-clip-text text-transparent bg-gradient-to-r from-zinc-100 to-zinc-500">
          CreditPath AI
        </h2>
        <p className="mt-2 text-center text-xs font-mono tracking-widest text-zinc-500 uppercase">
          {mode === "select" ? "Select your role to continue" : mode === "login" ? `Sign in as ${role}` : `Create ${role} account`}
        </p>
      </div>

      <div className="relative z-10 mt-10 sm:mx-auto sm:w-full sm:max-w-md">
        {mode === "select" ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-3xl mx-auto">
            {/* Institution Card */}
            <div
              onClick={() => handleRoleClick("institution")}
              className="cursor-pointer bg-[#0e0e11]/80 backdrop-blur-xl py-10 px-8 shadow-2xl rounded-2xl border border-cyan-500/20 shadow-[0_0_30px_-10px_rgba(34,211,238,0.2)] hover:border-cyan-500/50 transition-all group relative overflow-hidden flex flex-col items-center text-center"
            >
              <div className="absolute inset-0 bg-gradient-to-b from-cyan-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
              <div className="w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center mb-6 shadow-[0_0_15px_rgba(34,211,238,0.2)]">
                <Building2 className="w-8 h-8 text-cyan-400" />
              </div>
              <h3 className="text-xl font-black tracking-widest text-white uppercase mb-2">Institution</h3>
              <p className="text-[10px] font-mono tracking-widest text-zinc-400 mb-6 h-12">
                Access risk dashboards and recovery routing.
              </p>
              <div className="text-xs font-black tracking-widest text-cyan-400 border border-cyan-500/30 px-4 py-2 rounded-md">
                CONNECT AS LENDER →
              </div>
            </div>

            {/* Borrower Card */}
            <div
              onClick={() => handleRoleClick("borrower")}
              className="cursor-pointer bg-[#0e0e11]/80 backdrop-blur-xl py-10 px-8 shadow-2xl rounded-2xl border border-emerald-500/20 shadow-[0_0_30px_-10px_rgba(16,185,129,0.2)] hover:border-emerald-500/50 transition-all group relative overflow-hidden flex flex-col items-center text-center"
            >
              <div className="absolute inset-0 bg-gradient-to-b from-emerald-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
              <div className="w-16 h-16 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center mb-6 shadow-[0_0_15px_rgba(16,185,129,0.2)]">
                <User className="w-8 h-8 text-emerald-400" />
              </div>
              <h3 className="text-xl font-black tracking-widest text-white uppercase mb-2">Borrower</h3>
              <p className="text-[10px] font-mono tracking-widest text-zinc-400 mb-6 h-12">
                View financial health, simulate loans, and improve your credit score.
              </p>
              <div className="text-xs font-black tracking-widest text-emerald-400 border border-emerald-500/30 px-4 py-2 rounded-md">
                CONNECT AS BORROWER →
              </div>
            </div>
          </div>
        ) : (
          <div className={`bg-[#0e0e11]/90 backdrop-blur-xl px-8 py-10 rounded-2xl border border-${themeColor}-500/20 ${themeShadow}`}>
            <button onClick={() => setMode("select")} className="text-xs text-zinc-500 hover:text-zinc-300 mb-6 flex items-center gap-1 transition-colors">
              ← Back
            </button>

            <form onSubmit={(e) => handleSubmit(e, mode as "login" | "register")} className="space-y-5">
              {mode === "register" && (
                <>
                  <div>
                    <label className="block text-[10px] font-bold tracking-widest text-zinc-400 uppercase mb-1.5">Full Name</label>
                    <input
                      type="text"
                      required
                      value={form.full_name}
                      onChange={e => setForm({...form, full_name: e.target.value})}
                      className="block w-full bg-[#050505] border border-zinc-800 rounded-md text-zinc-100 text-sm px-3 py-2.5 focus:ring-1 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-colors"
                      placeholder="Arjun Sharma"
                    />
                  </div>
                  {role === "borrower" && (
                    <>
                      <div>
                        <label className="block text-[10px] font-bold tracking-widest text-zinc-400 uppercase mb-1.5">CIBIL / Credit Score</label>
                        <input
                          type="number"
                          required
                          min={300}
                          max={900}
                          value={form.credit_score}
                          onChange={e => setForm({...form, credit_score: e.target.value})}
                          className="block w-full bg-[#050505] border border-zinc-800 rounded-md text-zinc-100 text-sm px-3 py-2.5 focus:ring-1 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-colors"
                          placeholder="700"
                        />
                      </div>
                      <div>
                        <label className="block text-[10px] font-bold tracking-widest text-zinc-400 uppercase mb-1.5">Annual Income (₹)</label>
                        <input
                          type="number"
                          required
                          min={0}
                          value={form.annual_income}
                          onChange={e => setForm({...form, annual_income: e.target.value})}
                          className="block w-full bg-[#050505] border border-zinc-800 rounded-md text-zinc-100 text-sm px-3 py-2.5 focus:ring-1 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-colors"
                          placeholder="500000"
                        />
                      </div>
                      <div>
                        <label className="block text-[10px] font-bold tracking-widest text-zinc-400 uppercase mb-1.5">Employment Status</label>
                        <select
                          required
                          value={form.employment_status}
                          onChange={e => setForm({...form, employment_status: e.target.value})}
                          className="block w-full bg-[#050505] border border-zinc-800 rounded-md text-zinc-100 text-sm px-3 py-2.5 focus:ring-1 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-colors"
                        >
                          <option value="employed">Salaried / Full-Time</option>
                          <option value="self_employed">Self-Employed / Business</option>
                          <option value="retired">Retired</option>
                          <option value="unemployed">Unemployed</option>
                        </select>
                      </div>
                    </>
                  )}
                </>
              )}
              <div>
                <label className="block text-[10px] font-bold tracking-widest text-zinc-400 uppercase mb-1.5">Email</label>
                <input
                  type="email"
                  required
                  value={form.email}
                  onChange={e => setForm({...form, email: e.target.value})}
                  className="block w-full bg-[#050505] border border-zinc-800 rounded-md text-zinc-100 text-sm px-3 py-2.5 focus:ring-1 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-colors"
                  placeholder={role === "borrower" ? "borrower@creditpath.ai" : "lender@creditpath.ai"}
                />
              </div>
              <div>
                <label className="block text-[10px] font-bold tracking-widest text-zinc-400 uppercase mb-1.5">Password</label>
                <div className="relative">
                  <input
                    type={showPass ? "text" : "password"}
                    required
                    value={form.password}
                    onChange={e => setForm({...form, password: e.target.value})}
                    className="block w-full bg-[#050505] border border-zinc-800 rounded-md text-zinc-100 text-sm px-3 py-2.5 pr-10 focus:ring-1 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-colors"
                    placeholder="••••••••"
                  />
                  <button type="button" onClick={() => setShowPass(!showPass)} className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300">
                    {showPass ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              {error && (
                <div className="p-3 bg-rose-900/30 border border-rose-500/40 rounded-md text-xs font-mono text-rose-400">
                  ⚠ {error}
                </div>
              )}

              {mode === "login" && (
                <div className="text-[10px] text-zinc-500 font-mono">
                  Demo: <span className="text-zinc-400">{role === "borrower" ? "borrower@creditpath.ai / borrower123" : "lender@creditpath.ai / lender123"}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className={`w-full py-3 px-4 rounded-lg text-xs font-black tracking-widest uppercase ${btnClass} disabled:opacity-50 transition-all`}
              >
                {loading ? "AUTHENTICATING..." : mode === "login" ? "SIGN IN" : "CREATE ACCOUNT"}
              </button>

              <button
                type="button"
                onClick={() => setMode(mode === "login" ? "register" : "login")}
                className="w-full text-center text-[10px] font-mono tracking-widest text-zinc-500 hover:text-zinc-300 transition-colors"
              >
                {mode === "login" ? "Don't have an account? Register →" : "Already registered? Sign in →"}
              </button>
            </form>
          </div>
        )}

        <div className="mt-8 text-center">
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-zinc-800/80" />
            </div>
            <div className="relative flex justify-center text-[10px] font-mono tracking-widest uppercase">
              <span className="px-3 bg-[#050505] text-zinc-600 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                SECURE CONNECTION ESTABLISHED
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
