
import { useState } from "react";
import GaugeChart from "react-gauge-chart";
import { BarChart, Bar, XAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

export default function UserDashboard() {

  const [form, setForm] = useState({
    income: "",
    loan_amount: "",
    credit_score: "",
    ltv: "",
    dtir1: ""
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false); // 🔥 NEW

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async () => {
    try {
      setLoading(true); // 🔥 START

      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          loan_amount: Number(form.loan_amount),
          income: Number(form.income),
          credit_score: Number(form.credit_score),
          ltv: Number(form.ltv),
          dtir1: Number(form.dtir1)
        })
      });

      const data = await res.json();
      setResult(data);

    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false); // 🔥 STOP
    }
  };

  const getUserMessage = (risk) => {
    if (risk === "Low") return "You are in a safe financial position.";
    if (risk === "Medium") return "Consider improving income stability.";
    if (risk === "High") return "Reduce loan or improve credit score.";
    return "Critical financial risk. Immediate action needed.";
  };

  const chartData = [
    { name: "Credit", value: Number(form.credit_score) },
    { name: "LTV", value: Number(form.ltv) },
    { name: "DTI", value: Number(form.dtir1) }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#020617] via-[#020617] to-[#0f172a] text-white px-10 py-8">

      {/* HEADER */}
      <div className="flex justify-between items-center mb-12">

        <div>
          <h1 className="text-3xl font-bold tracking-wide">
            Credit Risk Dashboard
          </h1>
          <p className="text-sm text-gray-400 mt-1">
            AI-powered borrower insights
          </p>
        </div>

        <button
          onClick={() => window.location.href = "/"}
          className="bg-gray-700 px-4 py-2 rounded-lg hover:bg-gray-600 transition"
        >
          ⬅ Back
        </button>

      </div>

      {/* MAIN GRID */}
      <div className="max-w-6xl mx-auto grid grid-cols-2 gap-10">

        {/* LEFT FORM */}
        <div className="bg-white/5 backdrop-blur-xl p-8 rounded-2xl shadow-xl border border-white/10">

          <h2 className="text-lg mb-6 text-blue-400 font-semibold">
            Borrower Details
          </h2>

          <div className="space-y-5">

            <input name="income" placeholder="Monthly Income"
              onChange={handleChange}
              className="w-full p-3 rounded-lg bg-black/30 border border-gray-700 focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-500 transition" />

            <input name="loan_amount" placeholder="Loan Amount"
              onChange={handleChange}
              className="w-full p-3 rounded-lg bg-black/30 border border-gray-700 focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-500 transition" />

            <input name="credit_score" placeholder="Credit Score"
              onChange={handleChange}
              className="w-full p-3 rounded-lg bg-black/30 border border-gray-700 focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-500 transition" />

            <input name="ltv" placeholder="Loan-to-Value (%)"
              onChange={handleChange}
              className="w-full p-3 rounded-lg bg-black/30 border border-gray-700 focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-500 transition" />

            <input name="dtir1" placeholder="Debt-to-Income (%)"
              onChange={handleChange}
              className="w-full p-3 rounded-lg bg-black/30 border border-gray-700 focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-500 transition" />

            {/* 🔥 BUTTON WITH LOADING */}
            <button
              onClick={handleSubmit}
              disabled={loading}
              className={`w-full p-3 rounded-lg mt-4 font-semibold transition duration-300
                ${loading
                  ? "bg-gray-600 cursor-not-allowed"
                  : "bg-gradient-to-r from-blue-500 to-blue-700 hover:scale-105 hover:shadow-blue-500/30"
                }
              `}
            >
              {loading ? "Analyzing..." : "Analyze Risk"}
            </button>

          </div>
        </div>

        {/* RIGHT RESULT */}
        <div className="bg-white/5 backdrop-blur-xl p-8 rounded-2xl shadow-xl border border-white/10">

          <h2 className="text-lg mb-6 text-green-400 font-semibold">
            Risk Analysis
          </h2>

          {/* 🔥 LOADING SPINNER */}
          {loading && (
            <div className="flex justify-center mt-6">
              <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            </div>
          )}

          {!loading && !result && (
            <p className="text-gray-400">Enter details to analyze risk</p>
          )}

          {!loading && result && (
            <>
              {/* GAUGE */}
              <div className="flex justify-center mb-8">
                <GaugeChart
                  id="gauge-chart"
                  nrOfLevels={20}
                  percent={result.default_probability}
                  colors={["#22c55e", "#facc15", "#ef4444"]}
                  arcWidth={0.2}
                  needleColor="#ffffff"
                />
              </div>

              {/* METRICS */}
              <div className="flex justify-between mb-6">
                <div>
                  <p className="text-3xl font-bold text-blue-400">
                    {Math.round(result.default_probability * 100)}%
                  </p>
                  <p className="text-sm text-gray-400">Default Probability</p>
                </div>

                <div>
                  <p className="text-xl font-semibold text-red-400">
                    ₹{result.expected_loss}
                  </p>
                  <p className="text-sm text-gray-400">Expected Loss</p>
                </div>
              </div>

              {/* BADGE */}
              <div className={`px-5 py-2 rounded-full text-sm font-bold inline-block mb-4
                ${result.risk_level === "Low" && "bg-green-500/20 text-green-400"}
                ${result.risk_level === "Medium" && "bg-yellow-500/20 text-yellow-400"}
                ${result.risk_level === "High" && "bg-red-500/20 text-red-400"}
              `}>
                {result.risk_level} Risk
              </div>

              <p className="text-yellow-400 font-medium mb-6">
                {getUserMessage(result.risk_level)}
              </p>

              {/* CHART */}
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis dataKey="name" stroke="#ccc" />
                  <Tooltip />
                  <Bar dataKey="value" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>

              {/* FACTORS */}
              <div className="mt-6 border-t border-gray-700 pt-4">
                <p className="font-semibold mb-2">Key Risk Factors:</p>
                <ul className="space-y-2">
                  {result.key_risk_factors.map((f, i) => (
                    <li key={i} className="text-sm text-gray-300">• {f}</li>
                  ))}
                </ul>
              </div>
            </>
          )}

        </div>

      </div>
    </div>
  );
}

