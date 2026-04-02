import { useState } from "react";
import GaugeChart from "react-gauge-chart";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";

export default function BankDashboard() {

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

  const saveReport = () => {
    const reports = JSON.parse(localStorage.getItem("reports")) || [];
    reports.push({
      ...form,
      ...result,
      date: new Date().toLocaleString()
    });
    localStorage.setItem("reports", JSON.stringify(reports));
    alert("Report Saved ✅");
  };

  const exportPDF = async () => {
    const element = document.getElementById("report-section");
    const canvas = await html2canvas(element);
    const imgData = canvas.toDataURL("image/png");

    const pdf = new jsPDF();
    pdf.addImage(imgData, "PNG", 10, 10, 180, 0);
    pdf.save("loan-report.pdf");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#020617] via-[#020617] to-[#0f172a] text-white px-10 py-8">

      {/* HEADER */}
      <div className="flex justify-between items-center mb-12">

        <div>
          <h1 className="text-3xl font-bold">
            Bank Risk Dashboard
          </h1>
          <p className="text-sm text-gray-400 mt-1">
            AI-driven lending decisions
          </p>
        </div>

        <div className="flex gap-3">

          <button
            onClick={() => window.location.href = "/"}
            className="bg-gray-700 px-4 py-2 rounded-lg hover:bg-gray-600 transition"
          >
            ⬅ Back
          </button>

          <button
            onClick={() => window.location.href = "/history"}
            className="bg-gray-700 px-4 py-2 rounded-lg hover:bg-gray-600 transition"
          >
            📊 History
          </button>

        </div>

      </div>

      <div className="max-w-6xl mx-auto grid grid-cols-2 gap-10">

        {/* INPUT */}
        <div className="bg-white/5 backdrop-blur-xl p-8 rounded-2xl shadow-xl border border-white/10">

          <h2 className="text-lg mb-6 text-blue-400 font-semibold">
            Borrower Profile
          </h2>

          <div className="space-y-5">

            <input name="income" placeholder="Income"
              onChange={handleChange}
              className="w-full p-3 rounded-lg bg-black/30 border border-gray-700 focus:outline-none focus:border-blue-400 transition" />

            <input name="loan_amount" placeholder="Loan Amount"
              onChange={handleChange}
              className="w-full p-3 rounded-lg bg-black/30 border border-gray-700 focus:outline-none focus:border-blue-400 transition" />

            <input name="credit_score" placeholder="Credit Score"
              onChange={handleChange}
              className="w-full p-3 rounded-lg bg-black/30 border border-gray-700 focus:outline-none focus:border-blue-400 transition" />

            <input name="ltv" placeholder="LTV"
              onChange={handleChange}
              className="w-full p-3 rounded-lg bg-black/30 border border-gray-700 focus:outline-none focus:border-blue-400 transition" />

            <input name="dtir1" placeholder="DTI"
              onChange={handleChange}
              className="w-full p-3 rounded-lg bg-black/30 border border-gray-700 focus:outline-none focus:border-blue-400 transition" />

            {/* 🔥 BUTTON WITH LOADING */}
            <button
              onClick={handleSubmit}
              disabled={loading}
              className={`w-full p-3 rounded-lg font-semibold transition
                ${loading
                  ? "bg-gray-600 cursor-not-allowed"
                  : "bg-gradient-to-r from-blue-500 to-blue-700 hover:scale-105 hover:shadow-blue-500/30"
                }
              `}
            >
              {loading ? "Evaluating..." : "Evaluate Borrower"}
            </button>

          </div>
        </div>

        {/* OUTPUT */}
        <div id="report-section" className="bg-white/5 backdrop-blur-xl p-8 rounded-2xl shadow-xl border border-white/10">

          <h2 className="text-lg mb-6 text-green-400 font-semibold">
            Decision Insights
          </h2>

          {/* 🔥 LOADING SPINNER */}
          {loading && (
            <div className="flex justify-center mt-10">
              <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            </div>
          )}

          {!loading && !result && (
            <p className="text-gray-400">Run evaluation to see insights</p>
          )}

          {!loading && result && (
            <>
              {/* GAUGE */}
              <div className="flex justify-center mb-8">
                <GaugeChart
                  id="gauge-chart-bank"
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

              {/* BUTTONS */}
              <div className="mt-4 flex gap-3 flex-wrap">

                {(result.risk_level === "Low" || result.risk_level === "Medium") ? (
                  <button className="bg-green-500 hover:bg-green-600 px-5 py-2 rounded-lg font-semibold shadow">
                    ✅ Approve
                  </button>
                ) : (
                  <button className="bg-red-500 hover:bg-red-600 px-5 py-2 rounded-lg font-semibold shadow">
                    ❌ Reject
                  </button>
                )}

                <button
                  onClick={saveReport}
                  className="bg-blue-500 hover:bg-blue-600 px-5 py-2 rounded-lg shadow"
                >
                  💾 Save
                </button>

                <button
                  onClick={exportPDF}
                  className="bg-purple-500 hover:bg-purple-600 px-5 py-2 rounded-lg shadow"
                >
                  📄 Export
                </button>

              </div>

              {/* STRATEGY */}
              <div className={`mt-6 p-5 rounded-xl border text-sm
                ${result.risk_level === "Low" && "bg-green-500/10 border-green-500"}
                ${result.risk_level === "Medium" && "bg-yellow-500/10 border-yellow-500"}
                ${result.risk_level === "High" && "bg-red-500/10 border-red-500"}
              `}>
                <p className="text-gray-400 mb-2">Recommended Strategy</p>
                <p>{result.recommended_strategy}</p>
              </div>

              {/* FACTORS */}
              <div className="mt-6 border-t border-gray-700 pt-4">
                <p className="font-semibold mb-2">Key Risk Drivers:</p>
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