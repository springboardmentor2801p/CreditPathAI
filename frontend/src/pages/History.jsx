
import { useEffect, useState } from "react";

export default function History() {

  const [reports, setReports] = useState([]);

  useEffect(() => {
    const saved = JSON.parse(localStorage.getItem("reports")) || [];
    setReports(saved.reverse());
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#020617] via-[#020617] to-[#0f172a] text-white px-10 py-8">

      {/* 🔥 HEADER */}
      <div className="flex justify-between items-center mb-12">

        <div>
          <h1 className="text-3xl font-bold">
            Borrower History
          </h1>
          <p className="text-sm text-gray-400 mt-1">
            Previously analyzed loan profiles
          </p>
        </div>

        {/* 🔙 Back */}
        <button
          onClick={() => window.location.href = "/bank"}
          className="bg-gray-700 px-4 py-2 rounded-lg hover:bg-gray-600 transition"
        >
          ⬅ Back
        </button>

      </div>

      {/* CONTENT */}
      {reports.length === 0 ? (
        <p className="text-gray-400 text-center mt-20">
          No reports saved yet
        </p>
      ) : (
        <div className="max-w-6xl mx-auto space-y-6">

          {reports.map((r, index) => (
            <div
              key={index}
              className="bg-white/5 backdrop-blur-xl p-6 rounded-xl border border-white/10 hover:shadow-lg hover:shadow-blue-500/10 transition"
            >

              {/* TOP ROW */}
              <div className="flex justify-between items-center mb-4">

                <p className="text-sm text-gray-400">
                  {r.date}
                </p>

                <span className={`px-4 py-1 rounded-full text-sm font-semibold
                  ${r.risk_level === "Low" && "bg-green-500/20 text-green-400"}
                  ${r.risk_level === "Medium" && "bg-yellow-500/20 text-yellow-400"}
                  ${r.risk_level === "High" && "bg-red-500/20 text-red-400"}
                `}>
                  {r.risk_level} Risk
                </span>

              </div>

              {/* GRID DATA */}
              <div className="grid grid-cols-3 gap-4 text-sm mb-4">

                <div>
                  <p className="text-gray-400">Income</p>
                  <p>₹{r.income}</p>
                </div>

                <div>
                  <p className="text-gray-400">Loan Amount</p>
                  <p>₹{r.loan_amount}</p>
                </div>

                <div>
                  <p className="text-gray-400">Credit Score</p>
                  <p>{r.credit_score}</p>
                </div>

              </div>

              {/* METRICS */}
              <div className="flex justify-between mb-4">

                <div>
                  <p className="text-blue-400 font-bold text-lg">
                    {Math.round(r.default_probability * 100)}%
                  </p>
                  <p className="text-xs text-gray-400">
                    Default Probability
                  </p>
                </div>

                <div>
                  <p className="text-red-400 font-semibold">
                    ₹{r.expected_loss}
                  </p>
                  <p className="text-xs text-gray-400">
                    Expected Loss
                  </p>
                </div>

              </div>

              {/* STRATEGY */}
              <div className="bg-black/30 p-4 rounded-lg border border-gray-700 text-sm leading-relaxed">
                {r.recommended_strategy}
              </div>

            </div>
          ))}

        </div>
      )}
    </div>
  );
}

