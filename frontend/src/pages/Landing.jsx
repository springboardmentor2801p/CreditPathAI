
import { useNavigate } from "react-router-dom";

export default function Landing() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#020617] to-[#0f172a] flex flex-col items-center justify-center px-6">

      {/* Title */}
      <div className="text-center mb-16">
        <h1 className="text-5xl font-bold text-white tracking-wide">
          CreditPathAI
        </h1>
        <p className="text-gray-400 mt-3 text-lg">
          AI-powered Credit Risk Intelligence
        </p>
      </div>

      {/* Cards */}
      <div className="flex gap-10">

        {/* USER CARD */}
        <div
          onClick={() => navigate("/user")}
          className="w-80 cursor-pointer bg-[#1e293b] p-8 rounded-2xl shadow-xl border border-gray-700 hover:scale-105 hover:border-blue-400 hover:shadow-blue-500/20 transition duration-300"
        >
          <h2 className="text-2xl font-semibold text-blue-400">
            User
          </h2>
          <p className="text-gray-400 mt-4">
            Check your loan eligibility and understand your risk profile
          </p>
        </div>

        {/* BANK CARD */}
        <div
          onClick={() => navigate("/bank")}
          className="w-80 cursor-pointer bg-[#1e293b] p-8 rounded-2xl shadow-xl border border-gray-700 hover:scale-105 hover:border-green-400 hover:shadow-green-500/20 transition duration-300"
        >
          <h2 className="text-2xl font-semibold text-green-400">
            Bank / Institution
          </h2>
          <p className="text-gray-400 mt-4">
            Analyze borrower risk and make data-driven decisions
          </p>
        </div>

      </div>
    </div>
  );
}

