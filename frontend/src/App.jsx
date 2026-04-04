import { useState } from "react";
import LandingPage from "./pages/LandingPage";
import ApplicantPortal from "./pages/ApplicantPortal";
import BankPortal from "./pages/BankPortal";

export default function App() {
  const [page, setPage] = useState("landing");

  if (page === "applicant") return <ApplicantPortal onBack={() => setPage("landing")} />;
  if (page === "bank") return <BankPortal onBack={() => setPage("landing")} />;
  return <LandingPage onSelectApplicant={() => setPage("applicant")} onSelectBank={() => setPage("bank")} />;
}