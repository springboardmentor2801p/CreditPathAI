import { createContext, useContext, useState, type ReactNode } from "react";
import type { Borrower } from "@/data/mockData";

interface UserEnteredBorrower {
  formData: {
    person_age: string;
    person_income: string;
    person_emp_length: string;
    loan_amnt: string;
    loan_int_rate: string;
    loan_percent_income: string;
    cb_person_default_on_file: string;
    cb_person_cred_hist_length: string;
  };
  result: {
    score: number;
    category: string;
    action: string;
  };
  role: string;
  name: string;
}

interface BorrowerContextType {
  userBorrower: UserEnteredBorrower | null;
  setUserBorrower: (b: UserEnteredBorrower) => void;
  getUserAsBorrower: () => Borrower | null;
}

const BorrowerContext = createContext<BorrowerContextType | null>(null);

export function BorrowerProvider({ children }: { children: ReactNode }) {
  const [userBorrower, setUserBorrower] = useState<UserEnteredBorrower | null>(null);

  const getUserAsBorrower = (): Borrower | null => {
    if (!userBorrower) return null;
    const fd = userBorrower.formData;
    const loanAmount = parseFloat(fd.loan_amnt) || 0;
    const income = parseFloat(fd.person_income) || 0;
    return {
      id: "USR-0001",
      name: userBorrower.name || (userBorrower.role === "bank" ? "Assessed Borrower" : "You (Self-Check)"),
      loanAmount,
      outstandingBalance: loanAmount,
      riskScore: userBorrower.result.score,
      riskCategory: userBorrower.result.category as Borrower["riskCategory"],
      daysPastDue: 0,
      creditUtilization: Math.round(parseFloat(fd.loan_percent_income) * 100) || 0,
      repaymentVelocity: 0.5,
      recommendedAction: userBorrower.result.action,
      loanType: "Personal",
      lastPaymentDate: new Date().toISOString().split("T")[0],
      phone: "N/A",
      email: "N/A",
    };
  };

  return (
    <BorrowerContext.Provider value={{ userBorrower, setUserBorrower, getUserAsBorrower }}>
      {children}
    </BorrowerContext.Provider>
  );
}

export function useBorrowerContext() {
  const ctx = useContext(BorrowerContext);
  if (!ctx) throw new Error("useBorrowerContext must be used within BorrowerProvider");
  return ctx;
}
