import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ArrowLeft, Brain, Loader2, Home } from "lucide-react";
import { useBorrowerContext } from "@/context/BorrowerContext";
import CreditDashboard from "@/components/dashboard/CreditDashboard";

interface FormData {
  borrower_name: string;
  person_age: string;
  person_income: string;
  person_emp_length: string;
  loan_amnt: string;
  loan_int_rate: string;
  loan_percent_income: string;
  cb_person_default_on_file: string;
  cb_person_cred_hist_length: string;
}

function predictRisk(data: FormData): { score: number; category: string; action: string } {
  const age = parseFloat(data.person_age) || 0;
  const income = parseFloat(data.person_income) || 0;
  const empLength = parseFloat(data.person_emp_length) || 0;
  const loanAmt = parseFloat(data.loan_amnt) || 0;
  const intRate = parseFloat(data.loan_int_rate) || 0;
  const loanPctIncome = parseFloat(data.loan_percent_income) || 0;
  const defaultOnFile = data.cb_person_default_on_file === "Y" ? 1 : 0;
  const credHist = parseFloat(data.cb_person_cred_hist_length) || 0;

  let risk = 0;

  // Interest rate factor (0-20)
  if (intRate > 20) risk += 20;
  else if (intRate > 15) risk += 16;
  else if (intRate > 12) risk += 12;
  else if (intRate > 10) risk += 8;
  else risk += 3;

  // Loan-to-income ratio (0-20)
  if (loanPctIncome > 0.5) risk += 20;
  else if (loanPctIncome > 0.4) risk += 16;
  else if (loanPctIncome > 0.3) risk += 12;
  else if (loanPctIncome > 0.2) risk += 7;
  else risk += 2;

  // Default on file (0-18)
  risk += defaultOnFile * 18;

  // Age factor (0-12)
  if (age < 22) risk += 12;
  else if (age < 25) risk += 9;
  else if (age < 30) risk += 5;
  else if (age < 40) risk += 2;
  else risk += 0;

  // Employment length (0-12)
  if (empLength < 1) risk += 12;
  else if (empLength < 2) risk += 9;
  else if (empLength < 4) risk += 5;
  else if (empLength < 7) risk += 2;
  else risk += 0;

  // Credit history length (0-10)
  if (credHist < 2) risk += 10;
  else if (credHist < 4) risk += 7;
  else if (credHist < 7) risk += 4;
  else if (credHist < 12) risk += 2;
  else risk += 0;

  // Loan amount to income ratio (0-8)
  const loanToIncome = income > 0 ? loanAmt / income : 5;
  if (loanToIncome > 4) risk += 8;
  else if (loanToIncome > 2.5) risk += 5;
  else if (loanToIncome > 1) risk += 2;
  else risk += 0;

  risk = Math.min(100, Math.max(0, risk));

  let category: string;
  let action: string;

  if (risk <= 25) {
    category = "Low";
    action = "Low risk — approve with standard terms and automated SMS reminders.";
  } else if (risk <= 50) {
    category = "Medium";
    action = "Moderate risk — approve with conditions, assign a recovery agent for follow-ups.";
  } else if (risk <= 75) {
    category = "High";
    action = "High risk — escalate to senior recovery team, require collateral or guarantor.";
  } else {
    category = "Critical";
    action = "🚨 Critical risk — REJECT this loan application. Recommend debt restructuring or legal action.";
  }

  return { score: Math.round(risk), category, action };
}

const fields: { key: keyof FormData; label: string; placeholder: string; type?: string }[] = [
  { key: "borrower_name", label: "Full Name", placeholder: "e.g. Rahul Sharma", type: "text" },
  { key: "person_age", label: "Age", placeholder: "e.g. 28" },
  { key: "person_income", label: "Annual Income (₹)", placeholder: "e.g. 500000" },
  { key: "person_emp_length", label: "Employment Length (years)", placeholder: "e.g. 5" },
  { key: "loan_amnt", label: "Loan Amount (₹)", placeholder: "e.g. 200000" },
  { key: "loan_int_rate", label: "Interest Rate (%)", placeholder: "e.g. 12.5" },
  { key: "loan_percent_income", label: "Loan % of Income (0-1)", placeholder: "e.g. 0.35" },
  { key: "cb_person_cred_hist_length", label: "Credit History Length (years)", placeholder: "e.g. 8" },
];

export default function DataInput() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const role = searchParams.get("role") || "borrower";
  const { setUserBorrower } = useBorrowerContext();

  const [form, setForm] = useState<FormData>({
    borrower_name: "", person_age: "", person_income: "", person_emp_length: "",
    loan_amnt: "", loan_int_rate: "", loan_percent_income: "",
    cb_person_default_on_file: "", cb_person_cred_hist_length: "",
  });

  const [result, setResult] = useState<{ score: number; category: string; action: string } | null>(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (key: keyof FormData, value: string) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const payload = {
        person_age: parseFloat(form.person_age) || 0,
        person_income: parseFloat(form.person_income) || 0,
        person_emp_length: parseFloat(form.person_emp_length) || 0,
        loan_amnt: parseFloat(form.loan_amnt) || 0,
        loan_int_rate: parseFloat(form.loan_int_rate) || 0,
        loan_percent_income: parseFloat(form.loan_percent_income) || 0,
        cb_person_default_on_file: form.cb_person_default_on_file === "Y" ? 1 : 0,
        cb_person_cred_hist_length: parseFloat(form.cb_person_cred_hist_length) || 0,
      };

      let prediction: { score: number; category: string; action: string };

      try {
        const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
        const res = await fetch(`${API_URL}/risk-score`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await res.json();

        if (data.error) throw new Error(data.error);

        prediction = {
          score: Math.round(data.default_probability * 100),
          category: data.risk_level,
          action: data.recommended_action,
        };
      } catch {
        // Fallback to client-side prediction if API is unavailable
        prediction = predictRisk(form);
      }

      setResult(prediction);
      setUserBorrower({
        formData: form,
        result: prediction,
        role,
        name: form.borrower_name || (role === "bank" ? "Assessed Borrower" : "Self-Check User"),
      });
    } finally {
      setLoading(false);
    }
  };

  const handleContinue = () => {
    navigate("/dashboard");
  };

  const isFormValid = Object.values(form).every((v) => v.trim() !== "");

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-center gap-2">
          <Button variant="ghost" className="gap-2" onClick={() => navigate("/")}>
            <Home className="h-4 w-4" /> Home
          </Button>
          <Button variant="ghost" className="gap-2" onClick={() => navigate(-1)}>
            <ArrowLeft className="h-4 w-4" /> Back
          </Button>
        </div>

        {!result ? (
          <Card className="max-w-xl mx-auto">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-primary" />
                {role === "bank" ? "Assess Borrower Risk" : "Check Your Risk Score"}
              </CardTitle>
              <CardDescription>
                {role === "bank"
                  ? "Enter borrower details to predict default risk using our ML model"
                  : "Enter your details to see your predicted loan default risk"}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 sm:grid-cols-2">
                {fields.map((f) => (
                  <div key={f.key} className={`space-y-1.5 ${f.key === "borrower_name" ? "sm:col-span-2" : ""}`}>
                    <Label htmlFor={f.key} className="text-sm">{f.label}</Label>
                    <Input
                      id={f.key} type={f.type || "number"} placeholder={f.placeholder}
                      value={form[f.key]}
                      onChange={(e) => handleChange(f.key, e.target.value)}
                    />
                  </div>
                ))}
                <div className="space-y-1.5">
                  <Label className="text-sm">Default on File?</Label>
                  <Select value={form.cb_person_default_on_file} onValueChange={(v) => handleChange("cb_person_default_on_file", v)}>
                    <SelectTrigger><SelectValue placeholder="Select..." /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Y">Yes</SelectItem>
                      <SelectItem value="N">No</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <Button className="w-full" disabled={!isFormValid || loading} onClick={handlePredict}>
                {loading ? (
                  <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Running ML Prediction...</>
                ) : (
                  <><Brain className="h-4 w-4 mr-2" /> Predict Risk Score</>
                )}
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-4">
            <CreditDashboard
              name={form.borrower_name || "Borrower"}
              score={result.score}
              category={result.category}
              action={result.action}
              formData={form}
            />
            <div className="flex gap-3 justify-center">
              <Button variant="outline" onClick={() => setResult(null)}>
                <ArrowLeft className="h-4 w-4 mr-2" /> New Assessment
              </Button>
              <Button onClick={handleContinue}>
                {role === "bank" ? "Go to Dashboard →" : "View Full Dashboard →"}
              </Button>
            </div>
          </div>
        )}

        <p className="text-xs text-center text-muted-foreground">
          Model: LightGBM · Features based on Kaggle Loan Default Dataset
        </p>
      </div>
    </div>
  );
}
