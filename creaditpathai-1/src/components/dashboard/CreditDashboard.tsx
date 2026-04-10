import { useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Download, Loader2 } from "lucide-react";
import RiskBadge from "./RiskBadge";
import { downloadPdf } from "@/lib/downloadPdf";

interface CreditDashboardProps {
  name: string;
  score: number;
  category: string;
  action: string;
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
}

function getScoreLabel(score: number) {
  if (score <= 20) return { label: "Excellent", color: "text-emerald-500" };
  if (score <= 40) return { label: "Good", color: "text-accent" };
  if (score <= 60) return { label: "Fair", color: "text-yellow-500" };
  if (score <= 80) return { label: "Poor", color: "text-orange-500" };
  return { label: "Very Poor", color: "text-destructive" };
}

function getGrade(value: number, thresholds: [number, number]) {
  if (value <= thresholds[0]) return { grade: "A", color: "text-emerald-500" };
  if (value <= thresholds[1]) return { grade: "B", color: "text-yellow-500" };
  return { grade: "C", color: "text-orange-500" };
}

const formatCurrency = (n: number) =>
  new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(n);

export default function CreditDashboard({ name, score, category, action, formData }: CreditDashboardProps) {
  const reportRef = useRef<HTMLDivElement>(null);
  const [downloading, setDownloading] = useState(false);
  const [customRecommendation, setCustomRecommendation] = useState("");

  const handleDownload = async () => {
    if (!reportRef.current) return;
    setDownloading(true);
    try {
      await downloadPdf(reportRef.current, `${name.replace(/\s+/g, "_")}_credit_report.pdf`);
    } finally {
      setDownloading(false);
    }
  };

  const income = parseFloat(formData.person_income) || 0;
  const loanAmt = parseFloat(formData.loan_amnt) || 0;
  const intRate = parseFloat(formData.loan_int_rate) || 0;
  const loanPctIncome = parseFloat(formData.loan_percent_income) || 0;
  const credHist = parseFloat(formData.cb_person_cred_hist_length) || 0;
  const empLength = parseFloat(formData.person_emp_length) || 0;
  const age = parseFloat(formData.person_age) || 0;
  const defaultOnFile = formData.cb_person_default_on_file === "Y";

  const scoreInfo = getScoreLabel(score);
  const creditScore = Math.round(900 - (score / 100) * 600);
  const creditUtilization = Math.round(loanPctIncome * 100);

  const scoreRating = getGrade(score, [30, 60]);
  const paymentGrade = getGrade(defaultOnFile ? 80 : 10, [20, 50]);
  const creditUsageGrade = getGrade(creditUtilization, [30, 60]);
  const empGrade = getGrade(empLength < 2 ? 70 : empLength < 5 ? 40 : 10, [30, 60]);
  const historyGrade = getGrade(credHist < 3 ? 70 : credHist < 7 ? 40 : 10, [30, 60]);
  const ageGrade = getGrade(age < 25 ? 60 : age < 35 ? 30 : 10, [30, 60]);

  return (
    <div className="space-y-4 animate-in fade-in duration-500">
      <div ref={reportRef} className="space-y-4 bg-background p-4 rounded-lg">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-foreground">{name}'s Credit Report</h2>
          <div className="flex items-center gap-2">
            <RiskBadge category={category} />
            <Button variant="outline" size="sm" className="gap-1.5" onClick={handleDownload} disabled={downloading}>
              {downloading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
              Download PDF
            </Button>
          </div>
        </div>

        {/* Top Row: Score + Loan Overview */}
        <div className="grid gap-4 md:grid-cols-3">
          {/* Credit Score Gauge */}
          <Card className="col-span-1">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Risk Score</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col items-center gap-2">
              <div className="relative w-32 h-32">
                <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
                  <circle cx="60" cy="60" r="50" fill="none" stroke="hsl(var(--muted))" strokeWidth="10" />
                  <circle
                    cx="60" cy="60" r="50" fill="none"
                    stroke={score <= 25 ? "hsl(160,84%,39%)" : score <= 50 ? "hsl(45,93%,47%)" : score <= 75 ? "hsl(25,95%,53%)" : "hsl(0,84%,60%)"}
                    strokeWidth="10"
                    strokeDasharray={`${(score / 100) * 314} 314`}
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className={`text-3xl font-bold font-mono ${scoreInfo.color}`}>{score}</span>
                  <span className="text-[10px] text-muted-foreground">/100</span>
                </div>
              </div>
              <p className={`text-sm font-semibold ${scoreInfo.color}`}>Risk: {scoreInfo.label}</p>
              <p className="text-xs text-muted-foreground">Credit Score Equivalent: <span className="font-mono font-bold">{creditScore}</span></p>
            </CardContent>
          </Card>

          {/* Loan Details */}
          <Card className="col-span-1">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Loan Details</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="text-center">
                <p className="text-2xl font-bold text-foreground font-mono">{formatCurrency(loanAmt)}</p>
                <p className="text-xs text-muted-foreground">Loan Amount</p>
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Interest Rate</span>
                  <span className="font-mono font-medium">{intRate}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Loan/Income</span>
                  <span className="font-mono font-medium">{(loanPctIncome * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Annual Income</span>
                  <span className="font-mono font-medium">{formatCurrency(income)}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Overview Bars */}
          <Card className="col-span-1">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Overview</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-muted-foreground">Credit Utilization</span>
                  <span className="font-mono font-medium">{creditUtilization}%</span>
                </div>
                <Progress value={creditUtilization} className="h-2" />
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-muted-foreground">Debt-to-Income</span>
                  <span className="font-mono font-medium">{(loanPctIncome * 100).toFixed(0)}%</span>
                </div>
                <Progress value={loanPctIncome * 100} className="h-2" />
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-muted-foreground">Employment Stability</span>
                  <span className="font-mono font-medium">{Math.min(100, empLength * 10)}%</span>
                </div>
                <Progress value={Math.min(100, empLength * 10)} className="h-2" />
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-muted-foreground">Credit History</span>
                  <span className="font-mono font-medium">{Math.min(100, credHist * 10)}%</span>
                </div>
                <Progress value={Math.min(100, credHist * 10)} className="h-2" />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Score Analysis Cards */}
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <Card>
            <CardContent className="pt-4 pb-3 px-4">
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-xs font-medium text-muted-foreground">Score Rating</p>
                  <p className={`text-sm font-bold ${scoreRating.color}`}>{scoreInfo.label}</p>
                </div>
                <span className={`text-xs font-bold px-2 py-0.5 rounded ${scoreRating.color} bg-muted`}>Grade {scoreRating.grade}</span>
              </div>
              <p className="text-xs text-muted-foreground mt-2">Risk score of {score}/100 based on all input factors combined.</p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-4 pb-3 px-4">
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-xs font-medium text-muted-foreground">Payment History</p>
                  <p className="text-sm font-bold text-foreground">{defaultOnFile ? "Has Default" : "No Defaults"}</p>
                </div>
                <span className={`text-xs font-bold px-2 py-0.5 rounded ${paymentGrade.color} bg-muted`}>Grade {paymentGrade.grade}</span>
              </div>
              <p className="text-xs text-muted-foreground mt-2">{defaultOnFile ? "Previous default on file significantly impacts risk." : "Clean record with no previous defaults."}</p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-4 pb-3 px-4">
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-xs font-medium text-muted-foreground">Credit Usage</p>
                  <p className="text-sm font-bold text-foreground">{creditUtilization}%</p>
                </div>
                <span className={`text-xs font-bold px-2 py-0.5 rounded ${creditUsageGrade.color} bg-muted`}>Grade {creditUsageGrade.grade}</span>
              </div>
              <p className="text-xs text-muted-foreground mt-2">Keep loan-to-income below 30% for a better score.</p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-4 pb-3 px-4">
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-xs font-medium text-muted-foreground">Employment</p>
                  <p className="text-sm font-bold text-foreground">{empLength} Years</p>
                </div>
                <span className={`text-xs font-bold px-2 py-0.5 rounded ${empGrade.color} bg-muted`}>Grade {empGrade.grade}</span>
              </div>
              <p className="text-xs text-muted-foreground mt-2">Longer employment adds stability and reduces risk.</p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-4 pb-3 px-4">
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-xs font-medium text-muted-foreground">Credit Age</p>
                  <p className="text-sm font-bold text-foreground">{credHist} Years</p>
                </div>
                <span className={`text-xs font-bold px-2 py-0.5 rounded ${historyGrade.color} bg-muted`}>Grade {historyGrade.grade}</span>
              </div>
              <p className="text-xs text-muted-foreground mt-2">A longer credit history shows responsible borrowing.</p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-4 pb-3 px-4">
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-xs font-medium text-muted-foreground">Borrower Age</p>
                  <p className="text-sm font-bold text-foreground">{age} Years</p>
                </div>
                <span className={`text-xs font-bold px-2 py-0.5 rounded ${ageGrade.color} bg-muted`}>Grade {ageGrade.grade}</span>
              </div>
              <p className="text-xs text-muted-foreground mt-2">{age < 25 ? "Younger borrowers carry higher statistical risk." : "Age factor is favorable for risk assessment."}</p>
            </CardContent>
          </Card>
        </div>

        {/* Rejection Banner for score > 80 */}
        {score > 69 && (
          <Card className="border-destructive bg-destructive/10">
            <CardContent className="pt-4 pb-3 px-4 flex items-center gap-3">
              <span className="text-2xl">🚫</span>
              <div>
                <p className="text-sm font-bold text-destructive">APPLICATION REJECTED</p>
                <p className="text-xs text-destructive/80">Risk score exceeds 80/100 — this applicant should be rejected for this loan.</p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* AI Recommendation */}
        <Card className="border-primary/30 bg-primary/5">
          <CardContent className="pt-4 pb-3 px-4">
            <p className="text-xs font-semibold text-primary mb-1">🤖 AI Recommended Action</p>
            <p className="text-sm font-medium text-foreground">{action}</p>
          </CardContent>
        </Card>

        {/* Custom Recommendation - included in PDF */}
        {customRecommendation && (
          <Card className="border-accent/30 bg-accent/5">
            <CardContent className="pt-4 pb-3 px-4">
              <p className="text-xs font-semibold text-accent-foreground mb-1">📝 Agent Recommendation</p>
              <p className="text-sm font-medium text-foreground whitespace-pre-wrap">{customRecommendation}</p>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Custom Recommendation Input - outside PDF ref when empty */}
      <Card>
        <CardContent className="pt-4 pb-3 px-4 space-y-2">
          <Label className="text-sm font-semibold">📝 Add Your Recommendation</Label>
          <Textarea
            placeholder="Type your custom recommendation for this borrower..."
            value={customRecommendation}
            onChange={(e) => setCustomRecommendation(e.target.value)}
            rows={3}
          />
          <p className="text-xs text-muted-foreground">This recommendation will appear in the downloaded PDF report.</p>
        </CardContent>
      </Card>
    </div>
  );
}
