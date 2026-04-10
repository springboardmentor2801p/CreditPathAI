import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { borrowers, monthlyTrend, riskDistribution, loanTypeBreakdown } from "@/data/mockData";
import { useBorrowerContext } from "@/context/BorrowerContext";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
  AreaChart, Area, ScatterChart, Scatter, ZAxis,
} from "recharts";

const formatCurrency = (n: number) =>
  new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(n);

const COLORS = ["#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

export default function Analytics() {
  const { getUserAsBorrower } = useBorrowerContext();
  const userEntry = getUserAsBorrower();

  const allBorrowers = useMemo(() => {
    const list = [...borrowers];
    if (userEntry) list.unshift(userEntry);
    return list;
  }, [userEntry]);

  const scatterData = allBorrowers.map((b) => ({
    riskScore: b.riskScore,
    outstanding: b.outstandingBalance,
    dpd: b.daysPastDue,
    name: b.name,
    isUser: b.id === "USR-0001",
  }));

  const dpdBuckets = [
    { range: "0", count: allBorrowers.filter((b) => b.daysPastDue === 0).length },
    { range: "1-30", count: allBorrowers.filter((b) => b.daysPastDue >= 1 && b.daysPastDue <= 30).length },
    { range: "31-60", count: allBorrowers.filter((b) => b.daysPastDue >= 31 && b.daysPastDue <= 60).length },
    { range: "61-90", count: allBorrowers.filter((b) => b.daysPastDue >= 61 && b.daysPastDue <= 90).length },
    { range: "90+", count: allBorrowers.filter((b) => b.daysPastDue > 90).length },
  ];

  const updatedRiskDist = [
    { category: "Low", count: allBorrowers.filter((b) => b.riskCategory === "Low").length, color: "#10b981" },
    { category: "Medium", count: allBorrowers.filter((b) => b.riskCategory === "Medium").length, color: "#f59e0b" },
    { category: "High", count: allBorrowers.filter((b) => b.riskCategory === "High").length, color: "#f97316" },
    { category: "Critical", count: allBorrowers.filter((b) => b.riskCategory === "Critical").length, color: "#ef4444" },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Analytics & Reports</h1>
        <p className="text-muted-foreground">
          Deep dive into portfolio performance and model insights
          {userEntry && <span className="text-primary ml-2">· Includes your entered data</span>}
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle className="text-base">Recovery Trend (₹)</CardTitle></CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={monthlyTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="month" stroke="hsl(var(--muted-foreground))" tick={{ fontSize: 12 }} />
                <YAxis stroke="hsl(var(--muted-foreground))" tick={{ fontSize: 12 }} tickFormatter={(v) => `${(v / 1e6).toFixed(1)}M`} />
                <Tooltip formatter={(v: number) => formatCurrency(v)} />
                <Area type="monotone" dataKey="recovered" name="Recovered" stroke="#0ea5e9" fill="#0ea5e9" fillOpacity={0.2} />
                <Area type="monotone" dataKey="delinquent" name="Delinquent" stroke="#ef4444" fill="#ef4444" fillOpacity={0.1} />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle className="text-base">Loan Type Distribution</CardTitle></CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={loanTypeBreakdown} dataKey="count" nameKey="type" cx="50%" cy="50%" outerRadius={90} label={({ type, count }) => `${type}: ${count}`}>
                  {loanTypeBreakdown.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle className="text-base">Days Past Due Distribution</CardTitle></CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={dpdBuckets}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="range" stroke="hsl(var(--muted-foreground))" tick={{ fontSize: 12 }} />
                <YAxis stroke="hsl(var(--muted-foreground))" tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" name="Borrowers" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle className="text-base">Risk Score vs Outstanding Balance</CardTitle></CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="riskScore" name="Risk Score" stroke="hsl(var(--muted-foreground))" tick={{ fontSize: 12 }} />
                <YAxis dataKey="outstanding" name="Outstanding" stroke="hsl(var(--muted-foreground))" tick={{ fontSize: 12 }} tickFormatter={(v) => `${(v / 1e5).toFixed(0)}L`} />
                <ZAxis dataKey="dpd" range={[30, 300]} name="DPD" />
                <Tooltip formatter={(v: number, name: string) => name === "Outstanding" ? formatCurrency(v) : v} />
                <Scatter data={scatterData} fill="#8b5cf6" fillOpacity={0.7} />
              </ScatterChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader><CardTitle className="text-base">Model Performance — Risk Category Distribution</CardTitle></CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-4">
              {updatedRiskDist.map((r) => (
                <div key={r.category} className="rounded-lg border p-4 text-center">
                  <div className="text-3xl font-bold" style={{ color: r.color }}>{r.count}</div>
                  <div className="text-sm text-muted-foreground mt-1">{r.category} Risk</div>
                  <div className="text-xs text-muted-foreground">{((r.count / allBorrowers.length) * 100).toFixed(0)}% of portfolio</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
