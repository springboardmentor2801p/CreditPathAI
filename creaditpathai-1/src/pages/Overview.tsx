import { useMemo } from "react";
import {
  DollarSign, Users, AlertTriangle, TrendingUp, Target, Activity,
} from "lucide-react";
import MetricCard from "@/components/dashboard/MetricCard";
import RiskBadge from "@/components/dashboard/RiskBadge";
import { portfolioMetrics, borrowers, monthlyTrend, riskDistribution } from "@/data/mockData";
import { useBorrowerContext } from "@/context/BorrowerContext";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from "recharts";

const formatCurrency = (n: number) =>
  new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(n);

export default function Overview() {
  const { getUserAsBorrower } = useBorrowerContext();
  const userEntry = getUserAsBorrower();

  const allBorrowers = useMemo(() => {
    const list = [...borrowers];
    if (userEntry) list.unshift(userEntry);
    return list;
  }, [userEntry]);

  const topRisk = useMemo(() =>
    [...allBorrowers].sort((a, b) => b.riskScore - a.riskScore).slice(0, 5),
    [allBorrowers]
  );

  const metrics = useMemo(() => ({
    ...portfolioMetrics,
    totalLoans: allBorrowers.length,
    totalBorrowers: allBorrowers.length,
    totalOutstanding: allBorrowers.reduce((s, b) => s + b.outstandingBalance, 0),
    avgRiskScore: parseFloat((allBorrowers.reduce((s, b) => s + b.riskScore, 0) / allBorrowers.length).toFixed(1)),
    criticalAccounts: allBorrowers.filter(b => b.riskCategory === "Critical").length,
  }), [allBorrowers]);

  const updatedRiskDist = useMemo(() => [
    { category: "Low", count: allBorrowers.filter(b => b.riskCategory === "Low").length, color: "#10b981" },
    { category: "Medium", count: allBorrowers.filter(b => b.riskCategory === "Medium").length, color: "#f59e0b" },
    { category: "High", count: allBorrowers.filter(b => b.riskCategory === "High").length, color: "#f97316" },
    { category: "Critical", count: allBorrowers.filter(b => b.riskCategory === "Critical").length, color: "#ef4444" },
  ], [allBorrowers]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Portfolio Overview</h1>
        <p className="text-muted-foreground">
          Monitor loan recovery performance and borrower risk
          {userEntry && <span className="text-primary ml-2">· Your data is included</span>}
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        <MetricCard title="Total Loans" value={String(metrics.totalLoans)} icon={Users} change="+3 this month" changeType="positive" />
        <MetricCard title="Outstanding" value={formatCurrency(metrics.totalOutstanding)} icon={DollarSign} iconColor="#f59e0b" change="-4.2% vs last month" changeType="positive" />
        <MetricCard title="Delinquency Rate" value={`${metrics.delinquencyRate}%`} icon={AlertTriangle} iconColor="#ef4444" change="+1.2%" changeType="negative" />
        <MetricCard title="Recovery Rate" value={`${metrics.recoveryRate}%`} icon={TrendingUp} iconColor="#10b981" change="+5.1% vs target" changeType="positive" />
        <MetricCard title="Avg Risk Score" value={String(metrics.avgRiskScore)} icon={Activity} iconColor="#8b5cf6" />
        <MetricCard title="Critical Accounts" value={String(metrics.criticalAccounts)} icon={Target} iconColor="#ef4444" change="Needs attention" changeType="negative" />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle className="text-base">Monthly Recovery vs Target (₹)</CardTitle></CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={monthlyTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="month" tick={{ fontSize: 12 }} stroke="hsl(var(--muted-foreground))" />
                <YAxis tick={{ fontSize: 12 }} stroke="hsl(var(--muted-foreground))" tickFormatter={(v) => `${(v / 1e6).toFixed(1)}M`} />
                <Tooltip formatter={(v: number) => formatCurrency(v)} />
                <Bar dataKey="recovered" name="Recovered" fill="hsl(199, 89%, 48%)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="target" name="Target" fill="hsl(160, 84%, 39%)" radius={[4, 4, 0, 0]} opacity={0.5} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle className="text-base">Risk Distribution</CardTitle></CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={updatedRiskDist} dataKey="count" nameKey="category" cx="50%" cy="50%" outerRadius={90} label={({ category, count }) => `${category}: ${count}`}>
                  {updatedRiskDist.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader><CardTitle className="text-base">Top Risk Borrowers</CardTitle></CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Borrower</TableHead>
                <TableHead>Loan Type</TableHead>
                <TableHead className="text-right">Outstanding</TableHead>
                <TableHead className="text-center">Risk Score</TableHead>
                <TableHead>Risk Level</TableHead>
                <TableHead>Days Past Due</TableHead>
                <TableHead>Recommended Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {topRisk.map((b) => (
                <TableRow key={b.id} className={b.id === "USR-0001" ? "bg-primary/5 border-l-2 border-l-primary" : ""}>
                  <TableCell className="font-medium">
                    {b.id === "USR-0001" && <span className="text-primary mr-1">★</span>}
                    {b.name}
                  </TableCell>
                  <TableCell>{b.loanType}</TableCell>
                  <TableCell className="text-right font-mono">{formatCurrency(b.outstandingBalance)}</TableCell>
                  <TableCell className="text-center font-mono font-semibold">{b.riskScore}</TableCell>
                  <TableCell><RiskBadge category={b.riskCategory} /></TableCell>
                  <TableCell className="font-mono">{b.daysPastDue}</TableCell>
                  <TableCell className="text-xs text-muted-foreground">{b.recommendedAction}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
