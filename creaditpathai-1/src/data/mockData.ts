export interface Borrower {
  id: string;
  name: string;
  loanAmount: number;
  outstandingBalance: number;
  riskScore: number;
  riskCategory: 'Low' | 'Medium' | 'High' | 'Critical';
  daysPastDue: number;
  creditUtilization: number;
  repaymentVelocity: number;
  recommendedAction: string;
  loanType: string;
  lastPaymentDate: string;
  phone: string;
  email: string;
}

export interface PortfolioMetrics {
  totalLoans: number;
  totalOutstanding: number;
  delinquencyRate: number;
  recoveryRate: number;
  avgRiskScore: number;
  totalBorrowers: number;
  criticalAccounts: number;
  monthlyRecovery: number;
}

const riskActions: Record<string, string> = {
  Low: 'Automated reminder via SMS',
  Medium: 'Personal call from agent',
  High: 'Escalate to senior recovery team',
  Critical: 'Legal notice + restructuring offer',
};

const names = [
  'Aarav Sharma', 'Priya Patel', 'Rohit Kumar', 'Sneha Reddy', 'Vikram Singh',
  'Anjali Gupta', 'Rahul Verma', 'Deepika Nair', 'Arjun Mehta', 'Kavitha Iyer',
  'Suresh Rao', 'Meera Joshi', 'Aditya Das', 'Pooja Menon', 'Karthik Pillai',
  'Divya Chandra', 'Nikhil Saxena', 'Ritu Banerjee', 'Sanjay Malhotra', 'Anita Kulkarni',
  'Manish Tiwari', 'Swati Dubey', 'Rajesh Agarwal', 'Neha Choudhary', 'Amit Pandey',
];

const loanTypes = ['Personal', 'Home', 'Auto', 'Business', 'Education'];

function getRiskCategory(score: number): Borrower['riskCategory'] {
  if (score <= 25) return 'Low';
  if (score <= 50) return 'Medium';
  if (score <= 75) return 'High';
  return 'Critical';
}

export const borrowers: Borrower[] = names.map((name, i) => {
  const riskScore = Math.round(Math.random() * 100);
  const riskCategory = getRiskCategory(riskScore);
  const loanAmount = Math.round((50000 + Math.random() * 950000) / 1000) * 1000;
  const outstandingBalance = Math.round(loanAmount * (0.2 + Math.random() * 0.7));
  return {
    id: `BRW-${String(i + 1).padStart(4, '0')}`,
    name,
    loanAmount,
    outstandingBalance,
    riskScore,
    riskCategory,
    daysPastDue: riskCategory === 'Low' ? 0 : Math.round(Math.random() * (riskScore * 2)),
    creditUtilization: Math.round(30 + Math.random() * 60),
    repaymentVelocity: parseFloat((0.3 + Math.random() * 0.7).toFixed(2)),
    recommendedAction: riskActions[riskCategory],
    loanType: loanTypes[Math.floor(Math.random() * loanTypes.length)],
    lastPaymentDate: new Date(2024, Math.floor(Math.random() * 12), Math.floor(1 + Math.random() * 28)).toISOString().split('T')[0],
    phone: `+91 ${Math.floor(7000000000 + Math.random() * 3000000000)}`,
    email: `${name.toLowerCase().replace(/\s/g, '.')}@email.com`,
  };
});

export const portfolioMetrics: PortfolioMetrics = {
  totalLoans: borrowers.length,
  totalOutstanding: borrowers.reduce((s, b) => s + b.outstandingBalance, 0),
  delinquencyRate: parseFloat(((borrowers.filter(b => b.daysPastDue > 0).length / borrowers.length) * 100).toFixed(1)),
  recoveryRate: 67.3,
  avgRiskScore: parseFloat((borrowers.reduce((s, b) => s + b.riskScore, 0) / borrowers.length).toFixed(1)),
  totalBorrowers: borrowers.length,
  criticalAccounts: borrowers.filter(b => b.riskCategory === 'Critical').length,
  monthlyRecovery: 2340000,
};

export const monthlyTrend = [
  { month: 'Jan', recovered: 1800000, target: 2200000, delinquent: 5200000 },
  { month: 'Feb', recovered: 1950000, target: 2200000, delinquent: 5100000 },
  { month: 'Mar', recovered: 2100000, target: 2300000, delinquent: 4900000 },
  { month: 'Apr', recovered: 2250000, target: 2300000, delinquent: 4700000 },
  { month: 'May', recovered: 2340000, target: 2400000, delinquent: 4500000 },
  { month: 'Jun', recovered: 2400000, target: 2400000, delinquent: 4300000 },
];

export const riskDistribution = [
  { category: 'Low', count: borrowers.filter(b => b.riskCategory === 'Low').length, color: '#10b981' },
  { category: 'Medium', count: borrowers.filter(b => b.riskCategory === 'Medium').length, color: '#f59e0b' },
  { category: 'High', count: borrowers.filter(b => b.riskCategory === 'High').length, color: '#f97316' },
  { category: 'Critical', count: borrowers.filter(b => b.riskCategory === 'Critical').length, color: '#ef4444' },
];

export const loanTypeBreakdown = loanTypes.map(type => ({
  type,
  count: borrowers.filter(b => b.loanType === type).length,
  totalAmount: borrowers.filter(b => b.loanType === type).reduce((s, b) => s + b.loanAmount, 0),
}));
