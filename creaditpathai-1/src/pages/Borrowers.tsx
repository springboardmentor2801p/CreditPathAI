import { useState, useMemo, useRef } from "react";
import { borrowers } from "@/data/mockData";
import { useBorrowerContext } from "@/context/BorrowerContext";
import RiskBadge from "@/components/dashboard/RiskBadge";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { Search, Eye, Phone, Mail, ArrowUpDown, Star, Download, Loader2, MessageCircle, Send } from "lucide-react";
import type { Borrower } from "@/data/mockData";
import { downloadPdf } from "@/lib/downloadPdf";
import { useToast } from "@/hooks/use-toast";

const formatCurrency = (n: number) =>
  new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(n);

export default function Borrowers() {
  const { getUserAsBorrower, userBorrower } = useBorrowerContext();
  const userEntry = getUserAsBorrower();

  const allBorrowers = useMemo(() => {
    const list = [...borrowers];
    if (userEntry) list.unshift(userEntry);
    return list;
  }, [userEntry]);

  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState("all");
  const [sortField, setSortField] = useState<"riskScore" | "outstandingBalance" | "daysPastDue">("riskScore");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [selected, setSelected] = useState<Borrower | null>(null);

  const filtered = useMemo(() => {
    let list = [...allBorrowers];
    if (search) list = list.filter((b) => b.name.toLowerCase().includes(search.toLowerCase()) || b.id.includes(search));
    if (riskFilter !== "all") list = list.filter((b) => b.riskCategory === riskFilter);
    list.sort((a, b) => (sortDir === "desc" ? b[sortField] - a[sortField] : a[sortField] - b[sortField]));
    return list;
  }, [search, riskFilter, sortField, sortDir, allBorrowers]);

  const toggleSort = (field: typeof sortField) => {
    if (sortField === field) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortField(field); setSortDir("desc"); }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Borrower Risk Table</h1>
        <p className="text-muted-foreground">All borrowers with ML-predicted risk scores and recommended actions</p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="relative w-64">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input placeholder="Search by name or ID..." className="pl-9" value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>
        <Select value={riskFilter} onValueChange={setRiskFilter}>
          <SelectTrigger className="w-40"><SelectValue placeholder="Risk Level" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Levels</SelectItem>
            <SelectItem value="Low">Low</SelectItem>
            <SelectItem value="Medium">Medium</SelectItem>
            <SelectItem value="High">High</SelectItem>
            <SelectItem value="Critical">Critical</SelectItem>
          </SelectContent>
        </Select>
        <span className="ml-auto text-sm text-muted-foreground">{filtered.length} borrowers</span>
      </div>

      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Borrower</TableHead>
                <TableHead>Loan Type</TableHead>
                <TableHead className="text-right cursor-pointer" onClick={() => toggleSort("outstandingBalance")}>
                  <span className="inline-flex items-center gap-1">Outstanding <ArrowUpDown className="h-3 w-3" /></span>
                </TableHead>
                <TableHead className="text-center cursor-pointer" onClick={() => toggleSort("riskScore")}>
                  <span className="inline-flex items-center gap-1">Risk Score <ArrowUpDown className="h-3 w-3" /></span>
                </TableHead>
                <TableHead>Risk Level</TableHead>
                <TableHead className="text-center cursor-pointer" onClick={() => toggleSort("daysPastDue")}>
                  <span className="inline-flex items-center gap-1">DPD <ArrowUpDown className="h-3 w-3" /></span>
                </TableHead>
                <TableHead>Recommended Action</TableHead>
                <TableHead className="text-center">Details</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((b) => {
                const isUser = b.id === "USR-0001";
                return (
                  <TableRow key={b.id} className={isUser ? "bg-primary/5 border-l-2 border-l-primary" : ""}>
                    <TableCell className="font-mono text-xs">
                      {isUser && <Star className="h-3 w-3 inline mr-1 text-primary" />}
                      {b.id}
                    </TableCell>
                    <TableCell className="font-medium">{b.name}</TableCell>
                    <TableCell>{b.loanType}</TableCell>
                    <TableCell className="text-right font-mono">{formatCurrency(b.outstandingBalance)}</TableCell>
                    <TableCell className="text-center font-mono font-semibold">{b.riskScore}</TableCell>
                    <TableCell><RiskBadge category={b.riskCategory} /></TableCell>
                    <TableCell className="text-center font-mono">{b.daysPastDue}</TableCell>
                    <TableCell className="text-xs text-muted-foreground max-w-[200px]">{b.recommendedAction}</TableCell>
                    <TableCell className="text-center">
                      <Button variant="ghost" size="icon" onClick={() => setSelected(b)}>
                        <Eye className="h-4 w-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <BorrowerDialog selected={selected} onClose={() => setSelected(null)} userBorrower={userBorrower} />
    </div>
  );
}

function BorrowerDialog({ selected, onClose, userBorrower }: { selected: Borrower | null; onClose: () => void; userBorrower: any }) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const [downloading, setDownloading] = useState(false);
  const [emailBody, setEmailBody] = useState("");
  const [showCompose, setShowCompose] = useState<"email" | "whatsapp" | null>(null);
  const { toast } = useToast();

  const handleDownload = async () => {
    if (!dialogRef.current) return;
    setDownloading(true);
    try {
      await downloadPdf(dialogRef.current, `${selected?.name.replace(/\s+/g, "_")}_report.pdf`);
    } finally {
      setDownloading(false);
    }
  };

  const getDefaultMessage = () => {
    if (!selected) return "";
    return `Dear ${selected.name},\n\nThis is regarding your loan account (${selected.id}) with an outstanding balance of ${formatCurrency(selected.outstandingBalance)}.\n\nRisk Level: ${selected.riskCategory}\nRecommended Action: ${selected.recommendedAction}\n\nPlease contact us at your earliest convenience to discuss repayment options.\n\nBest regards,\nCreditPathAI Recovery Team`;
  };

  const handleSendEmail = () => {
    if (!selected) return;
    const subject = encodeURIComponent(`Loan Account ${selected.id} - Payment Reminder`);
    const body = encodeURIComponent(emailBody || getDefaultMessage());
    const email = selected.email !== "N/A" ? selected.email : "";
    window.open(`mailto:${email}?subject=${subject}&body=${body}`, "_blank");
    toast({ title: "Email client opened", description: `Composing email for ${selected.name}` });
    setShowCompose(null);
    setEmailBody("");
  };

  const handleSendWhatsApp = () => {
    if (!selected) return;
    const message = encodeURIComponent(emailBody || getDefaultMessage());
    const phone = selected.phone !== "N/A" ? selected.phone.replace(/[^0-9]/g, "") : "";
    window.open(`https://wa.me/${phone}?text=${message}`, "_blank");
    toast({ title: "WhatsApp opened", description: `Sending message to ${selected.name}` });
    setShowCompose(null);
    setEmailBody("");
  };

  return (
    <Dialog open={!!selected} onOpenChange={() => { onClose(); setShowCompose(null); setEmailBody(""); }}>
      <DialogContent className="max-w-lg max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{selected?.name}</DialogTitle>
        </DialogHeader>
        {selected && (
          <div className="space-y-4">
            <div ref={dialogRef} className="space-y-4 bg-background p-2 rounded">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div><span className="text-muted-foreground">ID:</span> <span className="font-mono">{selected.id}</span></div>
                <div><span className="text-muted-foreground">Loan Type:</span> {selected.loanType}</div>
                <div><span className="text-muted-foreground">Loan Amount:</span> <span className="font-mono">{formatCurrency(selected.loanAmount)}</span></div>
                <div><span className="text-muted-foreground">Outstanding:</span> <span className="font-mono">{formatCurrency(selected.outstandingBalance)}</span></div>
                <div><span className="text-muted-foreground">Risk Score:</span> <span className="font-mono font-bold">{selected.riskScore}</span></div>
                <div><span className="text-muted-foreground">Risk Level:</span> <RiskBadge category={selected.riskCategory} /></div>
                <div><span className="text-muted-foreground">Days Past Due:</span> <span className="font-mono">{selected.daysPastDue}</span></div>
                <div><span className="text-muted-foreground">Credit Utilization:</span> <span className="font-mono">{selected.creditUtilization}%</span></div>
                <div><span className="text-muted-foreground">Repayment Velocity:</span> <span className="font-mono">{selected.repaymentVelocity}</span></div>
                <div><span className="text-muted-foreground">Last Payment:</span> {selected.lastPaymentDate}</div>
                <div><span className="text-muted-foreground">Phone:</span> <span className="font-mono">{selected.phone}</span></div>
                <div><span className="text-muted-foreground">Email:</span> <span className="font-mono text-xs">{selected.email}</span></div>
              </div>

              {selected.id === "USR-0001" && userBorrower && (
                <div className="rounded-lg bg-primary/5 border border-primary/20 p-3">
                  <p className="text-xs font-semibold text-primary mb-2">📋 Entered Details</p>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                    <span className="text-muted-foreground">Age:</span> <span className="font-mono">{userBorrower.formData.person_age} yrs</span>
                    <span className="text-muted-foreground">Income:</span> <span className="font-mono">{formatCurrency(parseFloat(userBorrower.formData.person_income) || 0)}</span>
                    <span className="text-muted-foreground">Employment:</span> <span className="font-mono">{userBorrower.formData.person_emp_length} yrs</span>
                    <span className="text-muted-foreground">Interest Rate:</span> <span className="font-mono">{userBorrower.formData.loan_int_rate}%</span>
                    <span className="text-muted-foreground">Loan/Income:</span> <span className="font-mono">{userBorrower.formData.loan_percent_income}</span>
                    <span className="text-muted-foreground">Default on File:</span> <span className="font-mono">{userBorrower.formData.cb_person_default_on_file}</span>
                    <span className="text-muted-foreground">Credit History:</span> <span className="font-mono">{userBorrower.formData.cb_person_cred_hist_length} yrs</span>
                  </div>
                </div>
              )}

              <div className="rounded-lg bg-muted p-3">
                <p className="text-xs font-semibold text-muted-foreground mb-1">AI Recommended Action</p>
                <p className="text-sm font-medium">{selected.recommendedAction}</p>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-wrap gap-2">
              <Button size="sm" variant="outline" className="gap-1.5" onClick={() => {
                const phone = selected.phone !== "N/A" ? selected.phone : "";
                window.open(`tel:${phone}`);
              }}>
                <Phone className="h-3.5 w-3.5" /> Call
              </Button>
              <Button size="sm" variant="outline" className="gap-1.5" onClick={() => {
                setEmailBody(getDefaultMessage());
                setShowCompose("email");
              }}>
                <Mail className="h-3.5 w-3.5" /> Email
              </Button>
              <Button size="sm" variant="outline" className="gap-1.5" onClick={() => {
                setEmailBody(getDefaultMessage());
                setShowCompose("whatsapp");
              }}>
                <MessageCircle className="h-3.5 w-3.5" /> WhatsApp
              </Button>
              <Button size="sm" variant="outline" className="gap-1.5 ml-auto" onClick={handleDownload} disabled={downloading}>
                {downloading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
                PDF
              </Button>
            </div>

            {/* Compose Message Area */}
            {showCompose && (
              <div className="space-y-2 rounded-lg border p-3">
                <Label className="text-sm font-semibold">
                  {showCompose === "email" ? "📧 Compose Email" : "💬 Compose WhatsApp Message"}
                </Label>
                <Textarea
                  value={emailBody}
                  onChange={(e) => setEmailBody(e.target.value)}
                  rows={6}
                  placeholder="Type your message..."
                />
                <div className="flex gap-2 justify-end">
                  <Button size="sm" variant="ghost" onClick={() => { setShowCompose(null); setEmailBody(""); }}>
                    Cancel
                  </Button>
                  <Button size="sm" className="gap-1.5" onClick={showCompose === "email" ? handleSendEmail : handleSendWhatsApp}>
                    <Send className="h-3.5 w-3.5" />
                    {showCompose === "email" ? "Open Email Client" : "Open WhatsApp"}
                  </Button>
                </div>
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
