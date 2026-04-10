import { useNavigate } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { User, Building2, Shield } from "lucide-react";

export default function RoleSelection() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-6">
      <div className="w-full max-w-2xl space-y-8 text-center">
        {/* Logo / Title */}
        <div className="space-y-3">
          <div className="flex items-center justify-center gap-2">
            <Shield className="h-10 w-10 text-primary" />
            <h1 className="text-4xl font-bold tracking-tight text-foreground">
              CreditPath<span className="text-primary">AI</span>
            </h1>
          </div>
          <p className="text-muted-foreground text-lg">
            – AI-Based Credit Risk Prediction  &amp; Recommendation System
          </p>
        </div>

        {/* Role Cards */}
        <div className="grid gap-6 sm:grid-cols-2">
          <Card
            className="cursor-pointer border-2 border-transparent hover:border-primary transition-all duration-200 hover:shadow-lg group"
            onClick={() => navigate("/input?role=borrower")}
          >
            <CardContent className="flex flex-col items-center gap-4 p-8">
              <div className="rounded-full bg-primary/10 p-4 group-hover:bg-primary/20 transition-colors">
                <User className="h-10 w-10 text-primary" />
              </div>
              <div className="space-y-1">
                <h2 className="text-xl font-semibold text-foreground">Borrower / User</h2>
                <p className="text-sm text-muted-foreground">
                  Check your loan default risk score and get personalized recommendations
                </p>
              </div>
              <Button variant="outline" className="mt-2 w-full">
                Continue as Borrower
              </Button>
            </CardContent>
          </Card>

          <Card
            className="cursor-pointer border-2 border-transparent hover:border-accent transition-all duration-200 hover:shadow-lg group"
            onClick={() => navigate("/input?role=bank")}
          >
            <CardContent className="flex flex-col items-center gap-4 p-8">
              <div className="rounded-full bg-accent/10 p-4 group-hover:bg-accent/20 transition-colors">
                <Building2 className="h-10 w-10 text-accent" />
              </div>
              <div className="space-y-1">
                <h2 className="text-xl font-semibold text-foreground">Bank / Agent</h2>
                <p className="text-sm text-muted-foreground">
                  Assess borrower risk, view portfolio analytics, and manage recovery
                </p>
              </div>
              <Button variant="outline" className="mt-2 w-full">
                Continue as Bank
              </Button>
            </CardContent>
          </Card>
        </div>

        <p className="text-xs text-muted-foreground">
          Powered by LightGBM · Infosys Springboard Internship Project
        </p>
      </div>
    </div>
  );
}
