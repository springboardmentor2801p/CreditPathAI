import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  LayoutDashboard, Users, BarChart3, ChevronLeft, ChevronRight,
  BrainCircuit, Bell, Search, Home,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const navItems = [
  { label: "Home", icon: Home, path: "/" },
  { label: "Overview", icon: LayoutDashboard, path: "/dashboard" },
  { label: "Borrowers", icon: Users, path: "/borrowers" },
  { label: "Analytics", icon: BarChart3, path: "/analytics" },
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  return (
    <div className="flex h-screen overflow-hidden">
      <aside
        className={cn("flex flex-col border-r transition-all duration-300", collapsed ? "w-16" : "w-64")}
        style={{ background: "hsl(var(--sidebar-background))", color: "hsl(var(--sidebar-foreground))" }}
      >
        <div className="flex h-16 items-center gap-2 border-b border-white/10 px-4">
          <BrainCircuit className="h-7 w-7 shrink-0 text-[hsl(var(--primary))]" />
          {!collapsed && (
            <span className="text-lg font-bold tracking-tight">
              Credit<span className="text-[hsl(var(--primary))]">Path</span>AI
            </span>
          )}
        </div>

        <nav className="flex-1 space-y-1 p-2 pt-4">
          {navItems.map((item) => {
            const active = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                  active
                    ? "bg-[hsl(var(--sidebar-accent))] text-[hsl(var(--sidebar-primary))]"
                    : "text-[hsl(var(--sidebar-foreground))]/70 hover:bg-[hsl(var(--sidebar-accent))] hover:text-[hsl(var(--sidebar-foreground))]",
                )}
              >
                <item.icon className="h-5 w-5 shrink-0" />
                {!collapsed && item.label}
              </Link>
            );
          })}
        </nav>

        <div className="border-t border-white/10 p-2">
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="flex w-full items-center justify-center rounded-lg p-2 text-white/50 hover:bg-white/5 hover:text-white"
          >
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </button>
        </div>
      </aside>

      <div className="flex flex-1 flex-col overflow-hidden">
        <header className="flex h-16 items-center justify-between border-b bg-card px-6">
          <div className="relative w-72">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input placeholder="Search borrowers..." className="pl-9" />
          </div>
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" className="relative">
              <Bell className="h-5 w-5" />
              <span className="absolute -right-0.5 -top-0.5 h-2.5 w-2.5 rounded-full bg-destructive" />
            </Button>
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-full bg-primary/20 flex items-center justify-center text-primary text-sm font-semibold">
                A
              </div>
              <span className="text-sm font-medium">Agent</span>
            </div>
          </div>
        </header>
        <main className="flex-1 overflow-y-auto p-6">{children}</main>
      </div>
    </div>
  );
}
