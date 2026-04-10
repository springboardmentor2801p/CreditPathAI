import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

interface MetricCardProps {
  title: string;
  value: string;
  change?: string;
  changeType?: "positive" | "negative" | "neutral";
  icon: LucideIcon;
  iconColor?: string;
}

export default function MetricCard({ title, value, change, changeType = "neutral", icon: Icon, iconColor }: MetricCardProps) {
  return (
    <div className="rounded-xl border bg-card p-5 shadow-sm transition-shadow hover:shadow-md">
      <div className="flex items-start justify-between">
        <div className="space-y-2">
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          <p className="text-2xl font-bold tracking-tight">{value}</p>
          {change && (
            <p
              className={cn(
                "text-xs font-medium",
                changeType === "positive" && "text-[hsl(160,84%,39%)]",
                changeType === "negative" && "text-destructive",
                changeType === "neutral" && "text-muted-foreground",
              )}
            >
              {change}
            </p>
          )}
        </div>
        <div
          className={cn("rounded-lg p-2.5")}
          style={{ backgroundColor: iconColor ? `${iconColor}20` : "hsl(var(--primary) / 0.1)" }}
        >
          <Icon className="h-5 w-5" style={{ color: iconColor || "hsl(var(--primary))" }} />
        </div>
      </div>
    </div>
  );
}
