import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { BorrowerProvider } from "@/context/BorrowerContext";
import DashboardLayout from "@/components/layout/DashboardLayout";
import RoleSelection from "./pages/RoleSelection";
import DataInput from "./pages/DataInput";
import Overview from "./pages/Overview";
import Borrowers from "./pages/Borrowers";
import Analytics from "./pages/Analytics";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <BorrowerProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<RoleSelection />} />
            <Route path="/input" element={<DataInput />} />
            <Route path="/dashboard" element={<DashboardLayout><Overview /></DashboardLayout>} />
            <Route path="/borrowers" element={<DashboardLayout><Borrowers /></DashboardLayout>} />
            <Route path="/analytics" element={<DashboardLayout><Analytics /></DashboardLayout>} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </BorrowerProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
