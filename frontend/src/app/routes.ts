import { createBrowserRouter, redirect } from "react-router";
import { Layout } from "./components/Layout";
import { Login } from "./pages/Login";

// Institution Pages
import { Dashboard } from "./pages/Dashboard";
import { BorrowerForm } from "./pages/BorrowerForm";
import { RecoveryActions } from "./pages/RecoveryActions";
import { TeamAssignment } from "./pages/TeamAssignment";
import { HistoryTracking } from "./pages/HistoryTracking";
import { Analytics } from "./pages/Analytics";
import { LoanApplications } from "./pages/LoanApplications";

// Borrower Pages
import { BorrowerDashboard } from "./pages/BorrowerDashboard";
import { BorrowerEvaluator } from "./pages/BorrowerEvaluator";
import { Profile } from "./pages/Profile";
import { BorrowerApplications } from "./pages/BorrowerApplications";

export const router = createBrowserRouter([
  { path: "/login", Component: Login },
  
  // To handle the root layout with a component
  {
    path: "/",
    loader: () => redirect("/login"),
  },
  
  // Institution Routes
  {
    path: "/institution",
    Component: Layout,
    children: [
      { index: true, Component: Dashboard },
      { path: "borrower-input", Component: BorrowerForm },
      { path: "loan-applications", Component: LoanApplications },
      { path: "recovery-actions", Component: RecoveryActions },
      { path: "team-assignment", Component: TeamAssignment },
      { path: "history", Component: HistoryTracking },
      { path: "analytics", Component: Analytics },
      { path: "profile", Component: Profile },
    ],
  },
  
  // Borrower Routes
  {
    path: "/borrower",
    Component: Layout,
    children: [
      { index: true, Component: BorrowerDashboard },
      { path: "evaluator", Component: BorrowerEvaluator },
      { path: "applications", Component: BorrowerApplications },
      { path: "profile", Component: Profile },
    ],
  },
]);
