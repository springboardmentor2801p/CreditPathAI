import { Link, Outlet, useLocation, useNavigate } from "react-router";
import { 
  LayoutDashboard, 
  UserPlus, 
  UploadCloud, 
  ActivitySquare, 
  Users, 
  History, 
  BarChart3, 
  LogOut, 
  Menu,
  X,
  Zap,
  Target,
  UserCircle,
  FileText,
} from "lucide-react";
import { useState } from "react";

const institutionNavigation = [
  { name: 'SYSTEM_DASHBOARD',    href: '/institution',                    icon: LayoutDashboard },
  { name: 'EVALUATE_RISK',       href: '/institution/borrower-input',     icon: UserPlus },
  { name: 'LOAN_APPLICATIONS',   href: '/institution/loan-applications',  icon: FileText },
  { name: 'ACTIVE_RECOVERY',     href: '/institution/recovery-actions',   icon: ActivitySquare },
  { name: 'TEAM_CAPACITY',       href: '/institution/team-assignment',    icon: Users },
  { name: 'AUDIT_LOGS',          href: '/institution/history',            icon: History },
  { name: 'METRICS_&_ANALYTICS', href: '/institution/analytics',          icon: BarChart3 },
];

const borrowerNavigation = [
  { name: 'FINANCIAL_OVERVIEW', href: '/borrower', icon: LayoutDashboard },
  { name: 'LOAN_SIMULATOR', href: '/borrower/evaluator', icon: Target },
  { name: 'MY_APPLICATIONS', href: '/borrower/applications', icon: FileText },
  { name: 'IDENTITY_MATRIX', href: '/borrower/profile', icon: UserCircle },
];

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  const isBorrower = location.pathname.startsWith('/borrower');
  const navigation = isBorrower ? borrowerNavigation : institutionNavigation;
  
  const themeColor = isBorrower ? 'emerald' : 'cyan';
  const themeHex = isBorrower ? 'rgba(16,185,129,0.15)' : 'rgba(34,211,238,0.15)';

  // Read auth from localStorage
  const userName = localStorage.getItem('user_name') || (isBorrower ? 'BORROWER' : 'LENDER');
  const userEmail = localStorage.getItem('user_email') || '';
  const initials = userName.split(' ').map((n: string) => n[0]).join('').slice(0, 2).toUpperCase();

  const handleLogout = () => {
    localStorage.removeItem('user_id');
    localStorage.removeItem('user_role');
    localStorage.removeItem('user_name');
    localStorage.removeItem('user_email');
    navigate('/login');
  };

  return (
    <div className={`flex h-screen bg-[#09090b] text-zinc-100 overflow-hidden font-sans selection:bg-${themeColor}-500/30 selection:text-${themeColor}-50`}>
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black/80 backdrop-blur-sm lg:hidden" 
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`
        fixed inset-y-0 left-0 z-50 w-64 bg-[#0e0e11] border-r border-zinc-800/80 transform transition-transform duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] lg:static lg:translate-x-0
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <div className="flex items-center justify-between h-16 px-6 border-b border-zinc-800/80 bg-black/20">
          <div className="flex items-center gap-2">
            <Zap className={`h-5 w-5 text-${themeColor}-400`} />
            <span className={`text-xl font-black tracking-tighter bg-clip-text text-transparent bg-gradient-to-br from-${themeColor}-400 via-violet-500 to-fuchsia-500 uppercase`}>
              Credit Path AI
            </span>
          </div>
          <button onClick={() => setSidebarOpen(false)} className="lg:hidden text-zinc-500 hover:text-white transition-colors">
            <X size={20} />
          </button>
        </div>

        <nav className="flex-1 px-3 py-6 space-y-1.5 overflow-y-auto h-[calc(100vh-4rem)] flex flex-col no-scrollbar">
          <div className="px-3 mb-4">
            <p className="text-[10px] font-bold tracking-widest text-zinc-500 uppercase">
              {isBorrower ? 'Citizen Modules' : 'Core Modules'}
            </p>
          </div>
          {navigation.map((item) => {
            const isActive = location.pathname === item.href || (location.pathname === '/' && item.href === '/institution');
            const Icon = item.icon;
            
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`group flex items-center px-3 py-2.5 text-xs font-bold tracking-wider rounded-lg transition-all duration-200 ${
                  isActive 
                    ? `bg-${themeColor}-500/10 text-${themeColor}-400 border border-${themeColor}-500/20 shadow-[0_0_15px_-3px_${themeHex}]` 
                    : 'text-zinc-400 border border-transparent hover:bg-zinc-800/50 hover:text-zinc-100'
                }`}
                onClick={() => setSidebarOpen(false)}
              >
                <Icon className={`mr-3 h-4 w-4 transition-colors ${isActive ? `text-${themeColor}-400 drop-shadow-[0_0_8px_${themeHex.replace('0.15', '0.8')}]` : 'text-zinc-600 group-hover:text-zinc-300'}`} />
                {item.name}
              </Link>
            );
          })}
          
          <div className="mt-auto pt-6 px-3">
             <button
                onClick={handleLogout}
                className="flex w-full items-center px-3 py-2.5 text-xs font-bold tracking-wider rounded-lg text-rose-500/80 hover:bg-rose-500/10 hover:text-rose-400 border border-transparent hover:border-rose-500/20 transition-all duration-200"
              >
                <LogOut className="mr-3 h-4 w-4" />
                TERMINATE_SESSION
              </button>
          </div>
        </nav>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col w-0 overflow-hidden relative">
        {/* Subtle ambient glows */}
        <div className="absolute top-[-20%] left-[-10%] w-[40%] h-[40%] bg-violet-500/5 blur-[120px] rounded-full pointer-events-none" />
        <div className={`absolute bottom-[-10%] right-[-10%] w-[30%] h-[30%] bg-${themeColor}-500/5 blur-[120px] rounded-full pointer-events-none`} />

        <header className="flex-shrink-0 flex h-16 bg-[#0e0e11]/80 backdrop-blur-md border-b border-zinc-800/80 z-10">
          <button
            onClick={() => setSidebarOpen(true)}
            className="px-4 border-r border-zinc-800/80 text-zinc-400 focus:outline-none hover:text-white lg:hidden"
          >
            <span className="sr-only">Open sidebar</span>
            <Menu size={20} />
          </button>
          
          <div className="flex-1 px-6 flex justify-between items-center">
            <div className="flex items-center">
              <div className="hidden md:flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full bg-${themeColor}-400 shadow-[0_0_8px_${themeHex.replace('0.15', '0.8')}] animate-pulse`} />
                <span className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase">
                  {isBorrower ? 'Citizen Node // Active' : 'System Online // Node 04'}
                </span>
              </div>
            </div>
              <div className="flex items-center space-x-4">
               <div className="hidden sm:flex flex-col items-end">
                 <span className="text-xs font-bold tracking-wider text-zinc-200">
                   {userName.toUpperCase()}
                 </span>
                 <span className={`text-[10px] font-mono tracking-widest text-${themeColor}-500`}>
                   {userEmail || (isBorrower ? 'BORROWER' : 'INSTITUTION')}
                 </span>
               </div>
               <Link 
                 to={isBorrower ? '/borrower/profile' : '/institution/profile'}
                 className={`h-9 w-9 rounded-md border border-${themeColor}-500/30 bg-${themeColor}-500/10 flex items-center justify-center text-${themeColor}-400 font-black text-sm shadow-[0_0_10px_-2px_${themeHex}] hover:bg-${themeColor}-500/20 transition-colors cursor-pointer`}
               >
                 {initials || (isBorrower ? 'AM' : 'JD')}
               </Link>
            </div>
          </div>
        </header>

        <main className="flex-1 relative overflow-y-auto focus:outline-none custom-scrollbar z-10">
          <div className="py-8 px-4 sm:px-6 md:px-8 max-w-7xl mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
