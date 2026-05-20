import { NavLink, Outlet } from 'react-router-dom';
import { Activity, Database, Menu } from 'lucide-react';
import clsx from 'clsx';

const NAV_ITEMS = [
  { to: '/memory', label: 'Memory', icon: Database },
];

export default function AppShell() {
  return (
    <div className="flex min-h-screen bg-slate-50 text-slate-950">
      <aside className="hidden w-64 shrink-0 border-r border-slate-200 bg-white px-4 py-5 lg:flex lg:flex-col">
        <div className="mb-7 flex items-center gap-3 px-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-md border border-cyan-200 bg-cyan-50 text-cyan-700">
            <Activity size={18} />
          </div>
          <div>
            <div className="text-sm font-semibold tracking-wide text-slate-950">SpatialClaw</div>
            <div className="text-xs text-slate-500">Memory Console</div>
          </div>
        </div>

        <nav className="space-y-1">
          {NAV_ITEMS.map((item) => (
            <NavItem key={item.to} {...item} />
          ))}
        </nav>

        <div className="mt-auto rounded-md border border-slate-200 bg-slate-50 p-3 text-xs text-slate-500">
          <div className="mb-1 font-medium text-slate-700">API</div>
          <code className="text-cyan-700">/api</code>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-slate-200 bg-white/95 px-4 backdrop-blur lg:hidden">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Menu size={18} className="text-cyan-700" />
            SpatialClaw Memory
          </div>
          <nav className="flex items-center gap-1">
            {NAV_ITEMS.map((item) => (
              <NavIcon key={item.to} {...item} />
            ))}
          </nav>
        </header>

        <main className="min-h-0 flex-1 overflow-hidden">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

function NavItem({ to, label, icon: Icon }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        clsx(
          'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition',
          isActive
            ? 'bg-cyan-50 text-cyan-800 ring-1 ring-cyan-200'
            : 'text-slate-500 hover:bg-slate-100 hover:text-slate-950',
        )
      }
    >
      <Icon size={17} />
      {label}
    </NavLink>
  );
}

function NavIcon({ to, label, icon: Icon }) {
  return (
    <NavLink
      to={to}
      aria-label={label}
      title={label}
      className={({ isActive }) =>
        clsx(
          'rounded-md p-2 transition',
          isActive ? 'bg-cyan-50 text-cyan-800' : 'text-slate-500 hover:text-slate-950',
        )
      }
    >
      <Icon size={17} />
    </NavLink>
  );
}
