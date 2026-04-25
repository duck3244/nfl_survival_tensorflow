import { NavLink } from 'react-router-dom';
import { Activity } from 'lucide-react';

import { useHealth } from '../hooks/queries';

const linkClass = ({ isActive }: { isActive: boolean }) =>
  [
    'px-3 py-1.5 rounded text-sm font-medium transition',
    isActive ? 'bg-brand-600 text-white' : 'text-slate-600 hover:bg-slate-200',
  ].join(' ');

export function Header() {
  const { data, isLoading, isError } = useHealth();

  return (
    <header className="border-b bg-white">
      <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Activity className="w-5 h-5 text-brand-600" />
          <span className="font-semibold">NFL Survival</span>
        </div>

        <nav className="flex gap-1">
          <NavLink to="/" end className={linkClass}>Predict</NavLink>
          <NavLink to="/famous" className={linkClass}>Famous Players</NavLink>
        </nav>

        <div className="text-xs text-slate-500 text-right leading-tight min-w-[180px]">
          {isLoading && <span>loading…</span>}
          {isError && <span className="text-red-600">backend offline</span>}
          {data && (
            <>
              <div>
                <span className="font-medium text-slate-700">model</span>{' '}
                {data.training_meta.model_type ?? 'unknown'}
              </div>
              <div>
                test C-index{' '}
                <span className="font-mono">
                  {data.metrics.test_c_index?.toFixed(3) ?? '—'}
                </span>
              </div>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
