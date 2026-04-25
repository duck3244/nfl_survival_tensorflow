import { useMemo, useState } from 'react';

import { useFamousPlayers } from '../hooks/queries';
import type { FamousPlayer } from '../api/types';

type SortKey = 'predicted_career' | 'risk_score' | 'name';

export function FamousPage() {
  const { data, isLoading, isError, error } = useFamousPlayers();
  const [sortKey, setSortKey] = useState<SortKey>('predicted_career');
  const [showActualOnly, setShowActualOnly] = useState(false);

  const players = useMemo(() => {
    if (!data) return [];
    let list = data.players.slice();
    if (showActualOnly) list = list.filter((p) => p.actual_career != null);
    list.sort((a, b) => {
      if (sortKey === 'name') return a.name.localeCompare(b.name);
      if (sortKey === 'risk_score') return a.risk_score - b.risk_score;
      return b.predicted_career - a.predicted_career;
    });
    return list;
  }, [data, sortKey, showActualOnly]);

  if (isLoading) return <div className="p-6 text-slate-500">Loading…</div>;
  if (isError) {
    return (
      <div className="p-6 text-red-700 bg-red-50 rounded border border-red-200">
        {(error as Error).message}
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold">Famous NFL running backs</h2>
          <p className="text-sm text-slate-500">
            Predictions vs. actual career length, scored by the current model.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <label className="text-sm flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={showActualOnly}
              onChange={(e) => setShowActualOnly(e.target.checked)}
            />
            with actual data only
          </label>
          <select
            className="rounded border-slate-300 border text-sm px-2 py-1"
            value={sortKey}
            onChange={(e) => setSortKey(e.target.value as SortKey)}
          >
            <option value="predicted_career">Sort: predicted career</option>
            <option value="risk_score">Sort: risk score</option>
            <option value="name">Sort: name</option>
          </select>
        </div>
      </div>

      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {players.map((p) => <Card key={p.name} p={p} />)}
      </div>
    </div>
  );
}

function Card({ p }: { p: FamousPlayer }) {
  const error = p.actual_career != null
    ? Math.abs(p.predicted_career - p.actual_career)
    : null;

  return (
    <div className="bg-white border rounded-lg p-4 shadow-sm">
      <div className="flex justify-between items-start gap-2">
        <div>
          <div className="font-semibold">{p.name}</div>
          <div className="text-xs text-slate-500">{p.grade}</div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-brand-700">{p.predicted_career}</div>
          <div className="text-xs text-slate-500">predicted games</div>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
        <Stat label="Risk" value={p.risk_score.toFixed(3)} />
        <Stat label="P(100g)" value={`${p.survival_100.toFixed(1)}%`} />
        <Stat
          label="Actual"
          value={p.actual_career != null ? `${p.actual_career} g` : '—'}
        />
        <Stat
          label="|Δ|"
          value={error != null ? `${error}` : '—'}
          tone={error != null && error <= 30 ? 'good' : 'neutral'}
        />
      </div>

      <div className="mt-3 pt-3 border-t border-slate-100 text-xs text-slate-500 flex flex-wrap gap-x-3 gap-y-1">
        {Object.entries(p.features).map(([k, v]) => (
          <span key={k}>
            <span className="text-slate-400">{k}</span>{' '}
            <span className="font-mono text-slate-700">{v}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  tone = 'neutral',
}: {
  label: string;
  value: string;
  tone?: 'good' | 'neutral';
}) {
  const cls = tone === 'good' ? 'bg-emerald-50 text-emerald-700' : 'bg-slate-50 text-slate-700';
  return (
    <div className={`rounded ${cls} px-2 py-1`}>
      <span className="text-slate-400 mr-1">{label}</span>
      <span className="font-mono">{value}</span>
    </div>
  );
}
