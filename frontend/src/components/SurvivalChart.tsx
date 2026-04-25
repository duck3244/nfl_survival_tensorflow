import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

import type { SurvivalPoint } from '../api/types';

interface Props {
  curve: SurvivalPoint[];
  median?: number;
}

export function SurvivalChart({ curve, median }: Props) {
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <LineChart data={curve} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
          <XAxis
            dataKey="t"
            type="number"
            domain={[0, 'dataMax']}
            label={{ value: 'Games played', position: 'insideBottom', offset: -2, fontSize: 12 }}
            tick={{ fontSize: 11 }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v) => `${Math.round(v * 100)}%`}
            label={{ value: 'Survival', angle: -90, position: 'insideLeft', fontSize: 12 }}
            tick={{ fontSize: 11 }}
          />
          <Tooltip
            formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
            labelFormatter={(t) => `${t} games`}
          />
          <ReferenceLine y={0.5} stroke="#94a3b8" strokeDasharray="4 4" />
          {median !== undefined && (
            <ReferenceLine
              x={median}
              stroke="#dc2626"
              strokeDasharray="4 4"
              label={{ value: `median ${median}`, position: 'top', fill: '#dc2626', fontSize: 11 }}
            />
          )}
          <Line type="monotone" dataKey="s" stroke="#2563eb" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
