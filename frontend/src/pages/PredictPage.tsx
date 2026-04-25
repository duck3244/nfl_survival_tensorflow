import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { AlertTriangle } from 'lucide-react';

import { useHealth, usePredict } from '../hooks/queries';
import { SurvivalChart } from '../components/SurvivalChart';
import type { PredictResponse } from '../api/types';

// 합리적 기본값. 서버 feature_names에 없는 키는 무시되므로 모두 정의해도 안전.
const DEFAULTS: Record<string, number> = {
  YPC: 4.5,
  DrAge: 22,
  Pick: 30,
  Rnd: 2,
  BMI: 29.0,
};

const FIELD_HINT: Record<string, string> = {
  YPC: 'Yards per carry (career rate)',
  DrAge: 'Age at draft (years)',
  Pick: 'Draft pick number (lower = better)',
  Rnd: 'Draft round',
  BMI: 'Body mass index',
};

interface FormValues {
  features: Record<string, number>;
}

export function PredictPage() {
  const { data: health, isLoading: healthLoading } = useHealth();
  const predict = usePredict();
  const [last, setLast] = useState<PredictResponse | null>(null);

  const featureNames = health?.feature_names ?? [];

  const { register, handleSubmit, reset, formState: { errors } } = useForm<FormValues>({
    defaultValues: { features: {} },
  });

  // health 응답이 도착하면 feature 기본값 주입
  useEffect(() => {
    if (featureNames.length === 0) return;
    const values: Record<string, number> = {};
    featureNames.forEach((f) => { values[f] = DEFAULTS[f] ?? 0; });
    reset({ features: values });
  }, [featureNames.join('|'), reset]);

  const onSubmit = handleSubmit(async (data) => {
    // react-hook-form은 number input을 string으로 받기 쉬워 명시 변환
    const features: Record<string, number> = {};
    featureNames.forEach((f) => { features[f] = Number(data.features[f]); });
    const result = await predict.mutateAsync({ features });
    setLast(result);
  });

  if (healthLoading) {
    return <div className="p-6 text-slate-500">Loading model metadata…</div>;
  }
  if (!health) {
    return (
      <div className="p-6 text-red-700 bg-red-50 rounded border border-red-200">
        Backend not reachable. Start uvicorn at port 8001.
      </div>
    );
  }

  return (
    <div className="grid md:grid-cols-2 gap-6">
      <section className="bg-white border rounded-lg p-5 shadow-sm">
        <h2 className="text-lg font-semibold mb-1">Predict career length</h2>
        <p className="text-sm text-slate-500 mb-4">
          Model trained on {health.training_meta.sample_size ?? '?'} players.
          Using features: {featureNames.join(', ')}
        </p>

        <form onSubmit={onSubmit} className="space-y-4">
          {featureNames.map((name) => (
            <div key={name}>
              <label className="block text-sm font-medium text-slate-700">
                {name}
                <span className="text-slate-400 font-normal ml-2 text-xs">
                  {FIELD_HINT[name] ?? ''}
                </span>
              </label>
              <input
                type="number"
                step="any"
                {...register(`features.${name}` as const, {
                  required: 'required',
                  valueAsNumber: true,
                })}
                className="mt-1 w-full rounded border-slate-300 border px-3 py-2 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none"
              />
              {errors.features?.[name] && (
                <span className="text-xs text-red-600">
                  {errors.features[name]?.message as string}
                </span>
              )}
            </div>
          ))}

          <button
            type="submit"
            disabled={predict.isPending}
            className="w-full bg-brand-600 hover:bg-brand-700 text-white py-2 rounded font-medium disabled:opacity-50"
          >
            {predict.isPending ? 'Predicting…' : 'Predict'}
          </button>

          {predict.isError && (
            <div className="text-sm text-red-700 bg-red-50 rounded border border-red-200 p-2">
              {(predict.error as Error).message}
            </div>
          )}
        </form>
      </section>

      <section className="bg-white border rounded-lg p-5 shadow-sm">
        <h2 className="text-lg font-semibold mb-3">Result</h2>

        {!last && (
          <div className="text-sm text-slate-500">
            Submit the form to see predicted career length and survival curve.
          </div>
        )}

        {last && (
          <div className="space-y-4">
            {last.extrapolation_warnings.length > 0 && (
              <div className="flex gap-2 items-start text-amber-800 bg-amber-50 border border-amber-200 rounded p-2 text-sm">
                <AlertTriangle className="w-4 h-4 mt-0.5 flex-none" />
                <ul className="list-disc list-inside">
                  {last.extrapolation_warnings.map((w) => <li key={w}>{w}</li>)}
                </ul>
              </div>
            )}

            <div className="grid grid-cols-2 gap-3 text-sm">
              <Stat label="Median career" value={`${last.median_survival} games`} />
              <Stat label="Risk score" value={last.risk_score.toFixed(3)} />
              <Stat label="Risk level" value={last.risk_level} />
              <Stat label="Grade" value={last.grade} />
            </div>

            <p className="text-sm text-slate-600">{last.comment}</p>

            <SurvivalChart curve={last.survival_curve} median={last.median_survival} />

            <div className="grid grid-cols-3 gap-2 text-center text-xs">
              {Object.entries(last.survival_at).map(([t, p]) => (
                <div key={t} className="rounded bg-slate-100 py-2">
                  <div className="font-mono text-slate-600">@ {t} games</div>
                  <div className="font-semibold text-slate-900">{(p * 100).toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded border border-slate-200 px-3 py-2">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className="font-mono font-semibold text-slate-900">{value}</div>
    </div>
  );
}
