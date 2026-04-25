// Thin fetch wrapper. Vite dev proxy forwards /api → backend (port 8001).

import type {
  HealthResponse,
  PredictRequest,
  PredictResponse,
  FamousPlayersResponse,
} from './types';

const BASE = import.meta.env.VITE_API_BASE_URL ?? '/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  });

  if (!res.ok) {
    let detail = '';
    try {
      const body = await res.json();
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {
      detail = res.statusText;
    }
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }

  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>('/health'),
  predict: (body: PredictRequest) =>
    request<PredictResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  famousPlayers: () => request<FamousPlayersResponse>('/players/famous'),
};
