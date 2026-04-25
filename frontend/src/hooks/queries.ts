import { useMutation, useQuery } from '@tanstack/react-query';

import { api } from '../api/client';
import type { PredictRequest } from '../api/types';

// Health: 1회 조회 후 무한 캐시 (모델 메타는 서버 재시작 전엔 안 바뀜)
export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    staleTime: Infinity,
    retry: 1,
  });
}

// Famous players: 모델/입력 변화 없으면 결과 동일. 무한 캐시.
export function useFamousPlayers() {
  return useQuery({
    queryKey: ['famous-players'],
    queryFn: api.famousPlayers,
    staleTime: Infinity,
  });
}

// Predict: mutation — 사용자 입력에 따라 매번 호출.
export function usePredict() {
  return useMutation({
    mutationFn: (req: PredictRequest) => api.predict(req),
  });
}
