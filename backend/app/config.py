"""
런타임 설정 — 환경변수 기반. pydantic-settings 미도입 (단순 os.getenv).

기본값은 모두 개발용. 운영 시 환경변수로 덮어쓰기.
"""
from __future__ import annotations
import os
from pathlib import Path

# 백엔드 루트 (이 파일 기준 한 단계 위)
BACKEND_ROOT = Path(__file__).resolve().parent.parent

# 학습 산출물 prefix. train_cli가 `{MODEL_PATH}_model.h5` / `{MODEL_PATH}_meta.pkl` 생성.
MODEL_PATH: str = os.getenv(
    'MODEL_PATH',
    str(BACKEND_ROOT / 'output' / 'deepsurv_model'),
)

# CORS — 개발에선 Vite 프록시로 우회 가능하지만, 직접 호출도 안전하게 허용.
_default_origins = (
    'http://localhost:5173,http://127.0.0.1:5173,'
    'http://localhost:5174,http://127.0.0.1:5174'
)
CORS_ORIGINS: list[str] = [
    o.strip() for o in os.getenv('CORS_ORIGINS', _default_origins).split(',') if o.strip()
]

# CPU 강제 — TF가 GPU 메모리 전체를 잡지 않게. 서빙은 단일 추론이라 CPU로 충분.
FORCE_CPU: bool = os.getenv('FORCE_CPU', '1') == '1'

# 학습 데이터 분포 (extrapolation 경고에 사용)
# 값은 학습 데이터 IQR 기준 — 모델 학습 후 _meta.pkl에 동적 저장도 가능하지만
# MVP에선 정적 상수로 충분.
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    'BMI':   (24.0, 35.0),
    'YPC':   (2.5, 6.0),
    'DrAge': (20, 26),
    'Pick':  (1, 256),
    'Rnd':   (1, 7),
}

# 예측 곡선 길이 (경기 수). 응답 크기 ~2KB.
MAX_GAMES: int = int(os.getenv('MAX_GAMES', '200'))
