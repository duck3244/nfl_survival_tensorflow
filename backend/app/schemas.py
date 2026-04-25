"""
Pydantic v2 schemas — request / response contracts.

도메인 검증은 여기서, 학습 분포 밖 입력에 대한 경고 플래그는 service layer에서.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


# ---------- Health ----------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_names: list[str]
    metrics: dict
    training_meta: dict
    baseline_length: int

    # Pydantic v2: `model_` prefix는 보호되어 있어 fields가 충돌 없게 설정.
    model_config = ConfigDict(protected_namespaces=())


# ---------- Predict ----------

class PredictRequest(BaseModel):
    """
    features는 모델의 feature_names에 맞는 키만 사용된다.
    여분의 키는 무시, 부족한 키는 422.
    숫자 범위는 service layer에서 추가 검증 + 분포 밖이면 경고 플래그.
    """
    features: dict[str, float] = Field(
        ...,
        description="예: {'YPC': 4.5, 'DrAge': 21, 'Pick': 4}",
    )
    max_games: int = Field(default=200, ge=10, le=400)


class SurvivalPoint(BaseModel):
    t: int
    s: float


class PredictResponse(BaseModel):
    risk_score: float
    median_survival: int
    grade: str
    risk_level: str
    comment: str
    survival_at: dict[int, float]      # {50: 0.7, 100: 0.4, 150: 0.1}
    survival_curve: list[SurvivalPoint]
    extrapolation_warnings: list[str]  # 예: ["YPC outside training IQR (2.5, 6.0)"]


# ---------- Famous players ----------

class FamousPlayer(BaseModel):
    name: str
    features: dict[str, float]
    risk_score: float
    predicted_career: int
    actual_career: Optional[int]
    survival_100: float
    grade: str


class FamousPlayersResponse(BaseModel):
    players: list[FamousPlayer]
    feature_names: list[str]
