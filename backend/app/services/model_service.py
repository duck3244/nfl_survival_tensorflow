"""
모델 서비스 — DeepSurv 싱글톤 로딩 + 예측 래핑.

FastAPI lifespan에서 instance 1개를 생성해 app.state에 저장.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# 백엔드 루트를 sys.path에 추가 (model_architecture 등 import 가능하게)
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _force_cpu_if_requested():
    """TF import 전에 CPU 강제. lifespan 진입 직후 호출."""
    from app.config import FORCE_CPU
    if FORCE_CPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


class ModelService:
    """학습된 DeepSurv + PlayerPredictor를 감싸 API 응답에 맞는 dict를 반환."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.predictor = None

    def load(self):
        _force_cpu_if_requested()

        from model_architecture import DeepSurv  # noqa: WPS433
        from prediction_utils import PlayerPredictor  # noqa: WPS433

        meta_path = f'{self.model_path}_meta.pkl'
        h5_path = f'{self.model_path}_model.h5'
        if not (os.path.exists(meta_path) and os.path.exists(h5_path)):
            raise FileNotFoundError(
                f"Model artifacts not found at {self.model_path} (meta + h5). "
                f"Run `python train_cli.py --mode full --output output` first."
            )

        # input_dim은 load() 안에서 weights로 결정되므로 placeholder.
        self.model = DeepSurv(input_dim=1)
        self.model.load(self.model_path)
        self.predictor = PlayerPredictor(self.model)

        if not self.model.feature_names:
            raise RuntimeError(
                "Loaded model has no feature_names — re-train with the updated pipeline."
            )

    @property
    def feature_names(self) -> list[str]:
        return list(self.model.feature_names) if self.model else []

    @property
    def metrics(self) -> dict:
        return dict(getattr(self.model, 'metrics', {}) or {})

    @property
    def training_meta(self) -> dict:
        return dict(getattr(self.model, 'training_meta', {}) or {})

    @property
    def baseline_length(self) -> int:
        bs = getattr(self.model, 'baseline_survival', None)
        return int(len(bs)) if bs is not None else 0

    def predict(self, features: dict, max_games: int) -> dict:
        """예측 결과를 JSON 직렬화 가능한 dict로 반환."""
        # 누락 feature 체크 (Pydantic 422가 더 친화적이지만 service에서도 방어)
        missing = [n for n in self.feature_names if n not in features]
        if missing:
            raise KeyError(f"Missing features: {missing}")

        pred = self.predictor.predict_player(features=features, max_games=max_games)

        curve = [
            {'t': int(t), 's': float(s)}
            for t, s in zip(pred['survival_curve']['times'],
                            pred['survival_curve']['probabilities'])
        ]

        warnings = self._extrapolation_warnings(features)

        return {
            'risk_score': float(pred['risk_score']),
            'median_survival': int(pred['median_survival']),
            'grade': pred['interpretation']['grade'],
            'risk_level': pred['interpretation']['risk_level'],
            'comment': pred['interpretation']['comment'],
            'survival_at': {int(k): float(v) for k, v in pred['survival_at'].items()},
            'survival_curve': curve,
            'extrapolation_warnings': warnings,
        }

    @staticmethod
    def _extrapolation_warnings(features: dict) -> list[str]:
        from app.config import FEATURE_RANGES
        out: list[str] = []
        for name, value in features.items():
            rng = FEATURE_RANGES.get(name)
            if rng is None:
                continue
            lo, hi = rng
            if value < lo or value > hi:
                out.append(f"{name}={value} outside training range ({lo}, {hi})")
        return out

    def predict_famous(self) -> list[dict]:
        """학습 시점 famous DB를 현 모델로 일괄 예측."""
        from prediction_utils import FamousPlayersPredictor  # noqa: WPS433
        fp = FamousPlayersPredictor(self.predictor)
        df = fp.predict_all_famous_players()

        results: list[dict] = []
        for _, row in df.iterrows():
            feats = {f: float(row[f]) for f in self.feature_names if f in row}
            actual = row.get('Actual_Career')
            results.append({
                'name': row['Player'],
                'features': feats,
                'risk_score': float(row['Risk_Score']),
                'predicted_career': int(row['Predicted_Career']),
                'actual_career': int(actual) if actual and actual == actual else None,  # NaN check
                'survival_100': float(row['Survival_100']),
                'grade': row['Grade'],
            })
        return results
