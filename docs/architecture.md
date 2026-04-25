# 🏗️ 아키텍처 문서

NFL 러닝백 커리어 생존 분석 프로젝트의 전체 시스템 아키텍처를 설명합니다.

---

## 1. 시스템 개요

본 프로젝트는 **TensorFlow 기반 DeepSurv 신경망**으로 NFL 러닝백의 커리어 길이를 예측하는 풀스택 애플리케이션입니다.

| 계층 | 기술 스택 | 역할 |
|------|-----------|------|
| **Frontend** | React 18 + TypeScript + Vite | 사용자 입력 / 결과 시각화 |
| **Backend API** | FastAPI + Pydantic v2 | 추론 서비스 / 모델 서빙 |
| **ML Core** | TensorFlow 2.10+, Keras, scikit-learn, lifelines | DeepSurv 모델 학습 / 추론 |
| **Data** | nfl.csv (Pro-Football-Reference) | 학습 데이터셋 |

---

## 2. 디렉터리 구조

```
nfl_survival_tensorflow/
│
├── backend/                          # 백엔드 루트
│   ├── app/                          # FastAPI 애플리케이션
│   │   ├── api/                      # 라우터
│   │   │   ├── routes_health.py      # GET /api/health
│   │   │   ├── routes_predict.py     # POST /api/predict
│   │   │   └── routes_players.py     # GET /api/players/famous
│   │   ├── services/
│   │   │   └── model_service.py      # 모델 싱글톤 / 추론 래퍼
│   │   ├── config.py                 # 환경 설정 (CORS, MODEL_PATH 등)
│   │   ├── schemas.py                # Pydantic 요청/응답 스키마
│   │   └── main.py                   # FastAPI 엔트리 포인트
│   │
│   ├── data_preprocessing.py         # NFLDataPreprocessor
│   ├── model_architecture.py         # DeepSurv (Keras 모델 래퍼)
│   ├── model_training.py             # ModelTrainer / EnsembleTrainer
│   ├── prediction_utils.py           # PlayerPredictor / FamousPlayersPredictor
│   ├── visualization.py              # SurvivalVisualizer
│   ├── train_cli.py                  # 학습 파이프라인 CLI
│   ├── nfl.csv                       # 데이터셋
│   └── requirements.txt
│
├── frontend/                         # React + Vite 프런트엔드
│   ├── src/
│   │   ├── api/
│   │   │   ├── client.ts             # fetch 래퍼
│   │   │   └── types.ts              # 백엔드 스키마 미러링
│   │   ├── hooks/
│   │   │   └── queries.ts            # React Query 훅
│   │   ├── components/
│   │   │   ├── Header.tsx
│   │   │   ├── SurvivalChart.tsx     # Recharts 생존 곡선
│   │   │   └── ErrorBoundary.tsx
│   │   ├── pages/
│   │   │   ├── PredictPage.tsx       # 단일 예측 폼
│   │   │   └── FamousPage.tsx        # 유명 선수 비교
│   │   └── App.tsx
│   └── package.json
│
└── docs/                             # 문서
    ├── architecture.md               # 본 문서
    └── uml.md
```

---

## 3. 컴포넌트 계층 (Layered View)

```
┌──────────────────────────────────────────────────────────────┐
│  Presentation Layer (Frontend - React)                       │
│  ─ PredictPage / FamousPage / SurvivalChart                  │
│  ─ React Query (상태 캐싱) / React Hook Form (입력)          │
└──────────────────────────────────────────────────────────────┘
                            │ HTTP/JSON (REST)
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  API Layer (FastAPI)                                         │
│  ─ /api/health · /api/predict · /api/players/famous          │
│  ─ Pydantic 스키마 검증 / CORS 미들웨어                      │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Service Layer                                               │
│  ─ ModelService (싱글톤, 앱 lifespan에서 1회 로드)           │
│  ─ 외삽 경고 (FEATURE_RANGES 검사)                            │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  ML Core Layer                                               │
│  ─ DeepSurv (Keras 래퍼 + StandardScaler + Breslow baseline) │
│  ─ PlayerPredictor / FamousPlayersPredictor                  │
│  ─ ModelTrainer / EnsembleTrainer (학습 시점에만 사용)       │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Data Layer                                                  │
│  ─ NFLDataPreprocessor → nfl.csv                             │
│  ─ 모델 아티팩트: deepsurv_model_model.h5 / *_meta.pkl       │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. 핵심 모듈 설명

### 4.1 Data Layer — `data_preprocessing.py`

**`NFLDataPreprocessor`**
- 원시 CSV 로드 → 특징 엔지니어링 → 결측치/이상치 처리 → `(X, y_event, y_time)` 행렬 반환
- 파생 특징: `BMI`, `YPC`, `Years`, `PB_binary`, `AP1_binary`, `Retired`
- 기본 특징 우선순위: `['BMI', 'YPC', 'DrAge', 'Pick', 'Rnd']`

```python
preprocessor = NFLDataPreprocessor('nfl.csv')
df = preprocessor.preprocess(remove_outliers=True)
X, y_event, y_time = preprocessor.get_feature_matrix()
```

### 4.2 ML Core Layer — `model_architecture.py`

**`DeepSurv`** (Keras 모델 래퍼)
- 함수형 API로 `Input → Dense×N → BatchNorm → Dropout → Dense(1, linear)` 구조 빌드
- **Custom Loss**: `cox_partial_likelihood_loss` — Cox 부분 우도 손실
- **Baseline Survival**: Breslow 추정량을 학습 후 캐싱하여 추론 시 `S(t|X) = S₀(t)^exp(η)` 계산
- 아티팩트 저장: `_model.h5` (가중치) + `_meta.pkl` (scaler, baseline, metrics, feature_names)

| 변형 | hidden_layers |
|------|---------------|
| `standard` | `[64, 32, 16]` |
| `deep` | `[128, 64, 32, 16, 8]` |
| `wide` | `[256, 128, 64]` |
| `simple` | `[32, 16]` |

### 4.3 Training Layer — `model_training.py`

- **`ModelTrainer`**: train/test 분할 · 학습 · C-index 평가 · 시각화
- **`EnsembleTrainer`**: 부트스트랩 샘플링으로 N개 모델 학습 후 평균 위험 점수

### 4.4 Prediction Layer — `prediction_utils.py`

- **`PlayerPredictor`**: 단일/배치 예측, 등급 판정, 텍스트 리포트 생성
- **`FamousPlayersPredictor`**: 하드코딩된 유명 RB(LaDainian Tomlinson, Emmitt Smith, Barry Sanders 등)에 대한 일괄 예측 + 실제 커리어와 MAE/MAPE 비교

### 4.5 API Layer — `app/`

| 엔드포인트 | 메서드 | 응답 스키마 |
|------------|--------|-------------|
| `/api/health` | GET | `HealthResponse` (모델 메타데이터) |
| `/api/predict` | POST | `PredictResponse` (위험 점수 · 생존 곡선 · 해석) |
| `/api/players/famous` | GET | `FamousPlayersResponse` (유명 선수 일괄 예측) |

**`ModelService`** 는 FastAPI `lifespan`에서 1회 초기화되어 `app.state`에 부착되며, 이후 모든 요청이 동일 인스턴스를 공유합니다.

### 4.6 Frontend Layer — `frontend/src/`

- **App Shell**: `App.tsx` → `Header` + `<Routes>`
- **상태 관리**: TanStack React Query (서버 상태 캐싱), `useState` (로컬 결과)
- **데이터 페칭**: `useHealth`, `useFamousPlayers` (queries) / `usePredict` (mutation)
- **차트**: Recharts `LineChart` 기반 `SurvivalChart` 컴포넌트

---

## 5. 데이터 흐름 (Data Flow)

### 5.1 학습 파이프라인 (오프라인)

```
nfl.csv
  │
  ▼ NFLDataPreprocessor.preprocess()
(X, y_event, y_time)
  │
  ▼ ModelTrainer.train_test_split_data()
train / test split
  │
  ▼ DeepSurv.fit()
  │   ├─ StandardScaler.fit_transform(X_train)
  │   ├─ Keras 학습 (Adam + Cox loss + EarlyStopping)
  │   └─ Breslow 추정으로 baseline_survival 캐싱
  │
  ▼ DeepSurv.save()
[deepsurv_model_model.h5, deepsurv_model_meta.pkl]
```

### 5.2 추론 파이프라인 (온라인)

```
[React form] ──POST /api/predict──▶ [FastAPI]
                                       │
                                       ▼
                              ModelService.predict(features, max_games)
                                       │
                                       ▼
                              PlayerPredictor.predict_player()
                                       │  ├─ 특징 dict → 1×D 배열
                                       │  ├─ DeepSurv.predict_risk()
                                       │  ├─ baseline_survival_at(times)
                                       │  └─ S(t|X) = S₀(t)^exp(η)
                                       ▼
                              ModelService._extrapolation_warnings()
                                       │
                                       ▼
                              PredictResponse (JSON)
                                       │
[React] ◀──── survival_curve, risk_score, grade, warnings
   │
   ▼ SurvivalChart (Recharts)
```

---

## 6. 주요 설계 결정

### 6.1 모델 싱글톤 + lifespan 로딩
- TensorFlow 모델 로딩은 비싸므로 앱 시작 시 1회만 수행
- `app.state.model_service`로 모든 라우터가 공유

### 6.2 Breslow Baseline 사전 계산
- 추론 시 Kaplan-Meier 객체가 없어도 동작하도록 학습 시점에 baseline_times/baseline_survival을 `_meta.pkl`에 저장
- 이로써 서빙 환경에서 lifelines 의존성을 최소화

### 6.3 특징 외삽 경고
- `FEATURE_RANGES` (학습 IQR)를 벗어나는 입력에 대해 응답에 `extrapolation_warnings`를 포함
- 사용자가 비현실적 입력을 줬을 때 경고를 UI에 표시

### 6.4 React Query의 `staleTime: Infinity`
- 모델 메타데이터(`/health`)와 유명 선수(`/famous`)는 서버 재시작 전까지 변하지 않으므로 무한 캐싱
- 단일 예측(`usePredict`)만 mutation으로 매 호출 실행

### 6.5 Pydantic v2 / TS 타입 미러링
- `app/schemas.py`의 모델과 `frontend/src/api/types.ts`의 인터페이스가 1:1 대응
- 백엔드 변경 시 프런트 타입을 동기화해야 함

---

## 7. 외부 의존성

### Backend
- `tensorflow >= 2.10.0`, `keras >= 2.10.0` — DeepSurv 모델
- `lifelines >= 0.27.0` — Kaplan-Meier, C-index
- `scikit-learn >= 1.1.0` — StandardScaler, train_test_split
- `fastapi >= 0.115`, `uvicorn >= 0.30`, `pydantic >= 2.6` — API 서빙
- `pandas`, `numpy`, `matplotlib`, `seaborn`

### Frontend
- `react ^18.3.1`, `react-router-dom ^6.30.3`
- `@tanstack/react-query ^5.100.1`
- `react-hook-form ^7.73.1`, `zod ^3.25.76`
- `recharts ^2.15.4`, `lucide-react`, `tailwindcss`

---

## 8. 배포 구성 (개발 환경)

```
┌──────────────────────┐         ┌──────────────────────┐
│  Vite Dev Server     │  proxy  │  Uvicorn (FastAPI)   │
│  http://localhost    │ ──────▶ │  http://localhost    │
│  :5173 (or 5174)     │         │  :8000               │
└──────────────────────┘         └──────────────────────┘
        │                                   │
        ▼                                   ▼
   React 번들                          DeepSurv 모델
                                       (.h5 + .pkl)
```

`config.py`의 `CORS_ORIGINS`에 Vite 기본 포트(5173/5174)가 허용되어 있습니다.

---

## 9. 확장 가능성

| 영역 | 가능한 확장 |
|------|-------------|
| 모델 | EnsembleTrainer 결과를 ModelService에 통합 (현재는 단일 모델) |
| 특징 | 시즌별 시계열 입력 (RNN/Transformer) |
| 예측 | 일괄 CSV 업로드 엔드포인트 |
| 모니터링 | 추론 로그 + 분포 드리프트 추적 |
| 인증 | API 키 / OAuth (현재는 미구현) |
| 배포 | Docker Compose (frontend nginx + backend uvicorn) |
