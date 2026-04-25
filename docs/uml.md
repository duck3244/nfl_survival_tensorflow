# 📐 UML 다이어그램

NFL 러닝백 커리어 생존 분석 프로젝트의 UML 다이어그램 모음입니다. 모든 다이어그램은 [Mermaid](https://mermaid.js.org/) 문법으로 작성되었으며 GitHub, VS Code 등 대부분의 마크다운 뷰어에서 자동 렌더링됩니다.

---

## 1. 클래스 다이어그램 (Backend)

### 1.1 ML Core 클래스

```mermaid
classDiagram
    class NFLDataPreprocessor {
        +str filepath
        +DataFrame raw_data
        +DataFrame processed_data
        +list~str~ feature_columns_used
        +DEFAULT_FEATURE_PRIORITY: list
        +load_data() DataFrame
        +create_features(df) DataFrame
        +clean_data(df, remove_outliers) DataFrame
        +preprocess(remove_outliers) DataFrame
        +get_feature_matrix(feature_columns) tuple
        +get_summary_statistics() DataFrame
        +save_processed_data(output_path) None
    }

    class DeepSurv {
        +int input_dim
        +list~int~ hidden_layers
        +float dropout_rate
        +str activation
        +float l2_reg
        +list~str~ feature_names
        +keras.Model model
        +StandardScaler scaler
        +History history
        +ndarray baseline_times
        +ndarray baseline_survival
        +dict metrics
        +dict training_meta
        -_build_model() None
        +compile(learning_rate, optimizer, loss_fn) None
        +fit(X, y_event, y_time, ...) History
        -_compute_baseline_survival(X, y_event, y_time) None
        +baseline_survival_at(times) ndarray
        +predict_risk(X) ndarray
        +predict_survival(X, times, baseline) ndarray
        +save(filepath) None
        +load(filepath) None
        +summary() None
        +get_config() dict
    }

    class ModelTrainer {
        +DeepSurv model
        +int random_state
        +float train_c_index
        +float test_c_index
        +History history
        +train_test_split_data(X, y_event, y_time, test_size) tuple
        +train(X_train, y_event, y_time, ...) History
        +evaluate(X_train, y_event_train, ...) dict
        +cross_validate(X, y_event, y_time, n_splits) dict
        +plot_training_history(save_path) None
        +plot_cv_results(cv_results, save_path) None
    }

    class EnsembleTrainer {
        +int n_models
        +int random_state
        +list~DeepSurv~ models
        +train_ensemble(X, y_event, y_time, config, ...) None
        +predict_risk(X, method) ndarray
    }

    class PlayerPredictor {
        +DeepSurv model
        +KaplanMeierFitter kmf
        +list~str~ feature_names
        -_features_to_array(features) ndarray
        -_baseline_at(times) ndarray
        +predict_player(features, max_games) dict
        -_interpret_prediction(risk_score, median) dict
        +predict_multiple_players(players_df) DataFrame
        +compare_players(players_dict) DataFrame
        +generate_report(prediction, player_name) str
        +save_report(prediction, player_name, filepath) None
    }

    class FamousPlayersPredictor {
        +PlayerPredictor predictor
        +FAMOUS_PLAYERS: dict
        +predict_all_famous_players() DataFrame
        +compare_with_actual() dict
    }

    class SurvivalVisualizer {
        +str style
        +plot_kaplan_meier(y_time, y_event, label, save_path) None
        +plot_km_by_groups(y_time, y_event, groups, ...) None
        +plot_km_by_risk(y_time, y_event, risk_scores, n_groups) None
        +plot_individual_survival(times, probs, name, ...) None
        +plot_risk_distribution(risk_scores, group_labels) None
        +plot_feature_importance(names, scores) None
        +plot_comparison_dashboard(results_dict) None
    }

    NFLDataPreprocessor ..> DeepSurv : provides (X, y_event, y_time)
    ModelTrainer o--> DeepSurv : trains
    EnsembleTrainer o--> DeepSurv : trains N copies
    PlayerPredictor o--> DeepSurv : uses for inference
    FamousPlayersPredictor o--> PlayerPredictor : delegates
    ModelTrainer ..> SurvivalVisualizer : optional plotting
```

### 1.2 API Layer 클래스

```mermaid
classDiagram
    class FastAPI {
        <<framework>>
        +state: AppState
        +include_router(router) None
    }

    class ModelService {
        +str model_path
        +DeepSurv model
        +PlayerPredictor predictor
        +load() None
        +predict(features, max_games) dict
        +predict_famous() list
        -_extrapolation_warnings(features) list
        +feature_names: list
        +metrics: dict
        +training_meta: dict
        +baseline_length: int
    }

    class HealthResponse {
        +str status
        +bool model_loaded
        +list~str~ feature_names
        +dict metrics
        +dict training_meta
        +int baseline_length
    }

    class PredictRequest {
        +dict~str,float~ features
        +int max_games
    }

    class PredictResponse {
        +float risk_score
        +int median_survival
        +str grade
        +str risk_level
        +str comment
        +dict~int,float~ survival_at
        +list~SurvivalPoint~ survival_curve
        +list~str~ extrapolation_warnings
    }

    class SurvivalPoint {
        +int t
        +float s
    }

    class FamousPlayer {
        +str name
        +dict features
        +float risk_score
        +int predicted_career
        +int actual_career
        +float survival_100
        +str grade
    }

    class FamousPlayersResponse {
        +list~FamousPlayer~ players
        +list~str~ feature_names
    }

    class RouteHealth {
        <<router>>
        +health(request) HealthResponse
    }

    class RoutePredict {
        <<router>>
        +predict(payload, request) PredictResponse
    }

    class RoutePlayers {
        <<router>>
        +famous_players(request) FamousPlayersResponse
    }

    FastAPI o--> ModelService : app.state.model_service
    FastAPI ..> RouteHealth : include_router
    FastAPI ..> RoutePredict : include_router
    FastAPI ..> RoutePlayers : include_router

    ModelService o--> DeepSurv
    ModelService o--> PlayerPredictor

    RouteHealth ..> HealthResponse
    RoutePredict ..> PredictRequest
    RoutePredict ..> PredictResponse
    RoutePlayers ..> FamousPlayersResponse

    PredictResponse *-- SurvivalPoint
    FamousPlayersResponse *-- FamousPlayer
```

---

## 2. 컴포넌트 다이어그램 (Frontend)

```mermaid
classDiagram
    class App {
        <<component>>
        +render() JSX
    }

    class Header {
        <<component>>
        +useHealth()
        +render() JSX
    }

    class ErrorBoundary {
        <<component>>
        +componentDidCatch(error) void
        +render() JSX
    }

    class PredictPage {
        <<component>>
        +useHealth()
        +useForm()
        +usePredict()
        +last: PredictResponse
        +onSubmit(data) Promise
    }

    class FamousPage {
        <<component>>
        +useFamousPlayers()
        +sortBy: string
        +withActualOnly: boolean
    }

    class SurvivalChart {
        <<component>>
        +curve: SurvivalPoint[]
        +median: number
    }

    class Card {
        <<component>>
        +p: FamousPlayer
    }

    class apiClient {
        <<module>>
        +health() HealthResponse
        +predict(body) PredictResponse
        +famousPlayers() FamousPlayersResponse
    }

    class queries {
        <<module>>
        +useHealth()
        +useFamousPlayers()
        +usePredict()
    }

    App *-- Header
    App *-- ErrorBoundary
    ErrorBoundary *-- PredictPage : route="/"
    ErrorBoundary *-- FamousPage : route="/famous"
    PredictPage *-- SurvivalChart
    FamousPage *-- Card

    Header ..> queries : useHealth
    PredictPage ..> queries : useHealth, usePredict
    FamousPage ..> queries : useFamousPlayers
    queries ..> apiClient
```

---

## 3. 시퀀스 다이어그램

### 3.1 단일 선수 예측

```mermaid
sequenceDiagram
    actor User
    participant UI as PredictPage
    participant Hook as usePredict (RQ)
    participant Client as api.predict
    participant Route as /api/predict
    participant Service as ModelService
    participant Pred as PlayerPredictor
    participant Model as DeepSurv

    User->>UI: 폼 입력 (BMI, YPC, DrAge, ...)
    User->>UI: Submit 클릭
    UI->>Hook: mutateAsync({features, max_games})
    Hook->>Client: predict(body)
    Client->>Route: POST /api/predict (JSON)
    Route->>Service: predict(features, max_games)
    Service->>Pred: predict_player(features, max_games)
    Pred->>Pred: _features_to_array(features)
    Pred->>Model: predict_risk(X)
    Model-->>Pred: risk_score
    Pred->>Model: baseline_survival_at(times)
    Model-->>Pred: S₀(t)
    Pred->>Pred: S(t|X) = S₀(t)^exp(η)
    Pred-->>Service: prediction dict
    Service->>Service: _extrapolation_warnings(features)
    Service-->>Route: response dict + warnings
    Route-->>Client: PredictResponse JSON
    Client-->>Hook: PredictResponse
    Hook-->>UI: data
    UI->>UI: setLast(data)
    UI-->>User: SurvivalChart + 통계 카드 렌더
```

### 3.2 모델 로딩 (앱 시작 시)

```mermaid
sequenceDiagram
    participant Uvicorn
    participant App as FastAPI
    participant Lifespan
    participant Service as ModelService
    participant Model as DeepSurv

    Uvicorn->>App: 부팅
    App->>Lifespan: __aenter__()
    Lifespan->>Service: ModelService(MODEL_PATH)
    Service->>Model: DeepSurv.load(filepath)
    Model->>Model: load _model.h5 (가중치)
    Model->>Model: load _meta.pkl (scaler, baseline, metrics)
    Model-->>Service: 로드 완료
    Service->>Service: PlayerPredictor 생성
    Lifespan->>App: app.state.model_service = service
    App-->>Uvicorn: 서버 준비 완료

    Note over App: 이후 모든 요청이 동일 인스턴스 공유
```

### 3.3 유명 선수 페이지 로딩

```mermaid
sequenceDiagram
    actor User
    participant UI as FamousPage
    participant Hook as useFamousPlayers
    participant Route as /api/players/famous
    participant Service as ModelService
    participant Famous as FamousPlayersPredictor
    participant Pred as PlayerPredictor

    User->>UI: /famous 라우트 진입
    UI->>Hook: useFamousPlayers()
    Hook->>Route: GET /api/players/famous
    Route->>Service: predict_famous()
    Service->>Famous: FamousPlayersPredictor(predictor)
    Famous->>Famous: predict_all_famous_players()
    loop 각 유명 RB
        Famous->>Pred: predict_player(features)
        Pred-->>Famous: prediction
    end
    Famous-->>Service: DataFrame
    Service-->>Route: list of dicts
    Route-->>Hook: FamousPlayersResponse
    Hook-->>UI: data
    UI-->>User: Card 그리드 렌더링 (정렬/필터 가능)
```

---

## 4. 학습 파이프라인 액티비티 다이어그램

```mermaid
flowchart TD
    Start([시작]) --> Load[NFLDataPreprocessor.load_data]
    Load --> Feat[create_features<br/>BMI, YPC, Years, Retired]
    Feat --> Clean[clean_data<br/>결측치 / 이상치 제거]
    Clean --> Matrix[get_feature_matrix<br/>X, y_event, y_time]
    Matrix --> Split[ModelTrainer.train_test_split_data]
    Split --> Build[DeepSurv 인스턴스 생성<br/>_build_model 호출]
    Build --> Compile[model.compile<br/>Adam + Cox Loss]
    Compile --> Scale[StandardScaler.fit_transform]
    Scale --> Fit[Keras 학습<br/>EarlyStopping, BatchNorm, Dropout]
    Fit --> Baseline[_compute_baseline_survival<br/>Breslow 추정]
    Baseline --> Eval[ModelTrainer.evaluate<br/>C-index 계산]
    Eval --> CV{교차검증?}
    CV -->|Yes| KFold[cross_validate<br/>5-Fold]
    CV -->|No| Save
    KFold --> Save[DeepSurv.save<br/>_model.h5 + _meta.pkl]
    Save --> Plot[SurvivalVisualizer<br/>학습 곡선 / KM 그래프]
    Plot --> End([완료])
```

---

## 5. 추론 파이프라인 액티비티 다이어그램

```mermaid
flowchart TD
    Start([HTTP 요청]) --> Validate[Pydantic<br/>PredictRequest 검증]
    Validate --> GetService[app.state.model_service 조회]
    GetService --> Loaded{모델 로드됨?}
    Loaded -->|No| Err503[503 Service Unavailable]
    Loaded -->|Yes| ToArray[_features_to_array<br/>dict → 1×D 배열]
    ToArray --> Risk[DeepSurv.predict_risk<br/>η = 신경망 출력]
    Risk --> BaseLookup[baseline_survival_at<br/>Breslow 캐시 조회]
    BaseLookup --> Surv["S(t|X) = S₀(t)^exp(η)"]
    Surv --> Median[중앙 생존 시간 계산]
    Median --> Interp[_interpret_prediction<br/>등급/위험도/코멘트]
    Interp --> Warn[_extrapolation_warnings<br/>FEATURE_RANGES 검사]
    Warn --> Pack[PredictResponse 직렬화]
    Pack --> End([JSON 응답])
```

---

## 6. 패키지/모듈 의존성 그래프

```mermaid
flowchart LR
    subgraph Frontend
        AppTSX[App.tsx]
        Pages[pages/<br/>PredictPage<br/>FamousPage]
        Components[components/<br/>Header, SurvivalChart]
        Hooks[hooks/queries.ts]
        ApiClient[api/client.ts]
        ApiTypes[api/types.ts]

        AppTSX --> Pages
        AppTSX --> Components
        Pages --> Hooks
        Components --> Hooks
        Hooks --> ApiClient
        ApiClient --> ApiTypes
    end

    subgraph Backend_API[Backend API]
        Main[app/main.py]
        Routes[api/routes_*.py]
        Schemas[app/schemas.py]
        Service[services/model_service.py]
        Config[app/config.py]

        Main --> Routes
        Main --> Service
        Routes --> Schemas
        Routes --> Service
        Service --> Config
    end

    subgraph ML_Core[ML Core]
        Preproc[data_preprocessing.py]
        ModelArch[model_architecture.py]
        Training[model_training.py]
        PredUtils[prediction_utils.py]
        Viz[visualization.py]
        TrainCLI[train_cli.py]

        TrainCLI --> Preproc
        TrainCLI --> ModelArch
        TrainCLI --> Training
        TrainCLI --> PredUtils
        TrainCLI --> Viz
        Training --> ModelArch
        PredUtils --> ModelArch
    end

    ApiClient -.HTTP.-> Routes
    Service --> ModelArch
    Service --> PredUtils
```

---

## 7. 상태 다이어그램 — 모델 라이프사이클

```mermaid
stateDiagram-v2
    [*] --> Uninitialized
    Uninitialized --> Compiled: __init__ → _build_model
    Compiled --> Trained: fit() 완료
    Trained --> WithBaseline: _compute_baseline_survival
    WithBaseline --> Persisted: save()
    Persisted --> Loaded: DeepSurv.load()
    Loaded --> Serving: ModelService.load()
    Serving --> Serving: predict_risk / predict_survival
    Serving --> [*]: 앱 종료
```

---

## 8. 도메인 엔터티 (ER 스타일)

```mermaid
erDiagram
    PLAYER ||--o{ SEASON : has
    PLAYER {
        int Rk
        int DraftYear
        int Rnd
        int Pick
        int DrAge
        string Pos
        float BMI
        float Height
        float Weight
        string College
    }
    SEASON {
        int From
        int To
        int Games
        int GamesStarted
        int Att
        int Yds
        int TD
        int Rec
        int RecYds
        int RecTD
        int AP1
        int PB
    }
    PLAYER ||--|| SURVIVAL_RECORD : "derived"
    SURVIVAL_RECORD {
        int y_time "총 경기 수"
        int y_event "1=은퇴, 0=검열"
        float YPC
        float TD_per_game
        bool Retired
    }
```

---

## 9. 다이어그램 보는 방법

- **GitHub / GitLab**: `.md` 파일을 열면 Mermaid 코드 블록이 자동으로 렌더링됩니다.
- **VS Code**: [Markdown Preview Mermaid Support](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid) 확장 사용.
- **JetBrains IDE**: 내장 마크다운 미리보기에서 Mermaid 지원 (2023.2+).
- **CLI 렌더링**: `npx @mermaid-js/mermaid-cli -i uml.md -o uml.svg`

---

## 10. 다이어그램 갱신 가이드

이 문서의 다이어그램은 다음 변경 시 함께 업데이트해야 합니다.

| 변경 내용 | 영향받는 다이어그램 |
|-----------|---------------------|
| 새 클래스/모듈 추가 | §1 클래스 다이어그램, §6 의존성 그래프 |
| API 엔드포인트 추가 | §1.2, §3 시퀀스, §6 의존성 |
| Pydantic 스키마 변경 | §1.2 |
| 프런트 컴포넌트 추가 | §2 |
| 학습/추론 단계 변경 | §4, §5 액티비티 다이어그램 |
| 모델 저장 형식 변경 | §7 상태 다이어그램 |
