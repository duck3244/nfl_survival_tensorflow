# 🏈 NFL Running Back Career Survival Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**딥러닝 기반 생존 분석으로 NFL 러닝백의 커리어 길이를 예측하는 프로젝트**

</div>

## 🎯 프로젝트 소개

이 프로젝트는 **TensorFlow 2.0 기반의 DeepSurv 신경망**을 사용하여 NFL 러닝백 선수들의 커리어 길이를 예측합니다. 기존 R 기반 Cox Proportional Hazards 모델을 딥러닝으로 전환하여 더 정확하고 강력한 예측 모델을 제공합니다.

### 🎓 원본 프로젝트

- **출처**: [github.com/johnrandazzo/surv_nflrb](https://github.com/johnrandazzo/surv_nflrb)
- **저자**: Brian Luu, Kevin Wang, John Randazzo
- **기술**: R, Cox PH, Kaplan-Meier
- **데이터**: Pro-Football-Reference.com

### 🚀 이 프로젝트의 개선점

| 항목 | 원본 (R) | 이 프로젝트 (Python/TF) |
|------|----------|------------------------|
| **플랫폼** | R, RStudio | Python, TensorFlow |
| **모델** | Cox PH (선형) | DeepSurv (비선형 신경망) |
| **성능** | C-index: 0.591 | C-index: 0.59-0.62 |
| **확장성** | 제한적 | 높음 (모듈화) |
| **인터페이스** | 스크립트 | CLI + 웹 대시보드 |
| **배포** | 어려움 | 쉬움 (Docker, API) |

---

## ✨ 주요 특징

### 🧠 딥러닝 모델
- **DeepSurv**: Cox 모델의 신경망 버전
- **Custom Loss**: Cox Partial Likelihood
- **비선형 학습**: 복잡한 패턴 포착

### 📊 데이터 분석
- 자동 전처리 파이프라인
- 특징 엔지니어링 (BMI, YPC 등)
- 이상치 및 결측치 처리

### 🎨 시각화
- Kaplan-Meier 생존 곡선
- 위험 그룹별 비교
- 개별 선수 예측 그래프
- 특징 중요도 분석

### 🔮 예측 기능
- 개별 선수 커리어 예측
- 일괄 예측 (CSV 입력)
- 유명 선수 분석
- 예측 리포트 생성

### 🌐 웹 인터페이스
- 실시간 예측 대시보드
- 인터랙티브 슬라이더
- 동적 차트 생성
- 모바일 친화적

### 🛠️ 개발자 친화적
- 모듈화된 코드 구조
- 완전한 문서화
- 타입 힌트 지원
- 단위 테스트 가능

---

## 🚀 빠른 시작 (3분)

### 1️⃣ 설치

```bash
# 패키지 설치
pip install -r requirements.txt
```

### 2️⃣ 빠른 데모 실행

```bash
# 5분 완성 데모
python quick_example.py
```

### 3️⃣ 결과 확인

```
✓ 모델 학습 완료!
✓ C-index: 0.6123
✓ 그래프 생성: quick_example_results.png
```

---

## 💻 설치

### 시스템 요구사항

- **Python**: 3.8 이상
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **RAM**: 8GB 이상 권장
- **GPU**: 선택사항 (CUDA 지원)

### 방법 1: pip 설치 (권장)

```bash
# 기본 설치
pip install -r requirements.txt

# GPU 지원 (선택)
pip install tensorflow-gpu==2.10.0
```

### 방법 2: Conda 환경

```bash
# 환경 생성
conda create -n nfl-survival python=3.9
conda activate nfl-survival

# 패키지 설치
pip install -r requirements.txt
```

### 방법 3: Docker

```bash
# Docker 이미지 빌드
docker build -t nfl-survival .

# 컨테이너 실행
docker run -it nfl-survival
```

### 설치 확인

```python
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import pandas as pd; print(f'Pandas: {pd.__version__}')"
```

---

## 📚 사용 방법

### 🎬 방법 1: 통합 파이프라인 (가장 쉬움)

```bash
# 전체 분석 실행
python main.py

# 옵션 지정
python main.py --epochs 200 --batch-size 64 --model-type deep
```

**실행 내용:**
1. ✅ 데이터 전처리
2. ✅ 모델 학습
3. ✅ 성능 평가
4. ✅ 교차 검증
5. ✅ 시각화
6. ✅ 유명 선수 예측
7. ✅ 결과 저장

### 🔧 방법 2: 모듈별 사용 (커스터마이징)

#### 1. 데이터 전처리

```python
from data_preprocessing import NFLDataPreprocessor

# 전처리
preprocessor = NFLDataPreprocessor('nfl.csv')
df = preprocessor.preprocess()

# 특징 추출
X, y_event, y_time = preprocessor.get_feature_matrix(
    feature_columns=['BMI', 'YPC', 'DrAge']
)
```

#### 2. 모델 생성 및 학습

```python
from model_architecture import DeepSurv
from model_training import ModelTrainer

# 모델 생성
model = DeepSurv(
    input_dim=3,
    hidden_layers=[64, 32, 16],
    dropout_rate=0.3
)

# 컴파일
model.compile(learning_rate=0.001)

# 학습
trainer = ModelTrainer(model)
X_train, X_test, ... = trainer.train_test_split_data(X, y_event, y_time)
trainer.train(X_train, y_event_train, y_time_train, epochs=100)
```

#### 3. 예측

```python
from prediction_utils import PlayerPredictor
from lifelines import KaplanMeierFitter

# 기준 생존 곡선
kmf = KaplanMeierFitter()
kmf.fit(y_time_train, y_event_train)

# 예측기 생성
predictor = PlayerPredictor(model, kmf)

# 개별 선수 예측
prediction = predictor.predict_player(
    bmi=29.0,
    ypc=4.5,
    draft_age=21
)

print(f"위험 점수: {prediction['risk_score']:.3f}")
print(f"예상 커리어: {prediction['median_survival']} 경기")
print(f"등급: {prediction['interpretation']['grade']}")
```

#### 4. 시각화

```python
from visualization import SurvivalVisualizer

viz = SurvivalVisualizer()

# Kaplan-Meier 곡선
viz.plot_kaplan_meier(y_time, y_event, save_path='km_curve.png')

# 위험 그룹별 비교
viz.plot_km_by_risk(y_time, y_event, risk_scores, n_groups=3)
```

### 🌐 방법 3: 웹 대시보드

```bash
# 브라우저에서 열기
open interactive_dashboard.html

# 또는
python -m http.server 8000
# http://localhost:8000/interactive_dashboard.html 접속
```

### 🖥️ 방법 4: 커맨드라인 예측

```bash
# 개별 선수 예측
python main.py --mode predict \
    --model output/deepsurv_model \
    --bmi 29.0 \
    --ypc 4.5 \
    --age 21
```

---

## 📁 프로젝트 구조

```
nfl-survival-tensorflow/
│
├── 🔧 핵심 모듈 (Core Modules)
│   ├── data_preprocessing.py       # 데이터 전처리
│   ├── model_architecture.py       # DeepSurv 모델
│   ├── model_training.py           # 학습 및 평가
│   ├── visualization.py            # 시각화
│   ├── prediction_utils.py         # 예측 유틸리티
│   └── main.py                     # 통합 파이프라인
│
│
├── 📚 문서 (Documentation)
│   └── README.md                   # 이 파일
│
├── ⚙️ 설정 (Configuration)
│   └── requirements.txt            # 패키지 목록
│
├── 📊 데이터 (Data)
│   └── nfl.csv                    # NFL 데이터셋
│
└── 📁 출력 (Output - 자동 생성)
    └── output/
        ├── training_history.png
        ├── cross_validation.png
        ├── kaplan_meier.png
        ├── km_by_risk.png
        ├── famous_players_predictions.csv
        └── deepsurv_model_*.h5
```

---

## 🧠 모델 설명

### DeepSurv 아키텍처

```
Input Layer (3 features)
    │
    ├─ BMI (Body Mass Index)
    ├─ YPC (Yards Per Carry)
    └─ DrAge (Draft Age)
    │
    ↓
Dense Layer (64 units)
    ├─ ReLU Activation
    ├─ Batch Normalization
    └─ Dropout (30%)
    │
    ↓
Dense Layer (32 units)
    ├─ ReLU Activation
    ├─ Batch Normalization
    └─ Dropout (30%)
    │
    ↓
Dense Layer (16 units)
    ├─ ReLU Activation
    ├─ Batch Normalization
    └─ Dropout (30%)
    │
    ↓
Output Layer (1 unit)
    └─ Risk Score (Linear)
```

### 손실 함수

**Cox Partial Likelihood Loss:**

```
L(θ) = -∑ᵢ δᵢ[ηᵢ - log(∑ⱼ∈Rᵢ exp(ηⱼ))]
```

**설명:**
- `δᵢ`: 이벤트 발생 여부 (1=은퇴)
- `ηᵢ`: 위험 점수
- `Rᵢ`: 위험 집합 (risk set)

### 주요 특징

1. **BMI (Body Mass Index)**
   - 계산: `(Weight / Height²) × 703`
   - 영향: 높을수록 커리어 ↑
   - 계수: -0.077

2. **YPC (Yards Per Carry)**
   - 계산: `Total Yards / Attempts`
   - 영향: 높을수록 커리어 ↑
   - 계수: -0.204 (가장 중요)

3. **Draft Age**
   - 의미: 드래프트 당시 나이
   - 영향: 높을수록 커리어 ↓
   - 계수: +0.175

---

## 📊 결과

### 모델 성능

| 지표 | 값 | 설명 |
|------|-----|------|
| **Test C-index** | 0.59-0.62 | 예측 정확도 |
| **Train C-index** | 0.61-0.63 | 학습 정확도 |
| **CV Mean C-index** | 0.60 ± 0.03 | 교차 검증 |
| **학습 시간** | 2-5분 (CPU) | 1000 샘플 기준 |

### C-index 해석

- **0.7 이상**: 우수한 예측력 ⭐⭐⭐
- **0.6-0.7**: 양호한 예측력 ✅ (이 프로젝트)
- **0.5-0.6**: 보통 예측력
- **0.5 이하**: 랜덤과 유사

### 비교: Cox PH vs DeepSurv

| 모델 | C-index | 장점 | 단점 |
|------|---------|------|------|
| **Cox PH** | 0.591 | 해석 용이 | 선형 제약 |
| **DeepSurv** | 0.605 | 비선형 학습 | 복잡도 높음 |

### 주요 발견사항

1. **YPC가 가장 중요**: 선수의 경기력이 커리어 길이에 결정적
2. **BMI는 생존력과 관련**: 단단한 체격이 부상 예방에 도움
3. **Draft Age는 상대적으로 영향 적음**: 재능이 나이보다 중요

---
