"""
모델 아키텍처 모듈
DeepSurv 신경망 모델 정의 및 커스텀 손실 함수
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List
import pickle
import os


# ==================== Custom Loss Functions ====================

def cox_partial_likelihood_loss(y_true, y_pred):
    """
    Cox Partial Likelihood 손실 함수
    
    Parameters:
    -----------
    y_true : tensor
        [event, time] - 이벤트 발생 여부 및 생존 시간
    y_pred : tensor
        위험 점수 (risk scores)
    
    Returns:
    --------
    loss : scalar tensor
        Cox partial likelihood loss
    """
    # 이벤트와 시간 추출
    event = y_true[:, 0]
    time = y_true[:, 1]
    
    # 위험 점수
    risk_scores = y_pred[:, 0]
    
    # 시간 기준 내림차순 정렬
    sorted_indices = tf.argsort(time, direction='DESCENDING')
    sorted_event = tf.gather(event, sorted_indices)
    sorted_risk = tf.gather(risk_scores, sorted_indices)
    
    # 위험 비율 (Hazard Ratio)
    hazard_ratio = tf.exp(sorted_risk)
    
    # Log risk
    log_risk = sorted_risk
    
    # 누적 위험 비율 (Risk Set)
    cumsum_hazard = tf.cumsum(hazard_ratio)
    
    # Log cumulative hazard
    log_cumsum_hazard = tf.math.log(cumsum_hazard + 1e-7)
    
    # Partial likelihood
    likelihood = (log_risk - log_cumsum_hazard) * sorted_event
    
    # Negative log likelihood (최소화할 손실)
    loss = -tf.reduce_mean(likelihood)
    
    return loss


def weighted_cox_loss(y_true, y_pred, alpha=1.0):
    """
    가중치가 적용된 Cox 손실 함수
    
    Parameters:
    -----------
    alpha : float
        L2 정규화 가중치
    """
    cox_loss = cox_partial_likelihood_loss(y_true, y_pred)
    l2_loss = alpha * tf.reduce_sum(tf.square(y_pred))
    
    return cox_loss + l2_loss


# ==================== DeepSurv Model Architecture ====================

class DeepSurv:
    """
    DeepSurv: Cox Proportional Hazards 모델의 딥러닝 버전
    
    신경망을 사용하여 비선형 위험 함수를 학습합니다.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_layers: List[int] = [64, 32, 16],
                 dropout_rate: float = 0.3,
                 activation: str = 'relu',
                 l2_reg: float = 0.01,
                 feature_names: Optional[List[str]] = None):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.l2_reg = l2_reg
        self.feature_names = list(feature_names) if feature_names else []

        self.model = None
        self.scaler = StandardScaler()
        self.history = None

        # Cox baseline survival is computed once on the training set after fit()
        # (Breslow estimator) and persisted with the model so prediction-only
        # mode does not need a dummy KaplanMeierFitter.
        self.baseline_times: Optional[np.ndarray] = None
        self.baseline_survival: Optional[np.ndarray] = None

        # Trained-model metadata surfaced in /api/health so the UI can render
        # provenance without inspecting the model file directly.
        self.metrics: dict = {}
        self.training_meta: dict = {}

        self._build_model()
    
    def _build_model(self):
        """신경망 아키텍처 구축"""
        inputs = layers.Input(shape=(self.input_dim,), name='input')
        x = inputs
        
        # 은닉층
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(
                units,
                activation=self.activation,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                name=f'dense_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
        
        # 출력층: 단일 위험 점수 (활성화 함수 없음)
        outputs = layers.Dense(1, activation='linear', name='risk_score')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='DeepSurv')
    
    def compile(self, 
                learning_rate: float = 0.001,
                optimizer: str = 'adam',
                loss_fn=cox_partial_likelihood_loss):
        """
        모델 컴파일
        
        Parameters:
        -----------
        learning_rate : float
            학습률
        optimizer : str or optimizer
            최적화 알고리즘
        loss_fn : function
            손실 함수
        """
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=['mae']
        )
    
    def fit(self,
            X: np.ndarray,
            y_event: np.ndarray,
            y_time: np.ndarray,
            validation_split: float = 0.2,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 1,
            callbacks: Optional[List] = None):
        """
        모델 학습
        
        Parameters:
        -----------
        X : np.ndarray
            특징 행렬
        y_event : np.ndarray
            이벤트 발생 여부
        y_time : np.ndarray
            생존 시간
        validation_split : float
            검증 데이터 비율
        epochs : int
            학습 에폭 수
        batch_size : int
            배치 크기
        verbose : int
            출력 레벨 (0: 없음, 1: 진행바, 2: 에폭당 한 줄)
        callbacks : list
            콜백 함수 리스트
        
        Returns:
        --------
        history : History
            학습 이력
        """
        # 특징 정규화
        X_scaled = self.scaler.fit_transform(X)
        
        # 이벤트와 시간 결합
        y_combined = np.column_stack([y_event, y_time])
        
        # 기본 콜백 설정
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
        
        # 학습 실행
        self.history = self.model.fit(
            X_scaled, y_combined,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        # Compute baseline survival on the training set for later prediction.
        self._compute_baseline_survival(X_scaled, y_event, y_time)

        return self.history

    def _compute_baseline_survival(self,
                                    X_scaled: np.ndarray,
                                    y_event: np.ndarray,
                                    y_time: np.ndarray):
        """Breslow baseline cumulative hazard / survival on training data."""
        risk = self.model.predict(X_scaled, verbose=0).flatten()
        exp_risk = np.exp(risk)

        order = np.argsort(y_time)
        t_sorted = y_time[order]
        e_sorted = y_event[order]
        r_sorted = exp_risk[order]

        # Risk set at each time = sum of exp(risk) over individuals still at risk.
        # Walking forward in time, the risk set shrinks as people leave.
        risk_set = r_sorted[::-1].cumsum()[::-1]

        unique_t = np.unique(t_sorted)
        cum_hazard = np.zeros_like(unique_t, dtype=float)
        H = 0.0
        for i, t in enumerate(unique_t):
            mask = t_sorted == t
            d = e_sorted[mask].sum()
            denom = risk_set[np.argmax(t_sorted >= t)]
            if denom > 0 and d > 0:
                H += d / denom
            cum_hazard[i] = H

        self.baseline_times = unique_t
        self.baseline_survival = np.exp(-cum_hazard)

    def baseline_survival_at(self, times: np.ndarray) -> np.ndarray:
        """Step-function lookup of baseline survival at given times."""
        if self.baseline_times is None:
            raise RuntimeError("Baseline survival not available — call fit() first or load() a saved model.")
        idx = np.searchsorted(self.baseline_times, times, side='right') - 1
        idx = np.clip(idx, 0, len(self.baseline_times) - 1)
        out = self.baseline_survival[idx]
        # Times before the first event default to S(t)=1
        out = np.where(times < self.baseline_times[0], 1.0, out)
        return out
    
    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """
        위험 점수 예측
        
        Parameters:
        -----------
        X : np.ndarray
            특징 행렬
        
        Returns:
        --------
        risk_scores : np.ndarray
            위험 점수 (낮을수록 좋음)
        """
        X_scaled = self.scaler.transform(X)
        risk_scores = self.model.predict(X_scaled, verbose=0).flatten()
        return risk_scores
    
    def predict_survival(self,
                        X: np.ndarray,
                        times: np.ndarray,
                        baseline_survival: Optional[np.ndarray] = None) -> np.ndarray:
        """
        S(t|X) = S0(t) ^ exp(risk_score)

        baseline_survival가 None이면 모델에 저장된 학습 baseline을 사용한다.
        """
        risk_scores = self.predict_risk(X)
        if baseline_survival is None:
            baseline = self.baseline_survival_at(np.asarray(times))
        else:
            baseline = np.asarray(baseline_survival)

        baseline = np.clip(baseline, 1e-12, 1.0)
        return baseline[None, :] ** np.exp(risk_scores)[:, None]
    
    def save(self, filepath: str):
        """Save model weights and all metadata needed for prediction-only mode."""
        self.model.save(f'{filepath}_model.h5')

        meta = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'baseline_times': self.baseline_times,
            'baseline_survival': self.baseline_survival,
            'config': self.get_config(),
            'metrics': self.metrics,
            'training_meta': self.training_meta,
        }
        with open(f'{filepath}_meta.pkl', 'wb') as f:
            pickle.dump(meta, f)

        print(f"✓ 모델 저장 완료: {filepath}")

    def load(self, filepath: str):
        """Load model weights, scaler, baseline survival, and feature metadata."""
        self.model = keras.models.load_model(
            f'{filepath}_model.h5',
            custom_objects={'cox_partial_likelihood_loss': cox_partial_likelihood_loss}
        )

        # New unified metadata file; fall back to legacy `_scaler.pkl` if absent.
        meta_path = f'{filepath}_meta.pkl'
        legacy_scaler_path = f'{filepath}_scaler.pkl'
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.scaler = meta['scaler']
            self.feature_names = meta.get('feature_names', [])
            self.baseline_times = meta.get('baseline_times')
            self.baseline_survival = meta.get('baseline_survival')
            self.metrics = meta.get('metrics', {})
            self.training_meta = meta.get('training_meta', {})
        elif os.path.exists(legacy_scaler_path):
            with open(legacy_scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("⚠️  Legacy save format — baseline survival not available, "
                  "predictions will fall back to a uniform exponential baseline.")
        else:
            raise FileNotFoundError(f"No metadata found at {meta_path} or {legacy_scaler_path}")

        print(f"✓ 모델 로드 완료: {filepath}")
    
    def summary(self):
        """모델 구조 출력"""
        self.model.summary()
    
    def get_config(self) -> dict:
        """모델 설정 반환"""
        return {
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'l2_reg': self.l2_reg,
            'feature_names': self.feature_names,
        }


# ==================== 모델 빌더 유틸리티 ====================

def create_deepsurv_model(input_dim: int,
                          model_type: str = 'standard') -> DeepSurv:
    """
    사전 정의된 모델 아키텍처 생성
    
    Parameters:
    -----------
    input_dim : int
        입력 특징 차원
    model_type : str
        모델 타입 ('standard', 'deep', 'wide', 'simple')
    
    Returns:
    --------
    model : DeepSurv
        생성된 모델
    """
    architectures = {
        'standard': {
            'hidden_layers': [64, 32, 16],
            'dropout_rate': 0.3
        },
        'deep': {
            'hidden_layers': [128, 64, 32, 16, 8],
            'dropout_rate': 0.4
        },
        'wide': {
            'hidden_layers': [256, 128, 64],
            'dropout_rate': 0.3
        },
        'simple': {
            'hidden_layers': [32, 16],
            'dropout_rate': 0.2
        }
    }
    
    if model_type not in architectures:
        raise ValueError(f"Unknown model type: {model_type}")
    
    config = architectures[model_type]
    model = DeepSurv(input_dim=input_dim, **config)
    
    return model


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 70)
    print("DeepSurv 모델 아키텍처 테스트")
    print("=" * 70)
    
    # 샘플 데이터 생성
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    
    X = np.random.randn(n_samples, n_features)
    y_event = np.ones(n_samples)
    y_time = np.random.exponential(50, n_samples)
    
    # 모델 생성 및 컴파일
    model = DeepSurv(input_dim=n_features, hidden_layers=[32, 16])
    model.compile(learning_rate=0.001)
    
    print("\n모델 구조:")
    model.summary()
    
    # 학습
    print("\n모델 학습 중...")
    history = model.fit(X, y_event, y_time, epochs=10, verbose=0)
    
    print(f"✓ 학습 완료")
    print(f"  최종 Loss: {history.history['loss'][-1]:.4f}")
    
    # 예측
    risk_scores = model.predict_risk(X[:5])
    print(f"\n처음 5개 샘플의 위험 점수:")
    for i, risk in enumerate(risk_scores):
        print(f"  Sample {i+1}: {risk:.4f}")
