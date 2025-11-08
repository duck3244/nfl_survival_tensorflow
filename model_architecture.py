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
                 l2_reg: float = 0.01):
        """
        Parameters:
        -----------
        input_dim : int
            입력 특징 차원
        hidden_layers : list
            각 은닉층의 뉴런 수 리스트
        dropout_rate : float
            Dropout 비율 (0~1)
        activation : str
            활성화 함수
        l2_reg : float
            L2 정규화 계수
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.l2_reg = l2_reg
        
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
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
        
        return self.history
    
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
                        baseline_survival: np.ndarray,
                        times: np.ndarray) -> np.ndarray:
        """
        생존 확률 예측
        
        S(t|X) = S0(t)^exp(risk_score)
        
        Parameters:
        -----------
        X : np.ndarray
            특징 행렬
        baseline_survival : np.ndarray
            기준 생존 확률
        times : np.ndarray
            시간 포인트
        
        Returns:
        --------
        survival_probs : np.ndarray
            각 시간에서의 생존 확률
        """
        risk_scores = self.predict_risk(X)
        
        survival_probs = []
        for risk in risk_scores:
            surv = baseline_survival ** np.exp(risk)
            survival_probs.append(surv)
        
        return np.array(survival_probs)
    
    def save(self, filepath: str):
        """모델 저장"""
        self.model.save(f'{filepath}_model.h5')
        
        # Scaler 저장
        with open(f'{filepath}_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"✓ 모델 저장 완료: {filepath}")
    
    def load(self, filepath: str):
        """모델 로드"""
        self.model = keras.models.load_model(
            f'{filepath}_model.h5',
            custom_objects={'cox_partial_likelihood_loss': cox_partial_likelihood_loss}
        )
        
        # Scaler 로드
        with open(f'{filepath}_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
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
            'l2_reg': self.l2_reg
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
