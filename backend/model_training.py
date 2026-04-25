"""
모델 학습 및 평가 모듈
학습, 평가, 검증 기능 제공
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from lifelines.utils import concordance_index
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:
    """모델 학습 및 평가 클래스"""
    
    def __init__(self, model, random_state: int = 42):
        """
        Parameters:
        -----------
        model : DeepSurv
            학습할 모델
        random_state : int
            난수 시드
        """
        self.model = model
        self.random_state = random_state
        self.train_c_index = None
        self.test_c_index = None
        self.history = None
        
    def train_test_split_data(self,
                               X: np.ndarray,
                               y_event: np.ndarray,
                               y_time: np.ndarray,
                               test_size: float = 0.2) -> Tuple:
        """
        학습/테스트 데이터 분할
        
        Parameters:
        -----------
        X : np.ndarray
            특징 행렬
        y_event : np.ndarray
            이벤트 발생 여부
        y_time : np.ndarray
            생존 시간
        test_size : float
            테스트 데이터 비율
        
        Returns:
        --------
        tuple
            (X_train, X_test, y_event_train, y_event_test, 
             y_time_train, y_time_test)
        """
        return train_test_split(
            X, y_event, y_time,
            test_size=test_size,
            random_state=self.random_state
        )
    
    def train(self,
              X_train: np.ndarray,
              y_event_train: np.ndarray,
              y_time_train: np.ndarray,
              validation_split: float = 0.2,
              epochs: int = 100,
              batch_size: int = 32,
              verbose: int = 1):
        """
        모델 학습
        
        Parameters:
        -----------
        X_train : np.ndarray
            학습 특징
        y_event_train : np.ndarray
            학습 이벤트
        y_time_train : np.ndarray
            학습 시간
        validation_split : float
            검증 데이터 비율
        epochs : int
            에폭 수
        batch_size : int
            배치 크기
        verbose : int
            출력 레벨
        
        Returns:
        --------
        history
            학습 이력
        """
        print("=" * 70)
        print("모델 학습 시작")
        print("=" * 70)
        print(f"학습 데이터: {len(X_train)}개")
        print(f"검증 비율: {validation_split}")
        print(f"에폭: {epochs}, 배치 크기: {batch_size}")
        
        self.history = self.model.fit(
            X_train, y_event_train, y_time_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        print("\n✓ 학습 완료!")
        
        return self.history
    
    def evaluate(self,
                 X_train: np.ndarray,
                 y_event_train: np.ndarray,
                 y_time_train: np.ndarray,
                 X_test: np.ndarray,
                 y_event_test: np.ndarray,
                 y_time_test: np.ndarray) -> Dict[str, float]:
        """
        모델 평가 (C-index 계산)
        
        Parameters:
        -----------
        X_train, X_test : np.ndarray
            학습/테스트 특징
        y_event_train, y_event_test : np.ndarray
            학습/테스트 이벤트
        y_time_train, y_time_test : np.ndarray
            학습/테스트 시간
        
        Returns:
        --------
        metrics : dict
            평가 지표 딕셔너리
        """
        print("\n" + "=" * 70)
        print("모델 평가")
        print("=" * 70)
        
        # 위험 점수 예측
        train_risk = self.model.predict_risk(X_train)
        test_risk = self.model.predict_risk(X_test)

        # C-index 계산
        self.train_c_index = concordance_index(y_time_train, -train_risk, y_event_train)
        self.test_c_index = concordance_index(y_time_test, -test_risk, y_event_test)

        metrics = {
            'train_c_index': self.train_c_index,
            'test_c_index': self.test_c_index,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }

        print(f"\n📊 성능 지표:")
        print(f"  Training C-index:   {self.train_c_index:.4f}")
        print(f"  Testing C-index:    {self.test_c_index:.4f}")
        print(f"  Overfitting Gap:    {abs(self.train_c_index - self.test_c_index):.4f}")

        # 성능 평가
        if self.test_c_index > 0.7:
            print("  ⭐ 우수한 성능!")
        elif self.test_c_index > 0.6:
            print("  ✅ 양호한 성능!")
        elif self.test_c_index > 0.55:
            print("  📊 보통 성능")
        else:
            print("  ⚠️  성능 개선 필요")

        return metrics

    def cross_validate(self,
                       X: np.ndarray,
                       y_event: np.ndarray,
                       y_time: np.ndarray,
                       n_splits: int = 5,
                       epochs: int = 50,
                       verbose: int = 0) -> Dict[str, any]:
        """
        K-Fold 교차 검증

        Parameters:
        -----------
        X : np.ndarray
            전체 특징 행렬
        y_event : np.ndarray
            전체 이벤트
        y_time : np.ndarray
            전체 시간
        n_splits : int
            Fold 수
        epochs : int
            각 Fold당 에폭 수
        verbose : int
            출력 레벨

        Returns:
        --------
        cv_results : dict
            교차 검증 결과
        """
        print("\n" + "=" * 70)
        print(f"{n_splits}-Fold 교차 검증")
        print("=" * 70)

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        c_indices = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            print(f"\nFold {fold}/{n_splits}...")

            X_train, X_val = X[train_idx], X[val_idx]
            y_event_train, y_event_val = y_event[train_idx], y_event[val_idx]
            y_time_train, y_time_val = y_time[train_idx], y_time[val_idx]

            # 새 모델 생성 (각 Fold마다)
            from model_architecture import DeepSurv
            config = self.model.get_config()
            fold_model = DeepSurv(**config)
            fold_model.compile()

            # 학습
            fold_model.fit(
                X_train, y_event_train, y_time_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=32,
                verbose=verbose
            )

            # 평가
            risk_scores = fold_model.predict_risk(X_val)
            c_index = concordance_index(y_time_val, -risk_scores, y_event_val)
            c_indices.append(c_index)

            print(f"  C-index: {c_index:.4f}")

        # 결과 요약
        cv_results = {
            'c_indices': c_indices,
            'mean_c_index': np.mean(c_indices),
            'std_c_index': np.std(c_indices),
            'min_c_index': np.min(c_indices),
            'max_c_index': np.max(c_indices)
        }

        print("\n" + "-" * 70)
        print(f"📊 교차 검증 결과:")
        print(f"  평균 C-index:  {cv_results['mean_c_index']:.4f} ± {cv_results['std_c_index']:.4f}")
        print(f"  최소 C-index:  {cv_results['min_c_index']:.4f}")
        print(f"  최대 C-index:  {cv_results['max_c_index']:.4f}")

        if cv_results['std_c_index'] < 0.03:
            print("  ✅ 매우 안정적인 모델")
        elif cv_results['std_c_index'] < 0.05:
            print("  👍 안정적인 모델")
        else:
            print("  ⚠️  불안정한 모델 - 하이퍼파라미터 조정 필요")

        return cv_results

    def plot_training_history(self, save_path: Optional[str] = None):
        """학습 곡선 시각화"""
        if self.history is None:
            print("❌ 학습 이력이 없습니다. 먼저 모델을 학습하세요.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss 곡선
        axes[0].plot(self.history.history['loss'],
                    label='Training Loss', linewidth=2, color='#3498db')
        axes[0].plot(self.history.history['val_loss'],
                    label='Validation Loss', linewidth=2, color='#e74c3c')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(alpha=0.3)

        # MAE 곡선
        axes[1].plot(self.history.history['mae'],
                    label='Training MAE', linewidth=2, color='#3498db')
        axes[1].plot(self.history.history['val_mae'],
                    label='Validation MAE', linewidth=2, color='#e74c3c')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 학습 곡선 저장: {save_path}")

        plt.show()

    def plot_cv_results(self, cv_results: Dict, save_path: Optional[str] = None):
        """교차 검증 결과 시각화"""
        c_indices = cv_results['c_indices']
        n_splits = len(c_indices)

        fig, ax = plt.subplots(figsize=(10, 6))

        # C-index 플롯
        ax.plot(range(1, n_splits + 1), c_indices,
               marker='o', linewidth=2.5, markersize=10,
               color='#e74c3c', label='C-index')

        # 평균선
        mean_c = cv_results['mean_c_index']
        ax.axhline(y=mean_c, color='#3498db',
                  linestyle='--', linewidth=2,
                  label=f'Mean: {mean_c:.4f}')

        # 신뢰구간
        std_c = cv_results['std_c_index']
        ax.fill_between(range(1, n_splits + 1),
                        mean_c - std_c,
                        mean_c + std_c,
                        alpha=0.2, color='#3498db',
                        label=f'±1 SD: {std_c:.4f}')

        ax.set_xlabel('Fold', fontsize=12)
        ax.set_ylabel('C-index', fontsize=12)
        ax.set_title(f'{n_splits}-Fold Cross Validation Results',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xticks(range(1, n_splits + 1))
        ax.set_ylim([min(c_indices) - 0.05, max(c_indices) + 0.05])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 교차 검증 결과 저장: {save_path}")

        plt.show()


class EnsembleTrainer:
    """앙상블 모델 학습 클래스"""

    def __init__(self, n_models: int = 5, random_state: int = 42):
        """
        Parameters:
        -----------
        n_models : int
            앙상블할 모델 수
        random_state : int
            난수 시드
        """
        self.n_models = n_models
        self.random_state = random_state
        self.models = []

    def train_ensemble(self,
                       X: np.ndarray,
                       y_event: np.ndarray,
                       y_time: np.ndarray,
                       model_config: Dict,
                       epochs: int = 50,
                       verbose: int = 0):
        """
        앙상블 모델 학습

        Parameters:
        -----------
        X : np.ndarray
            특징 행렬
        y_event : np.ndarray
            이벤트
        y_time : np.ndarray
            시간
        model_config : dict
            모델 설정
        epochs : int
            에폭 수
        verbose : int
            출력 레벨
        """
        print(f"\n앙상블 모델 학습 시작 ({self.n_models}개 모델)")
        print("=" * 70)

        from model_architecture import DeepSurv

        for i in range(self.n_models):
            print(f"\n모델 {i+1}/{self.n_models} 학습 중...")

            # 모델 생성
            model = DeepSurv(**model_config)
            model.compile(learning_rate=0.001)

            # Bootstrap 샘플링
            np.random.seed(self.random_state + i)
            indices = np.random.choice(len(X), size=len(X), replace=True)

            X_boot = X[indices]
            y_event_boot = y_event[indices]
            y_time_boot = y_time[indices]

            # 학습
            model.fit(X_boot, y_event_boot, y_time_boot,
                     validation_split=0.2,
                     epochs=epochs,
                     batch_size=32,
                     verbose=verbose)

            self.models.append(model)

        print(f"\n✓ 앙상블 학습 완료! (총 {len(self.models)}개 모델)")

    def predict_risk(self, X: np.ndarray, method: str = 'mean') -> np.ndarray:
        """
        앙상블 예측

        Parameters:
        -----------
        X : np.ndarray
            특징 행렬
        method : str
            앙상블 방법 ('mean', 'median')

        Returns:
        --------
        risk_scores : np.ndarray
            앙상블 위험 점수
        """
        predictions = [model.predict_risk(X) for model in self.models]

        if method == 'mean':
            return np.mean(predictions, axis=0)
        elif method == 'median':
            return np.median(predictions, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")


def train_and_evaluate_model(X: np.ndarray,
                             y_event: np.ndarray,
                             y_time: np.ndarray,
                             model,
                             test_size: float = 0.2,
                             epochs: int = 100,
                             batch_size: int = 32,
                             plot: bool = True) -> Dict:
    """
    모델 학습 및 평가 원스텝 함수

    Parameters:
    -----------
    X, y_event, y_time : np.ndarray
        데이터
    model : DeepSurv
        학습할 모델
    test_size : float
        테스트 데이터 비율
    epochs : int
        에폭 수
    batch_size : int
        배치 크기
    plot : bool
        그래프 출력 여부

    Returns:
    --------
    results : dict
        학습 및 평가 결과
    """
    trainer = ModelTrainer(model)

    # 데이터 분할
    X_train, X_test, y_event_train, y_event_test, y_time_train, y_time_test = \
        trainer.train_test_split_data(X, y_event, y_time, test_size=test_size)

    # 학습
    history = trainer.train(X_train, y_event_train, y_time_train,
                           epochs=epochs, batch_size=batch_size)

    # 평가
    metrics = trainer.evaluate(X_train, y_event_train, y_time_train,
                              X_test, y_event_test, y_time_test)

    # 시각화
    if plot:
        trainer.plot_training_history()

    results = {
        'model': model,
        'trainer': trainer,
        'metrics': metrics,
        'history': history,
        'data_splits': {
            'X_train': X_train,
            'X_test': X_test,
            'y_event_train': y_event_train,
            'y_event_test': y_event_test,
            'y_time_train': y_time_train,
            'y_time_test': y_time_test
        }
    }

    return results


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 70)
    print("모델 학습 및 평가 모듈 테스트")
    print("=" * 70)

    # 샘플 데이터
    from data_preprocessing import create_sample_data, NFLDataPreprocessor

    create_sample_data(n_samples=300, output_path='test_data.csv')

    preprocessor = NFLDataPreprocessor('test_data.csv')
    X, y_event, y_time = preprocessor.get_feature_matrix()

    # 모델 생성
    from model_architecture import DeepSurv

    model = DeepSurv(input_dim=3, hidden_layers=[32, 16])
    model.compile()

    # 학습 및 평가
    results = train_and_evaluate_model(
        X, y_event, y_time,
        model,
        epochs=30,
        batch_size=32,
        plot=False
    )

    print(f"\n최종 테스트 C-index: {results['metrics']['test_c_index']:.4f}")

    # 교차 검증 테스트
    print("\n교차 검증 테스트...")
    trainer = results['trainer']
    cv_results = trainer.cross_validate(X, y_event, y_time, n_splits=3, epochs=20, verbose=0)

    print("\n✓ 모듈 테스트 완료!")