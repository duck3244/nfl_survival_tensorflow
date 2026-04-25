"""
데이터 전처리 모듈
NFL Running Back 데이터 로드 및 전처리 기능
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


class NFLDataPreprocessor:
    """NFL Running Back 데이터 전처리 클래스"""
    
    def __init__(self, filepath: str = 'nfl.csv'):
        """
        Parameters:
        -----------
        filepath : str
            CSV 파일 경로
        """
        self.filepath = filepath
        self.raw_data = None
        self.processed_data = None
        self.feature_columns_used: list = []
        
    def load_data(self) -> pd.DataFrame:
        """CSV 파일에서 데이터 로드"""
        try:
            self.raw_data = pd.read_csv(self.filepath)
            print(f"✓ 데이터 로드 완료: {len(self.raw_data)}개 행")
            return self.raw_data
        except FileNotFoundError:
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.filepath}")
        except Exception as e:
            raise Exception(f"데이터 로드 중 오류 발생: {str(e)}")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """파생 변수 생성"""
        df = df.copy()
        
        # Yards Per Carry (YPC)
        if 'YPC' not in df.columns and 'Yds' in df.columns and 'Att' in df.columns:
            df['YPC'] = df['Yds'] / df['Att']
            df['YPC'] = df['YPC'].replace([np.inf, -np.inf], np.nan)

        # Career Years
        if 'Years' not in df.columns and 'To' in df.columns and 'From' in df.columns:
            df['Years'] = df['To'] - df['From']

        # Pro Bowl Binary
        if 'PB_binary' not in df.columns and 'PB' in df.columns:
            df['PB_binary'] = (df['PB'] >= 1).astype(int)

        # All-Pro Binary
        if 'AP1_binary' not in df.columns and 'AP1' in df.columns:
            df['AP1_binary'] = (df['AP1'] >= 1).astype(int)

        # BMI (Body Mass Index) — only computed when raw inputs exist.
        # Random fallback was removed: it injected pure noise into a model feature.
        if 'BMI' not in df.columns and 'Weight' in df.columns and 'Height' in df.columns:
            df['BMI'] = (df['Weight'] / (df['Height'] ** 2)) * 703

        # Retired Indicator — derive real censoring from career end year.
        # Players whose final season equals the dataset snapshot year may still
        # be active, so they are right-censored (event=0).
        if 'Retired' not in df.columns:
            if 'To' in df.columns and df['To'].notna().any():
                snapshot_year = df['To'].max()
                df['Retired'] = (df['To'] < snapshot_year).astype(int)
                n_censored = (df['Retired'] == 0).sum()
                print(f"  검열 정의: snapshot={int(snapshot_year)}, "
                      f"censored={n_censored}/{len(df)} "
                      f"({n_censored/len(df)*100:.1f}%)")
            else:
                df['Retired'] = 1

        # 추가 유용한 특징들 (컬럼이 있을 때만)
        if 'TD' in df.columns and 'G' in df.columns:
            df['TD_per_game'] = df['TD'] / df['G']

        if 'Rec' in df.columns and 'G' in df.columns:
            df['Rec_per_game'] = df['Rec'] / df['G']

        if 'Att' in df.columns and 'G' in df.columns:
            df['Att_per_game'] = df['Att'] / df['G']

        return df

    def clean_data(self, df: pd.DataFrame, remove_outliers: bool = True) -> pd.DataFrame:
        """
        결측치 및 이상치 처리.

        remove_outliers=True 시 IQR 기반 이상치 제거를 수행한다. 이 단계는
        train/test 분할 전에 적용되므로 엄밀한 의미에서 약간의 정보 누수가
        있을 수 있으나, NFL 통계의 물리적 범위 검증 목적이 강해 실용적으로
        허용한다. 엄밀한 평가가 필요하면 False로 호출하라.
        """
        df = df.copy()

        # 필수 컬럼 확인
        required_cols = ['G']  # 최소 필수 컬럼
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # BMI, YPC, DrAge 결측치 제거 (있는 경우에만)
        essential_cols = []
        if 'G' in df.columns:
            essential_cols.append('G')
        if 'BMI' in df.columns:
            essential_cols.append('BMI')
        if 'YPC' in df.columns:
            essential_cols.append('YPC')
        if 'DrAge' in df.columns:
            essential_cols.append('DrAge')

        if essential_cols:
            df = df.dropna(subset=essential_cols)

        # 이상치 제거 (IQR 방법) - 컬럼이 있을 때만
        if remove_outliers:
            for col in ['BMI', 'YPC']:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        # 0 경기 선수 제거
        if 'G' in df.columns:
            df = df[df['G'] > 0]

        return df

    def preprocess(self, remove_outliers: bool = True) -> pd.DataFrame:
        """전체 전처리 파이프라인 실행"""
        # 1. 데이터 로드
        if self.raw_data is None:
            self.load_data()

        df = self.raw_data.copy()

        # 2. 파생 변수 생성
        df = self.create_features(df)

        # 3. 데이터 정제
        df = self.clean_data(df, remove_outliers=remove_outliers)

        self.processed_data = df

        print(f"✓ 전처리 완료: {len(df)}개 행 (제거된 행: {len(self.raw_data) - len(df)})")

        return df

    # When `feature_columns=None`, pick the first 3 available from this list.
    # BMI comes first (best feature when Weight/Height present); Pick/Rnd are
    # the pre-career fallbacks when BMI is unavailable.
    DEFAULT_FEATURE_PRIORITY = ['BMI', 'YPC', 'DrAge', 'Pick', 'Rnd']

    def get_feature_matrix(self,
                          feature_columns: list = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        모델 학습용 특징 행렬 반환.

        feature_columns=None이면 DEFAULT_FEATURE_PRIORITY 순서로 사용 가능한
        컬럼 3개를 자동 선택하고, 실제 사용된 컬럼은
        `self.feature_columns_used`에 저장한다.
        """
        if self.processed_data is None:
            self.preprocess()

        if feature_columns is None:
            available_features = [c for c in self.DEFAULT_FEATURE_PRIORITY
                                  if c in self.processed_data.columns][:3]
        else:
            available_features = [c for c in feature_columns
                                  if c in self.processed_data.columns]
            missing = [c for c in feature_columns if c not in available_features]
            if missing:
                print(f"⚠️  Missing features {missing}. Using only {available_features}")

        if not available_features:
            raise ValueError(
                f"None of the requested features are available. "
                f"Available numeric columns: {list(self.processed_data.select_dtypes('number').columns)}"
            )

        # 선택된 feature에 NaN 행이 남아있을 수 있으므로 제거
        subset = self.processed_data.dropna(subset=available_features)

        X = subset[available_features].values
        y_event = subset['Retired'].values
        y_time = subset['G'].values

        self.feature_columns_used = available_features
        print(f"✓ Features in use: {available_features}")

        return X, y_event, y_time

    def get_summary_statistics(self) -> pd.DataFrame:
        """데이터 요약 통계"""
        if self.processed_data is None:
            self.preprocess()

        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        summary = self.processed_data[numeric_cols].describe()

        return summary

    def save_processed_data(self, output_path: str = 'nfl_processed.csv'):
        """전처리된 데이터 저장"""
        if self.processed_data is None:
            self.preprocess()

        self.processed_data.to_csv(output_path, index=False)
        print(f"✓ 전처리된 데이터 저장: {output_path}")


def load_and_preprocess_data(filepath: str = 'nfl.csv') -> pd.DataFrame:
    """
    편의 함수: 데이터 로드 및 전처리를 한 번에 수행

    Parameters:
    -----------
    filepath : str
        CSV 파일 경로

    Returns:
    --------
    pd.DataFrame
        전처리된 데이터프레임
    """
    preprocessor = NFLDataPreprocessor(filepath)
    return preprocessor.preprocess()


def create_sample_data(n_samples: int = 500, output_path: str = 'nfl_sample.csv'):
    """
    샘플 데이터 생성 (테스트용)

    Parameters:
    -----------
    n_samples : int
        생성할 샘플 수
    output_path : str
        저장할 파일 경로
    """
    np.random.seed(42)

    sample_data = pd.DataFrame({
        'Player': [f'Player_{i}' for i in range(n_samples)],
        'Year': np.random.randint(2000, 2020, n_samples),
        'BMI': np.random.normal(29, 2, n_samples),
        'YPC': np.random.normal(4.2, 0.8, n_samples),
        'DrAge': np.random.randint(20, 26, n_samples),
        'G': np.random.exponential(60, n_samples).astype(int) + 10,
        'Retired': np.ones(n_samples),
        'Weight': np.random.normal(215, 15, n_samples),
        'Height': np.random.normal(70, 2, n_samples),
        'Yds': np.random.normal(2000, 800, n_samples),
        'Att': np.random.normal(400, 150, n_samples),
        'TD': np.random.randint(0, 50, n_samples),
        'Rec': np.random.randint(0, 100, n_samples),
        'From': 2000,
        'To': np.random.randint(2005, 2020, n_samples),
        'PB': np.random.randint(0, 5, n_samples),
        'AP1': np.random.randint(0, 3, n_samples),
    })

    sample_data.to_csv(output_path, index=False)
    print(f"✓ 샘플 데이터 생성: {output_path} ({n_samples}개 행)")

    return sample_data


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 70)
    print("데이터 전처리 모듈 테스트")
    print("=" * 70)

    # 샘플 데이터 생성
    create_sample_data(n_samples=500)

    # 전처리 실행
    preprocessor = NFLDataPreprocessor('nfl_sample.csv')
    df = preprocessor.preprocess()

    # 요약 통계
    print("\n데이터 요약 통계:")
    print(preprocessor.get_summary_statistics()[['BMI', 'YPC', 'DrAge', 'G']].T)

    # 특징 행렬 추출
    X, y_event, y_time = preprocessor.get_feature_matrix()
    print(f"\n특징 행렬 형태: {X.shape}")
    print(f"이벤트 수: {y_event.sum()}")
    print(f"평균 생존 시간: {y_time.mean():.1f} 경기")