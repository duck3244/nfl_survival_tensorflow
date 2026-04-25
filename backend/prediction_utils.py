"""
예측 유틸리티 모듈
개별 선수 예측 및 결과 해석 기능
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from typing import Dict, Tuple, Optional, List


class PlayerPredictor:
    """선수 커리어 예측 클래스 (feature-name 기반, 일반화 버전)"""

    def __init__(self, model, kmf: Optional[KaplanMeierFitter] = None,
                 feature_names: Optional[List[str]] = None):
        """
        feature_names가 명시되지 않으면 model.feature_names를 사용한다.
        kmf가 주어지면 baseline survival 출처로 우선 사용한다 (학습 직후 시나리오).
        그 외에는 model에 저장된 baseline_survival을 사용한다.
        """
        self.model = model
        self.kmf = kmf
        self.feature_names = list(feature_names) if feature_names \
            else list(getattr(model, 'feature_names', []) or [])
        if not self.feature_names:
            raise ValueError(
                "PlayerPredictor needs feature_names (either passed in or stored on the model)."
            )

    def _features_to_array(self, features: Dict) -> np.ndarray:
        """Dict → 학습에 쓰인 컬럼 순서대로 정렬된 1×D 배열."""
        try:
            row = [float(features[name]) for name in self.feature_names]
        except KeyError as e:
            raise KeyError(
                f"Missing feature {e} — predictor expects {self.feature_names}"
            ) from None
        return np.array([row])

    def _baseline_at(self, times: np.ndarray) -> np.ndarray:
        if self.kmf is not None:
            return self.kmf.survival_function_at_times(times).values
        if getattr(self.model, 'baseline_survival', None) is not None:
            return self.model.baseline_survival_at(times)
        # Last-resort fallback: simple exponential. Surface a warning so users
        # know predictions are coarse.
        print("⚠️  No baseline survival available — using exponential fallback")
        return np.exp(-0.015 * times)

    def predict_player(self,
                       features: Optional[Dict] = None,
                       max_games: int = 200,
                       **kwargs) -> Dict:
        """
        개별 선수 예측. features dict 또는 kwarg로 feature를 전달한다.

        예: predict_player({'BMI': 29.0, 'YPC': 4.5, 'DrAge': 21})
            predict_player(YPC=4.5, DrAge=21, Pick=4)
        """
        if features is None:
            features = kwargs

        X = self._features_to_array(features)
        risk_score = float(self.model.predict_risk(X)[0])

        times = np.arange(1, max_games + 1)
        baseline_surv = np.clip(self._baseline_at(times), 1e-12, 1.0)
        # NaN 안전 처리: NaN은 마지막 유효값으로 forward-fill
        if np.isnan(baseline_surv).any():
            baseline_surv = pd.Series(baseline_surv).ffill().bfill().values
        individual_surv = baseline_surv ** np.exp(risk_score)

        median_idx = np.where(individual_surv <= 0.5)[0]
        median_survival = int(times[median_idx[0]]) if len(median_idx) > 0 else max_games

        def at(t):
            return float(individual_surv[t - 1]) if t <= max_games else float('nan')

        return {
            'features': dict(features),
            'risk_score': risk_score,
            'median_survival': median_survival,
            'survival_curve': {'times': times, 'probabilities': individual_surv},
            'survival_at': {50: at(50), 100: at(100), 150: at(150)},
            'interpretation': self._interpret_prediction(risk_score, median_survival),
        }
    
    def _interpret_prediction(self, risk_score: float, median_survival: int) -> Dict:
        """
        예측 결과 해석
        
        Parameters:
        -----------
        risk_score : float
            위험 점수
        median_survival : int
            중간 생존 시간
        
        Returns:
        --------
        interpretation : dict
            해석 결과
        """
        # 등급 결정
        if median_survival > 120:
            grade = "🌟 Elite"
            comment = "Exceptional career length expected. This player has all the traits for a long, successful career."
        elif median_survival > 80:
            grade = "⭐ Above Average"
            comment = "Above average career expected. With proper management, this player can have a successful NFL career."
        elif median_survival > 50:
            grade = "📊 Average"
            comment = "Average career length expected. Performance and injury prevention will be key factors."
        else:
            grade = "⚠️ Below Average"
            comment = "Shorter career expected. Special attention to injury prevention and performance optimization needed."
        
        # 위험 수준
        if risk_score < -1.5:
            risk_level = "Very Low"
        elif risk_score < -0.5:
            risk_level = "Low"
        elif risk_score < 0.5:
            risk_level = "Moderate"
        elif risk_score < 1.5:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        return {
            'grade': grade,
            'comment': comment,
            'risk_level': risk_level,
            'career_outlook': f"Expected to play approximately {median_survival} games"
        }
    
    def predict_multiple_players(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """여러 선수 일괄 예측 — players_df는 self.feature_names 컬럼을 포함해야 한다."""
        results = []
        for idx, row in players_df.iterrows():
            feats = {name: row[name] for name in self.feature_names}
            pred = self.predict_player(feats)
            row_out = {
                'Player': row.get('Player', f'Player_{idx}'),
                **{name: feats[name] for name in self.feature_names},
                'Risk_Score': pred['risk_score'],
                'Median_Survival': pred['median_survival'],
                'Survival_50': pred['survival_at'][50],
                'Survival_100': pred['survival_at'][100],
                'Grade': pred['interpretation']['grade'],
                'Risk_Level': pred['interpretation']['risk_level'],
            }
            results.append(row_out)
        return pd.DataFrame(results)

    def compare_players(self, players_dict: Dict) -> pd.DataFrame:
        """{'player_name': {feature: value, ...}} 형식의 선수 비교."""
        comparisons = []
        for name, features in players_dict.items():
            pred = self.predict_player(features)
            comparisons.append({
                'Player': name,
                **{f: features.get(f) for f in self.feature_names},
                'Risk_Score': pred['risk_score'],
                'Expected_Career': pred['median_survival'],
                'Survival_100': pred['survival_at'][100] * 100,
                'Grade': pred['interpretation']['grade'],
            })
        return pd.DataFrame(comparisons).sort_values('Risk_Score')

    def generate_report(self, prediction: Dict, player_name: str = "Player") -> str:
        """
        예측 리포트 생성

        Parameters:
        -----------
        prediction : dict
            예측 결과
        player_name : str
            선수 이름

        Returns:
        --------
        report : str
            텍스트 리포트
        """
        report = []
        report.append("=" * 70)
        report.append(f"NFL RUNNING BACK CAREER PREDICTION REPORT")
        report.append("=" * 70)
        report.append("")

        # 선수 정보
        report.append(f"Player: {player_name}")
        report.append("-" * 70)
        for name, value in prediction['features'].items():
            try:
                report.append(f"  {name:24s} {float(value):.2f}")
            except (TypeError, ValueError):
                report.append(f"  {name:24s} {value}")
        report.append("")

        # 예측 결과
        report.append("PREDICTION RESULTS")
        report.append("-" * 70)
        report.append(f"  Risk Score:               {prediction['risk_score']:.4f}")
        report.append(f"  Risk Level:               {prediction['interpretation']['risk_level']}")
        report.append(f"  Expected Career:          {prediction['median_survival']} games")
        report.append(f"  Grade:                    {prediction['interpretation']['grade']}")
        report.append("")

        # 생존 확률
        report.append("SURVIVAL PROBABILITIES")
        report.append("-" * 70)
        for games, prob in prediction['survival_at'].items():
            if not np.isnan(prob):
                report.append(f"  At {games:3d} games:            {prob*100:.1f}%")
        report.append("")

        # 해석
        report.append("INTERPRETATION")
        report.append("-" * 70)
        report.append(f"  {prediction['interpretation']['comment']}")
        report.append(f"  {prediction['interpretation']['career_outlook']}")
        report.append("")
        report.append("=" * 70)

        return "\n".join(report)

    def save_report(self, prediction: Dict, player_name: str, filepath: str):
        """리포트를 파일로 저장"""
        report = self.generate_report(prediction, player_name)

        with open(filepath, 'w') as f:
            f.write(report)

        print(f"✓ 리포트 저장: {filepath}")


class FamousPlayersPredictor:
    """유명 NFL 선수 예측 클래스"""

    # 유명 선수 데이터베이스 (추정치). Pick은 실제 드래프트 순번이라 BMI fallback
    # (Pick/YPC/DrAge) 시나리오에서도 예측이 의미를 가진다.
    FAMOUS_PLAYERS = {
        'LaDainian Tomlinson': {'BMI': 30.5, 'YPC': 4.4, 'DrAge': 22, 'Pick': 5, 'Rnd': 1, 'Actual_Games': 170},
        'Emmitt Smith':        {'BMI': 31.0, 'YPC': 4.2, 'DrAge': 21, 'Pick': 17, 'Rnd': 1, 'Actual_Games': 226},
        'Barry Sanders':       {'BMI': 29.2, 'YPC': 5.0, 'DrAge': 21, 'Pick': 3, 'Rnd': 1, 'Actual_Games': 153},
        'Adrian Peterson':     {'BMI': 31.8, 'YPC': 4.8, 'DrAge': 22, 'Pick': 7, 'Rnd': 1, 'Actual_Games': 165},
        'Walter Payton':       {'BMI': 30.0, 'YPC': 4.4, 'DrAge': 21, 'Pick': 4, 'Rnd': 1, 'Actual_Games': 190},
        'Eric Dickerson':      {'BMI': 31.5, 'YPC': 4.6, 'DrAge': 22, 'Pick': 2, 'Rnd': 1, 'Actual_Games': 146},
        'Bo Jackson':          {'BMI': 32.5, 'YPC': 5.4, 'DrAge': 23, 'Pick': 183, 'Rnd': 7, 'Actual_Games': 38},
        'Ezekiel Elliott':     {'BMI': 29.0, 'YPC': 4.5, 'DrAge': 21, 'Pick': 4, 'Rnd': 1, 'Actual_Games': None},
        'Saquon Barkley':      {'BMI': 30.2, 'YPC': 4.6, 'DrAge': 21, 'Pick': 2, 'Rnd': 1, 'Actual_Games': None},
        'Derrick Henry':       {'BMI': 32.0, 'YPC': 5.0, 'DrAge': 22, 'Pick': 45, 'Rnd': 2, 'Actual_Games': None},
    }

    def __init__(self, predictor: PlayerPredictor):
        """
        Parameters:
        -----------
        predictor : PlayerPredictor
            선수 예측기
        """
        self.predictor = predictor

    def predict_all_famous_players(self) -> pd.DataFrame:
        """모든 유명 선수 예측 — predictor.feature_names에 따라 동적으로 컬럼 사용."""
        feature_names = self.predictor.feature_names
        results = []

        for name, data in self.FAMOUS_PLAYERS.items():
            missing = [f for f in feature_names if f not in data]
            if missing:
                print(f"⚠️  Skipping {name}: missing {missing} for current feature set")
                continue

            feats = {f: data[f] for f in feature_names}
            pred = self.predictor.predict_player(feats)

            result = {
                'Player': name,
                **feats,
                'Risk_Score': pred['risk_score'],
                'Predicted_Career': pred['median_survival'],
                'Actual_Career': data.get('Actual_Games'),
                'Survival_100': pred['survival_at'][100] * 100,
                'Grade': pred['interpretation']['grade'],
            }
            actual = result['Actual_Career']
            result['Prediction_Error'] = (
                abs(result['Predicted_Career'] - actual) if actual else None
            )
            results.append(result)

        return pd.DataFrame(results).sort_values('Predicted_Career', ascending=False)

    def compare_with_actual(self) -> Dict:
        """예측과 실제 비교 분석"""
        df = self.predict_all_famous_players()

        # 실제 데이터가 있는 선수만
        actual_data = df[df['Actual_Career'].notna()].copy()

        if len(actual_data) == 0:
            return {'message': 'No actual career data available for comparison'}

        # 평균 절대 오차
        mae = actual_data['Prediction_Error'].mean()

        # 평균 상대 오차
        mape = (actual_data['Prediction_Error'] / actual_data['Actual_Career'] * 100).mean()

        # 상관계수
        correlation = actual_data['Predicted_Career'].corr(actual_data['Actual_Career'])

        analysis = {
            'mean_absolute_error': mae,
            'mean_absolute_percentage_error': mape,
            'correlation': correlation,
            'n_players': len(actual_data),
            'best_prediction': actual_data.loc[actual_data['Prediction_Error'].idxmin(), 'Player'],
            'worst_prediction': actual_data.loc[actual_data['Prediction_Error'].idxmax(), 'Player']
        }

        return analysis


def create_player_profile(name: str,
                         height_inches: float,
                         weight_lbs: float,
                         yards: float,
                         attempts: float,
                         draft_age: int) -> Dict:
    """
    선수 프로필에서 예측용 특징 생성

    Parameters:
    -----------
    name : str
        선수 이름
    height_inches : float
        키 (인치)
    weight_lbs : float
        몸무게 (파운드)
    yards : float
        총 야드
    attempts : float
        시도 횟수
    draft_age : int
        드래프트 나이

    Returns:
    --------
    profile : dict
        예측용 프로필
    """
    # BMI 계산
    bmi = (weight_lbs / (height_inches ** 2)) * 703

    # YPC 계산
    ypc = yards / attempts if attempts > 0 else 0

    profile = {
        'name': name,
        'BMI': round(bmi, 1),
        'YPC': round(ypc, 1),
        'DrAge': draft_age,
        'height': height_inches,
        'weight': weight_lbs,
        'yards': yards,
        'attempts': attempts
    }

    return profile


def batch_predict_from_csv(csv_path: str,
                           model,
                           kmf=None,
                           output_path: Optional[str] = None) -> pd.DataFrame:
    """
    CSV 파일에서 일괄 예측

    Parameters:
    -----------
    csv_path : str
        입력 CSV 파일 경로
    model : DeepSurv
        학습된 모델
    kmf : KaplanMeierFitter
        기준 KM 객체
    output_path : str
        출력 CSV 경로

    Returns:
    --------
    results_df : pd.DataFrame
        예측 결과
    """
    # CSV 로드
    df = pd.read_csv(csv_path)

    # 필요한 컬럼 확인
    required_cols = ['BMI', 'YPC', 'DrAge']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # 예측기 생성
    predictor = PlayerPredictor(model, kmf)

    # 일괄 예측
    results = predictor.predict_multiple_players(df)

    # 저장
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"✓ 예측 결과 저장: {output_path}")

    return results


def find_similar_players(target_features: Dict,
                        players_df: pd.DataFrame,
                        n_similar: int = 5) -> pd.DataFrame:
    """
    유사한 특징을 가진 선수 찾기

    Parameters:
    -----------
    target_features : dict
        타겟 선수 특징 {'BMI': ..., 'YPC': ..., 'DrAge': ...}
    players_df : pd.DataFrame
        선수 데이터베이스
    n_similar : int
        반환할 유사 선수 수

    Returns:
    --------
    similar_df : pd.DataFrame
        유사 선수 목록
    """
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.preprocessing import StandardScaler

    # 특징 정규화
    features = ['BMI', 'YPC', 'DrAge']
    scaler = StandardScaler()

    players_features = players_df[features].values
    players_scaled = scaler.fit_transform(players_features)

    target_array = np.array([[target_features['BMI'],
                             target_features['YPC'],
                             target_features['DrAge']]])
    target_scaled = scaler.transform(target_array)

    # 거리 계산
    distances = euclidean_distances(target_scaled, players_scaled)[0]

    # 가장 가까운 선수들
    similar_indices = np.argsort(distances)[:n_similar]

    similar_df = players_df.iloc[similar_indices].copy()
    similar_df['Distance'] = distances[similar_indices]

    return similar_df


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 70)
    print("예측 유틸리티 모듈 테스트")
    print("=" * 70)

    # 샘플 모델 및 데이터
    from model_architecture import DeepSurv
    from lifelines import KaplanMeierFitter

    # 더미 모델 생성
    model = DeepSurv(input_dim=3, hidden_layers=[32, 16])
    model.compile()

    # 더미 학습 (실제로는 제대로 학습된 모델 사용)
    X_dummy = np.random.randn(100, 3)
    y_event_dummy = np.ones(100)
    y_time_dummy = np.random.exponential(60, 100)

    model.fit(X_dummy, y_event_dummy, y_time_dummy, epochs=5, verbose=0)

    # KMF 생성
    kmf = KaplanMeierFitter()
    kmf.fit(y_time_dummy, y_event_dummy)

    # 예측기 생성
    predictor = PlayerPredictor(model, kmf)

    # 1. 개별 선수 예측
    print("\n1. 개별 선수 예측")
    prediction = predictor.predict_player(bmi=29.0, ypc=4.5, draft_age=21)

    print(f"\n위험 점수: {prediction['risk_score']:.4f}")
    print(f"예상 커리어: {prediction['median_survival']} 경기")
    print(f"등급: {prediction['interpretation']['grade']}")

    # 2. 리포트 생성
    print("\n2. 리포트 생성")
    report = predictor.generate_report(prediction, "Test Player")
    print(report)

    # 3. 선수 비교
    print("\n3. 선수 비교")
    players_to_compare = {
        'Player A': {'BMI': 29.0, 'YPC': 4.5, 'DrAge': 21},
        'Player B': {'BMI': 31.0, 'YPC': 4.0, 'DrAge': 23},
        'Player C': {'BMI': 28.0, 'YPC': 5.0, 'DrAge': 20}
    }

    comparison = predictor.compare_players(players_to_compare)
    print(comparison[['Player', 'Risk_Score', 'Expected_Career', 'Grade']])

    # 4. 유명 선수 예측
    print("\n4. 유명 선수 예측")
    famous = FamousPlayersPredictor(predictor)
    famous_results = famous.predict_all_famous_players()

    print("\n상위 5명 (예측 커리어 기준):")
    print(famous_results[['Player', 'Predicted_Career', 'Grade']].head())

    # 5. 선수 프로필 생성
    print("\n5. 선수 프로필 생성")
    profile = create_player_profile(
        name="John Doe",
        height_inches=70,
        weight_lbs=215,
        yards=1500,
        attempts=350,
        draft_age=22
    )
    print(f"생성된 프로필: BMI={profile['BMI']}, YPC={profile['YPC']}")

    print("\n✓ 예측 유틸리티 테스트 완료!")