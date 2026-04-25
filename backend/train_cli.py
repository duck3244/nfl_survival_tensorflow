"""
메인 실행 파이프라인
전체 프로젝트를 실행하는 통합 스크립트
"""

import os
import sys
import argparse
import random
import warnings

# Suppress only the noisy categories instead of every warning, so genuine
# bugs (e.g. divide-by-zero) still surface.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def set_global_seed(seed: int = 42):
    """Reproducibility — covers Python random, NumPy, and TensorFlow."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    import numpy as _np
    _np.random.seed(seed)
    import tensorflow as _tf
    _tf.random.set_seed(seed)

# 모듈 임포트
from data_preprocessing import NFLDataPreprocessor, create_sample_data
from model_architecture import DeepSurv, create_deepsurv_model
from model_training import ModelTrainer, train_and_evaluate_model
from visualization import SurvivalVisualizer, plot_multiple_players_comparison
from prediction_utils import PlayerPredictor, FamousPlayersPredictor
from lifelines import KaplanMeierFitter
import numpy as np


def run_full_pipeline(data_path='nfl.csv',
                     model_type='standard',
                     epochs=100,
                     batch_size=32,
                     save_model=True,
                     output_dir='output'):
    """
    전체 파이프라인 실행
    
    Parameters:
    -----------
    data_path : str
        데이터 파일 경로
    model_type : str
        모델 타입 ('standard', 'deep', 'wide', 'simple')
    epochs : int
        학습 에폭 수
    batch_size : int
        배치 크기
    save_model : bool
        모델 저장 여부
    output_dir : str
        출력 디렉토리
    """
    print("=" * 70)
    print("NFL RUNNING BACK SURVIVAL ANALYSIS - FULL PIPELINE")
    print("=" * 70)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== 1. 데이터 전처리 ==========
    print("\n[STEP 1/6] 데이터 전처리")
    print("-" * 70)
    
    if not os.path.exists(data_path):
        print(f"⚠️  데이터 파일을 찾을 수 없습니다: {data_path}")
        print("샘플 데이터를 생성합니다...")
        create_sample_data(n_samples=500, output_path=data_path)
    
    preprocessor = NFLDataPreprocessor(data_path)
    df = preprocessor.preprocess()

    # 특징 행렬 추출 — 자동 폴백 (BMI 없으면 Pick으로 대체)
    X, y_event, y_time = preprocessor.get_feature_matrix()
    feature_names = preprocessor.feature_columns_used

    print(f"\n✓ 데이터 준비 완료")
    print(f"  - 샘플 수: {len(X)}")
    print(f"  - 사용 features: {feature_names}")
    print(f"  - 평균 커리어: {y_time.mean():.1f} 경기")
    print(f"  - 중간 커리어: {np.median(y_time):.1f} 경기")

    # ========== 2. 모델 생성 및 컴파일 ==========
    print("\n[STEP 2/6] 모델 생성")
    print("-" * 70)

    model = create_deepsurv_model(input_dim=len(feature_names), model_type=model_type)
    model.feature_names = feature_names
    model.compile(learning_rate=0.001)
    
    print(f"✓ {model_type.capitalize()} 모델 생성 완료")
    print("\n모델 아키텍처:")
    model.summary()
    
    # ========== 3. 모델 학습 ==========
    print("\n[STEP 3/6] 모델 학습")
    print("-" * 70)
    
    results = train_and_evaluate_model(
        X, y_event, y_time,
        model,
        test_size=0.2,
        epochs=epochs,
        batch_size=batch_size,
        plot=False
    )
    
    trainer = results['trainer']
    metrics = results['metrics']
    
    # 학습 곡선 저장
    trainer.plot_training_history(
        save_path=os.path.join(output_dir, 'training_history.png')
    )
    
    # ========== 4. 모델 평가 ==========
    print("\n[STEP 4/6] 모델 평가")
    print("-" * 70)
    
    print(f"\n최종 성능:")
    print(f"  Training C-index: {metrics['train_c_index']:.4f}")
    print(f"  Testing C-index:  {metrics['test_c_index']:.4f}")
    
    # 교차 검증
    print(f"\n교차 검증 수행 중...")
    cv_results = trainer.cross_validate(X, y_event, y_time, n_splits=5, 
                                       epochs=50, verbose=0)
    
    trainer.plot_cv_results(
        cv_results,
        save_path=os.path.join(output_dir, 'cross_validation.png')
    )
    
    # ========== 5. 시각화 ==========
    print("\n[STEP 5/6] 결과 시각화")
    print("-" * 70)
    
    viz = SurvivalVisualizer()
    
    # Kaplan-Meier 곡선
    viz.plot_kaplan_meier(
        y_time, y_event,
        save_path=os.path.join(output_dir, 'kaplan_meier.png')
    )
    
    # 위험 그룹별 KM 곡선
    test_risk = model.predict_risk(results['data_splits']['X_test'])
    viz.plot_km_by_risk(
        results['data_splits']['y_time_test'],
        results['data_splits']['y_event_test'],
        test_risk,
        n_groups=3,
        save_path=os.path.join(output_dir, 'km_by_risk.png')
    )
    
    # 위험 점수 분포
    viz.plot_risk_distribution(
        test_risk,
        save_path=os.path.join(output_dir, 'risk_distribution.png')
    )
    
    # ========== 6. 예측 ==========
    print("\n[STEP 6/6] 선수 예측")
    print("-" * 70)
    
    # Kaplan-Meier 기준 생성
    kmf = KaplanMeierFitter()
    kmf.fit(results['data_splits']['y_time_train'],
            results['data_splits']['y_event_train'])
    
    # 예측기 생성
    predictor = PlayerPredictor(model, kmf)
    
    # 유명 선수 예측
    famous = FamousPlayersPredictor(predictor)
    famous_df = famous.predict_all_famous_players()
    
    print("\n유명 NFL 러닝백 예측 결과 (Top 5):")
    print(famous_df[['Player', 'Predicted_Career', 'Grade']].head().to_string(index=False))
    
    # 결과 저장
    famous_df.to_csv(
        os.path.join(output_dir, 'famous_players_predictions.csv'),
        index=False
    )
    
    # 개별 선수 예측 예시 — 현재 feature 세트에 맞춰 합리적 기본값 구성
    print("\n예시 선수 예측:")
    example_defaults = {'BMI': 29.0, 'YPC': 4.5, 'DrAge': 21, 'Pick': 30, 'Rnd': 2}
    example_features = {f: example_defaults.get(f, 0) for f in feature_names}
    example_pred = predictor.predict_player(example_features)
    report = predictor.generate_report(example_pred, "Example Player")
    print(report)
    
    # 리포트 저장
    predictor.save_report(
        example_pred,
        "Example Player",
        os.path.join(output_dir, 'example_prediction_report.txt')
    )
    
    # ========== 모델 저장 ==========
    if save_model:
        print("\n모델 저장 중...")
        from datetime import datetime, timezone
        model.metrics = {
            'train_c_index': float(metrics['train_c_index']),
            'test_c_index': float(metrics['test_c_index']),
            'cv_mean_c_index': float(cv_results['mean_c_index']),
            'cv_std_c_index': float(cv_results['std_c_index']),
        }
        model.training_meta = {
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'sample_size': int(len(X)),
            'train_size': int(metrics['train_samples']),
            'test_size': int(metrics['test_samples']),
            'epochs': int(epochs),
            'model_type': model_type,
        }
        model.save(os.path.join(output_dir, 'deepsurv_model'))
        print(f"✓ 모델 저장 완료: {output_dir}/deepsurv_model")
    
    # ========== 최종 요약 ==========
    print("\n" + "=" * 70)
    print("파이프라인 실행 완료!")
    print("=" * 70)
    
    print(f"\n📊 생성된 파일:")
    print(f"  📁 {output_dir}/")
    print(f"    ├── training_history.png")
    print(f"    ├── cross_validation.png")
    print(f"    ├── kaplan_meier.png")
    print(f"    ├── km_by_risk.png")
    print(f"    ├── risk_distribution.png")
    print(f"    ├── famous_players_predictions.csv")
    print(f"    ├── example_prediction_report.txt")
    if save_model:
        print(f"    ├── deepsurv_model_model.h5")
        print(f"    └── deepsurv_model_meta.pkl")
    
    print(f"\n🎯 최종 성능:")
    print(f"  Test C-index: {metrics['test_c_index']:.4f}")
    print(f"  CV Mean C-index: {cv_results['mean_c_index']:.4f} ± {cv_results['std_c_index']:.4f}")
    
    return {
        'model': model,
        'trainer': trainer,
        'metrics': metrics,
        'cv_results': cv_results,
        'predictor': predictor,
        'famous_df': famous_df
    }


def run_quick_demo():
    """빠른 데모 실행 (5분 이내)"""
    print("=" * 70)
    print("빠른 데모 모드 (5분 완성)")
    print("=" * 70)
    
    return run_full_pipeline(
        data_path='nfl_sample.csv',
        model_type='simple',
        epochs=30,
        batch_size=32,
        save_model=False,
        output_dir='demo_output'
    )


def run_predict_mode(model_path, feature_overrides: dict):
    """예측 전용 모드 — 학습 시 저장된 feature_names + baseline survival 사용."""
    print("=" * 70)
    print("예측 모드")
    print("=" * 70)

    print("\n모델 로딩 중...")
    # input_dim은 load() 후 model.input_dim으로 갱신되지 않으므로 임시값 사용.
    # 실제 가중치/scaler는 load()가 채워주므로 예측에는 영향 없음.
    model = DeepSurv(input_dim=1)
    model.load(model_path)
    print(f"✓ 모델 로드 완료 (features: {model.feature_names})")

    if not model.feature_names:
        raise RuntimeError("Loaded model has no feature_names — re-train with the updated pipeline.")

    # 사용자 입력에 누락된 feature는 합리적 기본값으로 보충
    defaults = {'BMI': 29.0, 'YPC': 4.5, 'DrAge': 22, 'Pick': 50, 'Rnd': 3}
    feats = {name: feature_overrides.get(name, defaults.get(name, 0))
             for name in model.feature_names}
    print(f"  입력 features: {feats}")

    predictor = PlayerPredictor(model)
    prediction = predictor.predict_player(feats)

    report = predictor.generate_report(prediction, "Your Player")
    print("\n" + report)

    return prediction


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='NFL Running Back Survival Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 파이프라인 실행
  python main.py --mode full --data nfl.csv --epochs 100

  # 빠른 데모
  python main.py --mode demo

  # 선수 예측
  python main.py --mode predict --model output/deepsurv_model --bmi 29.0 --ypc 4.5 --age 21
        """
    )
    
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'demo', 'predict'],
                       help='실행 모드')
    
    parser.add_argument('--data', type=str, default='nfl.csv',
                       help='데이터 파일 경로')
    
    parser.add_argument('--model-type', type=str, default='standard',
                       choices=['standard', 'deep', 'wide', 'simple'],
                       help='모델 타입')
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='학습 에폭 수')
    
    parser.add_argument('--batch-size', type=int, default=32,
                       help='배치 크기')
    
    parser.add_argument('--output', type=str, default='output',
                       help='출력 디렉토리')
    
    parser.add_argument('--no-save', action='store_true',
                       help='모델 저장 안함')
    
    # 예측 모드 파라미터
    parser.add_argument('--model', type=str,
                       help='로드할 모델 경로 (예측 모드)')
    
    parser.add_argument('--bmi', type=float,
                       help='BMI (예측 모드)')
    
    parser.add_argument('--ypc', type=float,
                       help='Yards Per Carry (예측 모드)')
    
    parser.add_argument('--age', type=int,
                       help='Draft Age (예측 모드)')

    parser.add_argument('--pick', type=int,
                       help='Draft Pick (예측 모드, BMI fallback 시 사용)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (재현성)')

    args = parser.parse_args()

    set_global_seed(args.seed)

    # 모드별 실행
    if args.mode == 'full':
        results = run_full_pipeline(
            data_path=args.data,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_model=not args.no_save,
            output_dir=args.output
        )
        
    elif args.mode == 'demo':
        results = run_quick_demo()
        
    elif args.mode == 'predict':
        if not args.model:
            parser.error("예측 모드는 --model 필수 (저장된 모델 경로)")
        feature_overrides = {}
        if args.bmi is not None: feature_overrides['BMI'] = args.bmi
        if args.ypc is not None: feature_overrides['YPC'] = args.ypc
        if args.age is not None: feature_overrides['DrAge'] = args.age
        if args.pick is not None: feature_overrides['Pick'] = args.pick
        results = run_predict_mode(args.model, feature_overrides)
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ 프로그램이 성공적으로 완료되었습니다!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
