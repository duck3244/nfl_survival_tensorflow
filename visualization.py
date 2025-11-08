"""
시각화 모듈
생존 분석 결과 시각화 기능
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from typing import Optional, List, Dict


class SurvivalVisualizer:
    """생존 분석 시각화 클래스"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Parameters:
        -----------
        style : str
            Matplotlib 스타일
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # 색상 팔레트 설정
        self.colors = {
            'primary': '#3498db',
            'secondary': '#e74c3c',
            'success': '#2ecc71',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'info': '#3498db'
        }
    
    def plot_kaplan_meier(self,
                         y_time: np.ndarray,
                         y_event: np.ndarray,
                         label: str = 'All Players',
                         save_path: Optional[str] = None):
        """
        Kaplan-Meier 생존 곡선 그리기
        
        Parameters:
        -----------
        y_time : np.ndarray
            생존 시간
        y_event : np.ndarray
            이벤트 발생 여부
        label : str
            라벨
        save_path : str
            저장 경로
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        kmf = KaplanMeierFitter()
        kmf.fit(y_time, y_event, label=label)
        kmf.plot_survival_function(ax=ax, linewidth=2.5, color=self.colors['primary'])
        
        ax.set_xlabel('Games Played', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.set_title('Kaplan-Meier Survival Curve', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 저장 완료: {save_path}")
        
        plt.show()
    
    def plot_km_by_groups(self,
                         y_time: np.ndarray,
                         y_event: np.ndarray,
                         groups: np.ndarray,
                         group_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None):
        """
        그룹별 Kaplan-Meier 곡선
        
        Parameters:
        -----------
        y_time : np.ndarray
            생존 시간
        y_event : np.ndarray
            이벤트
        groups : np.ndarray
            그룹 라벨
        group_names : list
            그룹 이름
        save_path : str
            저장 경로
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        unique_groups = np.unique(groups)
        colors_list = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
        
        kmf = KaplanMeierFitter()
        
        for i, group in enumerate(unique_groups):
            mask = groups == group
            
            if group_names:
                label = group_names[i]
            else:
                label = f'Group {group}'
            
            color = colors_list[i % len(colors_list)]
            
            kmf.fit(y_time[mask], y_event[mask], label=label)
            kmf.plot_survival_function(ax=ax, linewidth=2.5, color=color)
        
        ax.set_xlabel('Games Played', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.set_title('Kaplan-Meier Curves by Group', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 저장 완료: {save_path}")
        
        plt.show()
    
    def plot_km_by_risk(self,
                       y_time: np.ndarray,
                       y_event: np.ndarray,
                       risk_scores: np.ndarray,
                       n_groups: int = 3,
                       save_path: Optional[str] = None):
        """
        위험 점수별 Kaplan-Meier 곡선
        
        Parameters:
        -----------
        y_time : np.ndarray
            생존 시간
        y_event : np.ndarray
            이벤트
        risk_scores : np.ndarray
            위험 점수
        n_groups : int
            그룹 수
        save_path : str
            저장 경로
        """
        # 위험 점수로 그룹 분할
        risk_groups = pd.qcut(risk_scores, q=n_groups, 
                             labels=['Low Risk', 'Medium Risk', 'High Risk'][:n_groups])
        
        self.plot_km_by_groups(y_time, y_event, risk_groups, 
                              group_names=risk_groups.categories.tolist(),
                              save_path=save_path)
    
    def plot_individual_survival(self,
                                times: np.ndarray,
                                survival_probs: np.ndarray,
                                player_name: str = 'Player',
                                actual_games: Optional[int] = None,
                                save_path: Optional[str] = None):
        """
        개별 선수 생존 곡선
        
        Parameters:
        -----------
        times : np.ndarray
            시간 포인트
        survival_probs : np.ndarray
            생존 확률
        player_name : str
            선수 이름
        actual_games : int
            실제 경기 수 (있을 경우)
        save_path : str
            저장 경로
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 생존 곡선
        ax.plot(times, survival_probs, linewidth=3, 
               color=self.colors['primary'], label='Predicted Survival')
        ax.fill_between(times, 0, survival_probs, alpha=0.3, 
                       color=self.colors['primary'])
        
        # 50% 생존선
        ax.axhline(y=0.5, color='gray', linestyle='--', 
                  alpha=0.5, label='50% Survival')
        
        # 중간 생존 시간
        median_idx = np.where(survival_probs <= 0.5)[0]
        if len(median_idx) > 0:
            median_time = times[median_idx[0]]
            ax.axvline(x=median_time, color=self.colors['warning'], 
                      linestyle='--', alpha=0.7,
                      label=f'Median Survival: {median_time} games')
        
        # 실제 경기 수
        if actual_games:
            ax.axvline(x=actual_games, color=self.colors['danger'], 
                      linestyle='-', linewidth=2,
                      label=f'Actual Career: {actual_games} games')
        
        ax.set_xlabel('Games Played', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.set_title(f'Career Survival Prediction: {player_name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_xlim(0, max(times))
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 저장 완료: {save_path}")
        
        plt.show()
    
    def plot_risk_distribution(self,
                              risk_scores: np.ndarray,
                              group_labels: Optional[np.ndarray] = None,
                              save_path: Optional[str] = None):
        """
        위험 점수 분포
        
        Parameters:
        -----------
        risk_scores : np.ndarray
            위험 점수
        group_labels : np.ndarray
            그룹 라벨 (있을 경우)
        save_path : str
            저장 경로
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 히스토그램
        axes[0].hist(risk_scores, bins=30, alpha=0.7, 
                    color=self.colors['primary'], edgecolor='black')
        axes[0].axvline(x=np.mean(risk_scores), color=self.colors['danger'], 
                       linestyle='--', linewidth=2, label=f'Mean: {np.mean(risk_scores):.3f}')
        axes[0].axvline(x=np.median(risk_scores), color=self.colors['success'], 
                       linestyle='--', linewidth=2, label=f'Median: {np.median(risk_scores):.3f}')
        axes[0].set_xlabel('Risk Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 박스플롯
        if group_labels is not None:
            df = pd.DataFrame({'Risk': risk_scores, 'Group': group_labels})
            df.boxplot(column='Risk', by='Group', ax=axes[1])
            axes[1].set_title('Risk Score by Group', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Group', fontsize=12)
            axes[1].set_ylabel('Risk Score', fontsize=12)
        else:
            axes[1].boxplot(risk_scores, vert=True)
            axes[1].set_title('Risk Score Box Plot', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Risk Score', fontsize=12)
            axes[1].set_xticklabels(['All'])
        
        axes[1].grid(alpha=0.3)
        plt.suptitle('')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 저장 완료: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self,
                               feature_names: List[str],
                               importance_scores: np.ndarray,
                               save_path: Optional[str] = None):
        """
        특징 중요도 시각화
        
        Parameters:
        -----------
        feature_names : list
            특징 이름
        importance_scores : np.ndarray
            중요도 점수
        save_path : str
            저장 경로
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 중요도 순으로 정렬
        sorted_idx = np.argsort(importance_scores)
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_scores = importance_scores[sorted_idx]
        
        # 색상 그라데이션
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(sorted_scores)))
        
        bars = ax.barh(sorted_features, sorted_scores, color=colors)
        
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Feature Importance Analysis', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 값 표시
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax.text(score + max(sorted_scores)*0.01, i, f'{score:.4f}',
                   va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 저장 완료: {save_path}")
        
        plt.show()
    
    def plot_comparison_dashboard(self,
                                 results_dict: Dict,
                                 save_path: Optional[str] = None):
        """
        종합 비교 대시보드
        
        Parameters:
        -----------
        results_dict : dict
            결과 딕셔너리 {'model_name': {'c_index': ..., 'risk_scores': ...}}
        save_path : str
            저장 경로
        """
        n_models = len(results_dict)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Comparison Dashboard', fontsize=16, fontweight='bold')
        
        # 1. C-index 비교
        models = list(results_dict.keys())
        c_indices = [results_dict[m]['c_index'] for m in models]
        
        colors = [self.colors['primary'], self.colors['secondary'], 
                 self.colors['success']][:len(models)]
        
        axes[0, 0].bar(models, c_indices, color=colors, alpha=0.7)
        axes[0, 0].set_ylabel('C-index', fontsize=11)
        axes[0, 0].set_title('Model Performance (C-index)', fontsize=12, fontweight='bold')
        axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        for i, (model, c_idx) in enumerate(zip(models, c_indices)):
            axes[0, 0].text(i, c_idx + 0.01, f'{c_idx:.4f}', 
                           ha='center', fontweight='bold')
        
        # 2. Risk Score 분포 비교
        for i, (model, color) in enumerate(zip(models, colors)):
            if 'risk_scores' in results_dict[model]:
                axes[0, 1].hist(results_dict[model]['risk_scores'], 
                              bins=20, alpha=0.5, label=model, color=color)
        
        axes[0, 1].set_xlabel('Risk Score', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Risk Score Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. 박스플롯
        risk_data = [results_dict[m].get('risk_scores', []) for m in models 
                    if 'risk_scores' in results_dict[m]]
        if risk_data:
            bp = axes[1, 0].boxplot(risk_data, labels=models, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        axes[1, 0].set_ylabel('Risk Score', fontsize=11)
        axes[1, 0].set_title('Risk Score Distribution (Box Plot)', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. 요약 통계 테이블
        axes[1, 1].axis('off')
        
        table_data = []
        for model in models:
            c_idx = results_dict[model]['c_index']
            if 'risk_scores' in results_dict[model]:
                risk = results_dict[model]['risk_scores']
                mean_risk = np.mean(risk)
                std_risk = np.std(risk)
            else:
                mean_risk = std_risk = np.nan
            
            table_data.append([model, f'{c_idx:.4f}', 
                             f'{mean_risk:.3f}', f'{std_risk:.3f}'])
        
        table = axes[1, 1].table(cellText=table_data,
                                colLabels=['Model', 'C-index', 'Mean Risk', 'Std Risk'],
                                cellLoc='center',
                                loc='center',
                                bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 헤더 스타일
        for i in range(4):
            table[(0, i)].set_facecolor(self.colors['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        axes[1, 1].set_title('Summary Statistics', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 저장 완료: {save_path}")
        
        plt.show()


def plot_multiple_players_comparison(players_dict: Dict,
                                     kmf_baseline,
                                     save_path: Optional[str] = None):
    """
    여러 선수 비교 시각화
    
    Parameters:
    -----------
    players_dict : dict
        {'player_name': {'features': [...], 'risk': ..., 'survival': [...]}}
    kmf_baseline : KaplanMeierFitter
        기준 KM 객체
    save_path : str
        저장 경로
    """
    n_players = len(players_dict)
    ncols = min(4, n_players)
    nrows = (n_players + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if n_players == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_players))
    
    for idx, (player_name, data) in enumerate(players_dict.items()):
        ax = axes[idx]
        
        times = data.get('times', np.arange(1, 201))
        survival = data['survival']
        risk = data['risk']
        
        # 생존 곡선
        ax.plot(times, survival, linewidth=2.5, color=colors[idx])
        ax.fill_between(times, 0, survival, alpha=0.3, color=colors[idx])
        
        # 중간 생존 시간
        median_idx = np.where(survival <= 0.5)[0]
        if len(median_idx) > 0:
            median_time = times[median_idx[0]]
            ax.axvline(x=median_time, color='gray', linestyle='--', alpha=0.7)
            ax.text(median_time, 0.9, f'{median_time} games', 
                   rotation=90, va='top')
        
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Games', fontsize=10)
        ax.set_ylabel('Survival Prob', fontsize=10)
        ax.set_title(f'{player_name}\n(Risk: {risk:.3f})', 
                    fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 1)
    
    # 빈 subplot 숨기기
    for idx in range(n_players, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Player Career Predictions', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 저장 완료: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 70)
    print("시각화 모듈 테스트")
    print("=" * 70)
    
    # 샘플 데이터
    np.random.seed(42)
    n_samples = 200
    
    y_time = np.random.exponential(60, n_samples)
    y_event = np.ones(n_samples)
    risk_scores = np.random.randn(n_samples)
    
    # 시각화 객체 생성
    viz = SurvivalVisualizer()
    
    # 1. Kaplan-Meier 곡선
    print("\n1. Kaplan-Meier 곡선 생성...")
    viz.plot_kaplan_meier(y_time, y_event)
    
    # 2. 위험 그룹별 KM 곡선
    print("\n2. 위험 그룹별 KM 곡선 생성...")
    viz.plot_km_by_risk(y_time, y_event, risk_scores, n_groups=3)
    
    # 3. 위험 점수 분포
    print("\n3. 위험 점수 분포 생성...")
    viz.plot_risk_distribution(risk_scores)
    
    print("\n✓ 시각화 테스트 완료!")
