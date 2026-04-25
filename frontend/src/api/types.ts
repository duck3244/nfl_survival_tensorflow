// Mirror of backend/app/schemas.py — keep in sync manually.

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  feature_names: string[];
  metrics: {
    train_c_index?: number;
    test_c_index?: number;
    cv_mean_c_index?: number;
    cv_std_c_index?: number;
  };
  training_meta: {
    trained_at?: string;
    sample_size?: number;
    train_size?: number;
    test_size?: number;
    epochs?: number;
    model_type?: string;
  };
  baseline_length: number;
}

export interface SurvivalPoint {
  t: number;
  s: number;
}

export interface PredictRequest {
  features: Record<string, number>;
  max_games?: number;
}

export interface PredictResponse {
  risk_score: number;
  median_survival: number;
  grade: string;
  risk_level: string;
  comment: string;
  survival_at: Record<string, number>; // JSON keys are strings: "50" / "100" / "150"
  survival_curve: SurvivalPoint[];
  extrapolation_warnings: string[];
}

export interface FamousPlayer {
  name: string;
  features: Record<string, number>;
  risk_score: number;
  predicted_career: number;
  actual_career: number | null;
  survival_100: number;
  grade: string;
}

export interface FamousPlayersResponse {
  players: FamousPlayer[];
  feature_names: string[];
}
