import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# =========================
# XGBoost Model Builder
# =========================
def build_xgboost():
    return xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        random_state=42
    )

# =========================
# Random Forest Model Builder
# =========================
def build_random_forest():
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )