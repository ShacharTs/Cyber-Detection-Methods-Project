import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# =========================
# XGBoost Model Builder
# =========================
def build_xgboost():
    return xgb.XGBClassifier(
        n_estimators=800,
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
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

# =========================
# Voting Ensemble Builder
# =========================
def build_voting_classifier(
    voting: str = "soft",
    weights: list | None = None
):
    xgb_model = build_xgboost()
    rf_model = build_random_forest()

    return VotingClassifier(
        estimators=[
            ("xgb", xgb_model),
            ("rf", rf_model),
        ],
        voting=voting,
        weights=weights,
        n_jobs=-1
    )
