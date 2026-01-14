import xgboost as xgb


# =========================
# Base models
# =========================
def build_xgboost():
    return xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )