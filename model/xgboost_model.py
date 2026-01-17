from xgboost import XGBClassifier

def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(set(y_train)),
        eval_metric="mlogloss",
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    return model
