from sklearn.model_selection import KFold, cross_val_score


def evaluate_model(model, X, y):
    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="neg_mean_squared_error"
    )

    return -scores.mean()