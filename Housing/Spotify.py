import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ======================================================
# 1. Load dataset
# ======================================================
spotify = pd.read_csv("spotify_churn_dataset.csv")

# ======================================================
# 2. Encode categorical variables
#    - Convert string categories (gender, country, country, subscription_type, device_type)
#      into dummy/indicator variables (0/1).
#    - drop_first=True avoids multicollinearity by dropping
#      one category per feature.
# ======================================================
spotify = pd.get_dummies(
    spotify,
    columns=["gender", "country", "subscription_type", "device_type"],
    drop_first=True
)

# ======================================================
# 3. Split into features (X) and target (y)
#    - "is_churned" is the label (0 = stayed, 1 = churned).
#    - We also drop "user_id" because it’s just an identifier.
# ======================================================
X = spotify.drop(columns=["user_id", "is_churned"])
y = spotify["is_churned"]

print("Class balance:\n", y.value_counts(normalize=True))
# Example output: 0 = 74%, 1 = 26% → indicates class imbalance

# ======================================================
# 4. Train/Test split
#    - 80% for training, 20% for testing.
#    - stratify=y ensures the churn ratio (74/26) is preserved
#      in both train and test sets.
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=500, stratify=y
)

# ======================================================
# 5. Pipeline definition
#    - StandardScaler ensures all numeric features have mean=0, std=1.
#      This is critical for Logistic Regression with L1/L2 penalties,
#      because unscaled features can distort coefficient magnitudes.
#    - LogisticRegression:
#        • saga solver supports L1, L2, and ElasticNet.
#        • class_weight="balanced" automatically adjusts weights
#          so minority churners (class 1) get more importance.
#        • max_iter=5000 ensures convergence.
# ======================================================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        solver="saga",
        max_iter=5000,
        class_weight="balanced"
    ))
])

# ======================================================
# 6. Hyperparameter grid
#    - C: regularization strength (smaller = stronger penalty).
#    - penalty: type of regularization:
#        • "l1" = feature selection (sparse coefficients)
#        • "l2" = ridge (smooth coefficients)
#        • "elasticnet" = mix of L1 + L2
#    - l1_ratio: only used if penalty="elasticnet"
#        • 0 = pure L2, 1 = pure L1, 0.5 = mix
# ======================================================
param_grid = [
    {
        "clf__C": [0.001, 0.01, 0.1, 1, 10],
        "clf__penalty": ["l2"]
    },
    {
        "clf__C": [0.001, 0.01, 0.1, 1, 10],
        "clf__penalty": ["elasticnet"],
        "clf__l1_ratio": [0, 0.5, 1]
    }
]

# ======================================================
# 7. Grid search setup
#    - scoring="roc_auc":
#        • ROC AUC is more reliable than F1 or accuracy
#          for imbalanced datasets.
#        • Measures the model’s ability to rank positives
#          higher than negatives.
#    - cv=5: 5-fold cross-validation.
#    - n_jobs=-1: use all CPU cores.
# ======================================================
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1
)

# ======================================================
# 8. Fit model (searches for best hyperparameters)
# ======================================================
grid_search.fit(X_train, y_train)

# ======================================================
# 9. Print best parameters and CV performance
# ======================================================
print("Best Parameters:", grid_search.best_params_)
print("Best CV ROC AUC:", grid_search.best_score_)

# ======================================================
# 10. Evaluate on test set
#     - Predict churn labels.
#     - Predict churn probabilities (for ROC AUC).
# ======================================================
best_lr = grid_search.best_estimator_
y_pred = best_lr.predict(X_test)
y_pred_prob = best_lr.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("Test ROC AUC:", roc_auc_score(y_test, y_pred_prob))
