import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

DATA_PATH = "data/healthcare-dataset-stroke-cleaned.csv"
RANDOM_STATE = 42

def evaluate_model(name, split, y_true, y_pred, y_prob):
    return {
        "Model": name,
        "Split": split,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    def make_preprocessor():
        return ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", StandardScaler(), numeric_cols),
        ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=180, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE),
        "K-Nearest Neighbors Classifier": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Support Vector Classifier": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_STATE),
    }

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    fitted_models = {}
    rows = []

    for name, clf in models.items():
        pipe = Pipeline([("preprocess", make_preprocessor()), ("model", clf)])
        if name == "Gradient Boosting Classifier":
            pipe.fit(X_train, y_train, model__sample_weight=sample_weights)
        else:
            pipe.fit(X_train, y_train)
        fitted_models[name] = pipe

        for split_name, X_split, y_split in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
            y_pred = pipe.predict(X_split)
            y_prob = pipe.predict_proba(X_split)[:, 1]
            rows.append(evaluate_model(name, split_name, y_split, y_pred, y_prob))

    results = pd.DataFrame(rows)
    top3 = results[results["Split"] == "Validation"].sort_values("ROC-AUC", ascending=False).head(3)["Model"].tolist()
    validation_auc = results[(results["Split"] == "Validation") & (results["Model"].isin(top3))].set_index("Model").loc[top3, "ROC-AUC"].values
    bayesian_weights = validation_auc / validation_auc.sum()

    for split_name, X_split, y_split in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        prob_matrix = np.column_stack([fitted_models[m].predict_proba(X_split)[:, 1] for m in top3])
        soft_prob = prob_matrix.mean(axis=1)
        soft_pred = (soft_prob >= 0.5).astype(int)
        rows.append(evaluate_model("Soft Voting Ensemble (Top 3)", split_name, y_split, soft_pred, soft_prob))

        bayes_prob = prob_matrix.dot(bayesian_weights)
        bayes_pred = (bayes_prob >= 0.5).astype(int)
        rows.append(evaluate_model("Bayesian Weighted Ensemble (Top 3)", split_name, y_split, bayes_pred, bayes_prob))

    final_results = pd.DataFrame(rows)
    print(final_results.round(4).sort_values(["Split", "F1-Score"], ascending=[True, False]))
    final_results.round(4).to_csv("results/model_metrics.csv", index=False)

if __name__ == "__main__":
    main()
