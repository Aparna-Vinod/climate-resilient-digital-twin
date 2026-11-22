"""
Task 1: train and evaluate the baseline, logistic regression and random forest.

Run from the project root:
    py src/machine_learning/train_models.py

Outputs
    results/model_metrics.csv          model, precision, recall, f1, roc_auc (evaluated on 2026)
    results/metrics_by_borough.csv     per-borough precision / recall / F1 on 2026
    results/feature_importances.csv    random forest importances, logistic regression coefficients
    results/fig5_confusion_matrices.png
    results/model_selection.json       how the best model was chosen (training data only)
    models/best_model.joblib           fitted scikit-learn Pipeline (preprocessing + model)

Rules followed
    * Label = flood_count >= 4 (a 311-complaint PROXY for flooding, not measured flooding).
    * Train = dates before 2026-01-01, test = 2026. Time-based, no shuffling.
    * All tuning (baseline threshold, hyper-parameters, decision thresholds, model choice)
      uses TRAINING data only. 2026 is touched once, at the very end, for reporting.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,  # noqa: E402
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.machine_learning.features import (CATEGORICAL_FEATURES, FEATURES,  # noqa: E402
                                           NUMERIC_FEATURES, ROOT, load_data,
                                           split_train_test)

RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FORBIDDEN = {"flood_count", "catch_basin_count", "manhole_count", "is_flood_day"}
SEED = 42
N_SPLITS = 5


# ---------------------------------------------------------------- helpers
def prf(y_true, y_pred):
    return (precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0))


def date_based_folds(train, n_splits=N_SPLITS):
    """TimeSeriesSplit on the unique DATES (so all 5 boroughs of a date stay together),
    returned as (train_row_index, validation_row_index) pairs. Validation is always later
    in time than the part used for fitting."""
    dates = np.sort(train["date"].unique())
    folds = []
    for tr_d, va_d in TimeSeriesSplit(n_splits=n_splits).split(dates):
        tr_idx = np.where(train["date"].isin(dates[tr_d]))[0]
        va_idx = np.where(train["date"].isin(dates[va_d]))[0]
        folds.append((tr_idx, va_idx))
    return folds


def make_pipeline(model, scale):
    num = StandardScaler() if scale else "passthrough"
    prep = ColumnTransformer([
        ("num", num, NUMERIC_FEATURES),
        ("borough", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def best_f1_threshold(y_true, score):
    """Score cut-off with the highest F1 (used on TRAINING out-of-fold scores only)."""
    best_t, best_f = 0.5, -1.0
    for t in np.unique(np.round(score, 4)):
        f = f1_score(y_true, score >= t, zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def make_clone(pipe, params):
    m = clone(pipe)
    m.set_params(**params)
    return m


def out_of_fold_scores(pipe, params, X, y, folds):
    """Fit on each earlier block, score the next block. Only validated rows get a score."""
    oof = np.full(len(y), np.nan)
    for tr_idx, va_idx in folds:
        m = make_clone(pipe, params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]
    mask = ~np.isnan(oof)
    return oof, mask


# ---------------------------------------------------------------- main
def main():
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    # leakage guard: the label columns must never be features
    assert not FORBIDDEN & set(FEATURES), "a label column is in FEATURES"

    df = load_data()
    train, test = split_train_test(df)
    train = train.reset_index(drop=True)
    X_tr, y_tr = train[FEATURES], train["is_flood_day"]
    X_te, y_te = test[FEATURES], test["is_flood_day"]
    print(f"Train: {len(train)} rows, {y_tr.mean():.1%} flood days "
          f"({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test : {len(test)} rows, {y_te.mean():.1%} flood days "
          f"({test['date'].min().date()} to {test['date'].max().date()})")

    folds = date_based_folds(train)
    print(f"TimeSeriesSplit on training dates: {len(folds)} folds, validation always after fitting data")

    # ---- 1. Baseline: flood if precip_mm >= t, t chosen on TRAINING data (best F1)
    t_base, f_base_train = best_f1_threshold(y_tr, train["precip_mm"].to_numpy())
    print(f"\nBaseline: flood if precip_mm >= {t_base} mm (training F1 {f_base_train:.3f})")

    # ---- 2. Tune LR and RF with TimeSeriesSplit on training data only
    candidates = {
        "logistic_regression": (
            make_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000), scale=True),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            make_pipeline(RandomForestClassifier(class_weight="balanced", random_state=SEED,
                                                 n_estimators=300, n_jobs=-1), scale=False),
            {"model__max_depth": [3, 5, 8],
             "model__min_samples_leaf": [5, 20, 50]},
        ),
    }
    fitted, selection = {}, {"baseline_threshold_mm": t_base,
                             "baseline_train_f1": f_base_train, "models": {}}
    for name, (pipe, grid) in candidates.items():
        gs = GridSearchCV(pipe, grid, cv=folds, scoring="average_precision", refit=True, n_jobs=-1)
        gs.fit(X_tr, y_tr)
        # decision threshold for yes/no metrics: best F1 on out-of-fold TRAINING scores
        oof, mask = out_of_fold_scores(pipe, gs.best_params_, X_tr, y_tr, folds)
        thr, cv_f1 = best_f1_threshold(y_tr[mask], oof[mask])
        fitted[name] = (gs.best_estimator_, thr)
        selection["models"][name] = {
            "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else v)
                            for k, v in gs.best_params_.items()},
            "cv_average_precision": float(gs.best_score_),
            "decision_threshold": thr,
            "cv_f1_at_threshold": float(cv_f1),
        }
        print(f"{name}: best params {gs.best_params_}, CV average precision "
              f"{gs.best_score_:.3f}, decision threshold {thr:.3f} (CV F1 {cv_f1:.3f})")

    # ---- 3. Choose the best model from TRAINING validation only (before looking at 2026)
    best_name = max(selection["models"], key=lambda n: selection["models"][n]["cv_average_precision"])
    selection["best_model"] = best_name
    selection["selection_rule"] = ("highest mean average precision over the 5 TimeSeriesSplit "
                                   "validation folds on 2024-2025 data")
    print(f"\nBest model (by training CV average precision): {best_name}")

    # ---- 4. Evaluate everything ONCE on 2026
    preds, scores = {}, {}
    preds["baseline"] = (test["precip_mm"] >= t_base).astype(int).to_numpy()
    scores["baseline"] = preds["baseline"]            # no probabilities: yes/no used as the score
    for name, (pipe, thr) in fitted.items():
        scores[name] = pipe.predict_proba(X_te)[:, 1]
        preds[name] = (scores[name] >= thr).astype(int)

    rows = []
    for name in ["baseline", "logistic_regression", "random_forest"]:
        p, r, f = prf(y_te, preds[name])
        rows.append({"model": name, "precision": p, "recall": r, "f1": f,
                     "roc_auc": roc_auc_score(y_te, scores[name])})
    metrics = pd.DataFrame(rows, columns=["model", "precision", "recall", "f1", "roc_auc"])
    metrics.to_csv(RESULTS / "model_metrics.csv", index=False)
    print("\nResults on the 2026 test period:")
    print(metrics.round(3).to_string(index=False))

    if (metrics["f1"] > 0.9).any():
        print("\n!!! WARNING: an F1 above 0.9 is suspiciously good. Check for label leakage "
              "before using these numbers.")

    # ---- 5. Per-borough metrics on 2026
    brows = []
    for name in ["baseline", "logistic_regression", "random_forest"]:
        for b, idx in test.reset_index(drop=True).groupby("borough").groups.items():
            yt, yp = y_te.to_numpy()[idx], preds[name][idx]
            p, r, f = prf(yt, yp)
            brows.append({"model": name, "borough": b, "precision": p, "recall": r, "f1": f,
                          "flood_days": int(yt.sum()), "rows": len(idx)})
    pd.DataFrame(brows).to_csv(RESULTS / "metrics_by_borough.csv", index=False)

    # ---- 6. Confusion-matrix figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, name in zip(axes, ["baseline", "logistic_regression", "random_forest"]):
        cm = confusion_matrix(y_te, preds[name], labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
        ax.set_xticks([0, 1], ["no flood", "flood day"])
        ax.set_yticks([0, 1], ["no flood", "flood day"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual (311 proxy)")
        ax.set_title(name + (" (best)" if name == best_name else ""))
    fig.suptitle("Confusion matrices on the 2026 test period")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig5_confusion_matrices.png", dpi=150)
    plt.close(fig)

    # ---- 7. Feature importances
    imp = []
    for name, (pipe, _) in fitted.items():
        names = [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
        model = pipe.named_steps["model"]
        if name == "random_forest":
            vals, kind = model.feature_importances_, "impurity importance"
        else:
            vals, kind = model.coef_[0], "coefficient on standardised features"
        imp += [{"model": name, "feature": f, "importance": float(v), "kind": kind}
                for f, v in zip(names, vals)]
    pd.DataFrame(imp).to_csv(RESULTS / "feature_importances.csv", index=False)

    # ---- 8. Save the selection record and the best model (pipeline includes preprocessing)
    selection["test_year_used_for_selection"] = False
    (RESULTS / "model_selection.json").write_text(json.dumps(selection, indent=2))
    joblib.dump(fitted[best_name][0], MODELS / "best_model.joblib")
    print(f"\nSaved best model ({best_name}) to models/best_model.joblib")
    print("Saved: results/model_metrics.csv, metrics_by_borough.csv, feature_importances.csv, "
          "fig5_confusion_matrices.png, model_selection.json")


if __name__ == "__main__":
    main()