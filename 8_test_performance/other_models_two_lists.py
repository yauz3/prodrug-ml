#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multiview K-Fold (Stratified) CV + Same-Model-Per-View Ensembling
- No Murcko scaffold here (only label-stratified CV)
- Preprocessing: to_numeric(coerce) -> SimpleImputer(median) -> MinMaxScaler
- OOF & TEST_full_train bootstrap 95% CI (AUC, AP)
- Fold-based holdout/test metrics and mean±std summaries
- Early enrichment: EF@1%, EF@5%, BEDROC(λ=20/50/80)
- Per-view AND ensemble metrics are saved. Ensemble = linear weighted average across VIEWS for the SAME model type.

Outputs:
  - model_cv_test_metrics.csv   (fold×model×feature_regime all metrics)
  - model_cv_test_summary.csv   (mean±std by model×feature_regime)
  - model_ci_summary.csv        (OOF & TEST_full_train AUC/AP + 95% CI; per-view and ensemble)
  - model_early_enrichment.csv  (EF@1/5% + BEDROC 20/50/80 for fold/test/OOF/full-train)
"""

import argparse
import numpy as np
import pandas as pd
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    balanced_accuracy_score, brier_score_loss, matthews_corrcoef
)

RANDOM_STATE = 42
random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
N_SPLITS = 5

# ---------------------------------------------------------------------
# DEFINE YOUR FEATURE LISTS (views) HERE
# list_1 prefilled with your shared columns; please fill list_2..list_4.
FEATURE_LISTS: dict[str, list[str]] = {
    "list_1": ['Avalon_FP_1309', 'Avalon_FP_93', 'Avalon_FP_599', 'Avalon_FP_1436', 'Avalon_FP_1250', 'Avalon_FP_1969', 'Avalon_FP_487', 'Avalon_FP_1119', 'Avalon_FP_492', 'Avalon_FP_2013', 'Avalon_FP_12', 'Avalon_FP_932', 'Avalon_FP_231', 'Avalon_FP_2000', 'Avalon_FP_451', 'Avalon_FP_1122', 'Avalon_FP_422', 'Avalon_FP_1581', 'Avalon_FP_1548', 'Avalon_FP_1499', 'Avalon_FP_1236', 'Avalon_FP_1237', 'Avalon_FP_1498', 'Avalon_FP_2024', 'Avalon_FP_1157', 'Avalon_FP_297', 'Avalon_FP_443', 'Avalon_FP_399', 'Avalon_FP_1794', 'Avalon_FP_712', 'Avalon_FP_1414', 'Avalon_FP_339', 'Avalon_FP_1292', 'Avalon_FP_1570', 'Avalon_FP_1996', 'Avalon_FP_762', 'Avalon_FP_101', 'Avalon_FP_255', 'Avalon_FP_178', 'Avalon_FP_1040'],
    "list_2": ['Avalon_FP_1924', 'Avalon_FP_604', 'Avalon_FP_117', 'Avalon_FP_339', 'Avalon_FP_1838', 'Avalon_FP_384', 'Avalon_FP_1585', 'Avalon_FP_1280', 'Avalon_FP_1557', 'Avalon_FP_552', 'Avalon_FP_1097', 'Avalon_FP_1530', 'Avalon_FP_1967', 'Avalon_FP_159', 'Avalon_FP_1696', 'Avalon_FP_1119', 'Avalon_FP_1662', 'Avalon_FP_1689', 'Avalon_FP_152', 'Avalon_FP_816', 'Avalon_FP_200', 'Avalon_FP_1454', 'Avalon_FP_1713', 'Avalon_FP_361', 'Avalon_FP_1237', 'Avalon_FP_1202', 'Avalon_FP_879', 'Avalon_FP_1352', 'Avalon_FP_579', 'Avalon_FP_122', 'Avalon_FP_846', 'Avalon_FP_595', 'Avalon_FP_2020', 'Avalon_FP_955', 'Avalon_FP_1645', 'Avalon_FP_1285', 'Avalon_FP_2000', 'Avalon_FP_1669', 'Avalon_FP_1250', 'Avalon_FP_1684', 'Avalon_FP_834', 'Avalon_FP_790', 'Avalon_FP_1073', 'Avalon_FP_327', 'Avalon_FP_568', 'Avalon_FP_661', 'Avalon_FP_209', 'Avalon_FP_1861', 'Avalon_FP_985', 'Avalon_FP_1157', 'Avalon_FP_1136', 'Avalon_FP_1232', 'Avalon_FP_1850', 'Avalon_FP_1211', 'Avalon_FP_521'],
    "list_3": [],
    "list_4": [],
    "list_5": [],
}
# ---------------------------------------------------------------------

def write_csv(df: pd.DataFrame, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"[WRITE] {p.resolve()}  (rows={len(df)})")

def _safe(fn, default=np.nan):
    try:
        return float(fn())
    except Exception:
        return float(default)

def proba_or_decision(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)

def compute_metrics(y_true, scores, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    finite_ok = np.isfinite(scores).all()
    if finite_ok and scores.min() >= 0.0 and scores.max() <= 1.0:
        y_pred = (scores >= threshold).astype(int)
        brier = _safe(lambda: brier_score_loss(y_true, scores))
    else:
        y_pred = (scores >= 0.0).astype(int)
        s_min, s_max = np.nanmin(scores), np.nanmax(scores)
        s_norm = (scores - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(scores)
        brier = _safe(lambda: brier_score_loss(y_true, s_norm))

    return {
        "roc_auc": _safe(lambda: roc_auc_score(y_true, scores)),
        "ap":      _safe(lambda: average_precision_score(y_true, scores)),
        "f1":       _safe(lambda: f1_score(y_true, y_pred, zero_division=0), default=0.0),
        "precision":_safe(lambda: precision_score(y_true, y_pred, zero_division=0), default=0.0),
        "recall":   _safe(lambda: recall_score(y_true, y_pred, zero_division=0), default=0.0),
        "bal_acc":  _safe(lambda: balanced_accuracy_score(y_true, y_pred)),
        "brier":    brier,
        "mcc":      _safe(lambda: matthews_corrcoef(y_true, y_pred), default=0.0),
    }

def stratified_bootstrap_ci(y, s, metric_fn, n_boot=2000, alpha=0.95, seed=42):
    rng = np.random.RandomState(seed)
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos)==0 or len(neg)==0:
        m = _safe(lambda: metric_fn(y, s))
        return float(m), float(m), float(m)
    stats = []
    for _ in range(n_boot):
        bpos = rng.choice(pos, size=len(pos), replace=True)
        bneg = rng.choice(neg, size=len(neg), replace=True)
        bidx = np.concatenate([bpos, bneg])
        stats.append(_safe(lambda: metric_fn(y[bidx], s[bidx])))
    lo, hi = np.percentile(stats, [(1-alpha)/2*100, (1+(alpha))/2*100])
    return float(np.mean(stats)), float(lo), float(hi)

# ---------------- Early Enrichment ----------------
def ef_at(y_true, scores, frac=0.01):
    """Enrichment Factor @frac (e.g., 0.01=1%)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    N = y.size
    n_pos = int(y.sum())
    if N == 0 or n_pos == 0:
        return np.nan
    k = max(1, int(np.ceil(frac * N)))
    order = np.argsort(-s, kind="mergesort")  # stable sort
    hits_topk = int(y[order][:k].sum())
    return float((hits_topk / k) / (n_pos / N))

def bedroc(y_true, scores, alpha=20.0):
    """Truchon & Bayly (2007) normalization."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    N = y.size
    n = int(y.sum())
    if N == 0 or n == 0:
        return np.nan
    if n == N:
        return 1.0

    order = np.argsort(-s, kind="mergesort")
    ranks = np.nonzero(y[order])[0] + 1  # 1..N

    C = alpha / (1.0 - np.exp(-alpha))
    S = np.exp(-alpha * (ranks / N)).sum()
    RIE = C * (S / n)

    i = np.arange(1, n+1)
    Smax = np.exp(-alpha * (i / N)).sum()
    RIE_max = C * (Smax / n)

    i = np.arange(N - n + 1, N + 1)
    Smin = np.exp(-alpha * (i / N)).sum()
    RIE_min = C * (Smin / n)

    denom = (RIE_max - RIE_min)
    if denom <= 0:
        return np.nan
    return float((RIE - RIE_min) / denom)

# ---------------- Data ----------------
def load_data(train_csv, test_csv, label_col="label"):
    train = pd.read_csv(train_csv)
    test  = pd.read_csv(test_csv)
    y_train = train[label_col].astype(int)
    y_test  = test[label_col].astype(int)
    return train, y_train, test, y_test

def coerce_numeric(df, cols):
    return df.loc[:, cols].apply(pd.to_numeric, errors="coerce")

def validate_views(train_df, test_df, view_names):
    for vn in view_names:
        if vn not in FEATURE_LISTS:
            raise ValueError(f"Unknown view '{vn}'. Available: {list(FEATURE_LISTS.keys())}")
        cols = FEATURE_LISTS[vn]
        if not cols:
            raise ValueError(f"FEATURE_LISTS['{vn}'] is empty. Please fill with column names.")
        miss_tr = [c for c in cols if c not in train_df.columns]
        miss_te = [c for c in cols if c not in test_df.columns]
        if miss_tr:
            raise ValueError(f"[{vn}] Missing columns in TRAIN: {miss_tr[:10]} ...")
        if miss_te:
            raise ValueError(f"[{vn}] Missing columns in TEST: {miss_te[:10]} ...")

def build_models():
    base_estimators = [
        ("rf",  RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")),
        ("svc", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        ("enet", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                    max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
    ]
    models = {
        "LogReg_L2": LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000,
                                        class_weight="balanced", random_state=RANDOM_STATE),
        "LogReg_ElasticNet": LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "SVM_RBF": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE,
                                               n_jobs=-1, class_weight="balanced"),
        "ExtraTrees": ExtraTreesClassifier(random_state=RANDOM_STATE, min_samples_split=7,
                                           n_jobs=-1, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "Bagging": BaggingClassifier(random_state=RANDOM_STATE, n_estimators=300, n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(200,), max_iter=1000, random_state=RANDOM_STATE),
        "Voting": VotingClassifier(estimators=base_estimators, voting="soft", n_jobs=-1),
        "Stacking": StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            passthrough=True, n_jobs=-1
        )
    }
    return models

def parse_views_and_weights(list_number: int, views_arg: str | None, weights_arg: str | None):
    keys = ["list_1","list_2","list_3","list_4"]
    if views_arg:
        view_names = [v.strip() for v in views_arg.split(",") if v.strip()]
    else:
        view_names = keys[:max(1, min(list_number, len(keys)))]
    # weights
    if weights_arg:
        w = np.array([float(x.strip()) for x in weights_arg.split(",") if x.strip()], dtype=float)
        if len(w) != len(view_names):
            raise ValueError(f"--view_weights has length {len(w)} but {len(view_names)} views selected.")
        w = np.maximum(w, 0.0)
        if w.sum() == 0:
            w = np.ones_like(w)
        w = w / w.sum()
    else:
        w = np.ones(len(view_names), dtype=float) / len(view_names)
    return view_names, w

def _fit_one_view_pipeline(model_name, estimator, X_tr, y_tr, sample_w=None):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
        ("model", clone(estimator))
    ])
    fit_params = {}
    if model_name in ("GradientBoosting", "AdaBoost") and sample_w is not None:
        fit_params["model__sample_weight"] = sample_w
    try:
        pipe.fit(X_tr, y_tr, **fit_params)
    except TypeError:
        pipe.fit(X_tr, y_tr)
    return pipe

def fit_kfold_cv_multiview(models: dict,
                           train_df: pd.DataFrame, y_train: pd.Series,
                           test_df: pd.DataFrame,  y_test: pd.Series,
                           view_names: list[str], view_weights: np.ndarray,
                           n_splits=N_SPLITS, n_boot=2000, alpha=0.95):
    """
    Trains SAME model type across all selected views, ensembles by weighted average.
    Saves per-view and ensemble metrics.
    """
    rows = []
    ci_rows = []
    ee_rows = []

    # Prepare X matrices per view
    Xtr_views = {vn: coerce_numeric(train_df, FEATURE_LISTS[vn]) for vn in view_names}
    Xte_views = {vn: coerce_numeric(test_df,  FEATURE_LISTS[vn]) for vn in view_names}

    view_tag = "+".join(view_names)
    ens_regime = f"ensemble[{view_tag}]"

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for model_name, estimator in models.items():

        # OOF collectors per regime (each view and ensemble)
        oof_store = {f"view:{vn}": {"idx": [], "scores": []} for vn in view_names}
        oof_store[ens_regime] = {"idx": [], "scores": []}

        for fold_id, (tr_idx, val_idx) in enumerate(skf.split(Xtr_views[view_names[0]], y_train)):
            # Split y
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            # Class imbalance weight (only used for GB/Ada)
            n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
            pos_w = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
            sample_w = np.where(y_tr.values==1, pos_w, 1.0).astype(float)

            # Fit per-view pipelines for this fold
            fold_pipes = {}
            val_pred_views = []
            test_pred_views = []

            for vi, vn in enumerate(view_names):
                X_tr = Xtr_views[vn].iloc[tr_idx]
                X_val = Xtr_views[vn].iloc[val_idx]

                pipe = _fit_one_view_pipeline(model_name, estimator, X_tr, y_tr,
                                              sample_w=sample_w if model_name in ("GradientBoosting","AdaBoost") else None)
                fold_pipes[vn] = pipe

                # Scores
                s_val = proba_or_decision(pipe, X_val)
                s_test = proba_or_decision(pipe, Xte_views[vn])

                val_pred_views.append(s_val)
                test_pred_views.append(s_test)

                # Log per-view metrics (cv_holdout + test) for traceability
                val_metrics = compute_metrics(y_val, s_val)
                rows.append({
                    "model": model_name, "base_model": model_name,
                    "feature_regime": f"view:{vn}",
                    "views": vn, "weights": "1.0",
                    "split": fold_id, "dataset": "cv_holdout", **val_metrics
                })
                ee_rows.append({
                    "model": model_name, "base_model": model_name,
                    "feature_regime": f"view:{vn}",
                    "views": vn, "weights": "1.0",
                    "split": fold_id, "dataset": "cv_holdout",
                    "ef_1pct": ef_at(y_val, s_val, 0.01),
                    "ef_5pct": ef_at(y_val, s_val, 0.05),
                    "bedroc_20": bedroc(y_val, s_val, 20.0),
                    "bedroc_50": bedroc(y_val, s_val, 50.0),
                    "bedroc_80": bedroc(y_val, s_val, 80.0),
                })

                # OOF accumulate per-view
                oof_store[f"view:{vn}"]["idx"].append(val_idx)
                oof_store[f"view:{vn}"]["scores"].append(s_val)

                # Test per-view
                test_metrics = compute_metrics(y_test, s_test)
                rows.append({
                    "model": model_name, "base_model": model_name,
                    "feature_regime": f"view:{vn}",
                    "views": vn, "weights": "1.0",
                    "split": fold_id, "dataset": "test", **test_metrics
                })
                ee_rows.append({
                    "model": model_name, "base_model": model_name,
                    "feature_regime": f"view:{vn}",
                    "views": vn, "weights": "1.0",
                    "split": fold_id, "dataset": "test",
                    "ef_1pct": ef_at(y_test, s_test, 0.01),
                    "ef_5pct": ef_at(y_test, s_test, 0.05),
                    "bedroc_20": bedroc(y_test, s_test, 20.0),
                    "bedroc_50": bedroc(y_test, s_test, 50.0),
                    "bedroc_80": bedroc(y_test, s_test, 80.0),
                })

            # Ensemble (linear weighted across views) for this fold
            val_pred_views = np.vstack(val_pred_views)  # V x n_val
            test_pred_views = np.vstack(test_pred_views)  # V x n_test
            s_val_ens = np.average(val_pred_views, axis=0, weights=view_weights)
            s_test_ens = np.average(test_pred_views, axis=0, weights=view_weights)

            # Log ensemble fold metrics
            val_metrics = compute_metrics(y_val, s_val_ens)
            rows.append({
                "model": model_name, "base_model": model_name,
                "feature_regime": ens_regime,
                "views": "+".join(view_names),
                "weights": ",".join([f"{w:.4f}" for w in view_weights]),
                "split": fold_id, "dataset": "cv_holdout", **val_metrics
            })
            ee_rows.append({
                "model": model_name, "base_model": model_name,
                "feature_regime": ens_regime,
                "views": "+".join(view_names),
                "weights": ",".join([f"{w:.4f}" for w in view_weights]),
                "split": fold_id, "dataset": "cv_holdout",
                "ef_1pct": ef_at(y_val, s_val_ens, 0.01),
                "ef_5pct": ef_at(y_val, s_val_ens, 0.05),
                "bedroc_20": bedroc(y_val, s_val_ens, 20.0),
                "bedroc_50": bedroc(y_val, s_val_ens, 50.0),
                "bedroc_80": bedroc(y_val, s_val_ens, 80.0),
            })

            # OOF ensemble accumulate
            oof_store[ens_regime]["idx"].append(val_idx)
            oof_store[ens_regime]["scores"].append(s_val_ens)

            # Ensemble test
            test_metrics = compute_metrics(y_test, s_test_ens)
            rows.append({
                "model": model_name, "base_model": model_name,
                "feature_regime": ens_regime,
                "views": "+".join(view_names),
                "weights": ",".join([f"{w:.4f}" for w in view_weights]),
                "split": fold_id, "dataset": "test", **test_metrics
            })
            ee_rows.append({
                "model": model_name, "base_model": model_name,
                "feature_regime": ens_regime,
                "views": "+".join(view_names),
                "weights": ",".join([f"{w:.4f}" for w in view_weights]),
                "split": fold_id, "dataset": "test",
                "ef_1pct": ef_at(y_test, s_test_ens, 0.01),
                "ef_5pct": ef_at(y_test, s_test_ens, 0.05),
                "bedroc_20": bedroc(y_test, s_test_ens, 20.0),
                "bedroc_50": bedroc(y_test, s_test_ens, 50.0),
                "bedroc_80": bedroc(y_test, s_test_ens, 80.0),
            })

        # ---- OOF CI + Early Enrichment (per-view and ensemble) ----
        for regime, store in oof_store.items():
            oidx = np.concatenate(store["idx"])
            osco = np.concatenate(store["scores"])
            order = np.argsort(oidx)
            y_oof = y_train.iloc[oidx[order]].values
            s_oof = osco[order]

            oof_auc = _safe(lambda: roc_auc_score(y_oof, s_oof))
            _, oof_auc_lo, oof_auc_hi = stratified_bootstrap_ci(y_oof, s_oof, roc_auc_score, n_boot=n_boot, alpha=alpha, seed=RANDOM_STATE)
            oof_ap  = _safe(lambda: average_precision_score(y_oof, s_oof))
            _, oof_ap_lo,  oof_ap_hi  = stratified_bootstrap_ci(y_oof, s_oof, average_precision_score, n_boot=n_boot, alpha=alpha, seed=RANDOM_STATE)

            ci_rows.append({
                "model": model_name, "base_model": model_name,
                "feature_regime": regime,
                "views": (regime.split("view:")[-1] if regime.startswith("view:") else "+".join(view_names)),
                "weights": ("1.0" if regime.startswith("view:") else ",".join([f"{w:.4f}" for w in view_weights])),
                "dataset": "OOF",
                "roc_auc": oof_auc, "roc_auc_lo": oof_auc_lo, "roc_auc_hi": oof_auc_hi,
                "ap": oof_ap, "ap_lo": oof_ap_lo, "ap_hi": oof_ap_hi
            })
            ee_rows.append({
                "model": model_name, "base_model": model_name,
                "feature_regime": regime,
                "views": (regime.split("view:")[-1] if regime.startswith("view:") else "+".join(view_names)),
                "weights": ("1.0" if regime.startswith("view:") else ",".join([f"{w:.4f}" for w in view_weights])),
                "split": -1, "dataset": "OOF",
                "ef_1pct": ef_at(y_oof, s_oof, 0.01),
                "ef_5pct": ef_at(y_oof, s_oof, 0.05),
                "bedroc_20": bedroc(y_oof, s_oof, 20.0),
                "bedroc_50": bedroc(y_oof, s_oof, 50.0),
                "bedroc_80": bedroc(y_oof, s_oof, 80.0),
            })

        # ---- Full-train → TEST CI + Early Enrichment (per-view and ensemble) ----
        # Fit per-view on full training
        s_test_views = []
        for vi, vn in enumerate(view_names):
            Xtr = Xtr_views[vn]; Xte = Xte_views[vn]
            # class weights only for GB/Ada
            n_pos = int(y_train.sum()); n_neg = len(y_train) - n_pos
            pos_w = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
            sample_w_full = np.where(y_train.values==1, pos_w, 1.0).astype(float)

            pipe = _fit_one_view_pipeline(
                model_name, estimator, Xtr, y_train,
                sample_w=sample_w_full if model_name in ("GradientBoosting","AdaBoost") else None
            )
            s_test = proba_or_decision(pipe, Xte)
            s_test_views.append(s_test)

            # CI per-view
            test_auc = _safe(lambda: roc_auc_score(y_test, s_test))
            _, test_auc_lo, test_auc_hi = stratified_bootstrap_ci(y_test.values, s_test, roc_auc_score, n_boot=n_boot, alpha=alpha, seed=RANDOM_STATE)
            test_ap  = _safe(lambda: average_precision_score(y_test, s_test))
            _, test_ap_lo,  test_ap_hi  = stratified_bootstrap_ci(y_test.values, s_test, average_precision_score, n_boot=n_boot, alpha=alpha, seed=RANDOM_STATE)

            ci_rows.append({
                "model": model_name, "base_model": model_name,
                "feature_regime": f"view:{vn}",
                "views": vn, "weights": "1.0",
                "dataset": "TEST_full_train",
                "roc_auc": test_auc, "roc_auc_lo": test_auc_lo, "roc_auc_hi": test_auc_hi,
                "ap": test_ap, "ap_lo": test_ap_lo, "ap_hi": test_ap_hi
            })
            ee_rows.append({
                "model": model_name, "base_model": model_name,
                "feature_regime": f"view:{vn}",
                "views": vn, "weights": "1.0",
                "split": -1, "dataset": "TEST_full_train",
                "ef_1pct": ef_at(y_test, s_test, 0.01),
                "ef_5pct": ef_at(y_test, s_test, 0.05),
                "bedroc_20": bedroc(y_test, s_test, 20.0),
                "bedroc_50": bedroc(y_test, s_test, 50.0),
                "bedroc_80": bedroc(y_test, s_test, 80.0),
            })

        # Ensemble full-train→test
        s_test_views = np.vstack(s_test_views)
        s_test_ens = np.average(s_test_views, axis=0, weights=view_weights)

        test_auc = _safe(lambda: roc_auc_score(y_test, s_test_ens))
        _, test_auc_lo, test_auc_hi = stratified_bootstrap_ci(y_test.values, s_test_ens, roc_auc_score, n_boot=n_boot, alpha=alpha, seed=RANDOM_STATE)
        test_ap  = _safe(lambda: average_precision_score(y_test, s_test_ens))
        _, test_ap_lo,  test_ap_hi  = stratified_bootstrap_ci(y_test.values, s_test_ens, average_precision_score, n_boot=n_boot, alpha=alpha, seed=RANDOM_STATE)

        ci_rows.append({
            "model": model_name, "base_model": model_name,
            "feature_regime": ens_regime,
            "views": "+".join(view_names),
            "weights": ",".join([f"{w:.4f}" for w in view_weights]),
            "dataset": "TEST_full_train",
            "roc_auc": test_auc, "roc_auc_lo": test_auc_lo, "roc_auc_hi": test_auc_hi,
            "ap": test_ap, "ap_lo": test_ap_lo, "ap_hi": test_ap_hi
        })
        ee_rows.append({
            "model": model_name, "base_model": model_name,
            "feature_regime": ens_regime,
            "views": "+".join(view_names),
            "weights": ",".join([f"{w:.4f}" for w in view_weights]),
            "split": -1, "dataset": "TEST_full_train",
            "ef_1pct": ef_at(y_test, s_test_ens, 0.01),
            "ef_5pct": ef_at(y_test, s_test_ens, 0.05),
            "bedroc_20": bedroc(y_test, s_test_ens, 20.0),
            "bedroc_50": bedroc(y_test, s_test_ens, 50.0),
            "bedroc_80": bedroc(y_test, s_test_ens, 80.0),
        })

    return rows, ci_rows, ee_rows

def summarize(df):
    agg = df.groupby(["model", "feature_regime", "dataset"]).agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for (m, s) in agg.columns]
    agg = agg.reset_index()
    return agg

def main():
    ap = argparse.ArgumentParser(description="Multiview Stratified K-Fold CV + OOF/TEST bootstrap CI (same-model-per-view ensemble)")
    ap.add_argument("--train_csv", default="filtered_all_data_training_1_6.csv")
    ap.add_argument("--test_csv",  default="test_ready.csv")
    ap.add_argument("--out_splits", default="model_cv_test_metrics_other_two.csv")
    ap.add_argument("--out_summary", default="model_cv_test_summary_other_two.csv")
    ap.add_argument("--out_ci", default="model_ci_summary_other_two.csv")
    ap.add_argument("--out_ee", default="model_early_enrichment_other_two.csv")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.95)
    # multiview controls
    ap.add_argument("--list_number", type=int, default=2, help="Use first N lists among list_1..list_4 (ignored if --views is provided)")
    ap.add_argument("--views", type=str, default=None, help="Comma-separated view names (e.g., 'list_1,list_3'); overrides --list_number")
    ap.add_argument("--view_weights", type=str, default=None, help="Comma-separated weights matching selected views (auto-normalized); default equal")
    args = ap.parse_args()

    # Data
    train_df, y_train, test_df, y_test = load_data(args.train_csv, args.test_csv)

    # Views & weights
    view_names, view_weights = parse_views_and_weights(args.list_number, args.views, args.view_weights)
    validate_views(train_df, test_df, view_names)
    print(f"[INFO] Views: {view_names} | Weights (normalized): {view_weights.round(4).tolist()}")

    # Models
    models = build_models()
    print(f"[INFO] Models: {list(models.keys())}")

    # Run CV
    print("[*] Running Multiview Stratified K-Fold CV ...")
    all_rows, ci_rows, ee_rows = fit_kfold_cv_multiview(
        models, train_df, y_train, test_df, y_test,
        view_names=view_names, view_weights=view_weights,
        n_splits=N_SPLITS, n_boot=args.n_boot, alpha=args.alpha
    )

    results = pd.DataFrame(all_rows)
    write_csv(results, args.out_splits)

    summary = summarize(results)
    write_csv(summary, args.out_summary)

    ci_df = pd.DataFrame(ci_rows)
    write_csv(ci_df, args.out_ci)

    ee_df = pd.DataFrame(ee_rows)
    write_csv(ee_df, args.out_ee)

    # Console summary
    print("\n=== Mean ± SD (fold-based) ===")
    for (name, reg) in summary[["model","feature_regime"]].drop_duplicates().itertuples(index=False):
        sub = summary[(summary["model"] == name) & (summary["feature_regime"] == reg)]
        def fmt(row, metric):
            mu = row.get(f"{metric}_mean", np.nan); sd = row.get(f"{metric}_std", np.nan)
            return f"{mu:.3f} ± {sd:.3f}"
        try:
            cv_row = sub[sub["dataset"] == "cv_holdout"].iloc[0]
            ts_row = sub[sub["dataset"] == "test"].iloc[0]
            print(f"{name:>18} [{reg}] | CV AUC: {fmt(cv_row,'roc_auc')} | TEST AUC: {fmt(ts_row,'roc_auc')} | "
                  f"CV AP: {fmt(cv_row,'ap')} | TEST AP: {fmt(ts_row,'ap')}")
        except Exception:
            pass

    print("\n=== 95% CI (bootstrap) for OOF and TEST_full_train ===")
    for (name, reg) in ci_df[["model","feature_regime"]].drop_duplicates().itertuples(index=False):
        sub = ci_df[(ci_df["model"] == name) & (ci_df["feature_regime"] == reg)]
        try:
            oof = sub[sub["dataset"] == "OOF"].iloc[0]
            tst = sub[sub["dataset"] == "TEST_full_train"].iloc[0]
            print(f"{name:>18} [{reg}] | OOF AUC: {oof['roc_auc']:.3f} (CI {oof['roc_auc_lo']:.3f}-{oof['roc_auc_hi']:.3f}) "
                  f"| OOF AP: {oof['ap']:.3f} (CI {oof['ap_lo']:.3f}-{oof['ap_hi']:.3f})")
            print(f"{'':>18} [{reg}] | TEST AUC: {tst['roc_auc']:.3f} (CI {tst['roc_auc_lo']:.3f}-{tst['roc_auc_hi']:.3f}) "
                  f"| TEST AP: {tst['ap']:.3f} (CI {tst['ap_lo']:.3f}-{tst['ap_hi']:.3f})")
        except Exception:
            pass

    print(f"\n[OK] Wrote: {args.out_splits}, {args.out_summary}, {args.out_ci}, {args.out_ee}")

if __name__ == "__main__":
    main()
