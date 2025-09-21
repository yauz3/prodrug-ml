#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hard-negative & domain-adversarial decoy selection
- Input: single CSV with columns: label, source_dataset, and fingerprint features
- Uses ONLY the provided feature_list (or auto-detects numeric FP columns if not given)
- Steps:
  (1) Compute max Tanimoto to positives (from a chosen FP subset, e.g., Morgan bits)
  (2) Train a domain classifier (ExtraTrees) to predict decoy source; take max prob (domain_conf)
  (3) Cluster negatives (PCA -> KMeans). Drop clusters:
        - dominated by a single source (low entropy), or
        - far from positives (low max_tanimoto_to_pos mean)
  (4) Filter negatives by:
        max_tanimoto_to_pos in [tani_min, tani_max]
        domain_conf <= domain_conf_max
  (5) Balance by source and cap negatives per positive (neg_pos_ratio)
- Output: writes a new CSV with all positives + selected negatives
"""

import argparse, math
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ------------ helpers ------------
def parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

NON_FEATURE_COLS = {
    "label", "source_dataset", "smiles", "SMILES",
    "inchikey", "InChIKey", "inchi", "InChI",
    "id", "ID", "molid", "MOLID", "name", "Name"
}

def detect_feature_columns(df: pd.DataFrame, feature_list: List[str]) -> List[str]:
    if feature_list:
        miss = [c for c in feature_list if c not in df.columns]
        if miss:
            raise ValueError(f"Missing features: {miss[:8]} ...")
        return feature_list
    # auto: all numeric except known non-feature cols
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num_cols if c not in NON_FEATURE_COLS]
    if not feats:
        raise ValueError("No feature columns found.")
    return feats

def get_prefixed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns found with prefix '{prefix}'.")
    return cols

def to_uint8_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    X = df.loc[:, cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(0, 1)
    return (X.values > 0.5).astype(np.uint8)

def tanimoto_max_to_pos(U_bits: np.ndarray, P_bits: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Vectorized max Tanimoto similarity from U to set P (binary matrices)."""
    if P_bits.size == 0 or U_bits.size == 0:
        return np.zeros(U_bits.shape[0], dtype=np.float32)
    u_ones = U_bits.sum(axis=1).astype(np.int32)
    p_ones = P_bits.sum(axis=1).astype(np.int32)
    PT = P_bits.T
    out = np.zeros(U_bits.shape[0], dtype=np.float32)
    for i in range(0, U_bits.shape[0], chunk):
        Uc = U_bits[i:i+chunk]
        inter = Uc @ PT
        union = (u_ones[i:i+Uc.shape[0], None] + p_ones[None, :]) - inter
        union = np.maximum(union, 1)
        sims = inter / union
        out[i:i+Uc.shape[0]] = sims.max(axis=1)
    return out

def entropy_of_counts(counts: np.ndarray) -> float:
    p = counts / max(1, counts.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

# ------------ main ------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="all_training.csv")
    ap.add_argument("--out_csv", default="filtered_all_data_training.csv")
    ap.add_argument("--pos_name", default="prodrug")
    ap.add_argument("--decoy_names", default="dude,chembl_random,chembl_high_conf")
    ap.add_argument("--feature_list", default="", help="Comma-separated feature columns to use. If empty, auto-detect numeric FPs.")
    ap.add_argument("--morgan_prefix", default="Avalon_FP_", help="Prefix for Tanimoto (binary) bits; used only for similarity-to-positives.")
    ap.add_argument("--neg_pos_ratio", type=float, default=6.0, help="Max selected negatives per positive overall.")
    ap.add_argument("--tani_min", type=float, default=0.25, help="Keep negatives with max_tanimoto_to_pos >= tani_min")
    ap.add_argument("--tani_max", type=float, default=0.90, help="and <= tani_max (avoid near-duplicates).")
    ap.add_argument("--domain_conf_max", type=float, default=0.80, help="Keep negatives with max domain prob <= this.")
    ap.add_argument("--n_clusters", type=int, default=16, help="KMeans cluster count for outlier/group filtering.")
    ap.add_argument("--cluster_entropy_min", type=float, default=0.8, help="Min source entropy per cluster (drop dominated clusters).")
    ap.add_argument("--cluster_tani_mean_min", type=float, default=0.3, help="Min mean max_tanimoto_to_pos for cluster to keep.")
    args = ap.parse_args()

    decoy_names = tuple(parse_list(args.decoy_names))
    df = pd.read_csv(args.data_csv)
    if "label" not in df.columns or "source_dataset" not in df.columns:
        raise ValueError("CSV must contain 'label' and 'source_dataset'.")

    # Split
    pos_df = df[df["source_dataset"] == args.pos_name].copy()
    neg_df = df[df["source_dataset"].isin(decoy_names)].copy()
    if pos_df.empty or neg_df.empty:
        raise ValueError("Positive or negative subset is empty. Check pos_name/decoy_names.")

    # Features for model & clustering
    feat_cols = detect_feature_columns(df, parse_list(args.feature_list))
    X_neg = neg_df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    X_pos = pos_df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values

    # Binary bits for Tanimoto (Morgan prefix)
    morgan_cols = get_prefixed_cols(df, args.morgan_prefix)
    P_bits = to_uint8_matrix(pos_df, morgan_cols)
    U_bits = to_uint8_matrix(neg_df, morgan_cols)

    # --- (1) Hardness: max Tanimoto to positives ---
    max_tani = tanimoto_max_to_pos(U_bits, P_bits)  # size: len(neg_df)

    # --- (2) Domain classifier on negatives (predict source) ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),  # for sparse-like numeric
        ("clf", ExtraTreesClassifier(
            n_estimators=600, random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])
    # OOF predicted proba
    y_src = neg_df["source_dataset"].astype(str).values
    classes = np.unique(y_src)
    oof_proba = np.zeros((len(neg_df), len(classes)), dtype=float)
    for tr, va in skf.split(X_neg, y_src):
        pipe.fit(X_neg[tr], y_src[tr])
        oof_proba[va, :] = pipe.predict_proba(X_neg[va])
    max_domain_conf = oof_proba.max(axis=1)  # smaller is better (harder to tell source)

    # --- (3) Cluster negatives for outlier/group filters ---
    # PCA to 50 dims (robust, speed), then KMeans
    n_comp = min(50, X_neg.shape[1])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    Z = pca.fit_transform(X_neg)
    km = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=RANDOM_STATE)
    cl = km.fit_predict(Z)
    neg_df["_cluster"] = cl

    # Per-cluster diagnostics
    keep_clusters = set()
    for k in range(args.n_clusters):
        idx = np.where(cl == k)[0]
        if idx.size == 0:
            continue
        # entropy over sources (higher is more mixed → good)
        counts = np.array([(y_src[idx] == s).sum() for s in classes], dtype=float)
        ent = entropy_of_counts(counts)
        # mean hardness
        mean_tani = float(max_tani[idx].mean())
        if ent >= args.cluster_entropy_min and mean_tani >= args.cluster_tani_mean_min:
            keep_clusters.add(k)

    # --- (4) Filter by criteria ---
    keep_mask = (
        neg_df["_cluster"].isin(keep_clusters).values &
        (max_tani >= args.tani_min) &
        (max_tani <= args.tani_max) &
        (max_domain_conf <= args.domain_conf_max)
    )
    neg_df["_max_tani_pos"] = max_tani
    neg_df["_domain_conf"] = max_domain_conf

    sel = neg_df.loc[keep_mask].copy()
    if sel.empty:
        raise RuntimeError("No negatives passed the filters. Relax thresholds or check inputs.")

    # --- (5) Balance by source and cap total negatives ---
    n_pos = len(pos_df)
    max_negs = int(math.ceil(args.neg_pos_ratio * n_pos))
    # equal share per source
    per_src_cap = max(1, max_negs // max(1, len(decoy_names)))
    chunks = []
    for src in decoy_names:
        sub = sel[sel["source_dataset"] == src].copy()
        if len(sub) > per_src_cap:
            # Take the best (lowest domain_conf, highest max_tani) via score
            score = (1.0 - sub["_max_tani_pos"]) + sub["_domain_conf"]  # smaller better
            sub = sub.assign(_score=score).sort_values("_score").head(per_src_cap)
        chunks.append(sub)
    neg_final = pd.concat(chunks, axis=0).sample(frac=1.0, random_state=RANDOM_STATE)

    out = pd.concat([pos_df, neg_final], axis=0).sample(frac=1.0, random_state=RANDOM_STATE)
    out.drop(columns=[c for c in out.columns if c.startswith("_")], inplace=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    # Quick report
    print(f"[OK] Wrote: {args.out_csv}")
    print(f"[INFO] Positives: {len(pos_df)}  | Selected negatives: {len(neg_final)}  (ratio={len(neg_final)/len(pos_df):.2f})")
    print("[INFO] Selected negatives by source:")
    print(neg_final["source_dataset"].value_counts())

if __name__ == "__main__":
    main()
