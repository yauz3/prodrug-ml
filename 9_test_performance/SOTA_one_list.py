#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SOTA (Multiview) — Stratified K-Fold CV (Murcko scaffold YOK) + OOF/TEST %95 CI
Fingerprint tarafında multiview (aynı model, farklı view'ler; lineer ağırlıklı ensemble),
SMILES tarafında değişiklik yok (ChemBERTa, GIN).

Eklenenler:
  - Early enrichment: EF@1%, EF@5%, BEDROC(λ=20/50/80)
  - Ensemble: view ağırlıkları (eşit veya kullanıcı tanımlı), per-view + ensemble metrikleri
  - Çıktılar: orijinal SOTA dosya adlarıyla aynı

Çıktılar:
  - sota_cv_test_metrics.csv   (fold×model metrikleri; feature_regime = view:*, ensemble[*] veya 'smiles')
  - sota_cv_test_summary.csv   (mean±std)
  - sota_ci_summary.csv        (OOF ve TEST_full_train için AUC/AP + %95 CI; feature_regime dahil)
  - sota_early_enrichment.csv  (EF@1/5% + BEDROC λ=20/50/80; CV holdout, TEST, OOF, TEST_full_train)
"""

import argparse
import numpy as np
import pandas as pd
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ----------------- Opsiyonel bağımlılık bayrakları -----------------
HAS_XGB = False
HAS_LGBM = False
HAS_CATB = False
HAS_HF   = False
HAS_TORCH = False
HAS_PYG   = False
HAS_RDKIT = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    pass
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    pass
try:
    from catboost import CatBoostClassifier
    HAS_CATB = True
except Exception:
    pass
try:
    import torch
    from torch import nn
    HAS_TORCH = True
except Exception:
    pass
if HAS_TORCH:
    try:
        from transformers import AutoTokenizer, AutoModel
        HAS_HF = True
    except Exception:
        pass
    try:
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GINConv, global_add_pool
        HAS_PYG = True
    except Exception:
        pass
try:
    from rdkit import Chem
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False

# ----------------- sklearn -----------------
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    balanced_accuracy_score, brier_score_loss, matthews_corrcoef
)

print("HAS_XGB", HAS_XGB)
print("HAS_LGBM", HAS_LGBM)
print("HAS_CATB", HAS_CATB)
print("HAS_HF", HAS_HF)
print("HAS_TORCH", HAS_TORCH)
print("HAS_PYG", HAS_PYG)
print("HAS_RDKIT", HAS_RDKIT)

RANDOM_STATE = 42
random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
N_SPLITS = 5

# ===================== VIEW TANIMLARI (Fingerprint) =====================
# list_1 sizin verdiğiniz sütun listesiyle dolu; lütfen list_2..list_4'ü doldurun.
FEATURE_LISTS: dict[str, list[str]] = {
    "list_1": ['Avalon_FP_1309', 'Avalon_FP_93', 'Avalon_FP_599', 'Avalon_FP_1436', 'Avalon_FP_1250', 'Avalon_FP_1969', 'Avalon_FP_487', 'Avalon_FP_1119', 'Avalon_FP_492', 'Avalon_FP_2013', 'Avalon_FP_12', 'Avalon_FP_932', 'Avalon_FP_231', 'Avalon_FP_2000', 'Avalon_FP_451', 'Avalon_FP_1122', 'Avalon_FP_422', 'Avalon_FP_1581', 'Avalon_FP_1548', 'Avalon_FP_1499', 'Avalon_FP_1236', 'Avalon_FP_1237', 'Avalon_FP_1498', 'Avalon_FP_2024', 'Avalon_FP_1157', 'Avalon_FP_297', 'Avalon_FP_443', 'Avalon_FP_399', 'Avalon_FP_1794', 'Avalon_FP_712', 'Avalon_FP_1414', 'Avalon_FP_339', 'Avalon_FP_1292', 'Avalon_FP_1570', 'Avalon_FP_1996', 'Avalon_FP_762', 'Avalon_FP_101', 'Avalon_FP_255', 'Avalon_FP_178', 'Avalon_FP_1040'],
    "list_2": ['Avalon_FP_1309', 'Avalon_FP_93', 'Avalon_FP_599', 'Avalon_FP_1436', 'Avalon_FP_1250', 'Avalon_FP_1969', 'Avalon_FP_487', 'Avalon_FP_1119', 'Avalon_FP_492', 'Avalon_FP_2013', 'Avalon_FP_12', 'Avalon_FP_932', 'Avalon_FP_231', 'Avalon_FP_2000', 'Avalon_FP_451', 'Avalon_FP_1122', 'Avalon_FP_422', 'Avalon_FP_1581', 'Avalon_FP_1548', 'Avalon_FP_1499', 'Avalon_FP_1236', 'Avalon_FP_1237', 'Avalon_FP_1498', 'Avalon_FP_2024', 'Avalon_FP_1157', 'Avalon_FP_297', 'Avalon_FP_443', 'Avalon_FP_399', 'Avalon_FP_1794', 'Avalon_FP_712', 'Avalon_FP_1414', 'Avalon_FP_339', 'Avalon_FP_1292', 'Avalon_FP_1570', 'Avalon_FP_1996', 'Avalon_FP_762', 'Avalon_FP_101', 'Avalon_FP_255', 'Avalon_FP_178', 'Avalon_FP_1040'],
    "list_3": ['Avalon_FP_1309', 'Avalon_FP_93', 'Avalon_FP_599', 'Avalon_FP_1436', 'Avalon_FP_1250', 'Avalon_FP_1969', 'Avalon_FP_487', 'Avalon_FP_1119', 'Avalon_FP_492', 'Avalon_FP_2013', 'Avalon_FP_12', 'Avalon_FP_932', 'Avalon_FP_231', 'Avalon_FP_2000', 'Avalon_FP_451', 'Avalon_FP_1122', 'Avalon_FP_422', 'Avalon_FP_1581', 'Avalon_FP_1548', 'Avalon_FP_1499', 'Avalon_FP_1236', 'Avalon_FP_1237', 'Avalon_FP_1498', 'Avalon_FP_2024', 'Avalon_FP_1157', 'Avalon_FP_297', 'Avalon_FP_443', 'Avalon_FP_399', 'Avalon_FP_1794', 'Avalon_FP_712', 'Avalon_FP_1414', 'Avalon_FP_339', 'Avalon_FP_1292', 'Avalon_FP_1570', 'Avalon_FP_1996', 'Avalon_FP_762', 'Avalon_FP_101', 'Avalon_FP_255', 'Avalon_FP_178', 'Avalon_FP_1040'],
    "list_4": ['Avalon_FP_1309', 'Avalon_FP_93', 'Avalon_FP_599', 'Avalon_FP_1436', 'Avalon_FP_1250', 'Avalon_FP_1969', 'Avalon_FP_487', 'Avalon_FP_1119', 'Avalon_FP_492', 'Avalon_FP_2013', 'Avalon_FP_12', 'Avalon_FP_932', 'Avalon_FP_231', 'Avalon_FP_2000', 'Avalon_FP_451', 'Avalon_FP_1122', 'Avalon_FP_422', 'Avalon_FP_1581', 'Avalon_FP_1548', 'Avalon_FP_1499', 'Avalon_FP_1236', 'Avalon_FP_1237', 'Avalon_FP_1498', 'Avalon_FP_2024', 'Avalon_FP_1157', 'Avalon_FP_297', 'Avalon_FP_443', 'Avalon_FP_399', 'Avalon_FP_1794', 'Avalon_FP_712', 'Avalon_FP_1414', 'Avalon_FP_339', 'Avalon_FP_1292', 'Avalon_FP_1570', 'Avalon_FP_1996', 'Avalon_FP_762', 'Avalon_FP_101', 'Avalon_FP_255', 'Avalon_FP_178', 'Avalon_FP_1040'],
    "list_5" : ['Avalon_FP_1309', 'Avalon_FP_93', 'Avalon_FP_599', 'Avalon_FP_1436', 'Avalon_FP_1250', 'Avalon_FP_1969', 'Avalon_FP_487', 'Avalon_FP_1119', 'Avalon_FP_492', 'Avalon_FP_2013', 'Avalon_FP_12', 'Avalon_FP_932', 'Avalon_FP_231', 'Avalon_FP_2000', 'Avalon_FP_451', 'Avalon_FP_1122', 'Avalon_FP_422', 'Avalon_FP_1581', 'Avalon_FP_1548', 'Avalon_FP_1499', 'Avalon_FP_1236', 'Avalon_FP_1237', 'Avalon_FP_1498', 'Avalon_FP_2024', 'Avalon_FP_1157', 'Avalon_FP_297', 'Avalon_FP_443', 'Avalon_FP_399', 'Avalon_FP_1794', 'Avalon_FP_712', 'Avalon_FP_1414', 'Avalon_FP_339', 'Avalon_FP_1292', 'Avalon_FP_1570', 'Avalon_FP_1996', 'Avalon_FP_762', 'Avalon_FP_101', 'Avalon_FP_255', 'Avalon_FP_178', 'Avalon_FP_1040'],
}
# -------------------- Yardımcılar --------------------
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

def find_smiles_col(df):
    for c in ["SMILES","smiles","Canonical_SMILES","canonical_smiles"]:
        if c in df.columns:
            return c
    return None

def load_data(train_csv, test_csv, label_col="label"):
    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)
    y_train = train_df[label_col].astype(int)
    y_test  = test_df[label_col].astype(int)
    smiles_col = find_smiles_col(train_df)
    if smiles_col is None:
        print("[WARN] SMILES kolonu bulunamadı (SMILES/smiles/Canonical_SMILES). SMILES tabanlı modeller atlanabilir.")
    return train_df, test_df, y_train, y_test, smiles_col

def coerce_numeric(df, cols):
    return df.loc[:, cols].apply(pd.to_numeric, errors="coerce")

def validate_views(train_df, test_df, view_names):
    for vn in view_names:
        if vn not in FEATURE_LISTS:
            raise ValueError(f"Bilinmeyen view '{vn}'. Mevcut: {list(FEATURE_LISTS.keys())}")
        cols = FEATURE_LISTS[vn]
        if not cols:
            raise ValueError(f"FEATURE_LISTS['{vn}'] boş. Lütfen sütun adlarını doldurun.")
        miss_tr = [c for c in cols if c not in train_df.columns]
        miss_te = [c for c in cols if c not in test_df.columns]
        if miss_tr:
            raise ValueError(f"[{vn}] Train eksik kolonlar: {miss_tr[:10]} ...")
        if miss_te:
            raise ValueError(f"[{vn}] Test eksik kolonlar: {miss_te[:10]} ...")

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
            raise ValueError(f"--view_weights uzunluğu {len(w)} ama view sayısı {len(view_names)}.")
        w = np.maximum(w, 0.0)
        if w.sum() == 0:
            w = np.ones_like(w)
        w = w / w.sum()
    else:
        w = np.ones(len(view_names), dtype=float) / len(view_names)
    return view_names, w

def proba_or_decision(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)

def compute_metrics(y_true, scores, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    if np.isfinite(scores).all() and scores.min() >= 0.0 and scores.max() <= 1.0:
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
    """Enrichment Factor @frac (örn. 0.01=1%)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    N = y.size
    n_pos = int(y.sum())
    if N == 0 or n_pos == 0:
        return np.nan
    k = max(1, int(np.ceil(frac * N)))
    order = np.argsort(-s, kind="mergesort")  # stabil sıralama
    hits_topk = int(y[order][:k].sum())
    return float((hits_topk / k) / (n_pos / N))

def bedroc(y_true, scores, alpha=20.0):
    """
    Truchon & Bayly (2007) normalizasyonu:
    RIE = (alpha/(1-exp(-alpha))) * (1/n) * sum(exp(-alpha * r_i / N))
    BEDROC = (RIE - RIE_min) / (RIE_max - RIE_min)
    """
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

# ===================== SOTA MODELLER =====================

def build_models_fingerprint():
    """Aynı fingerprint setlerinde adil kıyas: XGB, LGBM, CatBoost (+ L2 LR referansı)."""
    models = {}

    # Referans: L2 Logistic
    models["LogReg_L2"] = LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=2000,
        class_weight="balanced", random_state=RANDOM_STATE
    )

    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=800, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1
        )
        models["XGBoost"] = xgb
    else:
        print("[WARN] xgboost bulunamadı; XGBoost atlanıyor.")

    if HAS_LGBM:
        lgbm = LGBMClassifier(
            n_estimators=1200, num_leaves=64, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1
        )
        models["LightGBM"] = lgbm
    else:
        print("[WARN] lightgbm bulunamadı; LightGBM atlanıyor.")

    if HAS_CATB:
        catb = CatBoostClassifier(
            iterations=1500, depth=6, learning_rate=0.03,
            loss_function="Logloss", random_seed=RANDOM_STATE, verbose=False
        )
        models["CatBoost"] = catb
    else:
        print("[WARN] catboost bulunamadı; CatBoost atlanıyor.")

    return models

# ------- ChemBERTa (SMILES embedding + LR head) -------
class ChemBERTaClassifier:
    requires_smiles = True
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1", device=None):
        if not (HAS_TORCH and HAS_HF):
            raise ImportError("transformers/torch eksik")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device).eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.head = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, class_weight="balanced")

    @torch.no_grad()
    def _embed_batch(self, smiles_list, batch_size=64):
        embs = []
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            toks = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            out = self.encoder(**toks).last_hidden_state
            cls = out[:, 0, :]
            embs.append(cls.cpu().numpy())
        return np.vstack(embs)

    def fit_from_smiles(self, smiles_series: pd.Series, y: pd.Series, idx: np.ndarray):
        smi = smiles_series.iloc[idx].astype(str).tolist()
        Xemb = self._embed_batch(smi)
        self.head.fit(Xemb, y.iloc[idx].values)
        return self

    def predict_proba_by_index(self, smiles_series: pd.Series, idx: np.ndarray):
        smi = smiles_series.iloc[idx].astype(str).tolist()
        Xemb = self._embed_batch(smi)
        return self.head.predict_proba(Xemb)[:, 1]

# ------- GIN (GNN) -------
def mol_to_pyg(smiles: str, maxZ=100):
    if not HAS_RDKIT:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    N = m.GetNumAtoms()
    if N == 0:
        return None
    x = np.zeros((N, maxZ), dtype=np.float32)
    for i, atom in enumerate(m.GetAtoms()):
        Z = atom.GetAtomicNum()
        if 0 <= Z < maxZ:
            x[i, Z] = 1.0
    edges = []
    for b in m.GetBonds():
        i = b.GetBeginAtomIdx(); j = b.GetEndAtomIdx()
        edges.append([i, j]); edges.append([j, i])
    if len(edges) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        edge_index = np.array(edges, dtype=np.int64).T
    return Data(x=torch.tensor(x), edge_index=torch.tensor(edge_index))

class GINNet(nn.Module):
    def __init__(self, in_dim=100, hidden=128, layers=3, dropout=0.1):
        super().__init__()
        def block(in_ch, out_ch):
            mlp = nn.Sequential(nn.Linear(in_ch, out_ch), nn.ReLU(), nn.Linear(out_ch, out_ch))
            return GINConv(mlp)
        self.convs = torch.nn.ModuleList()
        dims = [in_dim] + [hidden]*layers
        for i in range(layers):
            self.convs.append(block(dims[i], dims[i+1]))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden, 1)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        hg = global_add_pool(x, batch)
        logit = self.lin(hg).view(-1)
        return logit

class GINClassifier:
    requires_smiles = True
    def __init__(self, in_dim=100, hidden=128, layers=3, epochs=30, lr=1e-3, batch_size=128, device=None):
        if not (HAS_TORCH and HAS_PYG and HAS_RDKIT):
            raise ImportError("torch/torch_geometric/rdkit eksik")
        self.in_dim=in_dim; self.hidden=hidden; self.layers=layers
        self.epochs=epochs; self.lr=lr; self.batch_size=batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GINNet(in_dim=in_dim, hidden=hidden, layers=layers).to(self.device)

    def _graphs_from_indices(self, smiles_series, idx):
        ds = []
        for i in idx:
            g = mol_to_pyg(str(smiles_series.iloc[i]), maxZ=self.in_dim)
            if g is None:
                g = Data(x=torch.zeros((1, self.in_dim), dtype=torch.float32),
                         edge_index=torch.zeros((2,0), dtype=torch.long))
            g.y = None
            ds.append(g)
        return ds

    def fit_from_smiles(self, smiles_series: pd.Series, y: pd.Series, idx: np.ndarray):
        yv = y.iloc[idx].values.astype(np.float32)
        graphs = self._graphs_from_indices(smiles_series, idx)
        for k, g in enumerate(graphs):
            g.y = torch.tensor([yv[k]], dtype=torch.float32)
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=True)
        self.model = GINNet(in_dim=self.in_dim, hidden=self.hidden, layers=self.layers).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        pos = float((yv==1).sum()); neg = float((yv==0).sum())
        pos_weight = torch.tensor([neg/pos], device=self.device) if pos>0 else torch.tensor([1.0], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.model.train()
        torch.manual_seed(RANDOM_STATE)
        for _ in range(self.epochs):
            for batch in loader:
                batch = batch.to(self.device)
                opt.zero_grad()
                logits = self.model(batch)
                loss = loss_fn(logits, batch.y.view(-1).to(self.device))
                loss.backward()
                opt.step()
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_proba_by_index(self, smiles_series: pd.Series, idx: np.ndarray):
        graphs = self._graphs_from_indices(smiles_series, idx)
        for g in graphs:
            g.y = torch.tensor([0.0])
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        probs = []
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(p)
        return np.concatenate(probs, axis=0)

# ===================== EĞİTİM DÖNGÜSÜ (Multiview + SOTA) =====================

def fit_kfold_cv_multiview_sota(fp_models: dict,
                                smiles_models: dict,
                                train_df: pd.DataFrame, test_df: pd.DataFrame,
                                y_train: pd.Series, y_test: pd.Series,
                                smiles_col: str | None,
                                view_names: list[str], view_weights: np.ndarray,
                                n_splits=N_SPLITS,
                                n_boot=2000,
                                alpha=0.95):
    rows = []
    ci_rows = []
    ee_rows = []

    # View bazlı X'ler
    Xtr_views = {vn: coerce_numeric(train_df, FEATURE_LISTS[vn]) for vn in view_names}
    Xte_views = {vn: coerce_numeric(test_df,  FEATURE_LISTS[vn]) for vn in view_names}

    view_tag = "+".join(view_names)
    ens_regime = f"ensemble[{view_tag}]"

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # ---------------- Fingerprint modelleri (multiview) ----------------
    for model_name, estimator in fp_models.items():

        # OOF toplayıcıları (her view ve ensemble için)
        oof_store = {f"view:{vn}": {"idx": [], "scores": []} for vn in view_names}
        oof_store[ens_regime] = {"idx": [], "scores": []}

        for fold_id, (tr_idx, val_idx) in enumerate(skf.split(Xtr_views[view_names[0]], y_train)):
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            # Sınıf dengesizliği (örnek ağırlığı; bazı modellerde yok sayılabilir)
            n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
            pos_w = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
            sample_w = np.where(y_tr.values==1, pos_w, 1.0).astype(float)

            fold_pipes = {}
            val_pred_views = []
            test_pred_views = []

            # Her view için AYNI modelin kopyası
            for vn in view_names:
                X_tr = Xtr_views[vn].iloc[tr_idx]
                X_val = Xtr_views[vn].iloc[val_idx]

                pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", MinMaxScaler()),
                    ("model", clone(estimator))
                ])
                fit_params = {"model__sample_weight": sample_w}
                try:
                    pipe.fit(X_tr, y_tr, **fit_params)
                except TypeError:
                    pipe.fit(X_tr, y_tr)

                fold_pipes[vn] = pipe

                s_val = proba_or_decision(pipe, X_val)
                s_test = proba_or_decision(pipe, Xte_views[vn])

                val_pred_views.append(s_val)
                test_pred_views.append(s_test)

                # Per-view metrikler
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

                # OOF per-view
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

            # Ensemble (view-weighted average)
            val_pred_views = np.vstack(val_pred_views)
            test_pred_views = np.vstack(test_pred_views)
            s_val_ens = np.average(val_pred_views, axis=0, weights=view_weights)
            s_test_ens = np.average(test_pred_views, axis=0, weights=view_weights)

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

            # OOF ensemble
            oof_store[ens_regime]["idx"].append(val_idx)
            oof_store[ens_regime]["scores"].append(s_val_ens)

            # Test ensemble
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

        # ---- OOF CI + Early Enrichment (per-view ve ensemble) ----
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

        # ---- Full-train → TEST CI + Early Enrichment (per-view ve ensemble) ----
        s_test_views = []
        for vn in view_names:
            Xtr = Xtr_views[vn]; Xte = Xte_views[vn]
            # full-train ağırlık
            n_pos = int(y_train.sum()); n_neg = len(y_train) - n_pos
            pos_w = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
            sample_w_full = np.where(y_train.values==1, pos_w, 1.0).astype(float)

            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler()),
                ("model", clone(estimator))
            ])
            try:
                pipe.fit(Xtr, y_train, model__sample_weight=sample_w_full)
            except TypeError:
                pipe.fit(Xtr, y_train)

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

        # Ensemble full-train → test
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

    # ---------------- SMILES tabanlı modeller (tek görünüm) ----------------
    if smiles_col is not None:
        smiles_train = train_df[smiles_col]
        smiles_test  = test_df[smiles_col]
        for name, est in smiles_models.items():
            feat_regime = "smiles"
            oof_idx, oof_scores = [], []
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            for fold_id, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_train)), y_train)):
                est.fit_from_smiles(smiles_train, y_train, tr_idx)

                # hold-out
                val_scores = est.predict_proba_by_index(smiles_train, val_idx)
                val_metrics = compute_metrics(y_train.iloc[val_idx], val_scores)
                rows.append({"model": name, "feature_regime": feat_regime,
                             "split": fold_id, "dataset": "cv_holdout", **val_metrics})
                ee_rows.append({
                    "model": name, "feature_regime": feat_regime,
                    "split": fold_id, "dataset": "cv_holdout",
                    "ef_1pct": ef_at(y_train.iloc[val_idx], val_scores, 0.01),
                    "ef_5pct": ef_at(y_train.iloc[val_idx], val_scores, 0.05),
                    "bedroc_20": bedroc(y_train.iloc[val_idx], val_scores, 20.0),
                    "bedroc_50": bedroc(y_train.iloc[val_idx], val_scores, 50.0),
                    "bedroc_80": bedroc(y_train.iloc[val_idx], val_scores, 80.0),
                })

                oof_idx.append(val_idx); oof_scores.append(val_scores)

                # test (fold modeliyle)
                test_scores = est.predict_proba_by_index(smiles_test, np.arange(len(y_test)))
                test_metrics = compute_metrics(y_test, test_scores)
                rows.append({"model": name, "feature_regime": feat_regime,
                             "split": fold_id, "dataset": "test", **test_metrics})
                ee_rows.append({
                    "model": name, "feature_regime": feat_regime,
                    "split": fold_id, "dataset": "test",
                    "ef_1pct": ef_at(y_test, test_scores, 0.01),
                    "ef_5pct": ef_at(y_test, test_scores, 0.05),
                    "bedroc_20": bedroc(y_test, test_scores, 20.0),
                    "bedroc_50": bedroc(y_test, test_scores, 50.0),
                    "bedroc_80": bedroc(y_test, test_scores, 80.0),
                })

            # OOF CI
            oidx = np.concatenate(oof_idx)
            osco = np.concatenate(oof_scores)
            order = np.argsort(oidx)
            y_oof = y_train.iloc[oidx[order]].values
            s_oof = osco[order]

            oof_auc = _safe(lambda: roc_auc_score(y_oof, s_oof))
            _, oof_auc_lo, oof_auc_hi = stratified_bootstrap_ci(y_oof, s_oof, roc_auc_score, n_boot=n_boot, alpha=alpha, seed=RANDOM_STATE)
            oof_ap  = _safe(lambda: average_precision_score(y_oof, s_oof))
            _, oof_ap_lo,  oof_ap_hi  = stratified_bootstrap_ci(y_oof, s_oof, average_precision_score, n_boot=n_boot, alpha=alpha, seed=RANDOM_STATE)

            ci_rows.append({
                "model": name, "feature_regime": feat_regime, "dataset": "OOF",
                "roc_auc": oof_auc, "roc_auc_lo": oof_auc_lo, "roc_auc_hi": oof_auc_hi,
                "ap": oof_ap, "ap_lo": oof_ap_lo, "ap_hi": oof_ap_hi
            })
            ee_rows.append({
                "model": name, "feature_regime": feat_regime,
                "split": -1, "dataset": "OOF",
                "ef_1pct": ef_at(y_oof, s_oof, 0.01),
                "ef_5pct": ef_at(y_oof, s_oof, 0.05),
                "bedroc_20": bedroc(y_oof, s_oof, 20.0),
                "bedroc_50": bedroc(y_oof, s_oof, 50.0),
                "bedroc_80": bedroc(y_oof, s_oof, 80.0),
            })

            # Full-train → TEST CI
            est.fit_from_smiles(smiles_train, y_train, np.arange(len(y_train)))
            s_test = est.predict_proba_by_index(smiles_test, np.arange(len(y_test)))

            test_auc = _safe(lambda: roc_auc_score(y_test, s_test))
            _, test_auc_lo, test_auc_hi = stratified_bootstrap_ci(y_test.values, s_test, roc_auc_score, n_boot=n_boot, alpha=alpha, seed=RANDOM_STATE)
            test_ap  = _safe(lambda: average_precision_score(y_test, s_test))
            _, test_ap_lo,  test_ap_hi  = stratified_bootstrap_ci(y_test.values, s_test, average_precision_score, n_boot=n_boot, alpha=alpha, seed=RANDOM_STATE)

            ci_rows.append({
                "model": name, "feature_regime": feat_regime, "dataset": "TEST_full_train",
                "roc_auc": test_auc, "roc_auc_lo": test_auc_lo, "roc_auc_hi": test_auc_hi,
                "ap": test_ap, "ap_lo": test_ap_lo, "ap_hi": test_ap_hi
            })
            ee_rows.append({
                "model": name, "feature_regime": feat_regime,
                "split": -1, "dataset": "TEST_full_train",
                "ef_1pct": ef_at(y_test, s_test, 0.01),
                "ef_5pct": ef_at(y_test, s_test, 0.05),
                "bedroc_20": bedroc(y_test, s_test, 20.0),
                "bedroc_50": bedroc(y_test, s_test, 50.0),
                "bedroc_80": bedroc(y_test, s_test, 80.0),
            })

    return rows, ci_rows, ee_rows

def summarize(df):
    agg = df.groupby(["model", "feature_regime", "dataset"]).agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for (m, s) in agg.columns]
    agg = agg.reset_index()
    return agg

def main():
    ap = argparse.ArgumentParser(description="SOTA (multiview FP + SMILES) — Stratified K-Fold CV + OOF/TEST bootstrap CI")
    ap.add_argument("--train_csv", default="filtered_all_data_training_1_6.csv")
    ap.add_argument("--test_csv",  default="test_ready.csv")
    ap.add_argument("--out_splits", default="sota_cv_test_metrics_one.csv")
    ap.add_argument("--out_summary", default="sota_cv_test_summary_one.csv")
    ap.add_argument("--out_ci", default="sota_ci_summary_one.csv")
    ap.add_argument("--out_ee", default="sota_early_enrichment_one.csv")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.95)
    # multiview kontrolleri
    ap.add_argument("--list_number", type=int, default=1,
                    help="list_1..list_4 arasından ilk N view'ü kullan (eğer --views verilmediyse)")
    ap.add_argument("--views", type=str, default=None,
                    help="Virgülle view adları: 'list_1,list_3' (verilirse --list_number yok sayılır)")
    ap.add_argument("--view_weights", type=str, default=None,
                    help="Virgülle ağırlıklar (view sayısıyla aynı uzunlukta); otomatik normalize edilir")
    args = ap.parse_args()

    # Veri
    train_df, test_df, y_train, y_test, smiles_col = load_data(args.train_csv, args.test_csv)

    # View & ağırlıklar
    view_names, view_weights = parse_views_and_weights(args.list_number, args.views, args.view_weights)
    validate_views(train_df, test_df, view_names)
    print(f"[INFO] Views: {view_names} | Weights (normalized): {view_weights.round(4).tolist()}")

    # Modeller
    fp_models = build_models_fingerprint()

    smiles_models = {}
    if smiles_col is not None and HAS_TORCH and HAS_HF:
        try:
            smiles_models["ChemBERTa+LR"] = ChemBERTaClassifier()
        except Exception as e:
            print(f"[WARN] ChemBERTa init başarısız: {e}")
    else:
        print("[WARN] ChemBERTa için transformers/torch veya SMILES kolonu yok; atlanıyor.")

    if smiles_col is not None and HAS_TORCH and HAS_PYG and HAS_RDKIT:
        try:
            smiles_models["GIN"] = GINClassifier(epochs=30, hidden=128, layers=3)
        except Exception as e:
            print(f"[WARN] GIN init başarısız: {e}")
    else:
        print("[WARN] GIN için torch_geometric/rdkit veya SMILES kolonu yok; atlanıyor.")

    # Çalıştır
    print("[*] Running Multiview SOTA Stratified K-Fold CV ...")
    all_rows, ci_rows, ee_rows = fit_kfold_cv_multiview_sota(
        fp_models, smiles_models,
        train_df, test_df, y_train, y_test, smiles_col,
        view_names=view_names, view_weights=view_weights,
        n_splits=N_SPLITS, n_boot=args.n_boot, alpha=args.alpha
    )

    # Kayıtlar
    results = pd.DataFrame(all_rows)
    write_csv(results, args.out_splits)

    summary = summarize(results)
    write_csv(summary, args.out_summary)

    ci_df = pd.DataFrame(ci_rows)
    write_csv(ci_df, args.out_ci)

    ee_df = pd.DataFrame(ee_rows)
    write_csv(ee_df, args.out_ee)

    # Konsol özeti
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
