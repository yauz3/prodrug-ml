#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import numpy as np
import pandas as pd


def choose_domain_col(df: pd.DataFrame, user_col: Optional[str]) -> str:
    if user_col:
        if user_col not in df.columns:
            raise ValueError(f"domain_col='{user_col}' bulunamadı. Mevcut sütunlar: {list(df.columns)}")
        return user_col
    # Auto-detect common names
    candidates = ["decoy_name", "decoy_domain", "domain", "source"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Negatif domain sütunu bulunamadı. Lütfen '--domain_col' verin "
        "veya veri setinizde 'decoy_name/decoy_domain/domain/source' gibi bir sütun bulunsun."
    )


def even_allocation(
    required_total: int,
    avail: Dict[str, int],
    per_domain_cap: Optional[int] = None,
) -> Dict[str, int]:
    """
    Allocate required_total across domains as evenly as possible.
    Respect availability and an optional per-domain cap.
    If total availability < required_total, return 'all you can' (best-effort).
    """
    names: List[str] = list(avail.keys())
    k = len(names)
    # Initial even split
    base, rem = divmod(required_total, k)
    alloc = {n: base + (i < rem) for i, n in enumerate(names)}

    # Apply per-domain cap (if any) and availability
    for n in names:
        if per_domain_cap is not None:
            alloc[n] = min(alloc[n], per_domain_cap)
        alloc[n] = min(alloc[n], avail[n])

    # Compute shortfall and redistribute greedily
    total = sum(alloc.values())
    need = required_total - total
    if need <= 0:
        # If we exceeded (can happen due to integer division + caps), trim greedily
        excess = -need
        if excess > 0:
            # Remove 1-by-1 from domains with largest alloc
            order = sorted(names, key=lambda x: alloc[x], reverse=True)
            i = 0
            while excess > 0 and sum(alloc.values()) > 0:
                n = order[i % k]
                if alloc[n] > 0:
                    alloc[n] -= 1
                    excess -= 1
                i += 1
        return alloc

    # need > 0: try to top up from domains with spare capacity
    # Spare = min(cap, avail) - current alloc
    def cap_of(n: str) -> int:
        return min(per_domain_cap, avail[n]) if per_domain_cap is not None else avail[n]

    rounds = 0
    while need > 0:
        progressed = False
        for n in names:
            spare = cap_of(n) - alloc[n]
            if spare > 0 and need > 0:
                take = min(spare, need)
                alloc[n] += take
                need -= take
                progressed = True
                if need == 0:
                    break
        rounds += 1
        if not progressed or rounds > 3 * k:
            # Could not fill completely; return best-effort
            break

    return alloc


def build_test_split(
    df: pd.DataFrame,
    label_col: str,
    domain_col: str,
    decoy_names: Tuple[str, ...],
    pos_frac: float = 0.20,
    ratio_neg_per_pos: int = 6,
    seed: int = 42,
    neg_per_domain: Optional[int] = None,
):
    rng = np.random.RandomState(seed)

    # Split by label
    df_pos = df[df[label_col] == 1].copy()
    df_neg = df[df[label_col] == 0].copy()

    # === 1) Select test positives: 20% of positives (at least 1 if possible) ===
    n_pos = len(df_pos)
    if n_pos == 0:
        raise ValueError("Pozitif (label=1) örnek bulunamadı; test seti oluşturulamıyor.")
    n_test_pos = max(1, min(n_pos - 1, int(round(n_pos * pos_frac)))) if n_pos > 1 else 1
    test_pos_idx = rng.choice(df_pos.index.to_numpy(), size=n_test_pos, replace=False)
    test_pos = df_pos.loc[test_pos_idx]

    # === 2) Select test negatives to achieve 1:6 ratio ===
    target_neg = ratio_neg_per_pos * n_test_pos

    # Availability per domain (restrict to provided decoy_names)
    avail_by_domain = {}
    for name in decoy_names:
        avail_by_domain[name] = int((df_neg[domain_col] == name).sum())

    total_avail = sum(avail_by_domain.values())
    if total_avail == 0:
        raise ValueError("Negatif örnek bulunamadı (label=0).")

    # Preferred per-domain cap (e.g., 200) if provided
    per_domain_cap = neg_per_domain if neg_per_domain and neg_per_domain > 0 else None

    # If the total availability (or total cap) is insufficient, we will best-effort
    if per_domain_cap is not None:
        total_cap = sum(min(avail_by_domain[n], per_domain_cap) for n in decoy_names)
        if total_cap < target_neg:
            print(
                f"[UYARI] --neg_per_domain={per_domain_cap} ile toplam üst sınır {total_cap}, "
                f"hedef {target_neg} negatif için yetersiz. "
                f"Elde edilebildiği kadar örnek seçilecek (oran < 1:{ratio_neg_per_pos} olabilir)."
            )

    alloc = even_allocation(target_neg, avail_by_domain, per_domain_cap)

    # Draw negatives per domain
    test_neg_parts = []
    for name, k in alloc.items():
        if k <= 0:
            continue
        pool = df_neg[df_neg[domain_col] == name]
        choose_idx = rng.choice(pool.index.to_numpy(), size=k, replace=False)
        test_neg_parts.append(pool.loc[choose_idx])

    test_neg = pd.concat(test_neg_parts, axis=0) if test_neg_parts else df_neg.iloc[0:0]

    # Combine test set
    test_df = pd.concat([test_pos, test_neg], axis=0).sample(frac=1.0, random_state=seed)
    # Train = everything else
    train_df = df.drop(index=test_df.index).copy()

    # Summaries
    test_pos_ct = int((test_df[label_col] == 1).sum())
    test_neg_ct = int((test_df[label_col] == 0).sum())
    achieved_ratio = f"1:{(test_neg_ct // test_pos_ct) if test_pos_ct else 'NA'}" if test_pos_ct else "NA"

    return {
        "test_df": test_df.reset_index(drop=True),
        "train_df": train_df.reset_index(drop=True),
        "n_pos_total": n_pos,
        "n_test_pos": test_pos_ct,
        "n_test_neg": test_neg_ct,
        "alloc": alloc,
        "avail_by_domain": avail_by_domain,
        "achieved_ratio": achieved_ratio,
        "target_neg": target_neg,
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Pozitiflerin %%20'sini test'e al; test'te 1:6 (poz:neg) oranı olacak şekilde "
            "negatifleri decoy domain'lerinden seç. Kalanlar all_training.csv'e yazılır."
        )
    )
    ap.add_argument("-i", "--input", default="all_data_avalon_morgan_maccs.csv", help="Girdi CSV")
    ap.add_argument("-te", "--test_out", default="test_ready.csv", help="Test çıktısı (CSV)")
    ap.add_argument("-tr", "--train_out", default="all_training.csv", help="Eğitim (kalan) çıktısı (CSV)")
    ap.add_argument("--label_col", default="label", help="Etiket sütunu (vars: label)")
    ap.add_argument(
        "--domain_col",
        default="source_dataset",
        help="Negatif domain sütunu (örn: decoy_name). Boş bırakılırsa otomatik bulunur."
    )
    ap.add_argument(
        "--decoy_names",
        default="dude,chembl_random,chembl_high_conf",
        help="Negatif domain isimleri; virgülle ayır (vars: 'dude,chembl_random,chembl_high_conf')",
    )
    ap.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu (vars: 42)")
    ap.add_argument("--pos_frac", type=float, default=0.20, help="Test'e alınacak pozitif oranı (vars: 0.20)")
    ap.add_argument(
        "--ratio",
        type=int,
        default=6,
        help="Test'te negatif/pozitif oranının payı (1:ratio; vars: 6)"
    )
    ap.add_argument(
        "--neg_per_domain",
        type=int,
        default=None,
        help="İstersen test negatifleri için domain başına üst sınır/tercih (örn: 200). "
             "Oranla çelişirse oran öncelikli şekilde ayarlanır."
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {in_path}")

    df = pd.read_csv(in_path)

    if args.label_col not in df.columns:
        raise ValueError(f"'{args.label_col}' kolonu bulunamadı. Mevcut sütunlar: {list(df.columns)}")

    domain_col = choose_domain_col(df, args.domain_col)

    decoy_names: Tuple[str, ...] = tuple([s.strip() for s in args.decoy_names.split(",") if s.strip()])

    res = build_test_split(
        df=df,
        label_col=args.label_col,
        domain_col=domain_col,
        decoy_names=decoy_names,
        pos_frac=args.pos_frac,
        ratio_neg_per_pos=args.ratio,
        seed=args.seed,
        neg_per_domain=args.neg_per_domain,
    )

    # Save
    res["test_df"].to_csv(args.test_out, index=False)
    res["train_df"].to_csv(args.train_out, index=False)

    # ==== PRINT SUMMARY ====
    total = len(df)
    total_pos = int((df[args.label_col] == 1).sum())
    total_neg = int((df[args.label_col] == 0).sum())
    print("\n=== ÖZET ===")
    print(f"Girdi: {in_path} | Toplam: {total} (pos={total_pos}, neg={total_neg})")
    print(f"Pozitiflerin %{int(round(args.pos_frac*100))}'i test'e alındı -> test_pos={res['n_test_pos']}")
    print(f"Hedef test_neg (1:{args.ratio}) = {res['target_neg']}")
    print(f"Gerçekleşen test_neg = {res['n_test_neg']}  | Elde edilen oran ≈ {res['achieved_ratio']}")
    print("\nNegatif domain kullanılabilirlikleri:")
    for n, a in res["avail_by_domain"].items():
        print(f"  - {n}: {a} mevcut")
    print("\nTest'te seçilen negatif dağılımı (domain: seçilen):")
    for n in decoy_names:
        print(f"  - {n}: {res['alloc'].get(n, 0)}")
    print("\nÇıktılar:")
    print(f"  Test CSV      : {args.test_out} (satır={len(res['test_df'])})")
    print(f"  Eğitim (kalan): {args.train_out} (satır={len(res['train_df'])})\n")


if __name__ == "__main__":
    main()
