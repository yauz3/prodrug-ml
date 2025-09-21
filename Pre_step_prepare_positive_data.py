import pandas as pd
import numpy as np

# === CONFIG ===
input_file = "Prodrug_data.xlsx"
output_file = "Prodrug_identifiers.csv"

# Columns we want to keep
identifier_cols = [
    "Entry", "Prodrugid", "Prodrug", "SMILES",
    "PubChem", "CAS No.", "inchi", "inchi_k"
]

# Helper: pick the "best" row within a group of identical SMILES
# Preference: row with the most non-null identifier fields
def pick_best_row(group: pd.DataFrame) -> pd.Series:
    scoring_cols = [c for c in ["Prodrugid", "Prodrug", "PubChem", "CAS No.", "inchi", "inchi_k"] if c in group.columns]
    scores = group[scoring_cols].notna().sum(axis=1)
    # If tie, keep the first
    return group.loc[scores.idxmax()]

# === STEP 1: Read all sheets ===
xls = pd.ExcelFile(input_file)
dfs = [pd.read_excel(input_file, sheet_name=sheet) for sheet in xls.sheet_names]

# === STEP 2: Concatenate & remove exact duplicates ===
combined_df = pd.concat(dfs, ignore_index=True).drop_duplicates()

# === STEP 2.1: Clean SMILES: strip spaces; empty→NaN; keep case (SMILES is case-sensitive)
if "SMILES" in combined_df.columns:
    combined_df["SMILES"] = (
        combined_df["SMILES"]
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )

# === STEP 2.2: Deduplicate by SMILES but DO NOT collapse NaN rows ===
if "SMILES" in combined_df.columns:
    with_smiles = combined_df[combined_df["SMILES"].notna()].copy()
    without_smiles = combined_df[combined_df["SMILES"].isna()].copy()

    # Group by SMILES and pick a representative row using the scoring function
    dedup_by_smiles = (
        with_smiles
        .groupby("SMILES", as_index=False, group_keys=False)
        .apply(pick_best_row)
        .reset_index(drop=True)
    )

    combined_df = pd.concat([dedup_by_smiles, without_smiles], ignore_index=True)
    # (Optional) Remove exact duplicates again if anything re-merged identical rows
    combined_df = combined_df.drop_duplicates()

# === STEP 3: Keep only identifier columns that exist in the file ===
available_cols = [col for col in identifier_cols if col in combined_df.columns]
identifier_df = combined_df[available_cols].copy()

# === STEP 4: Add label column ===
identifier_df["prodrug_label"] = 1

# === STEP 5: Save to CSV ===
identifier_df.to_csv(output_file, index=False)

# === DIAGNOSTICS ===
total_rows = len(pd.concat(dfs, ignore_index=True))
after_exact = len(pd.concat(dfs, ignore_index=True).drop_duplicates())
with_smiles_n = combined_df["SMILES"].notna().sum() if "SMILES" in combined_df.columns else 0
without_smiles_n = combined_df["SMILES"].isna().sum() if "SMILES" in combined_df.columns else 0

print(f"✅ Processed dataframe saved to {output_file}")
print("Columns kept:", available_cols)
print(f"Total raw rows (all sheets): {total_rows}")
print(f"After exact drop_duplicates: {after_exact}")
if "SMILES" in combined_df.columns:
    print(f"Rows WITH SMILES after dedup-by-SMILES: {with_smiles_n}")
    print(f"Rows WITHOUT SMILES preserved: {without_smiles_n}")
print("Final number of entries:", len(identifier_df))

# Extra: show top duplicated SMILES (before dedup) to inspect what was collapsed
if "SMILES" in combined_df.columns:
    all_concat = pd.concat(dfs, ignore_index=True)
    all_concat["SMILES"] = (
        all_concat.get("SMILES")
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    dup_counts = (all_concat["SMILES"]
                  .value_counts(dropna=True)
                  .loc[lambda s: s > 1]
                  .sort_values(ascending=False))
    print("\nSMILES that appeared more than once (before dedup):")
    print(dup_counts.head(20))
