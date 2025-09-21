# Prodrug-ML: Prodrug-Likeness Prediction via Machine Learning on Sampled Negative Decoys

This repository contains the experiment pipeline used in **“Prodrug-ML: Prodrug-Likeness Prediction via Machine Learning on Sampled Negative Decoys.”**  
It includes scripts for overlap control, fingerprint preparation, test-set isolation, hardness/ratio checks, domain-bias analysis, cross-decoy validation, and final evaluation.

---

## Data Availability & Acknowledgement

- **Positive class** molecules must be requested from **smProdrugs: A repository of small-molecule prodrugs**.  
  ➤ Positive data files are **not** redistributed here (no authority to distribute).  
  ➤ Many thanks to the **smProdrugs** team for sharing data that made this research possible.

---

## Environment Setup

You need **Python**, **RDKit**, and **scikit-learn**.

### Option A — Conda (recommended)
```bash
# Create and activate a clean environment
conda create -n prodrug-ml python=3.10 -y
conda activate prodrug-ml

# Install RDKit (choose a build/channel compatible with your OS)
conda install -c conda-forge rdkit -y

# ML/utility packages
pip install scikit-learn pandas numpy matplotlib
```
### Repository Structure & Steps

Pre_step_prepare_positive_data.py
Sanity checks for positive samples (required fields, duplicates, missing entries) before running the main pipeline.

#### 1_overlap_control/
De-duplication and overlap control across sources (e.g., InChIKey standardization, parent–prodrug reconciliation) to prevent leakage between classes/splits.

#### 2_preparation_of_fingerprint/
RDKit-based featurization to generate molecular fingerprints/descriptors (e.g., Avalon/Morgan) and export feature matrices.

#### 3_test_set_isolation/
Early, untouched test split created before any preprocessing or model selection; persists consistent train.csv / test.csv.

#### 4_and_5_hardness_control_and_ratio/
Hardness filtering for negatives (remove trivially easy decoys) and class/source ratio balancing to stabilize learning/evaluation.

#### 6_domain_bias/
Domain-bias checks (e.g., training a source/domain classifier) to detect distribution shortcuts unrelated to true prodrug-likeness.

#### 7_cross_decoy/
Cross-decoy validation by rotating negative sources/folds to stress-test generalization; can inform robust feature list selection.

#### 8_test_performance/
Final training and evaluation on the untouched test set; reports early-recognition (EF@1%, EF@5%, BEDROC) and global metrics (ROC-AUC, AP, F1, etc.).

.gitignore
Ignores caches, editor files, and other non-source artifacts.

### Typical Usage (High Level)

Obtain positive data from smProdrugs and place files in the expected input paths.

Run: overlap control → fingerprint preparation → test-set isolation (in order).

Apply hardness/ratio checks, then run domain-bias analysis.

Execute cross-decoy validation to assess robustness.

Train final models and compute test performance metrics.

Use each script’s -h/--help for arguments and I/O details.

## Reproducibility Notes

Fix random seeds where supported, and version the generated train.csv / test.csv to avoid leakage.

If adding new decoy sources, re-run overlap control, hardness checks, and domain-bias analysis before comparing results.

## Citation

If this repository or pipeline is useful in academic work, please cite the Prodrug-ML paper (citation details will be added upon publication).

## License & Usage

Academic usage is no change (as-is).

For commercial usage (or any usage beyond the above), please contact: s.yavuz.ugurlu@gmail.com
.

Note: The positive dataset from smProdrugs is not included here and must be requested from the original source under their terms.

## Acknowledgements

We thank the smProdrugs team for dataset access and the developers of RDKit and scikit-learn for essential open-source tools.

