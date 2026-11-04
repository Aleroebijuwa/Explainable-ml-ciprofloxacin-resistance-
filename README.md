# Explainable Machine Learning for Predicting Ciprofloxacin Resistance in *E. coli*

MSc project (Artificial Intelligence with Data Analytics, Teesside University).

Can machine-learning models predict ciprofloxacin resistance in *Escherichia coli* from whole-genome
sequencing (WGS) data, do they rely on known biological resistance mechanisms, and do they still work
on bacterial lineages they have never seen?

*Academic research prototype — not a clinically validated diagnostic tool.*

## Data

704 *E. coli* isolates with ciprofloxacin minimum inhibitory concentrations (MICs) from
Pataki et al. (2020), [AMR_ciprofloxacin](https://github.com/patbaa/AMR_ciprofloxacin)
(`Data/meta.tsv`). Isolates are labelled **resistant** when MIC > 0.5 mg/L (EUCAST breakpoint).

Genomic features were derived from each isolate's own raw sequencing reads in the European
Nucleotide Archive (`aws/extract_features.py`):

| Feature type | How |
|---|---|
| Amino-acid substitutions in *gyrA*, *gyrB*, *parC*, *parE*, *marR*, *acrR*, *soxR* | Reads mapped to *E. coli* K-12 MG1655 genes (BWA-MEM), consensus called, translated, K-12 numbering |
| Acquired quinolone resistance genes (*qnr*, *aac(6')-Ib-cr*, *oqxAB*, *qepA*) | KMA against ResFinder |
| Lineage (for clade-aware validation) | Mash distances, average-linkage clustering at distance 0.01 |

Features were extracted for 703 of 704 isolates (one isolate's reads could not be downloaded);
691 passed sequencing-quality checks and were used for modelling.

## Methods

- **Models:** Logistic Regression, Random Forest and XGBoost, compared with a majority-class baseline
  and a rule that counts known resistance mechanisms.
- **Validation:** nested cross-validation (hyperparameters tuned in the inner loop), with **random**
  and **clade-aware** (lineage-grouped) outer folds, plus the dataset's original held-out test set
  with bootstrap 95% confidence intervals.
- **Further analyses:** ablation (known mechanisms only), sensitivity to the lineage threshold,
  and performance on test isolates from seen vs unseen lineages.
- **Explainability:** SHAP values for every model; feature attribution grouped into known
  mechanisms vs other substitutions (likely lineage markers).

## Key results

Clade-aware 5-fold cross-validation (mean ± SD):

| Model | AUROC | Balanced accuracy | MCC |
|---|---|---|---|
| Majority class | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 |
| Known-mechanism rule | 0.981 ± 0.027 | 0.935 ± 0.047 | 0.886 ± 0.085 |
| Logistic Regression | 0.984 ± 0.027 | 0.917 ± 0.047 | 0.848 ± 0.048 |
| Random Forest | 0.983 ± 0.028 | 0.911 ± 0.075 | 0.861 ± 0.126 |
| XGBoost | 0.984 ± 0.029 | 0.942 ± 0.058 | 0.912 ± 0.062 |

- All models predict resistance very accurately (held-out AUROC ≈ 0.99, MCC 0.96–0.97), but they
  do not clearly outperform the simple known-mechanism rule.
- Performance drops very little on unseen lineages (random minus clade-aware CV: AUROC ≤ 0.005).
- SHAP ranks **parC S80I, gyrA D87N and gyrA S83L** as the three most important features in every
  model, followed by the acquired gene *qnrS*.
- Logistic Regression attributes about 30% of its importance to non-mechanism substitutions
  (likely lineage markers), compared with about 9% for Random Forest and 15% for XGBoost.

All tables are in [`results/`](results) and figures in [`results/figures/`](results/figures).

## Repository structure

| Path | Contents |
|---|---|
| `ecoli_ciprofloxacin_resistance.ipynb` | Main notebook: data exploration, modelling and validation, SHAP |
| `app.py` | Streamlit app for interactive, explained predictions |
| `src/amr_utils.py` | Shared helpers: feature categories, pipeline components, evaluation |
| `aws/` | Feature extraction from raw reads on a Linux server (see [`aws/README.md`](aws/README.md)) |
| `Data/` | Metadata, extracted features, QC and Mash distance matrix |
| `models/` | Final fitted models |
| `results/` | Result tables and figures |

## How to run

Requires Python 3.10–3.12.

```bash
pip install -r requirements.txt
```

**Notebook:** open `ecoli_ciprofloxacin_resistance.ipynb` in Jupyter and run all cells
(about 10 minutes). It reads the files in `Data/` and writes `models/` and `results/`.

**App:**

```bash
streamlit run app.py
```

Choose a model, then pick a held-out test isolate or build your own mutation profile. The app
shows the predicted probability of resistance and the SHAP contributions behind it.

**Feature extraction** (only needed to rebuild `Data/features.csv`, `qc.csv` and
`mash_distances.csv` from raw reads): see [`aws/README.md`](aws/README.md).

## Reference

Pataki, B. Á. et al. (2020). Understanding and predicting ciprofloxacin minimum inhibitory concentration in *Escherichia coli*
with machine learning. *Scientific Reports*, 10, 15026.
