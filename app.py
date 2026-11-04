"""Streamlit app: explainable prediction of ciprofloxacin resistance in E. coli.

Loads the final models trained in ecoli_ciprofloxacin_resistance.ipynb (models/*.joblib)
and explains each prediction with SHAP.

Run from the project folder:
    streamlit run app.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.amr_utils import CATEGORY_LABELS, feature_category

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_FILES = {
    "XGBoost": "xgboost.joblib",
    "Random Forest": "random_forest.joblib",
    "Logistic Regression": "logistic_regression.joblib",
}
RESISTANT_COLOUR = "#c0392b"
SUSCEPTIBLE_COLOUR = "#2471a3"
TOP_N = 10

st.set_page_config(page_title="Ciprofloxacin resistance predictor", layout="wide")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_data():
    return joblib.load(MODELS_DIR / "datasets.joblib")


@st.cache_resource
def load_model(name):
    return joblib.load(MODELS_DIR / MODEL_FILES[name])


@st.cache_resource
def load_rule():
    return joblib.load(MODELS_DIR / "known_mechanism_rule.joblib")


@st.cache_resource
def load_explainer(name):
    """SHAP explainer for the classifier inside the pipeline (after the feature filter)."""
    pipeline = load_model(name)
    data = load_data()
    background = pipeline.named_steps["filter"].transform(data["X_dev"])
    model = pipeline.named_steps["model"]
    if name == "Logistic Regression":
        return shap.LinearExplainer(model, background)
    return shap.TreeExplainer(model)


def explain(name, X):
    """SHAP values for the resistant class, one row per isolate."""
    pipeline = load_model(name)
    X_filtered = pipeline.named_steps["filter"].transform(X)
    values = load_explainer(name).shap_values(X_filtered)
    if isinstance(values, list):          # older shap: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:                  # Random Forest: (isolates, features, classes)
        values = values[:, :, 1]
    return pd.DataFrame(values, columns=X_filtered.columns, index=X.index)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def category_label(feature):
    return CATEGORY_LABELS[feature_category(feature)]


def contribution_chart(contributions, present, model_name):
    """Horizontal bar chart of the largest SHAP contributions for one isolate."""
    top = contributions.reindex(contributions.abs().sort_values(ascending=False).index)[:TOP_N][::-1]
    labels = [f"{f} ({'present' if present[f] else 'absent'})" for f in top.index]
    colours = [RESISTANT_COLOUR if v > 0 else SUSCEPTIBLE_COLOUR for v in top]

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(top) + 1))
    ax.barh(labels, top.values, color=colours)
    ax.axvline(0, color="black", linewidth=0.8)
    unit = "probability" if model_name == "Random Forest" else "log-odds"
    ax.set_xlabel(f"SHAP contribution ({unit})  ←  susceptible  |  resistant  →")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def show_figure(filename, caption):
    path = FIGURES_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_column_width=True)
    else:
        st.info(f"{filename} not found. Run the notebook to create it.")


def read_result(filename):
    path = RESULTS_DIR / filename
    return pd.read_csv(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

data = load_data()
feature_cols = data["feature_cols"]
X_all = pd.concat([data["X_dev"], data["X_test"]])
y_all = pd.concat([data["y_dev"], data["y_test"]])
metadata = data["metadata"]
breakpoint_mic = data["breakpoint"]

st.title("Ciprofloxacin resistance in *E. coli*")
st.caption(
    "Explainable machine-learning prediction from whole-genome sequencing features. "
    "Academic research prototype — not a clinically validated diagnostic tool."
)

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", list(MODEL_FILES))
    st.markdown(
        f"**Resistant** = MIC > {breakpoint_mic} mg/L (EUCAST breakpoint).\n\n"
        f"Trained on {len(data['X_dev'])} isolates; tested on {len(data['X_test'])} held-out isolates."
    )

predict_tab, performance_tab, explain_tab, about_tab = st.tabs(
    ["Predict", "Model performance", "What the models learned", "About"]
)

# --- Predict ---------------------------------------------------------------
with predict_tab:
    mode = st.radio(
        "Input",
        ["Isolate from the held-out test set", "Build a mutation profile"],
        horizontal=True,
    )

    if mode == "Isolate from the held-out test set":
        isolate = st.selectbox("Isolate", data["X_test"].index)
        profile = data["X_test"].loc[[isolate]]
        mic = metadata.loc[isolate, "measurement"]
        truth = "Resistant" if y_all.loc[isolate] == 1 else "Susceptible"
        st.markdown(f"Measured MIC: **{mic} mg/L** → **{truth}**  ·  Country: {metadata.loc[isolate, 'country']}")
    else:
        known = sorted(f for f in feature_cols if feature_category(f) != "other")
        other = sorted(f for f in feature_cols if feature_category(f) == "other")
        left, right = st.columns(2)
        chosen_known = left.multiselect(
            "Known resistance mechanisms",
            known,
            default=["gyrA_S83L"],
            help="QRDR mutations in gyrA/gyrB/parC/parE, acquired genes (qnr, aac(6')-Ib-cr, ...) "
                 "and efflux-regulator truncations.",
        )
        chosen_other = right.multiselect(
            "Other substitutions (mostly lineage markers)", other,
        )
        profile = pd.DataFrame(0, index=["custom profile"], columns=feature_cols)
        profile.loc["custom profile", chosen_known + chosen_other] = 1

    present = profile.iloc[0]
    probability = load_model(model_name).predict_proba(profile)[0, 1]
    rule = load_rule()
    rule_count = int(rule._count(profile)[0])
    detected = [f for f in rule.columns_ if present[f] == 1]

    col1, col2, col3 = st.columns(3)
    col1.metric(f"{model_name}: probability resistant", f"{probability:.0%}")
    col2.metric("Prediction", "Resistant" if probability >= 0.5 else "Susceptible")
    col3.metric(
        "Known mechanisms detected", rule_count,
        help=f"The rule-based baseline predicts resistant at ≥ {rule.threshold_} mechanisms.",
    )
    st.markdown("**Known mechanisms in this profile:** " + (", ".join(detected) if detected else "none"))

    st.subheader("Why this prediction?")
    contributions = explain(model_name, profile).iloc[0]
    if contributions.abs().sum() == 0:
        st.info("No feature moved the prediction away from the model's baseline.")
    else:
        chart_col, text_col = st.columns([3, 2])
        chart_col.pyplot(contribution_chart(contributions, present, model_name))
        top = contributions.reindex(contributions.abs().sort_values(ascending=False).index)[:TOP_N]
        table = pd.DataFrame({
            "feature": top.index,
            "present": ["yes" if present[f] else "no" for f in top.index],
            "category": [category_label(f) for f in top.index],
            "SHAP": top.values.round(3),
        })
        text_col.dataframe(table, hide_index=True, use_container_width=True)
        text_col.caption(
            "Red bars push towards resistant, blue towards susceptible. Features the model "
            "removed as too rare (seen in < 3 training isolates) are not shown."
        )

# --- Model performance -----------------------------------------------------
with performance_tab:
    st.subheader("Held-out test set (95% bootstrap confidence intervals)")
    holdout = read_result("holdout_results.csv")
    if holdout is not None:
        holdout["value"] = holdout.apply(
            lambda r: f"{r.estimate:.3f} ({r.ci_low:.3f}–{r.ci_high:.3f})", axis=1)
        st.dataframe(holdout.pivot(index="model", columns="metric", values="value"),
                     use_container_width=True)

    left, right = st.columns(2)
    with left:
        show_figure("cv_comparison.png", "Random vs clade-aware cross-validation")
    with right:
        show_figure("roc_holdout.png", "ROC curves on the held-out test set")

    gap = read_result("generalisation_gap.csv")
    if gap is not None:
        st.subheader("Generalisation gap (random CV minus clade-aware CV)")
        st.dataframe(gap, hide_index=True, use_container_width=True)
        st.caption("Positive values mean the model performs worse on lineages it has not seen.")

# --- What the models learned -----------------------------------------------
with explain_tab:
    model_file = model_name.lower().replace(" ", "_")
    left, right = st.columns(2)
    with left:
        show_figure(f"shap_beeswarm_{model_file}.png", f"SHAP summary — {model_name}")
    with right:
        show_figure("shap_share_by_category.png", "Share of SHAP attribution by feature category")
    show_figure("shap_importance_all_models.png", "Top features across the three models")
    show_figure("shap_dependence_gyrA_parC.png", "Interaction between gyrA and parC mutations")

# --- About -------------------------------------------------------------------
with about_tab:
    st.markdown(
        """
**Data.** 704 *E. coli* isolates with ciprofloxacin MIC measurements from Pataki et al. (2020)
([AMR_ciprofloxacin](https://github.com/patbaa/AMR_ciprofloxacin)). Genomic features were derived
from each isolate's raw sequencing reads (European Nucleotide Archive): amino-acid substitutions in
*gyrA*, *gyrB*, *parC*, *parE*, *marR*, *acrR* and *soxR*, and acquired quinolone resistance genes.

**Models.** Logistic Regression, Random Forest and XGBoost, compared with a majority-class baseline and
a rule that counts known resistance mechanisms. Hyperparameters were tuned with nested
cross-validation; clade-aware cross-validation (grouping isolates by Mash-distance lineage) tests
whether the models generalise to unseen lineages.

**Explanations.** SHAP values show how much each feature pushed a prediction towards resistant or
susceptible.

**Limitations.** Predictions are only as good as the features: resistance mechanisms not included
(e.g. other efflux changes, plasmid genes not in the reference set) cannot be detected. Isolates
close to the breakpoint are the hardest to classify. This tool is for research and teaching only.
"""
    )
