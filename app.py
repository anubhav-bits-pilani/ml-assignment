# Dry Bean Classifier Lab - the Streamlit front end.
# it only LOADS the pickled models from model/ and scores whatever csv it's given -
# nothing gets trained or retrained in here.

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import matplotlib

# no display on the server, so draw straight into image buffers
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from beanml import metrics as M
from beanml.io import load_pickle
from beanml.validation import prepare

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"

# dark palette - keep in sync with .streamlit/config.toml
BG, PANEL, BORDER = "#0D1117", "#161B22", "#30363D"
TEXT, MUTED = "#E6EDF3", "#8B949E"
AMBER, TEAL = "#F0A050", "#3FB8AF"
CLASS_COLOURS = ["#F0A050", "#3FB8AF", "#E5534B", "#A371F7", "#57AB5A", "#DB61A2", "#539BF5"]
# dark -> brown -> amber -> cream, so low counts fade into the background and big ones pop
BEAN_CMAP = LinearSegmentedColormap.from_list("bean", [PANEL, "#4A2F14", AMBER, "#FFE3B8"])

METRIC_LABELS = [("accuracy", "Accuracy"), ("auc", "AUC"), ("precision", "Precision"),
                 ("recall", "Recall"), ("f1", "F1"), ("mcc", "MCC")]
METRIC_HELP = {
    "accuracy": "Share of rows predicted correctly.",
    "auc": "One-vs-rest ROC AUC per class, then averaged (macro).",
    "precision": "Per-class precision, plain average over classes (macro).",
    "recall": "Per-class recall, plain average over classes (macro).",
    "f1": "Per-class F1, plain average over classes (macro).",
    "mcc": "Multiclass Matthews correlation. 1 = perfect, 0 = coin flip.",
}
# one-liners shown under the model picker
MODEL_BLURBS = {
    "logistic_regression": "Softmax regression trained with mini-batch gradient descent. Linear boundaries, calibrated-ish probabilities.",
    "decision_tree": "CART tree with entropy/gini splits. Easy to read, but jumpy - small data changes move the splits.",
    "knn": "Looks up the k closest training beans (after scaling) and lets them vote, closer ones count more.",
    "naive_bayes": "Gaussian Naive Bayes - pretends the 16 features are independent given the class (they really aren't).",
    "svm": "RBF-kernel SVM solved with SMO, one-vs-one over all 21 class pairs.",
    "linear_svm": "Linear SVM, one-vs-rest, trained in the primal with hinge-loss subgradients.",
    "perceptron": "Averaged multiclass perceptron - the oldest trick in the book, still decent here.",
    "lda": "Linear discriminant analysis: one shared Gaussian covariance for every class.",
    "qda": "Quadratic discriminant analysis: each class gets its own (regularised) covariance.",
    "random_forest": "100 bootstrapped trees, each split only looks at 4 random features.",
    "adaboost": "SAMME AdaBoost on depth-2 trees, re-weights the beans it keeps getting wrong.",
    "gradient_boosting": "Softmax gradient boosting with depth-3 regression trees and early stopping.",
    "mlp": "Two hidden ReLU layers trained with Adam, early-stopped on a validation slice.",
    "soft_voting": "Averages the probabilities of LR, KNN, Random Forest, Gradient Boosting and the MLP.",
}

st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="🫘", layout="wide",
                   initial_sidebar_state="expanded")

# make every matplotlib chart match the dark theme
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
    "grid.color": BORDER, "legend.facecolor": PANEL, "legend.edgecolor": BORDER,
    "font.size": 10,
})

# a bit of CSS for the header card, numbered step badges, status bar, jump bar etc.
st.markdown(f"""
<style>
.block-container {{ padding-top: 1.6rem; }}
.hero {{
    background: linear-gradient(120deg, #1C2430 0%, #2A1D10 100%);
    border: 1px solid {BORDER}; border-radius: 16px; padding: 22px 28px; margin-bottom: 18px;
}}
.hero h1 {{ font-size: 2.1rem; margin: 0 0 4px 0; color: {TEXT}; }}
.hero p {{ color: #C9D1D9; margin: 0 0 12px 0; font-size: 1.02rem; }}
.chip {{
    display: inline-block; padding: 3px 12px; margin: 2px 6px 2px 0; border-radius: 999px;
    background: rgba(240,160,80,0.12); border: 1px solid rgba(240,160,80,0.45);
    color: {AMBER}; font-size: 0.82rem; font-weight: 600;
}}
.chip.teal {{ background: rgba(63,184,175,0.12); border-color: rgba(63,184,175,0.45); color: {TEAL}; }}
.step {{ font-size: 1.12rem; font-weight: 700; color: {TEXT}; margin-bottom: 2px; }}
.step .num {{
    display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px;
    border-radius: 50%; background: {AMBER}; color: {BG}; font-weight: 800; margin-right: 10px;
}}
.step-sub {{ color: {MUTED}; font-size: 0.88rem; margin: 0 0 8px 38px; }}
.status {{
    background: {PANEL}; border: 1px solid {BORDER}; border-left: 4px solid {AMBER};
    border-radius: 10px; padding: 10px 16px; margin: 6px 0 14px 0; color: #C9D1D9;
}}
.status b {{ color: {TEXT}; }}
.badge {{ padding: 2px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 700; }}
.badge.req {{ background: {AMBER}; color: {BG}; }}
.badge.bonus {{ background: {TEAL}; color: {BG}; }}
.blurb {{ color: #C9D1D9; font-size: 0.92rem; margin-top: 8px; }}
[data-testid="stMetric"] {{ background: {PANEL}; border-radius: 12px; }}
[data-testid="stMetricValue"] {{ color: {AMBER}; }}
[data-testid="stFileUploaderDropzone"] {{
    border: 2px dashed rgba(240,160,80,0.55); background: rgba(240,160,80,0.05);
}}
.nav {{
    position: sticky; top: 3.2rem; z-index: 50; display: flex; gap: 8px; flex-wrap: wrap;
    background: {BG}; padding: 8px 0 10px 0; border-bottom: 1px solid {BORDER}; margin-bottom: 6px;
}}
.nav a {{
    text-decoration: none; color: {TEXT}; background: {PANEL}; border: 1px solid {BORDER};
    border-radius: 999px; padding: 5px 16px; font-size: 0.9rem; font-weight: 600;
}}
.nav a:hover {{ border-color: {AMBER}; color: {AMBER}; }}
.section-head {{
    font-size: 1.45rem; font-weight: 800; color: {TEXT}; margin: 28px 0 12px 0; padding-bottom: 6px;
    border-bottom: 2px solid {AMBER}; scroll-margin-top: 7rem;
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- loading bits

@st.cache_resource
def load_meta():
    with open(MODEL_DIR / "meta.json") as f:
        return json.load(f)


@st.cache_resource
def load_model(slug):
    meta = load_meta()
    info = next(m for m in meta["models"] if m["slug"] == slug)
    model = load_pickle(MODEL_DIR / info["file"])
    # the voter only stores member names, so hand it the (cached) members
    if slug == "soft_voting":
        model.attach({s: load_model(s) for s in model.member_names})
    return model


@st.cache_data
def load_json_file(name):
    p = MODEL_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def read_csv_bytes(raw: bytes):
    return pd.read_csv(io.BytesIO(raw))


def fmt(v, nd=4):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


# cached per (model, file): data_key is a hash of the uploaded bytes, and the leading
# underscore on _X tells streamlit not to try hashing the DataFrame itself
@st.cache_data(show_spinner=False)
def score_model(slug, data_key, _X):
    model = load_model(slug)
    s = np.asarray(model.scores(_X), dtype=float)
    return s, np.argmax(s, axis=1)


# ---------------------------------------------------------------- plots

def plot_confusion(cm, names):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap=BEAN_CMAP)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    # dark text on bright cells, light text on dark cells, so every number stays readable
    thresh = cm.max() * 0.55 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9, fontweight="bold",
                    color=BG if cm[i, j] > thresh else TEXT)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_edgecolor(BORDER)
    fig.tight_layout()
    return fig


# one ROC curve per class (that class vs the rest); the legend shows each class's AUC
# computed as the area under its curve with the trapezoid rule
def plot_roc(y, scores, names):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for k, name in enumerate(names):
        # a curve needs at least one positive and one negative
        if (y == k).any() and (y != k).any():
            fpr, tpr = M.roc_curve(y == k, scores[:, k])
            ax.plot(fpr, tpr, color=CLASS_COLOURS[k % len(CLASS_COLOURS)], lw=1.8,
                    label=f"{name}  ({M.trapezoid(tpr, fpr):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=9, loc="lower right", title="class (AUC)", title_fontsize=9)
    fig.tight_layout()
    return fig


def show(fig):
    st.pyplot(fig)
    # free tier memory is tight, don't leak figures
    plt.close(fig)


# leaderboard table: ranked by macro-F1, MCC breaks ties
def board_frame(rows):
    tab = pd.DataFrame([{"Model": r["name"], "Type": "★ required" if r["required"] else "bonus",
                         **{lbl: r["metrics"][k] for k, lbl in METRIC_LABELS}} for r in rows])
    tab = tab.sort_values(["F1", "MCC"], ascending=False).reset_index(drop=True)
    # rank starts at 1, not 0
    tab.index = tab.index + 1
    return tab


def render_board(tab, caption):
    best = tab.iloc[0]
    st.success(f"🏆 Winner: **{best['Model']}** - macro-F1 {best['F1']:.4f}, MCC {best['MCC']:.4f}")
    num = [lbl for _, lbl in METRIC_LABELS]
    st.dataframe(tab.style.format({c: "{:.4f}" for c in num}, na_rep="n/a")
                 .highlight_max(subset=num, props=f"background-color:#4A2F14;color:{AMBER};font-weight:700"),
                 height=min(35 * (len(tab) + 1) + 3, 600))
    st.caption(caption)
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(tab) + 0.9))
    t2 = tab.iloc[::-1]
    colours = [AMBER if t.startswith("★") else TEAL for t in t2["Type"]]
    bars = ax.barh(t2["Model"], t2["F1"], color=colours)
    for b, v in zip(bars, t2["F1"]):
        ax.text(v + 0.001, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=8, color=TEXT)
    lo = max(0.0, float(np.nanmin(t2["F1"])) - 0.03)
    ax.set_xlim(lo, min(1.0, float(np.nanmax(t2["F1"])) + 0.02))
    ax.set_xlabel("macro F1   (amber = required model, teal = bonus)")
    ax.grid(axis="x", alpha=0.4)
    fig.tight_layout()
    show(fig)


# ---------------------------------------------------------------- page

meta = load_meta()
models = meta["models"]
names = meta["class_names"]
slugs = [m["slug"] for m in models]
by_slug = {m["slug"]: m for m in models}

with st.sidebar:
    st.markdown("### 🫘 How to use")
    st.markdown(
        "1. **Upload** a test CSV in panel ① (or just use the bundled one)\n"
        "2. **Pick a model** from the dropdown in panel ②\n"
        "3. Scroll down (or use the jump bar) for results, leaderboard and predictions")
    st.divider()
    st.markdown("### 📦 Dataset at a glance")
    ds = meta["dataset"]
    st.markdown(
        f"- UCI **Dry Bean** dataset\n- {ds['rows_clean']:,} beans after de-dup\n"
        f"- 16 shape features, 7 varieties\n- test split: {meta['n_test']:,} rows (seed {meta['seed']})")
    st.divider()
    st.caption("Every model is hand-written in NumPy - no scikit-learn. They were trained once offline; "
               "this app only loads the saved pickles and never retrains.")

# numpy 2 pickles don't load on numpy 1, so warn if the versions' major numbers differ
trained_np = meta["env"]["numpy"].split(".")[0]
if trained_np != np.__version__.split(".")[0]:
    st.warning(f"Models were saved with numpy {meta['env']['numpy']} but this box runs {np.__version__}. "
               "If loading fails, install the versions in requirements.txt.")

st.markdown(
    """
<div class="hero">
  <h1>🫘 Dry Bean Classifier Lab</h1>
  <p>Which of the 7 dry bean varieties is this grain? Each row is one bean, photographed and measured into
  16 shape features. Upload test data, pick a model, and see how well it does.</p>
  <span class="chip">16 features</span><span class="chip">7 classes</span>
  <span class="chip">14 models</span><span class="chip teal">NumPy only · no sklearn</span>
  <span class="chip teal">no retraining in the app</span>
</div>
""", unsafe_allow_html=True)

# the two controls people actually need, front and centre
c_data, c_model = st.columns([1.15, 1], gap="large")
with c_data:
    with st.container(border=True):
        st.markdown('<div class="step"><span class="num">1</span>Load your test data</div>'
                    '<div class="step-sub">CSV with the 16 feature columns + <code>Class</code>. '
                    'Test data only - 10 MB max.</div>', unsafe_allow_html=True)
        upload = st.file_uploader("Upload test CSV", type=["csv"], label_visibility="collapsed",
                                  help="Column order doesn't matter, extra columns are ignored.")
        dl, note = st.columns([1, 1.6])
        dl.download_button("⬇️ Sample test_data.csv", TEST_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")
        note.caption("No file? The bundled hold-out set is used automatically.")

with c_model:
    with st.container(border=True):
        st.markdown('<div class="step"><span class="num">2</span>Pick a model</div>'
                    '<div class="step-sub">★ = one of the 5 required models, the rest are bonus.</div>',
                    unsafe_allow_html=True)
        chosen = st.selectbox(
            "Model", slugs, label_visibility="collapsed",
            format_func=lambda s: ("★  " if by_slug[s]["required"] else "      ") + by_slug[s]["name"])
        info = by_slug[chosen]
        badge = '<span class="badge req">REQUIRED</span>' if info["required"] \
            else '<span class="badge bonus">BONUS</span>'
        tuned = "tuned with CV" if info["tuned"] else "fixed hyper-parameters"
        st.markdown(f'{badge} &nbsp;<span style="color:{MUTED};font-size:0.85rem">{tuned}</span>'
                    f'<div class="blurb">{MODEL_BLURBS.get(chosen, "")}</div>', unsafe_allow_html=True)

# no upload yet -> fall back to the bundled hold-out split so the page isn't empty
if upload is not None:
    raw, source = upload.getvalue(), f"your file <b>{upload.name}</b>"
else:
    raw, source = TEST_CSV.read_bytes(), "the bundled <b>test_data.csv</b>"

try:
    df_in = read_csv_bytes(raw)
# bad csv, wrong delimiter, binary junk...
except Exception as e:
    st.error(f"Couldn't read that as a CSV: {e}")
    st.stop()

# if there's no obvious Class column, let the user point at the label column (or skip it)
target_choice = None
if meta["target"].lower() not in {c.strip().lower() for c in df_in.columns}:
    opts = ["(no target - just predict)"] + [c for c in df_in.columns if c not in meta["feature_names"]]
    pick = st.selectbox("Couldn't find a `Class` column - which column holds the true label?", opts)
    target_choice = None if pick == opts[0] else pick

X, y, notes, err = prepare(df_in, meta, target_choice)
data_key = hashlib.md5(raw).hexdigest() + str(target_choice)
if err:
    st.error(err)
    st.stop()

st.markdown(f'<div class="status">Scoring <b>{info["name"]}</b> on {source} · '
            f'<b>{len(X):,}</b> rows · {len(meta["feature_names"])} features</div>', unsafe_allow_html=True)
for n in notes:
    st.info(n)
if y is None:
    st.warning("No target column, so there's nothing to compare against - showing predictions only.")

scores, pred = score_model(chosen, data_key, X)

# everything stays on one page (no tabs) so nothing is hidden - the jump bar just scrolls to each part
SECTIONS = [("results", "📊 Model results"), ("leaderboard", "🏆 Leaderboard"),
            ("predictions", "🔮 Predictions"), ("about", "ℹ️ About")]
st.markdown('<div class="nav">' + "".join(f'<a href="#{a}">{t}</a>' for a, t in SECTIONS) + "</div>",
            unsafe_allow_html=True)


def section(anchor):
    title = dict(SECTIONS)[anchor]
    st.markdown(f'<div class="section-head" id="{anchor}">{title}</div>', unsafe_allow_html=True)


with st.container():
    section("results")
    if y is None:
        st.info("Upload a file with the `Class` column to see metrics.")
    else:
        # all six metrics come from our own beanml.metrics, nothing from sklearn
        res = M.evaluate(y, pred, scores, len(names))
        cols = st.columns(6)
        for c, (key, label) in zip(cols, METRIC_LABELS):
            c.metric(label, fmt(res[key]), help=METRIC_HELP[key], border=True)
        st.caption("Precision / Recall / F1 are macro averages (every class counts equally), "
                   "AUC is macro one-vs-rest.")
        # a hand-made test file might not contain every bean type
        if len(np.unique(y)) < len(names):
            st.caption(f"Heads up: only {len(np.unique(y))} of {len(names)} classes are in this file - "
                       "AUC skips the missing ones, macro scores average over classes that appear.")
        with st.expander("Weighted averages (weighted by class support)", expanded=True):
            w = st.columns(3)
            w[0].metric("Precision (weighted)", fmt(res["precision_weighted"]), border=True)
            w[1].metric("Recall (weighted)", fmt(res["recall_weighted"]), border=True)
            w[2].metric("F1 (weighted)", fmt(res["f1_weighted"]), border=True)

        left, right = st.columns(2, gap="large")
        with left:
            with st.container(border=True):
                st.markdown("**Confusion matrix** · rows = actual, columns = predicted")
                show(plot_confusion(M.confusion_matrix(y, pred, len(names)), names))
        with right:
            with st.container(border=True):
                st.markdown("**ROC curves** · one-vs-rest")
                show(plot_roc(y, scores, names))
                if not info["has_proba"]:
                    st.caption("No probabilities for this model - the curves use its raw decision scores "
                               "(ROC only cares about ranking, so that's fine).")

        st.markdown("#### Classification report")
        rep = pd.DataFrame(M.classification_report(y, pred, names)).set_index("class")
        st.dataframe(rep.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"})
                     .background_gradient(subset=["f1"], cmap=BEAN_CMAP, vmin=0.75, vmax=1.0))

    with st.expander("Hyper-parameters used for this model", expanded=True):
        st.json({"params": info["params"], "tuned with CV": info["tuned"],
                 "fit time (s)": info["fit_seconds"]})

with st.container():
    section("leaderboard")
    # numbers saved at training time, so every model's results show up even before any upload
    saved = load_json_file("test_metrics.json")
    if saved:
        st.markdown("**All 14 models on the held-out test set**")
        render_board(board_frame(saved["models"].values()),
                     f"Pre-computed when the models were trained (n = {saved['n_test']:,} test rows). "
                     "Winner = highest macro-F1, MCC breaks ties.")
    if y is not None:
        st.divider()
        st.markdown("#### Re-score on the data loaded above")
        # re-score every saved model on whatever file is loaded (still no retraining)
        if st.button("🔁 Run all 14 models on this data", type="primary"):
            rows = []
            with st.spinner("Scoring 14 models..."):
                for s in slugs:
                    sc, pr = score_model(s, data_key, X)
                    rows.append({"name": by_slug[s]["name"], "required": by_slug[s]["required"],
                                 "metrics": M.evaluate(y, pr, sc, len(names))})
            render_board(board_frame(rows), f"Computed live on {len(X):,} rows.")

with st.container():
    section("predictions")
    out = X.copy()
    out.insert(0, "predicted", np.asarray(names)[pred])
    if info["has_proba"]:
        # confidence = the top class probability (only makes sense for models that output probabilities)
        out.insert(1, "confidence", scores.max(axis=1).round(4))
    if y is not None:
        out.insert(1, "actual", np.asarray(names)[y])
        out.insert(2, "correct", pred == y)
        n_wrong = int((pred != y).sum())
        a, b, c = st.columns([1, 1, 2])
        a.metric("Rows", f"{len(y):,}", border=True)
        b.metric("Mistakes", f"{n_wrong:,}", border=True)
        with c:
            only_wrong = st.toggle("Show only the mistakes")
        if only_wrong:
            out = out[~out["correct"]]
    st.dataframe(out, height=460)
    st.download_button("⬇️ Download these predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{chosen}.csv", mime="text/csv")

with st.container():
    section("about")
    ds = meta["dataset"]
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### About the data")
        st.markdown(
            f"""
**UCI Dry Bean dataset** (Koklu & Ozkan, 2020, CC BY 4.0): images of 13,611 grains from 7 registered
dry bean varieties, each turned into 12 dimension features + 4 shape-form features.

* raw rows: **{ds['rows_raw']:,}**, exact duplicates dropped: **{ds['exact_duplicates_dropped']}**,
  rows used: **{ds['rows_clean']:,}**
* stratified 80/20 split, seed **{meta['seed']}** -> {meta['n_train']:,} train / {meta['n_test']:,} test rows
""")
        counts = pd.Series(meta["class_counts_clean"]).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.bar(counts.index, counts.values, color=[CLASS_COLOURS[names.index(c)] for c in counts.index])
        ax.set_ylabel("beans")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.4)
        fig.tight_layout()
        show(fig)
    with right:
        st.markdown("#### How the metrics are computed")
        st.markdown(
            """
* **Accuracy** - share of rows predicted right.
* **AUC** - one-vs-rest ROC AUC per class, then averaged (macro). Uses predicted probabilities, or the raw
  decision scores for the SVMs / perceptron (AUC only looks at ranking, so that's fine).
* **Precision / Recall / F1** - per class, then plain average (macro), so the small BOMBAY class counts as
  much as DERMASON. Weighted versions are in the expander on the results tab.
* **MCC** - multiclass Matthews correlation (Gorodkin's generalisation), 1 = perfect, 0 = no better than chance.
""")
    # best CV setting for every tuned model, all computed on the training split only
    cv = load_json_file("cv_results.json")
    if cv:
        st.markdown("#### Cross-validation (training split only)")
        cv_tab = pd.DataFrame([{"Model": by_slug[s]["name"], "folds": r["folds"], "best params": str(r["best"]["params"]),
                                "CV macro-F1": r["best"]["f1_mean"], "± std": r["best"]["f1_std"]}
                               for s, r in cv.items() if s in by_slug])
        st.dataframe(cv_tab.style.format({"CV macro-F1": "{:.4f}", "± std": "{:.4f}"}), hide_index=True)
    env = meta["env"]
    st.caption(f"Trained with python {env['python']}, numpy {env['numpy']}, pandas {env['pandas']} "
               f"on {env['timestamp']}.")
