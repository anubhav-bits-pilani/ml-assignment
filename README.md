# Dry Bean Classifier Lab

**ML Assignment 2 · AIMLZG565 · M.Tech AIML (WILP) · Student ID: 2026AJ05327**

Live app: **https://YOUR-APP-NAME.streamlit.app**

---

## a. Problem statement

Dry beans are one of the most widely grown legumes, and seed quality depends heavily on variety purity — but
sorting beans by hand is slow and inconsistent. The goal here is to build classifiers that tell **which of 7
registered dry bean varieties** (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA) a single grain belongs to,
using only 16 shape measurements extracted from a photo of that grain. This is a **multi-class classification**
problem.

The assignment asks for five classifiers — Logistic Regression, Decision Tree, KNN, Naive Bayes and SVM — on the
same data, evaluated with Accuracy, AUC, Precision, Recall, F1 and MCC, and shown through an interactive Streamlit
app. I implemented **all models and all metrics from scratch in NumPy** (no scikit-learn anywhere), and added nine
extra models on top of the required five.

## b. Dataset description

| | |
|---|---|
| Source | [UCI Machine Learning Repository — Dry Bean Dataset (id 602)](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) |
| Citation | Koklu, M. & Ozkan, I.A. (2020). *Multiclass classification of dry beans using computer vision and machine learning techniques.* Computers and Electronics in Agriculture, 174, 105507 |
| License | CC BY 4.0 |
| Instances | 13,611 raw → **13,543** after removing 68 exact duplicate rows (all HOROZ) |
| Features | **16**, all numeric, no missing values |
| Target | `Class` — 7 bean varieties (multi-class) |

**Features** (12 dimension + 4 shape-form, all computed from the segmented grain image):
`Area` (pixel count inside the bean outline), `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRation`
(major/minor axis — spelled this way in the original data), `Eccentricity`, `ConvexArea`, `EquivDiameter`
(diameter of a circle with the same area), `Extent`, `Solidity` (area / convex area), `roundness`, `Compactness`,
`ShapeFactor1`–`ShapeFactor4`.

**Class distribution after de-duplication** (imbalanced — largest class is ~6.8× the smallest):

| DERMASON | SIRA | SEKER | HOROZ | CALI | BARBUNYA | BOMBAY |
|---|---|---|---|---|---|---|
| 3546 | 2636 | 2027 | 1860 | 1630 | 1322 | 522 |

**Preparation.** `data/dry_bean.csv` is a straight conversion of the UCI `.arff` file. Duplicates are dropped
*before* splitting (so the same bean can't be in both train and test), then a **stratified 80/20 train/test split**
with seed **5327** (from my student ID) gives 10,836 training rows and **2,707 test rows**. The test rows are
saved as `test_data.csv` — that's the file to upload in the app.

Worth knowing: several features are nearly copies of each other (`Area` vs `ConvexArea` r = 0.9999,
`Compactness` vs `ShapeFactor3` r = 0.999 — see the correlation heatmap in the notebook), and size features are
strongly right-skewed (`Area` skew ≈ 2.95). Standardisation is fit on the training split only and stored inside
each saved model.

## c. GitHub repository link

**https://github.com/YOUR-USERNAME/drybean-ml**

```
drybean-ml/
├── app.py                      Streamlit app (loads saved models, never retrains)
├── requirements.txt            pinned versions used to save the pickles
├── README.md
├── test_data.csv               the 20% hold-out split (16 features + Class)
├── train_and_evaluate.ipynb    EDA + training + evaluation (run on BITS Virtual Lab)
├── train.py                    same pipeline from the command line: python train.py [--quick]
├── data/dry_bean.csv           full dataset (converted from the UCI .arff)
├── beanml/                     the from-scratch ML library
│   ├── preprocessing.py        scaler, label encoder, de-dup, stratified split / k-fold
│   ├── metrics.py              confusion matrix, accuracy, P/R/F1, MCC, ROC, AUC, report
│   ├── linear.py               softmax regression, linear SVM, averaged perceptron
│   ├── tree.py                 CART classification + regression trees
│   ├── neighbors.py            KNN
│   ├── bayes.py                Gaussian Naive Bayes
│   ├── svm.py                  kernel SVM (SMO solver, RBF kernel, one-vs-one)
│   ├── discriminant.py         LDA, QDA
│   ├── ensemble.py             random forest, AdaBoost (SAMME), gradient boosting, soft voting
│   ├── neural.py               MLP
│   ├── tuning.py               grid search with stratified k-fold CV
│   ├── experiment.py           the full train/evaluate run used by train.py and the notebook
│   ├── validation.py           checks on uploaded CSVs for the app
│   └── io.py                   pickle/json helpers
└── model/                      saved models (*.pkl) + meta.json, cv_results.json, test_metrics.json
```

## d. Models used

All models below are my own NumPy implementations; metrics come from `beanml/metrics.py`.
Because this is a 7-class problem, **Precision, Recall and F1 are macro averages** (computed per class, then
averaged, so BOMBAY counts as much as DERMASON), **AUC is macro one-vs-rest ROC-AUC** (rank / Mann-Whitney formula
with proper tie handling; uses predicted probabilities, or raw decision scores for the SVM and perceptron), and
**MCC is the multiclass Matthews correlation** (Gorodkin's generalisation). All numbers are on the
**2,707-row held-out test set** (`test_data.csv`), which was never used for tuning.

### Comparison table — required models

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9250 | 0.9944 | 0.9359 | 0.9355 | 0.9356 | 0.9093 |
| Decision Tree | 0.8969 | 0.9777 | 0.9069 | 0.9056 | 0.9059 | 0.8755 |
| KNN | 0.9224 | 0.9927 | 0.9377 | 0.9321 | 0.9347 | 0.9061 |
| Naive Bayes | 0.8977 | 0.9907 | 0.9073 | 0.9053 | 0.9058 | 0.8765 |
| Support Vector Machine | 0.9346 | 0.9898 | 0.9457 | 0.9430 | 0.9443 | 0.9209 |

### Extra models (bonus)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Linear SVM | 0.9128 | 0.9894 | 0.9257 | 0.9218 | 0.9237 | 0.8945 |
| Averaged Perceptron | 0.9261 | 0.9263 | 0.9355 | 0.9374 | 0.9363 | 0.9107 |
| LDA | 0.9040 | 0.9919 | 0.9294 | 0.9129 | 0.9183 | 0.8855 |
| QDA | 0.9213 | 0.9920 | 0.9323 | 0.9337 | 0.9329 | 0.9049 |
| Random Forest | 0.9235 | 0.9936 | 0.9358 | 0.9307 | 0.9331 | 0.9075 |
| AdaBoost (SAMME) | 0.8921 | 0.9800 | 0.9079 | 0.9065 | 0.9057 | 0.8703 |
| Gradient Boosting | 0.9291 | 0.9944 | 0.9408 | 0.9350 | 0.9378 | 0.9142 |
| MLP Neural Net | 0.9313 | 0.9954 | 0.9427 | 0.9405 | 0.9414 | 0.9169 |
| Soft Voting Ensemble | 0.9339 | 0.9955 | 0.9465 | 0.9431 | 0.9447 | 0.9200 |

### Implementation and tuning details

Hyper-parameters were picked by grid search with **stratified k-fold CV on the training 80% only**, scored by
macro-F1; Random Forest, AdaBoost, Gradient Boosting and the Perceptron use fixed settings (GB early-stops on an
inner validation slice instead). The final model is refit on the full training split. Every model is saved as
`Pipeline(scaler, model)` (trees skip the scaler).

| Model | Implementation notes | Chosen hyper-parameters (CV macro-F1) |
|---|---|---|
| Logistic Regression | multinomial softmax, L2 penalty (not on bias), mini-batch gradient descent with momentum, stops once the loss flattens | lr 0.1, L2 0 (0.9357, 5-fold) |
| Decision Tree | CART, exact best-split search with vectorised cumulative class counts, splits only between distinct values, flat-array storage | entropy, max_depth 10, min_samples_leaf 5 (0.9186) |
| KNN | standardised features, chunked distance matrix, distance-weighted voting | k = 15, distance weights, Euclidean (0.9367) |
| Naive Bayes | Gaussian NB in log space with variance smoothing | var_smoothing 0.01 (0.9059) |
| SVM | RBF-kernel SVM solved with **SMO using second-order working-set selection** (as in libsvm), one-vs-one over all 21 class pairs, no subsampling | C = 1000, γ = 0.003 (0.9407, 3-fold) |

The SVM's CV surface is a flat ridge running from high-C/low-γ to low-C/high-γ. (C=100, γ=0.01) scores exactly the same 0.9407,
and going further along the ridge doesn't help, so the corner pick isn't hiding a better setting.
Extras: Linear SVM (one-vs-rest primal hinge loss), averaged perceptron, LDA/QDA with covariance shrinkage,
Random Forest (100 trees, √d features per split), AdaBoost-SAMME (150 depth-2 trees), softmax gradient boosting
(Newton leaf values, early-stopped at 92 rounds), a 2-layer ReLU MLP trained with Adam, and a soft-voting ensemble
of LR + KNN + RF + GB + MLP.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A strong baseline: **0.9356 macro-F1 and the joint-best AUC among the required models (0.9944)** — its probabilities rank classes very well even when the top pick is wrong. The shape features separate most varieties close to linearly once standardised. Almost all of its errors are SIRA↔DERMASON (40 SIRA→DERMASON, 38 the other way), two varieties of similar size and shape. L2 regularisation made no real difference in CV (λ = 0 to 1e-3 all within 0.001), which is expected with 10.8k rows and only 16 features. |
| Decision Tree | The **weakest required model** (F1 0.9059, MCC 0.8755). Even tuned (depth 10, min 5 samples per leaf, 523 nodes), axis-aligned splits approximate the curved class boundaries with staircases; SIRA recall drops to 0.805 with 66 SIRA beans called DERMASON. Its AUC (0.978) is also clearly lower because leaf probabilities are coarse and tie heavily. As one of the extras shows, averaging 100 trees (Random Forest) recovers +0.027 F1 — the problem is the single tree's variance, not the features. |
| KNN | Very close to Logistic Regression (F1 0.9347, AUC 0.9927). Standardising the features was essential, since `Area` is in the tens of thousands while `ShapeFactor2` is ~0.001. CV preferred a fairly large neighbourhood (k = 15) with distance weighting, which smooths the noisy SIRA/DERMASON border. Perfect on BOMBAY. The main cost is prediction time: it compares every test bean with all 10,836 training beans. |
| Naive Bayes | Ties with the decision tree (F1 0.9058) but for a completely different reason: the **independence assumption is badly violated** — many features are near-duplicates (Area/ConvexArea r = 0.9999), so the same evidence is counted several times and the posteriors become over-confident. BARBUNYA suffers most (recall 0.788, often confused with CALI). Ranking quality is still good (AUC 0.9907). Interestingly, CV preferred a large smoothing value (0.01 rather than 1e-9), which softens that over-confidence. QDA, which models the full covariance, gets F1 0.9329 — showing exactly what NB loses by ignoring correlations. |
| Support Vector Machine | The **best of the five required models** on Accuracy (0.9346), F1 (0.9443) and MCC (0.9209) — the highest MCC of all 14 models. The RBF kernel draws smooth curved boundaries, which is exactly what the SIRA/DERMASON overlap needs: SIRA recall 0.884 and DERMASON 0.946, the best pair among the required models. Its AUC (0.9898) is slightly below LR/KNN because one-vs-one votes are a cruder ranking score than real probabilities, but it makes the fewest mistakes of all 14 models (177 of 2,707). Linear SVM (F1 0.9237) shows how much the kernel adds. |
| **Overall, Winner for your dataset?** | **Among the 5 required models: the RBF-kernel SVM** — top Accuracy, F1 and MCC, with Logistic Regression a close second and the best AUC. **Across all 14 models**, the soft-voting ensemble edges ahead on macro-F1 (0.9447 vs 0.9443), but the SVM still has higher accuracy and MCC; the gap is within one or two beans out of 2,707, so in practice they're tied, and a single SVM is simpler than a five-model ensemble — I'd pick the **SVM** for this dataset. Across every model the hard part is the same: SIRA↔DERMASON alone is 32–42% of every model's errors, while BOMBAY is recognised essentially perfectly (recall 1.0 for most models) because it's far larger than every other bean. |

## Streamlit app

**https://YOUR-APP-NAME.streamlit.app**

Controls at the top of the page: **① upload a test CSV** (the bundled `test_data.csv` is used if nothing is
uploaded, and there's a button to download it) and **② a model dropdown** with all 14 models (★ = required).
Everything else is on one page with a jump bar: metric cards for all six metrics (plus weighted averages),
**confusion matrix**, one-vs-rest ROC curves, **per-class classification report**, a leaderboard of all 14 models on
the test set (with a re-score-on-your-upload button), and a predictions table with download and a mistakes-only
filter. The app only **loads** the pickles in `model/` — it never trains anything. Uploaded files are checked for
missing columns, unknown labels and non-numeric values, with a clear message for each.

## How to run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py               # the app
python train.py --quick            # retrain with the saved best params (~1 min)
python train.py                    # full retrain incl. CV grid search (~2-3 min)
```

The pickles were saved with Python 3.12 / NumPy 2.5 — on Streamlit Cloud pick **Python 3.12** under *Advanced
settings*.

An AI assistant was used as a coding and learning aid while building this project.
