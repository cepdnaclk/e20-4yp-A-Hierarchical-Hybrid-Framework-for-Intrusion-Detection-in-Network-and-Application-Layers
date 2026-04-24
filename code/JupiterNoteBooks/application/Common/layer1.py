"""
================================================================
  LAYER 1 — ANOMALY DETECTOR
  Hierarchical Application-Layer IDS
================================================================
  INPUT   : processed_dataset/balanced_application_data.csv
  LABELS  : Uses ONLY benign samples for training (one-class)
  MODELS  : Isolation Forest  (primary)
            One-Class SVM     (secondary / ensemble)
  SPLITS  : 70% train  |  10% validation  |  20% test
            Threshold tuned on validation — test never touched
  SCALER  : RobustScaler fitted on benign-only training data
  OUTPUT  : models/layer1_*
            results/layer1/*
================================================================
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import RobustScaler
from sklearn.ensemble        import IsolationForest
from sklearn.svm             import OneClassSVM
from sklearn.decomposition   import PCA
from sklearn.metrics         import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score,
)

warnings.filterwarnings("ignore")
np.random.seed(42)


# ================================================================
#  CONFIGURATION  ← only change these if needed
# ================================================================
DATASET_PATH = "/home/fyp_ids_e20/processed_dataset/balanced_application_data.csv"
LABEL_COLUMN = "Label"
BENIGN_LABEL = "Benign"

MODELS_DIR  = "models"
RESULTS_DIR = os.path.join("results", "layer1")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_SIZE   = 0.20    # held out completely until final evaluation
VAL_SIZE    = 0.10    # used only for threshold tuning
RANDOM_SEED = 42

# Isolation Forest
IF_N_ESTIMATORS  = 300    # more trees = more stable boundary
IF_CONTAMINATION = "auto" # sklearn estimates from training data
IF_MAX_SAMPLES   = 256    # 256 is optimal per the original IF paper

# One-Class SVM
OCSVM_KERNEL = "rbf"
OCSVM_NU     = 0.05   # upper bound on outlier fraction
OCSVM_GAMMA  = "scale"

# 19 application-layer features (intersection of MI + Pearson top-30)
APPLICATION_FEATURES = [
    "Fwd Header Len",    "Dst Port",          "TotLen Fwd Pkts",
    "Fwd Seg Size Avg",  "Pkt Size Avg",      "Init Fwd Win Byts",
    "Init Bwd Win Byts", "Fwd Seg Size Min",  "Subflow Fwd Byts",
    "Fwd Pkt Len Mean",  "Pkt Len Std",       "Pkt Len Mean",
    "Fwd Pkt Len Max",   "Subflow Fwd Pkts",  "Bwd Pkt Len Mean",
    "Tot Fwd Pkts",      "Bwd Seg Size Avg",  "Pkt Len Max",
    "Fwd Pkt Len Std",
]


# ================================================================
#  STEP 1 — LOAD DATASET
# ================================================================
def load_dataset():
    print("\n" + "="*62)
    print("  STEP 1 — Loading Dataset")
    print("="*62)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"\n  File not found: {DATASET_PATH}"
            "\n  Place balanced_application_data.csv inside processed_dataset/"
        )

    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df.columns       = df.columns.str.strip()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()

    print(f"\n  Path       : {DATASET_PATH}")
    print(f"  Rows       : {len(df):,}")
    print(f"  Columns    : {df.shape[1]}")

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found. "
                         f"Available: {list(df.columns)}")

    found   = [f for f in APPLICATION_FEATURES if f in df.columns]
    missing = [f for f in APPLICATION_FEATURES if f not in df.columns]
    print(f"\n  Features found   : {len(found)} / {len(APPLICATION_FEATURES)}")
    if missing:
        print(f"  Features missing : {missing}")
    if not found:
        raise ValueError("None of the 19 application features found in dataset.")

    counts = df[LABEL_COLUMN].value_counts()
    print("\n  Class distribution:")
    for cls, cnt in counts.items():
        tag = "  ← used for training (benign only)" if cls == BENIGN_LABEL else ""
        print(f"      {cls:<44}  {cnt:>8,}{tag}")

    n_b = (df[LABEL_COLUMN] == BENIGN_LABEL).sum()
    n_a = (df[LABEL_COLUMN] != BENIGN_LABEL).sum()
    print(f"\n  Benign : {n_b:,}  ({100*n_b/len(df):.1f}%)")
    print(f"  Attack : {n_a:,}  ({100*n_a/len(df):.1f}%)")
    return df, found


# ================================================================
#  STEP 2 — PREPARE FEATURES & SPLITS
#
#  Three-way stratified split:
#    Test (20%)        — completely held out, never seen until eval
#    Validation (10%)  — used only for threshold selection
#    Train (70%)       — fits the scaler and both models
#
#  Why stratify?
#    Ensures the class ratio (benign/attack) is the same in all
#    three splits, giving an unbiased evaluation.
# ================================================================
def prepare_data(df, feature_cols):
    print("\n" + "="*62)
    print("  STEP 2 — Feature Preparation & Three-Way Split")
    print("="*62)

    X = df[feature_cols].values.astype(np.float64)
    y = (df[LABEL_COLUMN] != BENIGN_LABEL).astype(int).values
    # y: 0 = Benign  |  1 = Attack

    stats_path = os.path.join(RESULTS_DIR, "feature_statistics.csv")
    pd.DataFrame(X, columns=feature_cols).describe().T.to_csv(stats_path)
    print(f"\n  Feature statistics saved  → {stats_path}")

    # First cut: hold out 20% as test set
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

    # Second cut: take 10% of total (= 12.5% of remaining) as validation
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, random_state=RANDOM_SEED, stratify=y_tv)

    print(f"\n  {'Split':<12} {'Total':>8}  {'Benign':>8}  {'Attack':>8}  {'Benign%':>8}")
    print(f"  {'─'*12} {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for name, X_, y_ in [("Train", X_train, y_train),
                          ("Validation", X_val, y_val),
                          ("Test", X_test, y_test)]:
        nb = (y_==0).sum(); na = (y_==1).sum()
        print(f"  {name:<12} {len(y_):>8,}  {nb:>8,}  {na:>8,}  {100*nb/len(y_):>7.1f}%")

    print(f"\n  Training benign-only samples available : {(y_train==0).sum():,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ================================================================
#  STEP 3 — SCALE FEATURES
#
#  Why RobustScaler instead of StandardScaler?
#    Network traffic features have extreme outliers (e.g., a single
#    flood packet can have byte counts 1000× the median). StandardScaler
#    uses mean and std, which are heavily influenced by outliers.
#    RobustScaler uses the median and IQR, making it resistant to
#    these extremes and producing better-calibrated anomaly scores.
#
#  Why fit on benign-only?
#    If we fit the scaler on all samples, the scale of attack
#    features influences the normalisation — leaking attack
#    information into the scaler before any training. Fitting
#    on benign samples only keeps the baseline pure.
# ================================================================
def scale_features(X_train, X_val, X_test, y_train):
    print("\n" + "="*62)
    print("  STEP 3 — Scaling  (RobustScaler, benign-only fit)")
    print("="*62)

    X_benign = X_train[y_train == 0]
    print(f"\n  Fitting on {len(X_benign):,} benign training samples only")
    print("  Using RobustScaler (median + IQR) — robust to traffic outliers")

    scaler    = RobustScaler()
    scaler.fit(X_benign)

    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    path = os.path.join(MODELS_DIR, "layer1_scaler.pkl")
    joblib.dump(scaler, path)
    print(f"  Scaler saved  → {path}")
    return X_train_s, X_val_s, X_test_s


# ================================================================
#  STEP 4a — TRAIN ISOLATION FOREST
#
#  How it works:
#    Builds an ensemble of random "isolation trees". Each tree
#    recursively partitions the feature space by randomly picking
#    a feature and a split value. Anomalous points (attacks) are
#    rare and distinct, so they get isolated near the root of the
#    tree with fewer splits. Normal points (benign) cluster together
#    and require many splits to isolate — they receive higher scores.
#
#  Why n_estimators=300?
#    More trees stabilise the anomaly score, especially important
#    when classes are imbalanced. 100 is the sklearn default; 300
#    gives noticeably more stable decision boundaries.
#
#  Why max_samples=256?
#    The original Isolation Forest paper shows that the algorithm
#    saturates quickly — 256 sub-samples per tree is often enough.
#    Using fewer samples per tree keeps training fast.
# ================================================================
def train_isolation_forest(X_train, y_train):
    print("\n" + "="*62)
    print("  STEP 4a — Isolation Forest  (benign only)")
    print("="*62)

    X_benign = X_train[y_train == 0]
    print(f"\n  Training on {len(X_benign):,} benign samples")
    print(f"  n_estimators={IF_N_ESTIMATORS}  "
          f"contamination={IF_CONTAMINATION}  "
          f"max_samples={IF_MAX_SAMPLES}")

    t0    = time.time()
    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        max_samples=IF_MAX_SAMPLES,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_benign)
    print(f"  ✔  Done in {time.time()-t0:.1f}s")

    path = os.path.join(MODELS_DIR, "layer1_isolation_forest.pkl")
    joblib.dump(model, path)
    print(f"  Saved  → {path}")
    return model


# ================================================================
#  STEP 4b — TRAIN ONE-CLASS SVM
#
#  How it works:
#    Maps the benign samples into a high-dimensional kernel space
#    using the RBF kernel, then finds the smallest hypersphere that
#    encloses most benign points. At inference, points outside this
#    boundary are flagged as anomalies (attacks).
#
#  Why combine with Isolation Forest?
#    IF and OC-SVM learn different geometric properties of the
#    data. IF detects points that are far from the cluster centers;
#    OC-SVM detects points outside the learned boundary surface.
#    Their ensemble catches anomalies that either model alone misses.
#
#  Why subsample to 50k?
#    OC-SVM training complexity is O(n²) — it becomes very slow on
#    large datasets. 50k samples is typically sufficient to learn a
#    reliable boundary while keeping training time acceptable.
# ================================================================
def train_ocsvm(X_train, y_train):
    print("\n" + "="*62)
    print("  STEP 4b — One-Class SVM  (benign only)")
    print("="*62)

    X_benign = X_train[y_train == 0]
    MAX_N    = 50_000
    if len(X_benign) > MAX_N:
        idx   = np.random.choice(len(X_benign), MAX_N, replace=False)
        X_fit = X_benign[idx]
        print(f"\n  Subsampled to {MAX_N:,} samples  (OC-SVM is O(n²) — keeps training fast)")
    else:
        X_fit = X_benign
        print(f"\n  Training on {len(X_fit):,} benign samples")
    print(f"  kernel={OCSVM_KERNEL}  nu={OCSVM_NU}  gamma={OCSVM_GAMMA}")

    t0    = time.time()
    model = OneClassSVM(kernel=OCSVM_KERNEL, nu=OCSVM_NU, gamma=OCSVM_GAMMA)
    model.fit(X_fit)
    print(f"  ✔  Done in {time.time()-t0:.1f}s")

    path = os.path.join(MODELS_DIR, "layer1_one_class_svm.pkl")
    joblib.dump(model, path)
    print(f"  Saved  → {path}")
    return model


# ================================================================
#  STEP 5 — THRESHOLD TUNING  (validation set only)
#
#  Both models produce a continuous score, not a hard 0/1 label.
#  The default sklearn threshold (0.0) is rarely optimal.
#
#  Strategy:
#    Use precision_recall_curve to evaluate every possible threshold,
#    compute F1 at each point, and select the threshold that gives
#    the best F1 on the VALIDATION set. The test set is never used
#    here — it stays completely unseen until Step 7.
#
#  Score direction:
#    Isolation Forest : lower raw score → more anomalous → attack
#    One-Class SVM    : lower raw score → outside boundary → attack
#    We negate both so that a HIGHER score means more likely attack,
#    which is the convention expected by precision_recall_curve.
# ================================================================
def tune_threshold(model, X_val, y_val, name, invert):
    raw   = model.decision_function(X_val)
    score = -raw if invert else raw   # higher = more likely attack

    prec, rec, thr = precision_recall_curve(y_val, score)
    f1s    = 2 * prec * rec / (prec + rec + 1e-9)
    best   = int(np.argmax(f1s[:-1]))
    best_t = float(thr[best])
    orig_t = -best_t if invert else best_t   # restore original score space
    f1v    = float(f1s[best])

    print(f"  {name:<22}  threshold = {orig_t:+.6f}   val-F1 = {f1v:.4f}")
    return orig_t, f1v, prec, rec, thr, best


# ================================================================
#  STEP 6 — ENSEMBLE PREDICTION  (majority vote)
#
#  Each model independently predicts 0 (benign) or 1 (attack).
#  Majority vote: if EITHER model flags a sample as attack,
#  the ensemble labels it attack. This is the "OR" rule and
#  maximises recall — i.e., it minimises missed attacks at the
#  cost of slightly more false positives compared to "AND".
#
#  For an IDS, missing a real attack (false negative) is usually
#  worse than a false alarm (false positive), so this trade-off
#  is appropriate.
# ================================================================
def ensemble_predict(iforest, ocsvm, X, if_thr, svm_thr):
    if_s  = iforest.decision_function(X)
    svm_s = ocsvm.decision_function(X)
    if_p  = (if_s  < if_thr ).astype(int)   # below threshold → attack
    svm_p = (svm_s < svm_thr).astype(int)
    ens_p = np.clip(if_p + svm_p, 0, 1)     # majority vote (OR rule)
    return if_p, svm_p, ens_p, if_s, svm_s


# ================================================================
#  STEP 7 — EVALUATE ON TEST SET
# ================================================================
def evaluate(y_test, if_p, svm_p, ens_p, if_s, svm_s):
    print("\n" + "="*62)
    print("  STEP 7 — Evaluation on Test Set")
    print("="*62)

    rows  = []
    lines = ["LAYER 1 — ANOMALY DETECTOR : EVALUATION REPORT\n" + "="*62 + "\n"]

    for name, preds, auc_sc in [
        ("Isolation Forest", if_p,  -if_s),
        ("One-Class SVM",    svm_p, -svm_s),
        ("Ensemble (Vote)",  ens_p, (-if_s + -svm_s) / 2),
    ]:
        rpt = classification_report(
            y_test, preds, target_names=["Benign", "Attack"], digits=4)
        print(f"\n  ── {name} ──\n")
        print(rpt)
        lines += [f"\n{'─'*40}\n  {name}\n{'─'*40}\n", rpt]

        try:
            auc = roc_auc_score(y_test, auc_sc)
            print(f"  ROC-AUC : {auc:.4f}")
            lines.append(f"  ROC-AUC : {auc:.4f}\n")
        except Exception:
            auc = None

        rows.append({
            "Model"    : name,
            "Accuracy" : round(float((y_test == preds).mean()), 4),
            "Precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "Recall"   : round(float(recall_score(y_test,    preds, zero_division=0)), 4),
            "F1"       : round(float(f1_score(y_test,        preds, zero_division=0)), 4),
            "ROC-AUC"  : round(auc, 4) if auc is not None else "N/A",
        })

    txt_path = os.path.join(RESULTS_DIR, "layer1_evaluation_report.txt")
    with open(txt_path, "w") as fh:
        fh.write("\n".join(lines))

    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(os.path.join(RESULTS_DIR, "layer1_model_summary.csv"), index=False)

    print("\n" + "─"*52)
    print(df_sum.to_string(index=False))
    print("─"*52)
    return rows


# ================================================================
#  STEP 8 — PLOTS  (7 diagnostic charts)
# ================================================================
def _save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  → {path}")


def generate_plots(y_test, if_p, svm_p, ens_p, if_s, svm_s,
                   if_thr, svm_thr, if_pr, svm_pr, X_test, feature_cols):
    print("\n" + "="*62)
    print("  STEP 8 — Generating Diagnostic Plots")
    print("="*62)

    # ── 1. Confusion matrices (3 models side-by-side) ────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (nm, p, cmap) in zip(axes, [
        ("Isolation Forest", if_p,  "Blues"),
        ("One-Class SVM",    svm_p, "Oranges"),
        ("Ensemble (Vote)",  ens_p, "Greens"),
    ]):
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, p), display_labels=["Benign", "Attack"]
        ).plot(ax=ax, colorbar=False, cmap=cmap)
        ax.set_title(f"{nm}\nF1 = {f1_score(y_test,p,zero_division=0):.4f}",
                     fontsize=11, fontweight="bold")
    plt.suptitle("Layer 1 — Confusion Matrices (Test Set)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "layer1_confusion_matrices.png")

    # ── 2. Score distributions with threshold marker ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (nm, sc, thr, cb, ca) in zip(axes, [
        ("Isolation Forest", if_s,  if_thr,  "steelblue", "tomato"),
        ("One-Class SVM",    svm_s, svm_thr, "darkorange", "purple"),
    ]):
        ax.hist(sc[y_test == 0], bins=100, alpha=0.65,
                color=cb, label="Benign", density=True)
        ax.hist(sc[y_test == 1], bins=100, alpha=0.65,
                color=ca, label="Attack", density=True)
        ax.axvline(thr, color="black", linestyle="--", lw=2,
                   label=f"Threshold = {thr:.4f}")
        ax.set_xlabel("Anomaly Score", fontsize=11)
        ax.set_ylabel("Density",       fontsize=11)
        ax.set_title(f"{nm}\nScore Distribution", fontweight="bold")
        ax.legend(fontsize=9)
    plt.suptitle("Layer 1 — Decision Score Distributions",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "layer1_score_distributions.png")

    # ── 3. Precision-Recall curves (validation set) ───────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (nm, (prec, rec, thr_arr, bi), col) in zip(axes, [
        ("Isolation Forest", if_pr,  "steelblue"),
        ("One-Class SVM",    svm_pr, "darkorange"),
    ]):
        ax.plot(rec[:-1], prec[:-1], color=col, lw=2)
        f1v = 2 * prec[bi] * rec[bi] / (prec[bi] + rec[bi] + 1e-9)
        ax.scatter(rec[bi], prec[bi], color="red", s=120, zorder=5,
                   label=f"Best threshold  (F1 = {f1v:.3f})")
        ax.set_xlabel("Recall",    fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title(f"{nm}\nPrecision-Recall Curve", fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    plt.suptitle("Layer 1 — PR Curves  (Validation Set)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "layer1_pr_curves.png")

    # ── 4. ROC curves ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for nm, sc, col in [
        ("Isolation Forest", -if_s,              "steelblue"),
        ("One-Class SVM",    -svm_s,             "darkorange"),
        ("Ensemble",         (-if_s + -svm_s)/2, "seagreen"),
    ]:
        try:
            fpr, tpr, _ = roc_curve(y_test, sc)
            auc         = roc_auc_score(y_test, sc)
            ax.plot(fpr, tpr, color=col, lw=2,
                    label=f"{nm}  (AUC = {auc:.3f})")
        except Exception:
            pass
    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("Layer 1 — ROC Curves", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, "layer1_roc_curves.png")

    # ── 5. Model comparison bar chart ─────────────────────────────
    model_names = ["Isolation\nForest", "One-Class\nSVM", "Ensemble\n(Vote)"]
    all_preds   = [if_p, svm_p, ens_p]
    metrics     = {
        "F1 Score" : [f1_score(y_test,p,zero_division=0)       for p in all_preds],
        "Precision": [precision_score(y_test,p,zero_division=0) for p in all_preds],
        "Recall"   : [recall_score(y_test,p,zero_division=0)    for p in all_preds],
    }
    x, w = np.arange(3), 0.25
    colors = ["steelblue", "darkorange", "seagreen"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (met, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + i*w, vals, w, label=met,
                      color=colors[i], edgecolor="white", alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
    ax.set_xticks(x + w)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Layer 1 — Model Comparison (Test Set)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, "layer1_model_comparison.png")

    # ── 6. PCA visualisation ──────────────────────────────────────
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X2d = pca.fit_transform(X_test)
    var = pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (lbl, title) in zip(axes, [
        (y_test, "True Labels"),
        (ens_p,  "Ensemble Predictions"),
    ]):
        ax.scatter(X2d[lbl==0, 0], X2d[lbl==0, 1],
                   c="steelblue", s=5, alpha=0.4, label="Benign")
        ax.scatter(X2d[lbl==1, 0], X2d[lbl==1, 1],
                   c="tomato",    s=5, alpha=0.4, label="Attack")
        ax.set_xlabel(f"PC1 ({100*var[0]:.1f}% var)", fontsize=10)
        ax.set_ylabel(f"PC2 ({100*var[1]:.1f}% var)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(markerscale=3, fontsize=9)
    plt.suptitle("Layer 1 — PCA Visualisation (Test Set)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "layer1_pca_visualisation.png")

    # ── 7. Feature correlation heatmap (benign test samples) ──────
    n = len(feature_cols)
    df_b = pd.DataFrame(X_test[y_test == 0], columns=feature_cols)
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(df_b.corr(),
                mask=np.triu(np.ones((n, n), dtype=bool)),
                annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.4, ax=ax, annot_kws={"size": 7})
    ax.set_title("Layer 1 — Feature Correlation  (Benign Samples Only)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "layer1_feature_correlation.png")

    print(f"\n  All 7 plots saved to  {RESULTS_DIR}/")


# ================================================================
#  MAIN
# ================================================================
def main():
    t_start = time.time()
    print("\n" + "="*62)
    print("  LAYER 1 — ANOMALY DETECTOR")
    print("  Hierarchical Application-Layer IDS")
    print("="*62)

    # 1. Load
    df, feature_cols = load_dataset()

    # 2. Prepare
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df, feature_cols)

    # 3. Scale (benign-only fit)
    X_train_s, X_val_s, X_test_s = scale_features(X_train, X_val, X_test, y_train)

    # 4. Train both models on benign samples only
    iforest = train_isolation_forest(X_train_s, y_train)
    ocsvm   = train_ocsvm(X_train_s, y_train)

    # 5. Tune thresholds on validation set (test set untouched)
    print("\n" + "="*62)
    print("  STEP 5 — Threshold Tuning  (Validation Set)")
    print("="*62)
    if_thr,  _, if_prec,  if_rec,  if_thr_arr,  if_bi  =         tune_threshold(iforest, X_val_s, y_val, "Isolation Forest", invert=True)
    svm_thr, _, svm_prec, svm_rec, svm_thr_arr, svm_bi =         tune_threshold(ocsvm,   X_val_s, y_val, "One-Class SVM",    invert=False)

    if_pr  = (if_prec,  if_rec,  if_thr_arr,  if_bi)
    svm_pr = (svm_prec, svm_rec, svm_thr_arr, svm_bi)

    # 6. Ensemble prediction on test set
    print("\n" + "="*62)
    print("  STEP 6 — Ensemble Prediction  (Test Set)")
    print("="*62)
    if_p, svm_p, ens_p, if_s, svm_s = ensemble_predict(
        iforest, ocsvm, X_test_s, if_thr, svm_thr)
    print(f"  Isolation Forest : {(if_p==1).sum():>7,} attacks  /  {(if_p==0).sum():>7,} benign")
    print(f"  One-Class SVM    : {(svm_p==1).sum():>7,} attacks  /  {(svm_p==0).sum():>7,} benign")
    print(f"  Ensemble (Vote)  : {(ens_p==1).sum():>7,} attacks  /  {(ens_p==0).sum():>7,} benign")
    print(f"  True labels      : {(y_test==1).sum():>7,} attacks  /  {(y_test==0).sum():>7,} benign")

    # 7. Evaluate
    evaluate(y_test, if_p, svm_p, ens_p, if_s, svm_s)

    # 8. Plots
    generate_plots(y_test, if_p, svm_p, ens_p, if_s, svm_s,
                   if_thr, svm_thr, if_pr, svm_pr, X_test_s, feature_cols)

    # 9. Save config
    print("\n" + "="*62)
    print("  STEP 9 — Saving Config & Thresholds")
    print("="*62)
    cfg = {
        "isolation_forest_threshold" : if_thr,
        "one_class_svm_threshold"    : svm_thr,
        "label_column"               : LABEL_COLUMN,
        "benign_label"               : BENIGN_LABEL,
        "features"                   : APPLICATION_FEATURES,
    }
    cfg_path = os.path.join(MODELS_DIR, "layer1_config.json")
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    with open(os.path.join(MODELS_DIR, "layer1_if_threshold.txt"),  "w") as fh:
        fh.write(str(if_thr))
    with open(os.path.join(MODELS_DIR, "layer1_svm_threshold.txt"), "w") as fh:
        fh.write(str(svm_thr))
    print(f"  Config saved  → {cfg_path}")

    elapsed = time.time() - t_start
    print(f"\n{'='*62}")
    print(f"  LAYER 1 COMPLETE  —  total time: {elapsed:.1f}s")
    print(f"{'='*62}")
    print("""
  ┌─ Models saved in  models/ ─────────────────────────────┐
  │  layer1_scaler.pkl              RobustScaler            │
  │  layer1_isolation_forest.pkl    Primary model           │
  │  layer1_one_class_svm.pkl       Secondary model         │
  │  layer1_config.json             Thresholds + features   │
  │  layer1_if_threshold.txt                                │
  │  layer1_svm_threshold.txt                               │
  ├─ Results saved in  results/layer1/ ────────────────────┤
  │  layer1_confusion_matrices.png  3 models side-by-side  │
  │  layer1_score_distributions.png Scores + threshold line │
  │  layer1_pr_curves.png           Precision-Recall        │
  │  layer1_roc_curves.png          ROC + AUC               │
  │  layer1_model_comparison.png    F1 / Prec / Recall bars │
  │  layer1_pca_visualisation.png   2D decision quality     │
  │  layer1_feature_correlation.png Feature heatmap         │
  │  layer1_evaluation_report.txt   Full classification rpt │
  │  layer1_model_summary.csv       Metrics table           │
  │  feature_statistics.csv         Descriptive statistics  │
  └────────────────────────────────────────────────────────┘

  Next  →  run  layer2_known_unknown.py
""")


if __name__ == "__main__":
    main()