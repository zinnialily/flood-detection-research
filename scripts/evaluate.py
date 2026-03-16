"""
evaluate.py -- Comprehensive evaluation for any saved flood detection model.

PR-AUC is the PRIMARY discrimination metric (see Davis & Goadrich, 2006).
Produces metrics tables, bootstrap CIs, confusion matrix, PR/ROC curves,
severity-stratified recall, swimming-pool FP analysis with Clopper-Pearson
CIs, and optional McNemar's test for pairwise model comparison.

Compute: Colab T4 GPU for inference; CPU for metrics and plots.
Dependencies: tensorflow, numpy, pandas, scikit-learn, scipy, matplotlib,
              seaborn.  statsmodels is optional (McNemar fallback included).
"""

import argparse
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from utils import PREPROCESS_FN, set_all_seeds  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a saved flood detection model.  PR-AUC is the "
        "primary metric."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved .keras model file.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["efficientnet", "resnet50"],
        help="Backbone architecture (efficientnet or resnet50).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/FloodingDataset2",
        help="Root directory of the dataset (default: ./data/FloodingDataset2).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save evaluation outputs (default: ./results).",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for CIs (default: 1000).",
    )
    parser.add_argument(
        "--compare_predictions_csv",
        type=str,
        default=None,
        help="Path to another model's predictions CSV for McNemar's test.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap 95 % confidence interval for a metric.

    Args:
        y_true: Ground-truth labels.
        y_score: Predicted probabilities or binary predictions.
        metric_fn: Callable(y_true, y_score) -> float.
        n: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (lower_bound, upper_bound) at 2.5 / 97.5 percentiles.
    """
    rng = np.random.RandomState(seed)
    scores: List[float] = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_score[idx]))
        except (ValueError, ZeroDivisionError):
            # Can happen when a resample has a single class.
            continue
    if len(scores) == 0:
        return (np.nan, np.nan)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ---------------------------------------------------------------------------
# Clopper-Pearson exact binomial CI
# ---------------------------------------------------------------------------


def clopper_pearson_ci(
    k: int, n: int, alpha: float = 0.05
) -> Tuple[float, float]:
    """Clopper-Pearson exact binomial confidence interval.

    Args:
        k: Number of successes (e.g. false positives).
        n: Number of trials (e.g. total pool images).
        alpha: Significance level (default 0.05 for 95 % CI).

    Returns:
        Tuple (lower, upper) bounds on the true rate.
    """
    from scipy.stats import beta as beta_dist

    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------


def mcnemar_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    alpha: float = 0.05,
    n_comparisons: int = 1,
) -> Dict[str, float]:
    """Run McNemar's test with continuity correction.

    Attempts to use statsmodels; falls back to manual scipy implementation.

    Args:
        correct_a: Boolean array -- True where model A was correct.
        correct_b: Boolean array -- True where model B was correct.
        alpha: Nominal significance level before Bonferroni correction.
        n_comparisons: Number of pairwise comparisons for Bonferroni.

    Returns:
        Dict with keys: b, c, chi2, p_value, adjusted_alpha, significant.
    """
    # b: A correct, B wrong; c: A wrong, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    try:
        from statsmodels.stats.contingency_tables import mcnemar as _mcnemar

        table = np.array(
            [
                [int(np.sum(correct_a & correct_b)), b],
                [c, int(np.sum(~correct_a & ~correct_b))],
            ]
        )
        result = _mcnemar(table, exact=False, correction=True)
        chi2 = float(result.statistic)
        p_value = float(result.pvalue)
    except ImportError:
        from scipy.stats import chi2 as chi2_dist

        if (b + c) > 0:
            chi2 = float((abs(b - c) - 1) ** 2 / (b + c))
        else:
            chi2 = 0.0
        p_value = float(chi2_dist.sf(chi2, df=1))

    adjusted_alpha = alpha / n_comparisons
    return {
        "b": b,
        "c": c,
        "chi2": chi2,
        "p_value": p_value,
        "adjusted_alpha": adjusted_alpha,
        "significant": p_value < adjusted_alpha,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> None:
    """Plot and save a confusion matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Flood", "Flood"],
        yticklabels=["Non-Flood", "Flood"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (flood = positive class)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _save_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pr_auc: float,
    save_path: str,
) -> None:
    """Plot and save the Precision-Recall curve (primary figure)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, precision_score, recall_score

    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_prob)

    # Operating point at threshold = 0.5
    y_pred_05 = (y_prob >= 0.5).astype(int)
    op_prec = precision_score(y_true, y_pred_05, zero_division=0)
    op_rec = recall_score(y_true, y_pred_05, zero_division=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall_vals, precision_vals, lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.plot(
        op_rec,
        op_prec,
        marker="*",
        markersize=15,
        color="red",
        label=f"Threshold = 0.5 (R={op_rec:.3f}, P={op_prec:.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Primary Metric)")
    ax.legend(loc="lower left")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _save_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    roc_auc: float,
    save_path: str,
) -> None:
    """Plot and save the ROC curve."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        "ROC Curve\n"
        "(ROC-AUC may be optimistic under class imbalance -- prefer PR-AUC)"
    )
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _save_severity_recall(
    severity_results: Dict[str, Dict[str, float]],
    save_path: str,
) -> None:
    """Plot and save severity-stratified recall bar chart."""
    import matplotlib.pyplot as plt

    labels = list(severity_results.keys())
    recalls = [severity_results[s]["recall"] for s in labels]
    counts = [severity_results[s]["total"] for s in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, recalls, color=["#d32f2f", "#f57c00", "#fbc02d"])
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Recall")
    ax.set_title("Flood Recall by Severity Level")
    ax.set_ylim([0.0, 1.1])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Severity-stratified recall
# ---------------------------------------------------------------------------


def _compute_severity_recall(
    filenames: List[str],
    y_true_flood: np.ndarray,
    y_pred_flood: np.ndarray,
    data_dir: str,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Compute recall stratified by flood severity.

    Attempts to load category information from test_split.csv first; falls
    back to parsing filename prefixes.

    Returns:
        Dict mapping severity name -> {recall, correct, total}, or None if
        severity information is unavailable.
    """
    severity_map: Dict[int, str] = {}
    severity_levels = {"MajorFlood", "ModerateFlood", "MinorFlood"}

    # Strategy 1: Look for test_split.csv with a 'category' column.
    split_csv = os.path.join(data_dir, "test_split.csv")
    if os.path.exists(split_csv):
        df = pd.read_csv(split_csv)
        if "category" in df.columns and "path" in df.columns:
            fname_to_cat = {}
            for _, row in df.iterrows():
                fname = os.path.basename(str(row["path"]))
                fname_to_cat[fname] = str(row["category"])
            for idx, fpath in enumerate(filenames):
                fname = os.path.basename(fpath)
                cat = fname_to_cat.get(fname, "")
                if cat in severity_levels:
                    severity_map[idx] = cat

    # Strategy 2: Parse from filename prefix (before first underscore or dot).
    if len(severity_map) == 0:
        for idx, fpath in enumerate(filenames):
            fname = os.path.basename(fpath)
            for sev in severity_levels:
                if sev.lower() in fname.lower():
                    severity_map[idx] = sev
                    break

    if len(severity_map) == 0:
        return None

    results: Dict[str, Dict[str, float]] = {}
    for sev in sorted(severity_levels):
        indices = [i for i, s in severity_map.items() if s == sev]
        if len(indices) == 0:
            continue
        n_total = len(indices)
        n_correct = int(
            np.sum(
                y_pred_flood[indices][y_true_flood[indices] == 1]
                == 1
            )
        )
        n_flood = int(np.sum(y_true_flood[indices] == 1))
        recall = n_correct / n_flood if n_flood > 0 else float("nan")
        results[sev] = {"recall": recall, "correct": n_correct, "total": n_flood}

    return results if len(results) > 0 else None


# ---------------------------------------------------------------------------
# Swimming pool FP analysis
# ---------------------------------------------------------------------------


def _pool_fp_analysis(
    filenames: List[str],
    y_true_flood: np.ndarray,
    y_pred_flood: np.ndarray,
) -> Optional[Dict]:
    """Analyse false positive rate on swimming pool images.

    Pool images are identified by 'swimmingpool' (case-insensitive) in path.

    Returns:
        Dict with pool analysis results, or None if no pool images found.
    """
    pool_mask = np.array(
        ["swimmingpool" in f.lower() for f in filenames], dtype=bool
    )
    n_pool = int(pool_mask.sum())
    if n_pool == 0:
        return None

    # Pool images are non-flood (y_true_flood==0). FP = predicted as flood.
    pool_pred_flood = y_pred_flood[pool_mask]
    n_fp = int(pool_pred_flood.sum())
    rate = n_fp / n_pool

    lo, hi = clopper_pearson_ci(n_fp, n_pool)
    return {
        "n_pool": n_pool,
        "n_fp": n_fp,
        "rate": rate,
        "ci_lo": lo,
        "ci_hi": hi,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def main() -> None:
    """Run full evaluation pipeline."""
    args = parse_args()
    set_all_seeds(42)

    # Deferred heavy imports so argparse --help is fast.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F811
    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # ------------------------------------------------------------------
    # Startup assertions
    # ------------------------------------------------------------------
    data_dir = args.data_dir
    test_dir = os.path.join(data_dir, "processed_data", "binary", "test")
    assert "binary_hnm" not in str(test_dir), (
        f"evaluate.py must load from original test partition, got: {test_dir}"
    )
    assert os.path.exists(test_dir), f"Test directory not found: {test_dir}"

    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    print(f"\n{'=' * 60}")
    print(f"  Evaluation: {model_name}")
    print(f"  Architecture: {args.arch}")
    print(f"  Test dir: {test_dir}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Create output directories
    # ------------------------------------------------------------------
    for subdir in ["tables", "predictions", "figures"]:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("Loading model ...")
    model = tf.keras.models.load_model(args.model_path)
    print(f"  Model loaded: {args.model_path}")
    print(f"  Total parameters: {model.count_params():,}")

    # ------------------------------------------------------------------
    # Load test data -- NO rescale; preprocessing via PREPROCESS_FN
    # ------------------------------------------------------------------
    preprocess_fn = PREPROCESS_FN[args.arch]
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        shuffle=False,
    )

    # Class mapping: flood=0, non_flood=1 (alphabetical).
    print(f"  Class indices: {test_gen.class_indices}")
    print(f"  Test samples: {test_gen.samples}")

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    print("\nRunning inference ...")
    y_true = test_gen.classes  # 0=flood, 1=non_flood
    y_prob_nonflood = model.predict(test_gen, verbose=1).flatten()  # P(non_flood)
    y_prob = 1.0 - y_prob_nonflood  # P(flood)

    # Convert to flood-positive convention (flood=1).
    y_true_flood = (y_true == 0).astype(int)
    y_pred_flood = (y_prob >= 0.5).astype(int)

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------
    pr_auc = average_precision_score(y_true_flood, y_prob)
    recall = recall_score(y_true_flood, y_pred_flood, zero_division=0)
    precision = precision_score(y_true_flood, y_pred_flood, zero_division=0)
    f1 = f1_score(y_true_flood, y_pred_flood, zero_division=0)
    roc_auc = roc_auc_score(y_true_flood, y_prob)
    accuracy = accuracy_score(y_true_flood, y_pred_flood)

    # ------------------------------------------------------------------
    # Bootstrap CIs
    # ------------------------------------------------------------------
    print(f"\nComputing bootstrap CIs ({args.n_bootstrap} resamples) ...")
    pr_auc_ci = bootstrap_ci(
        y_true_flood, y_prob, average_precision_score, n=args.n_bootstrap
    )
    recall_ci = bootstrap_ci(
        y_true_flood,
        y_pred_flood,
        lambda yt, yp: recall_score(yt, yp, zero_division=0),
        n=args.n_bootstrap,
    )
    f1_ci = bootstrap_ci(
        y_true_flood,
        y_pred_flood,
        lambda yt, yp: f1_score(yt, yp, zero_division=0),
        n=args.n_bootstrap,
    )
    roc_auc_ci = bootstrap_ci(
        y_true_flood, y_prob, roc_auc_score, n=args.n_bootstrap
    )

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print(f"\n=== Evaluation: {model_name} ===")
    print("PRIMARY METRIC")
    print(
        f"  PR-AUC:    {pr_auc:.4f}   "
        f"[bootstrap 95% CI: {pr_auc_ci[0]:.4f}\u2013{pr_auc_ci[1]:.4f}]"
    )
    print("\nSECONDARY METRICS")
    print(
        f"  Recall:    {recall:.4f}   "
        f"[CI: {recall_ci[0]:.4f}\u2013{recall_ci[1]:.4f}]"
    )
    print(f"  Precision: {precision:.4f}")
    print(
        f"  F1:        {f1:.4f}        "
        f"[CI: {f1_ci[0]:.4f}\u2013{f1_ci[1]:.4f}]"
    )
    print(
        f"  ROC-AUC:   {roc_auc:.4f}  "
        f"[CI: {roc_auc_ci[0]:.4f}\u2013{roc_auc_ci[1]:.4f}]  "
        "* See Davis & Goadrich (2006) re: imbalance"
    )
    print(f"  Accuracy:  {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true_flood,
            y_pred_flood,
            target_names=["Non-Flood", "Flood"],
        )
    )

    # ------------------------------------------------------------------
    # Swimming pool FP analysis
    # ------------------------------------------------------------------
    filenames = test_gen.filenames
    pool_result = _pool_fp_analysis(filenames, y_true_flood, y_pred_flood)
    if pool_result is not None:
        print("\nSWIMMING POOL FALSE POSITIVE ANALYSIS")
        print(
            f"  Pool FP rate: {pool_result['n_fp']}/{pool_result['n_pool']} "
            f"= {pool_result['rate']:.1%}, "
            f"95% CI: [{pool_result['ci_lo']:.1%}, {pool_result['ci_hi']:.1%}]"
        )
    else:
        print("\n  [WARN] No swimming pool images found in test set.")

    # ------------------------------------------------------------------
    # McNemar's test (if comparison CSV provided)
    # ------------------------------------------------------------------
    if args.compare_predictions_csv is not None:
        print("\nMcNEMAR'S TEST")
        compare_path = args.compare_predictions_csv
        assert os.path.exists(compare_path), (
            f"Comparison CSV not found: {compare_path}"
        )
        compare_df = pd.read_csv(compare_path)

        # Build current predictions keyed by filename.
        current_correct = {}
        for idx, fpath in enumerate(filenames):
            fname = os.path.basename(fpath)
            current_correct[fname] = bool(y_pred_flood[idx] == y_true_flood[idx])

        # Align with comparison predictions.
        compare_correct = {}
        for _, row in compare_df.iterrows():
            fname = os.path.basename(str(row["filename"]))
            compare_correct[fname] = bool(row["correct"])

        common_files = sorted(
            set(current_correct.keys()) & set(compare_correct.keys())
        )
        if len(common_files) == 0:
            print("  [WARN] No overlapping filenames -- cannot run McNemar.")
        else:
            correct_a = np.array([current_correct[f] for f in common_files])
            correct_b = np.array([compare_correct[f] for f in common_files])

            compare_model = os.path.splitext(
                os.path.basename(compare_path)
            )[0].replace("_predictions", "")

            result = mcnemar_test(
                correct_a, correct_b, alpha=0.05, n_comparisons=3
            )
            sig_str = "significant" if result["significant"] else "NOT significant"
            print(
                f"  McNemar test vs {compare_model}: "
                f"chi2={result['chi2']:.4f}, p={result['p_value']:.6f} "
                f"({sig_str} at alpha=0.05 with Bonferroni, "
                f"adjusted alpha={result['adjusted_alpha']:.4f})"
            )
            print(
                f"  Discordant pairs: b={result['b']} "
                f"(current correct, compare wrong), "
                f"c={result['c']} (current wrong, compare correct)"
            )

    # ------------------------------------------------------------------
    # Fisher exact test (pool FP -- only with comparison)
    # ------------------------------------------------------------------
    if pool_result is not None and args.compare_predictions_csv is not None:
        from scipy.stats import fisher_exact

        # Load compare model's pool FP count from its predictions CSV.
        compare_df_full = pd.read_csv(args.compare_predictions_csv)
        compare_pool = compare_df_full[
            compare_df_full["filename"]
            .str.lower()
            .str.contains("swimmingpool", na=False)
        ]
        if len(compare_pool) > 0:
            compare_pool_fp = int(
                (compare_pool["predicted_label"] == 1).sum()
                if "predicted_label" in compare_pool.columns
                else 0
            )
            compare_pool_n = len(compare_pool)

            # Contingency: [[correct_compare, fp_compare], [correct_current, fp_current]]
            table = [
                [compare_pool_n - compare_pool_fp, compare_pool_fp],
                [
                    pool_result["n_pool"] - pool_result["n_fp"],
                    pool_result["n_fp"],
                ],
            ]
            odds_ratio, p_value = fisher_exact(table, alternative="greater")
            sig_str = "Significant" if p_value < 0.05 else "NOT significant"
            print(
                f"\n  Fisher exact test (pool FP, one-sided): p = {p_value:.4f}"
            )
            print(f"    {sig_str} at alpha = 0.05")

    # ------------------------------------------------------------------
    # Severity-stratified recall
    # ------------------------------------------------------------------
    severity_results = _compute_severity_recall(
        filenames, y_true_flood, y_pred_flood, data_dir
    )
    if severity_results is not None:
        print("\nSEVERITY-STRATIFIED RECALL (flood images only)")
        for sev, vals in severity_results.items():
            print(
                f"  {sev:15s}: recall={vals['recall']:.4f}  "
                f"({vals['correct']}/{vals['total']})"
            )
    else:
        print(
            "\n  [WARN] Severity information not available -- "
            "could not stratify recall. "
            "Checked test_split.csv and filename patterns."
        )

    # ------------------------------------------------------------------
    # Save predictions CSV
    # ------------------------------------------------------------------
    pred_rows = []
    for idx, fpath in enumerate(filenames):
        fname = os.path.basename(fpath)
        # Determine category from directory path or filename.
        category = ""
        for sev in ["MajorFlood", "ModerateFlood", "MinorFlood"]:
            if sev.lower() in fname.lower():
                category = sev
                break
        if "swimmingpool" in fname.lower():
            category = "SwimmingPool"

        pred_rows.append(
            {
                "filename": fname,
                "true_label": int(y_true_flood[idx]),
                "predicted_label": int(y_pred_flood[idx]),
                "flood_probability": float(y_prob[idx]),
                "correct": bool(y_pred_flood[idx] == y_true_flood[idx]),
                "is_swimming_pool": "swimmingpool" in fname.lower(),
                "category": category,
            }
        )
    pred_df = pd.DataFrame(pred_rows)
    pred_path = os.path.join(
        args.output_dir, "predictions", f"{model_name}_predictions.csv"
    )
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  Predictions saved: {pred_path}")

    # ------------------------------------------------------------------
    # Save metrics CSV
    # ------------------------------------------------------------------
    metrics_dict = {
        "model": model_name,
        "architecture": args.arch,
        "pr_auc": pr_auc,
        "pr_auc_ci_lo": pr_auc_ci[0],
        "pr_auc_ci_hi": pr_auc_ci[1],
        "recall": recall,
        "recall_ci_lo": recall_ci[0],
        "recall_ci_hi": recall_ci[1],
        "precision": precision,
        "f1": f1,
        "f1_ci_lo": f1_ci[0],
        "f1_ci_hi": f1_ci[1],
        "roc_auc": roc_auc,
        "roc_auc_ci_lo": roc_auc_ci[0],
        "roc_auc_ci_hi": roc_auc_ci[1],
        "accuracy": accuracy,
        "n_test": len(y_true),
    }
    if pool_result is not None:
        metrics_dict["pool_fp_rate"] = pool_result["rate"]
        metrics_dict["pool_fp_count"] = pool_result["n_fp"]
        metrics_dict["pool_n"] = pool_result["n_pool"]
        metrics_dict["pool_fp_ci_lo"] = pool_result["ci_lo"]
        metrics_dict["pool_fp_ci_hi"] = pool_result["ci_hi"]

    metrics_df = pd.DataFrame([metrics_dict])
    metrics_path = os.path.join(
        args.output_dir, "tables", f"{model_name}_metrics.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Metrics saved: {metrics_path}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\nGenerating figures ...")

    _save_confusion_matrix(
        y_true_flood,
        y_pred_flood,
        os.path.join(
            args.output_dir, "figures", f"{model_name}_confusion_matrix.png"
        ),
    )

    _save_pr_curve(
        y_true_flood,
        y_prob,
        pr_auc,
        os.path.join(
            args.output_dir, "figures", f"{model_name}_pr_curve.png"
        ),
    )

    _save_roc_curve(
        y_true_flood,
        y_prob,
        roc_auc,
        os.path.join(
            args.output_dir, "figures", f"{model_name}_roc_curve.png"
        ),
    )

    if severity_results is not None:
        _save_severity_recall(
            severity_results,
            os.path.join(
                args.output_dir, "figures", f"{model_name}_severity_recall.png"
            ),
        )

    print(f"\n{'=' * 60}")
    print(f"  Evaluation complete: {model_name}")
    print(f"  All outputs in: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
