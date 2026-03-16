"""
grad_cam.py -- GradCAM++ visualization for flood detection models.

Generates heatmap overlays for four image sets extracted from a predictions
CSV produced by evaluate.py:
  - false_negatives: floods the model missed
  - false_positives: non-flood images predicted as flood
  - pool_images: swimming pool images (hard negatives)
  - random_correct: randomly sampled correct predictions

Compute: Colab T4 GPU for inference; CPU for plotting.
Dependencies: tensorflow, numpy, pandas, matplotlib, tf-keras-vis.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from utils import PREPROCESS_FN, set_all_seeds  # noqa: E402

# ---------------------------------------------------------------------------
# GradCAM++ layer selection per architecture
# ---------------------------------------------------------------------------
GRADCAM_LAYER: Dict[str, str] = {
    "efficientnet": "top_conv",
    "resnet50": "conv5_block3_out",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate GradCAM++ visualizations for flood detection models."
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
        "--predictions_csv",
        type=str,
        required=True,
        help="Path to predictions CSV from evaluate.py.",
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
        help="Directory to save GradCAM outputs (default: ./results).",
    )
    parser.add_argument(
        "--n_per_set",
        type=int,
        default=10,
        help="Maximum images per visualization set (default: 10).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Image set selection from predictions CSV
# ---------------------------------------------------------------------------


def select_image_sets(
    predictions_csv: str,
    n_per_set: int,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Select four image sets from the predictions CSV.

    Sets:
      false_negatives: true_label==1 (flood) AND correct==False
      false_positives: true_label==0 (non-flood) AND correct==False
      pool_images: 'swimmingpool' in filename (case-insensitive)
      random_correct: correct==True, balanced 50/50 flood/non-flood

    Args:
        predictions_csv: Path to predictions CSV from evaluate.py.
        n_per_set: Maximum images per set.
        seed: Random seed for sampling.

    Returns:
        Dict mapping set name to DataFrame subset.
    """
    df = pd.read_csv(predictions_csv)
    rng = np.random.RandomState(seed)

    sets: Dict[str, pd.DataFrame] = {}

    # False negatives: actual floods predicted as non-flood
    fn_mask = (df["true_label"] == 1) & (df["correct"] == False)  # noqa: E712
    fn_df = df[fn_mask]
    if len(fn_df) > n_per_set:
        fn_df = fn_df.sample(n=n_per_set, random_state=rng)
    sets["false_negatives"] = fn_df

    # False positives: actual non-floods predicted as flood
    fp_mask = (df["true_label"] == 0) & (df["correct"] == False)  # noqa: E712
    fp_df = df[fp_mask]
    if len(fp_df) > n_per_set:
        fp_df = fp_df.sample(n=n_per_set, random_state=rng)
    sets["false_positives"] = fp_df

    # Swimming pool images
    pool_mask = df["filename"].str.lower().str.contains("swimmingpool", na=False)
    pool_df = df[pool_mask]
    if len(pool_df) > n_per_set:
        pool_df = pool_df.sample(n=n_per_set, random_state=rng)
    sets["pool_images"] = pool_df

    # Random correct: balanced flood/non-flood
    correct_flood = df[(df["correct"] == True) & (df["true_label"] == 1)]  # noqa: E712
    correct_nonflood = df[(df["correct"] == True) & (df["true_label"] == 0)]  # noqa: E712
    n_half = n_per_set // 2
    if len(correct_flood) > n_half:
        correct_flood = correct_flood.sample(n=n_half, random_state=rng)
    if len(correct_nonflood) > n_half:
        correct_nonflood = correct_nonflood.sample(n=n_half, random_state=rng)
    sets["random_correct"] = pd.concat([correct_flood, correct_nonflood])

    return sets


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def find_image_path(filename: str, data_dir: str) -> str:
    """Locate an image file within the dataset directory tree.

    Searches the test split directories (processed_data/binary/test/) for the
    given filename.

    Args:
        filename: Image filename (e.g. 'swimmingpool_001.jpg').
        data_dir: Root dataset directory.

    Returns:
        Absolute path to the image file.

    Raises:
        FileNotFoundError: If the image cannot be located.
    """
    test_dir = os.path.join(data_dir, "processed_data", "binary", "test")
    for class_dir in ["flood", "non_flood"]:
        candidate = os.path.join(test_dir, class_dir, filename)
        if os.path.exists(candidate):
            return candidate

    # Fallback: walk the entire data_dir
    for root, _dirs, files in os.walk(data_dir):
        if filename in files:
            return os.path.join(root, filename)

    raise FileNotFoundError(
        f"Image not found: {filename} (searched {data_dir})"
    )


def load_image_pair(
    path: str, arch: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image for display and for model input.

    Args:
        path: Absolute path to the image file.
        arch: Architecture name for preprocessing selection.

    Returns:
        Tuple of (display_image, preprocessed_batch):
          - display_image: uint8 array [0, 255], shape (224, 224, 3)
          - preprocessed_batch: float32 array, shape (1, 224, 224, 3),
            preprocessed for the given backbone
    """
    from tensorflow.keras.preprocessing.image import img_to_array, load_img

    img = load_img(path, target_size=(224, 224))
    x = img_to_array(img)  # float32 [0, 255]
    display_img = x.astype(np.uint8)
    preprocessed = PREPROCESS_FN[arch](x[np.newaxis, ...].copy())
    return display_img, preprocessed


# ---------------------------------------------------------------------------
# GradCAM++ generation
# ---------------------------------------------------------------------------


def make_gradcam(
    model, arch: str, images_preprocessed: np.ndarray
) -> np.ndarray:
    """Generate GradCAM++ heatmaps for a batch of preprocessed images.

    Args:
        model: Loaded Keras model.
        arch: Architecture name for layer selection.
        images_preprocessed: Array of shape (N, 224, 224, 3), already
            preprocessed via PREPROCESS_FN[arch].

    Returns:
        Heatmap array of shape (N, 224, 224) with values in [0, 1].
    """
    try:
        from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
        from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
        from tf_keras_vis.utils.scores import BinaryScore
    except ImportError:
        print("tf-keras-vis not installed. Run: pip install tf-keras-vis")
        sys.exit(1)

    gradcam = GradcamPlusPlus(model, model_modifier=ReplaceToLinear(), clone=True)
    # BinaryScore(False) targets class index 0 in binary output.
    # Since the model outputs P(non_flood) via sigmoid, and we want to
    # visualize flood-relevant features, we use False to target the
    # flood-positive direction.
    score = BinaryScore(False)
    layer_name = GRADCAM_LAYER[arch]
    cam = gradcam(score, images_preprocessed, penultimate_layer=layer_name)
    return cam  # shape (N, 224, 224)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def save_gradcam_grid(
    display_images: List[np.ndarray],
    heatmaps: np.ndarray,
    filenames: List[str],
    set_name: str,
    arch: str,
    save_path: str,
) -> None:
    """Save a grid of original images with GradCAM++ heatmap overlays.

    Each image is shown as a pair: original on top, overlay on bottom.

    Args:
        display_images: List of uint8 arrays (224, 224, 3).
        heatmaps: Array of shape (N, 224, 224).
        filenames: List of filenames for subplot titles.
        set_name: Name of the image set (for figure title).
        arch: Architecture name (for figure title).
        save_path: Output file path.
    """
    import matplotlib.pyplot as plt

    n = len(display_images)
    if n == 0:
        return

    ncols = min(n, 5)
    nrows = 2 * ((n + ncols - 1) // ncols)  # 2 rows per image row (original + overlay)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows)
    )
    if nrows == 2 and ncols == 1:
        axes = axes.reshape(2, 1)
    elif nrows == 2:
        axes = axes.reshape(2, ncols)

    fig.suptitle(
        f"GradCAM++ -- {set_name} ({arch})",
        fontsize=16,
        fontweight="bold",
    )

    for idx in range(n):
        row_block = (idx // ncols) * 2
        col = idx % ncols

        # Original image
        ax_orig = axes[row_block, col] if nrows > 2 else axes[0, col]
        ax_orig.imshow(display_images[idx])
        short_name = filenames[idx][:25] + "..." if len(filenames[idx]) > 25 else filenames[idx]
        ax_orig.set_title(short_name, fontsize=8)
        ax_orig.axis("off")

        # Overlay
        ax_overlay = axes[row_block + 1, col] if nrows > 2 else axes[1, col]
        ax_overlay.imshow(display_images[idx])
        ax_overlay.imshow(heatmaps[idx], cmap="jet", alpha=0.4)
        ax_overlay.set_title("GradCAM++", fontsize=8)
        ax_overlay.axis("off")

    # Hide unused axes
    for idx in range(n, nrows // 2 * ncols):
        row_block = (idx // ncols) * 2
        col = idx % ncols
        if nrows > 2:
            axes[row_block, col].axis("off")
            axes[row_block + 1, col].axis("off")
        else:
            axes[0, col].axis("off")
            axes[1, col].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run GradCAM++ visualization pipeline."""
    args = parse_args()
    set_all_seeds(42)

    import matplotlib
    matplotlib.use("Agg")
    import tensorflow as tf

    # Validate inputs
    assert os.path.exists(args.model_path), f"Model not found: {args.model_path}"
    assert os.path.exists(args.predictions_csv), (
        f"Predictions CSV not found: {args.predictions_csv}"
    )
    assert args.arch in GRADCAM_LAYER, (
        f"Unknown arch '{args.arch}'. Valid: {list(GRADCAM_LAYER.keys())}"
    )

    # Create output directory
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Load model
    print(f"\nLoading model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    print(f"  Parameters: {model.count_params():,}")

    # Verify GradCAM layer exists
    layer_name = GRADCAM_LAYER[args.arch]
    layer_names = [layer.name for layer in model.layers]
    # The target layer may be inside a nested model (backbone)
    all_layer_names = []
    for layer in model.layers:
        all_layer_names.append(layer.name)
        if hasattr(layer, "layers"):
            for sub_layer in layer.layers:
                all_layer_names.append(sub_layer.name)
    assert layer_name in all_layer_names, (
        f"GradCAM layer '{layer_name}' not found in model. "
        f"Available layers (first 20): {all_layer_names[:20]}"
    )
    print(f"  GradCAM target layer: {layer_name}")

    # Select image sets
    print(f"\nSelecting image sets from: {args.predictions_csv}")
    image_sets = select_image_sets(args.predictions_csv, args.n_per_set)

    for set_name, subset_df in image_sets.items():
        n_images = len(subset_df)
        if n_images == 0:
            print(f"\n  [WARN] {set_name}: 0 images -- skipping.")
            continue

        print(f"\n  Processing {set_name}: {n_images} images")

        display_images: List[np.ndarray] = []
        preprocessed_list: List[np.ndarray] = []
        filenames: List[str] = []

        for _, row in subset_df.iterrows():
            fname = str(row["filename"])
            try:
                path = find_image_path(fname, args.data_dir)
                display_img, preprocessed = load_image_pair(path, args.arch)
                display_images.append(display_img)
                preprocessed_list.append(preprocessed[0])
                filenames.append(fname)
            except FileNotFoundError as e:
                print(f"    [WARN] {e}")
                continue

        if len(display_images) == 0:
            print(f"    [WARN] No images loaded for {set_name} -- skipping.")
            continue

        # Stack preprocessed images and generate heatmaps
        preprocessed_batch = np.array(preprocessed_list)
        print(f"    Generating GradCAM++ heatmaps ...")
        heatmaps = make_gradcam(model, args.arch, preprocessed_batch)

        # Save grid
        save_path = os.path.join(
            fig_dir, f"gradcam_{args.arch}_{set_name}.png"
        )
        save_gradcam_grid(
            display_images, heatmaps, filenames,
            set_name, args.arch, save_path,
        )

    print(f"\n{'=' * 60}")
    print(f"  GradCAM++ visualization complete.")
    print(f"  Outputs in: {fig_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
