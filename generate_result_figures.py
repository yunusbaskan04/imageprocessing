import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

IMAGE_QUALITY_CSV = os.path.join(RESULTS_DIR, "image_quality_results.csv")
DETECTION_CSV = os.path.join(RESULTS_DIR, "detection_results.csv")
RUNTIME_CSV = os.path.join(RESULTS_DIR, "runtime_results.csv")

METHOD_LABELS = {
    "degraded": "Degraded",
    "global_he": "Global HE",
    "clahe_only": "CLAHE-only",
    "proposed": "CLAHE+Bilateral",
    "morphology_ablation": "Proposed-Final",
    "clahe_bilateral": "CLAHE+Bilateral",
    "proposed_final": "Proposed-Final",
    "original": "Original"
}

DEGRADATION_LABELS = {
    "low_light": "Low-light",
    "fog": "Fog",
    "noise": "Noise",
    "clean": "Clean"
}

METHOD_ORDER = [
    "degraded",
    "global_he",
    "clahe_only",
    "proposed",
    "morphology_ablation"
]

RUNTIME_METHOD_ORDER = [
    "global_he",
    "clahe_only",
    "clahe_bilateral",
    "proposed_final"
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def prettify_method(method):
    return METHOD_LABELS.get(method, method)


def prettify_degradation(degradation):
    return DEGRADATION_LABELS.get(degradation, degradation)


def plot_grouped_bar(df, value_col, title, ylabel, output_path):
    degradations = ["low_light", "fog", "noise"]
    methods = METHOD_ORDER

    x = range(len(degradations))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(methods):
        values = []

        for degradation in degradations:
            subset = df[
                (df["degradation"] == degradation) &
                (df["method"] == method)
            ]

            if len(subset) == 0:
                values.append(0)
            else:
                values.append(float(subset.iloc[0][value_col]))

        positions = [pos + (i - 2) * width for pos in x]
        ax.bar(positions, values, width, label=prettify_method(method))

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(x))
    ax.set_xticklabels([prettify_degradation(d) for d in degradations])
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_runtime(df):
    methods = RUNTIME_METHOD_ORDER
    degradations = ["low_light", "fog", "noise"]

    x = range(len(degradations))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(methods):
        values = []

        for degradation in degradations:
            subset = df[
                (df["degradation"] == degradation) &
                (df["method"] == method)
            ]

            if len(subset) == 0:
                values.append(0)
            else:
                values.append(float(subset.iloc[0]["fps"]))

        positions = [pos + (i - 1.5) * width for pos in x]
        ax.bar(positions, values, width, label=prettify_method(method))

    ax.set_title("Pre-processing Runtime Comparison")
    ax.set_ylabel("FPS")
    ax.set_xticks(list(x))
    ax.set_xticklabels([prettify_degradation(d) for d in degradations])
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, "fig5_runtime_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def draw_pipeline():
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.axis("off")

    steps = [
        "Degraded\nInput",
        "BGR to LAB",
        "CLAHE\nL-channel",
        "LAB to BGR",
        "Bilateral\nFilter",
        "Morphological\nClosing",
        "YOLOv8n\nDetection"
    ]

    x_positions = list(range(len(steps)))

    for i, step in enumerate(steps):
        ax.text(
            x_positions[i],
            0,
            step,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.35", linewidth=1.5),
            fontsize=9
        )

        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i] + 0.42, 0),
                xytext=(x_positions[i] + 0.9, 0),
                arrowprops=dict(arrowstyle="<-", linewidth=1.5)
            )

    ax.set_xlim(-0.6, len(steps) - 0.4)
    ax.set_ylim(-0.8, 0.8)

    output_path = os.path.join(FIGURES_DIR, "fig1_pipeline.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    ensure_dir(FIGURES_DIR)

    if not os.path.exists(DETECTION_CSV):
        print(f"Missing file: {DETECTION_CSV}")
        return

    if not os.path.exists(IMAGE_QUALITY_CSV):
        print(f"Missing file: {IMAGE_QUALITY_CSV}")
        return

    if not os.path.exists(RUNTIME_CSV):
        print(f"Missing file: {RUNTIME_CSV}")
        return

    detection_df = pd.read_csv(DETECTION_CSV)
    quality_df = pd.read_csv(IMAGE_QUALITY_CSV)
    runtime_df = pd.read_csv(RUNTIME_CSV)

    draw_pipeline()

    plot_grouped_bar(
        detection_df,
        value_col="map50",
        title="YOLOv8n mAP@50 Comparison under Synthetic Degradations",
        ylabel="mAP@50",
        output_path=os.path.join(FIGURES_DIR, "fig2_map50_comparison.png")
    )

    plot_grouped_bar(
        quality_df,
        value_col="psnr",
        title="PSNR Comparison under Synthetic Degradations",
        ylabel="PSNR (dB)",
        output_path=os.path.join(FIGURES_DIR, "fig3_psnr_comparison.png")
    )

    plot_grouped_bar(
        quality_df,
        value_col="ssim",
        title="SSIM Comparison under Synthetic Degradations",
        ylabel="SSIM",
        output_path=os.path.join(FIGURES_DIR, "fig4_ssim_comparison.png")
    )

    plot_runtime(runtime_df)

    print("\nAll result figures generated successfully.")
    print(f"Figures folder: {FIGURES_DIR}")


if __name__ == "__main__":
    main()