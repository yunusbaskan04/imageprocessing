# Comparative Evaluation of Classical Image Pre-processing Pipelines for UAV Detection under Adverse Visual Conditions

This repository contains the implementation code and experimental results for a Digital Image Processing term project. The study investigates whether lightweight classical image pre-processing methods can improve zero-shot UAV detection performance under adverse visual conditions such as low light, fog, and Gaussian noise.

The project compares several classical enhancement pipelines under identical conditions using the same dataset, synthetic degradation settings, YOLOv8n evaluation protocol, and performance metrics.

## Project Summary

Real-time UAV detection is challenging under degraded visual conditions. Instead of retraining a large deep learning model, this project evaluates classical image processing pipelines as plug-and-play pre-processing modules before object detection.

The evaluated methods are:

1. Degraded input without enhancement
2. Global Histogram Equalization
3. CLAHE-only
4. CLAHE + Bilateral Filtering
5. Proposed-Final: CLAHE + Bilateral Filtering + Morphological Closing

The main objective is not to train a new UAV detector, but to compare how different image enhancement pipelines affect the performance of a pre-trained YOLOv8n model under the same degradation and evaluation settings.

## Dataset

The experiments use the **Drone Detection v2** dataset from Roboflow Universe.

* Dataset name: Drone Detection v2
* Roboflow project: `drone-detection-adly5`
* Workspace: `elevator-6m2yb`
* Version: 2
* Source: https://universe.roboflow.com/elevator-6m2yb/drone-detection-adly5/dataset/2
* License: CC BY 4.0
* Annotation format: YOLOv8
* Number of classes: 1
* Class name: `drone`
* Image size: 640 × 640 pixels

Dataset split statistics:

| Split      | Images | Labels | Bounding Boxes | Empty Labels | Resolution |
| ---------- | -----: | -----: | -------------: | -----------: | ---------- |
| Train      |   2021 |   2021 |           2078 |           24 | 640×640    |
| Validation |    129 |    129 |            138 |            2 | 640×640    |
| Test       |     40 |     40 |             39 |            1 | 640×640    |
| Total      |   2190 |   2190 |           2255 |           27 | 640×640    |

Since this study does not train or fine-tune a model, the validation and test subsets were combined only for zero-shot evaluation. The final evaluation subset contains:

* 169 images
* 177 drone annotations

## Evaluation Protocol

A pre-trained YOLOv8n model is used in zero-shot mode.

Important note: COCO-pretrained YOLOv8n does not include a dedicated `drone` class. Therefore, for evaluation consistency, the original drone labels are mapped to the closest aerial COCO class, `aeroplane` with class ID 4. This mapping is applied equally to all methods. The reported detection results should therefore be interpreted as relative zero-shot comparisons rather than fully trained UAV detector accuracy.

## Synthetic Degradations

The following synthetic degradations are applied to the evaluation images:

| Degradation    | Method                             | Parameters                        |
| -------------- | ---------------------------------- | --------------------------------- |
| Low light      | Gamma correction                   | gamma = 2.5                       |
| Fog            | Alpha blending with gray fog layer | alpha = 0.5, fog intensity = 200  |
| Gaussian noise | Additive Gaussian noise            | mean = 0, standard deviation = 60 |

A fixed random seed is used for Gaussian noise generation to improve reproducibility.

## Enhancement Methods

### 1. Global Histogram Equalization

Global histogram equalization is applied to the luminance channel in YCrCb color space. This method is used as a simple classical baseline.

### 2. CLAHE-only

Contrast Limited Adaptive Histogram Equalization is applied to the L channel in LAB color space.

Parameters:

* clipLimit = 2.0
* tileGridSize = 8 × 8

### 3. CLAHE + Bilateral Filtering

This variant applies CLAHE followed by bilateral filtering.

Bilateral filter parameters:

* diameter = 9
* sigmaColor = 75
* sigmaSpace = 75

### 4. Proposed-Final Pipeline

The proposed final pipeline consists of:

1. BGR to LAB color space conversion
2. CLAHE on the L channel
3. LAB to BGR conversion
4. Bilateral filtering
5. Morphological closing using a 7 × 7 elliptical structuring element

This pipeline is evaluated as the final detection-oriented enhancement method.

## Repository Structure

```text
imageprocessing/
│
├── dataset_original/              # Original Roboflow YOLOv8 dataset
├── dataset_eval/                  # Combined validation + test evaluation subset
├── dataset_degraded/              # Synthetic low-light, fog, and noise images
├── outputs/                       # Enhancement outputs for each method
│   ├── degraded/
│   ├── global_he/
│   ├── clahe_only/
│   ├── proposed/
│   └── morphology_ablation/
│
├── results/
│   ├── image_quality_results.csv
│   ├── detection_results.csv
│   ├── runtime_results.csv
│   └── figures/
│
├── prepare_eval_dataset.py
├── generate_degradations.py
├── enhance_images.py
├── calculate_metrics.py
├── evaluate_yolo.py
├── measure_runtime.py
├── generate_result_figures.py
├── create_visual_comparisons.py
├── requirements.txt
└── README.md
```

## Installation

Create a Python environment and install the required libraries:

```bash
pip install opencv-python numpy pandas matplotlib scikit-image ultralytics pyyaml
```

The experiments were conducted with:

* Python 3.13.3
* Ultralytics 8.4.60
* PyTorch 2.10.0 + CUDA 12.6
* GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM

## How to Reproduce the Experiments

Run the following commands from the project root directory.

### 1. Prepare the evaluation subset

This combines the validation and test subsets into `dataset_eval/`.

```bash
python prepare_eval_dataset.py
```

Expected output:

```text
Total copied images: 169
Total copied labels: 169
Output folder: dataset_eval
```

### 2. Generate synthetic degradations

```bash
python generate_degradations.py
```

This creates:

```text
dataset_degraded/
├── low_light/
├── fog/
└── noise/
```

### 3. Generate enhancement outputs

```bash
python enhance_images.py --method all
```

This creates:

```text
outputs/
├── degraded/
├── global_he/
├── clahe_only/
├── proposed/
└── morphology_ablation/
```

### 4. Calculate PSNR and SSIM

```bash
python calculate_metrics.py
```

Output:

```text
results/image_quality_results.csv
```

### 5. Evaluate YOLOv8n detection performance

```bash
python evaluate_yolo.py
```

Output:

```text
results/detection_results.csv
results/yolo_visuals/
```

### 6. Measure pre-processing runtime

```bash
python measure_runtime.py
```

Output:

```text
results/runtime_results.csv
```

### 7. Generate result figures

```bash
python generate_result_figures.py
python create_visual_comparisons.py
```

Output:

```text
results/figures/
```

## Main Results

### Detection Performance

The Proposed-Final pipeline achieved the highest mAP@50 in all tested degradation scenarios.

| Degradation | Best Method    | mAP@50 | Precision | Recall |
| ----------- | -------------- | -----: | --------: | -----: |
| Low-light   | Proposed-Final | 0.0812 |    0.0665 | 0.2825 |
| Fog         | Proposed-Final | 0.1095 |    0.2801 | 0.1073 |
| Noise       | Proposed-Final | 0.0634 |    0.0956 | 0.2203 |

Compared with the degraded input, the Proposed-Final pipeline improved mAP@50 approximately:

* 9.9× under low-light degradation
* 3.5× under fog degradation
* 2.3× under Gaussian noise degradation

### Image Quality Metrics

The image-quality metrics showed that the method with the best PSNR or SSIM is not always the best method for object detection. For example, Global Histogram Equalization and CLAHE-only achieved strong PSNR/SSIM values in some cases, while the Proposed-Final pipeline achieved the best detection-level mAP@50.

This indicates that pixel-level similarity metrics and detection-level metrics may not always correlate directly.

### Runtime

The Proposed-Final pipeline achieved approximately 23–24 FPS for pre-processing alone on the tested RTX 3050 laptop environment.

| Method                        | Approximate FPS |
| ----------------------------- | --------------: |
| Global Histogram Equalization |     365–525 FPS |
| CLAHE-only                    |     228–320 FPS |
| CLAHE + Bilateral Filtering   |       24–25 FPS |
| Proposed-Final                |       23–24 FPS |

The main computational cost comes from bilateral filtering. Morphological closing adds only limited additional overhead.

## Key Observation

The experiments show that image enhancement methods should not be evaluated only with PSNR and SSIM. Although morphological closing may reduce pixel-level fidelity and produce a slightly simplified visual appearance, it improved zero-shot YOLOv8n detection performance in all tested degradation scenarios.

Therefore, this project emphasizes a detection-oriented evaluation approach rather than relying only on image restoration metrics.

## Limitations

This project has several limitations:

1. YOLOv8n is used in zero-shot mode without UAV-specific fine-tuning.
2. Since COCO does not include a drone class, labels are mapped to the surrogate aeroplane class for relative evaluation.
3. The evaluation subset contains 169 images, which is useful for controlled comparison but still limited.
4. The degradations are synthetic and may not fully represent real-world fog, illumination, and sensor noise.
5. The Proposed-Final pipeline improves detection metrics but may reduce visual fidelity in some cases.

## Future Work

Future work may include:

* Fine-tuning YOLOv8n or another lightweight detector directly on the UAV dataset
* Testing on real adverse-weather UAV images
* Comparing the classical pipelines with deep-learning-based dehazing or low-light enhancement methods
* Performing parameter optimization for CLAHE, bilateral filtering, and morphological operations
* Evaluating the pipeline on embedded edge devices such as Jetson Nano or Jetson Orin Nano

## AI Usage Declaration

Generative AI tools were used for language refinement, document structuring, debugging support, and README organization. All experimental results, code execution, parameter settings, and interpretations were manually reviewed and verified by the author.

## Author

Yunus Başkan
Department of Computer Engineering
Abdullah Gül University
Kayseri, Türkiye

## Repository

https://github.com/yunusbaskan04/imageprocessing
