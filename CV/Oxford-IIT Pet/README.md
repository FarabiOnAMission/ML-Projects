# Oxford-IIIT Pet Image Classification

A computer vision project focused on exploring image classification using Convolutional Neural Networks (CNNs), transfer learning, and fine-tuning with PyTorch.

The project uses the **Oxford-IIIT Pet Dataset**, containing images of 37 cat and dog breeds. The goal is to build, train, evaluate, and compare different deep learning architectures for multi-class image classification.

## Project Objectives

- Understand and implement CNN architectures from scratch.
- Explore transfer learning using pretrained models.
- Experiment with different fine-tuning strategies.
- Evaluate models using accuracy, precision, recall, F1 score, and confusion matrices.
- Develop a reproducible training workflow using PyTorch and Kaggle GPUs.
- Extend the experiments to more advanced architectures, including Vision Transformers (ViTs).

## Dataset

**Oxford-IIIT Pet Dataset**

- **Classes:** 37 cat and dog breeds
- **Training images:** 3,680 (before validation split)
- **Test images:** 3,669
- **Task:** Multi-class image classification

Dataset source: [Hugging Face — Oxford-IIIT Pet](https://huggingface.co/datasets/timm/oxford-iiit-pet)

## Models and Experiments

The project currently includes experiments with:

1. **Custom CNN:** A lightweight convolutional network trained from scratch.
2. **ResNet18 (Transfer Learning):** An ImageNet-pretrained ResNet18 with a newly trained classification head.
3. **ResNet18 (Partial Fine-Tuning):** Fine-tuning the final residual block and classification head while preserving pretrained BatchNorm running statistics.

### Current Results

| Model | Validation Accuracy | Test Accuracy |
|---|---:|---:|
| Custom CNN | 10.19% | — |
| ResNet18 (Head Only) | 92.26% | 89.53% |
| ResNet18 (Partial Fine-Tuning) | **92.39%** | **89.86%** |

The best-performing model currently achieves a **test macro F1 score of 0.8973**.

## Project Structure

```text
Oxford-IIIT Pet/
├── configs/          # Experiment configurations
├── data/             # Dataset-related files
├── kaggle/           # Kaggle training and deployment setup
├── notebooks/        # Exploratory analysis and experiments
├── outputs/          # Model checkpoints and experiment outputs
├── src/              # Reusable source code
├── requirements.txt  # Python dependencies
└── README.md
```

## Technologies

- Python
- PyTorch
- Torchvision
- timm
- Hugging Face Datasets
- Scikit-learn
- Pandas, NumPy, and Matplotlib
- Kaggle GPU

## Getting Started

Clone the repository:

```bash
git clone https://github.com/FarabiOnAMission/ML-Projects.git
cd ML-Projects/"Oxford-IIIT Pet"
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebooks in the `notebooks/` directory to explore the dataset, preprocessing pipeline, and model experiments.

GPU training experiments are configured through the Kaggle CLI. Some experiments require separately stored model checkpoints.

## Future Work

- Per-class performance evaluation and confusion matrix analysis.
- Error analysis and visualization of misclassified images.
- Experiments with stronger CNN backbones.
- Vision Transformer (ViT) implementation and fine-tuning.
- Comparison between CNN-based and Transformer-based image classification.

## Project Status

**In Progress**

This project is part of my ongoing exploration of deep learning and computer vision, with an emphasis on understanding model architectures, training strategies, and practical experimentation.