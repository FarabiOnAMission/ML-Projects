# SST-2 Sentiment Classification

An end-to-end Natural Language Processing (NLP) project exploring sentiment classification using traditional machine learning and pretrained transformer models.

The project focuses on understanding the complete machine learning workflow, from exploratory data analysis and baseline development to BERT fine-tuning, evaluation, model serialization, and GPU-based training.

The implementation uses **PyTorch, Hugging Face Transformers, scikit-learn, and Kaggle GPU resources**.

## 1. Project Overview

Sentiment analysis is a fundamental NLP task that involves identifying the sentiment expressed in a piece of text.

This project uses the Stanford Sentiment Treebank 2 (SST-2) dataset to classify movie reviews into two categories:

| Label | Sentiment |
|-------|-----------|
| 0 | Negative |
| 1 | Positive |

For example:

**Input:**
> A beautifully directed movie with outstanding performances.

**Expected output:**
```text
Positive (1)
```

### Objectives

- Explore and understand a real-world NLP classification dataset.
- Analyze class distribution, text characteristics, and data quality.
- Establish a baseline using TF-IDF and Logistic Regression.
- Implement a group-aware validation strategy.
- Fine-tune a pretrained BERT model for sentiment classification.
- Compare traditional ML and transformer-based approaches.
- Analyze classification errors and generalization behavior.
- Train models using Kaggle GPUs.
- Save trained models for future inference without retraining.
- Develop a reproducible NLP experimentation workflow.

---

## 2. Dataset

**Dataset:** Stanford Sentiment Treebank 2 (SST-2)

**Source:** [Hugging Face — stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2)

SST-2 contains movie-review text labeled according to its sentiment.

### Dataset Statistics

| Split | Examples |
|-------|---------:|
| Training | 67,349 |
| Validation | 872 |
| Test | 1,821 |

The official test split does not provide publicly accessible sentiment labels.

### Dataset Format

Each labeled example contains:

| Column | Description |
|--------|-------------|
| `sentence` | Movie-review text |
| `label` | Sentiment classification target |

The dataset also provides an `idx` identifier.

### Exploratory Data Analysis

Our analysis identified several important characteristics:

- The training dataset contains approximately 56% positive and 44% negative examples.
- Training examples are generally short.
- The official validation set contains substantially longer sentences on average.
- Duplicate and related review phrases require careful handling during validation.
- Classification difficulty varies with sentence length and linguistic structure.

Approximate average sentence lengths:

| Split | Mean word count |
|-------|----------------:|
| Training | 9.41 |
| Official validation | 19.55 |

These differences are important when interpreting model performance.

---

## 3. Project Structure

The project is organized inside the main ML-Projects repository.

```text
ML-Projects/
└── LLM/
    └── SST2-Sentiment/
        │
        ├── .venv/
        ├── .gitignore
        ├── README.md
        ├── requirements.txt
        │
        ├── configs/
        │
        ├── data/
        │   ├── raw/
        │   └── processed/
        │
        ├── notebooks/
        │   └── 01_data_exploration.ipynb
        │
        ├── kaggle/
        │   ├── bert_training.ipynb
        │   ├── kernel-metadata.json
        │   └── train.py
        │
        ├── outputs/
        │
        ├── src/
        │
        ├── tests/
        │
        └── test_environment.py
```

### Directory Descriptions

| Directory | Purpose |
|-----------|---------|
| `configs/` | Experiment configurations |
| `data/raw/` | Original datasets |
| `data/processed/` | Processed datasets and intermediate data |
| `notebooks/` | Exploratory analysis and experiments |
| `kaggle/` | Kaggle training notebook and execution configuration |
| `outputs/` | Generated metrics, predictions, and model artifacts |
| `src/` | Reusable project code |
| `tests/` | Project tests and verification |

The `kaggle/train.py` file was used during initial remote-execution testing. The notebook is the primary implementation for BERT fine-tuning.

Large datasets, virtual environments, and trained model checkpoints should not be committed to Git.

---

## 4. Environment Setup

### Requirements

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- scikit-learn
- pandas
- NumPy
- Jupyter
- Matplotlib
- tqdm

Use a Python version compatible with the dependencies specified in `requirements.txt`.

### Step 1: Navigate to the Project

```bash
cd /Users/farabi/Farabi/ML-Projects/LLM/SST2-Sentiment
```

### Step 2: Create a Virtual Environment

For example, using Python 3.11:

```bash
python3.11 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Configure Jupyter

```bash
python -m pip install ipykernel

python -m ipykernel install --user \
    --name sst2-sentiment \
    --display-name "Python (SST2-Sentiment)"
```

Inside VS Code, select:

```text
Python (SST2-Sentiment)
```

as the notebook kernel.

### Step 5: Verify Hardware Acceleration

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
```

The training implementation supports CUDA, Apple Silicon MPS, and CPU execution.

---

## 5. Data Preparation and Validation

To evaluate models consistently, we implemented a group-aware validation strategy using:

```python
StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Sentences were normalized by:

- Converting text to lowercase.
- Removing leading and trailing whitespace.
- Normalizing repeated whitespace.

Normalized sentences were used as grouping identifiers.

This helps prevent identical normalized sentences from appearing in both training and validation folds.

### Internal Validation Split

The initial internal split contains:

| Subset | Examples |
|--------|---------:|
| Training | 53,879 |
| Validation | 13,470 |

Group-aware splitting reduces exact-duplicate leakage but does not eliminate every form of related-text or substring overlap.

For model comparison, the same validation protocol should be maintained.

---

## 6. Baseline Model: TF-IDF + Logistic Regression

Our initial approach used a traditional machine learning pipeline:

```text
Input Sentence
      ↓
Text Vectorization
      ↓
TF-IDF Features
      ↓
Logistic Regression
      ↓
Sentiment Prediction
```

### Model Configuration

**Feature Extraction:** TF-IDF

**N-gram range:** `(1, 2)`

**Classifier:** Logistic Regression

**Regularization parameter:** `C = 10`

The final baseline used a token pattern that better preserved words containing apostrophes.

### Baseline Results

#### Five-Fold Internal Cross-Validation

| Metric | Result |
|--------|-------:|
| Mean Accuracy | 92.63% |
| Accuracy Standard Deviation | 0.16 percentage points |
| Mean Macro F1 | 0.9253 |

The first internal fold achieved approximately 92.81% accuracy.

#### Official Validation

After fitting the baseline on the full official training split:

| Metric | Result |
|--------|-------:|
| Accuracy | 81.77% |
| Macro F1 | 0.8168 |

### Analysis

The substantial difference between internal cross-validation and official validation performance motivated further investigation.

Our analysis found:

- The official validation set contains longer sentences.
- Negative sentiment examples were particularly challenging at longer lengths.
- Related phrases and sentence fragments may make internal validation easier.
- Differences in sample characteristics can affect generalization.

These observations suggest limitations in the baseline's ability to generalize, although they do not establish a single definitive cause of the performance gap.

The internal and official validation scores represent different evaluation protocols and should not be treated as directly interchangeable.

---

## 7. Transformer Model: BERT

To improve contextual understanding, we implemented a pretrained transformer-based sentiment classifier.

### Model

**Checkpoint:** `bert-base-uncased`

**Architecture:** BERT for Sequence Classification

**Framework:** PyTorch + Hugging Face Transformers

### BERT Base Configuration

| Component | Configuration |
|-----------|--------------|
| Transformer layers | 12 |
| Hidden dimension | 768 |
| Attention heads | 12 |
| Classification classes | 2 |
| Maximum input length used | 128 tokens |

### Architecture

```text
Movie Review
     ↓
BERT Tokenizer
     ↓
Input IDs + Attention Masks
     ↓
Token and Position Embeddings
     ↓
Pretrained BERT Encoder
     ↓
[CLS] Representation
     ↓
Classification Head
     ↓
Negative / Positive Logits
```

The model uses pretrained BERT parameters together with a task-specific classification head.

During fine-tuning, the model parameters are updated using labeled SST-2 examples.

---

## 8. BERT Data Pipeline

The training notebook implements the following pipeline:

```text
SST-2 Dataset
      ↓
Group-Aware Train/Validation Split
      ↓
Hugging Face Dataset
      ↓
BERT Tokenization
      ↓
Dynamic Padding
      ↓
PyTorch DataLoader
      ↓
BERT Sequence Classifier
```

### Tokenization

```python
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)
```

Tokenization uses:

```python
truncation=True
max_length=128
padding=False
```

Padding is applied dynamically at the batch level using:

```python
DataCollatorWithPadding
```

This reduces unnecessary padding compared with padding every example to a fixed sequence length.

### DataLoader Configuration

```python
batch_size = 16
```

Training batches are shuffled, while validation batches retain a consistent ordering.

### Input Dimensions

For a batch containing 16 examples:

```text
input_ids:      (16, sequence_length)
attention_mask: (16, sequence_length)
labels:         (16,)
```

The sequence length depends on the longest tokenized input within the batch.

BERT produces classification logits with shape:

```text
(16, 2)
```

---

## 9. BERT Fine-Tuning Configuration

Our initial BERT experiment was configured with:

| Hyperparameter | Value |
|----------------|-------|
| Model | bert-base-uncased |
| Epochs | 2 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Maximum sequence length | 128 |
| Random seed | 42 |
| Model-selection metric | Validation Macro F1 |

### Training Procedure

For each training batch:

1. Move input tensors to the selected device.
2. Perform forward propagation.
3. Calculate cross-entropy loss.
4. Calculate gradients through backpropagation.
5. Update model parameters using AdamW.

After each epoch:

1. Evaluate on the internal validation set.
2. Calculate validation loss.
3. Calculate accuracy.
4. Calculate Macro F1.
5. Save the model if validation Macro F1 improves.

### Local Sanity Checks

Before full training, we tested:

- BERT model initialization.
- Forward propagation.
- Logit dimensions.
- Classification loss.
- Gradient computation.
- Optimizer updates.
- Small-batch training.

These tests verified the basic model and training operations on an Apple Silicon MPS device.

A successful small-batch test does not by itself establish final model performance.

---

## 10. Training on Kaggle GPU

Because full transformer fine-tuning is computationally expensive, the project uses Kaggle for remote GPU execution.

The intended workflow is:

```text
Local Development
       ↓
VS Code + Jupyter
       ↓
Kaggle CLI
       ↓
Remote GPU Training
       ↓
Model Evaluation
       ↓
Save Best Checkpoint
       ↓
Download Saved Model
```

### Install Kaggle CLI

From the activated project environment:

```bash
python -m pip install kaggle
```

Verify:

```bash
kaggle --version
```

Authenticate:

```bash
kaggle auth login
```

Check quota:

```bash
kaggle quota
```

### Kaggle Configuration

The `kaggle/kernel-metadata.json` file configures remote execution.

The intended accelerator is:

```json
{
  "enable_gpu": true,
  "machine_shape": "NvidiaTeslaT4"
}
```

This requests Kaggle's Tesla T4 ×2 accelerator environment.

The training implementation uses a single CUDA device by default.

### Submit Training

From the project root:

```bash
cd kaggle
```

Submit the notebook:

```bash
kaggle kernels push -p .
```

Check execution status:

```bash
kaggle kernels status code51/sst2-bert-training
```

Possible statuses include:

```text
QUEUED
RUNNING
COMPLETE
ERROR
```

A queued job has not yet started executing.

GPU quota availability does not guarantee immediate resource allocation.

Training should be considered successful only after confirming that the remote execution completed and the expected artifacts were saved.

---

## 11. Model Saving and Retrieval

The BERT training notebook is configured to save the best-performing model using:

```python
model.save_pretrained(
    "/kaggle/working/bert_sst2_best"
)

tokenizer.save_pretrained(
    "/kaggle/working/bert_sst2_best"
)
```

A successful model export should contain files such as:

```text
bert_sst2_best/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt
```

### Download from Kaggle

From the project root:

```bash
kaggle kernels output \
    code51/sst2-bert-training \
    -p outputs/models/
```

Verify that the downloaded files exist and contain valid model data.

The exact output directory should be checked after downloading.

### Important

The saved model can be used for inference without repeating training.

An inference-ready model export is not necessarily a complete resumable training checkpoint containing optimizer and scheduler state.

---

## 12. Inference

The intended inference workflow is:

```text
New Sentence
      ↓
Load Saved Tokenizer
      ↓
Tokenization
      ↓
Load Fine-Tuned BERT
      ↓
Forward Pass
      ↓
Classification Logits
      ↓
Predicted Sentiment
```

### Example

Once the trained checkpoint has been retrieved and verified:

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_PATH = "outputs/models/bert_sst2_best"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

sentence = "The movie was absolutely fantastic."

inputs = tokenizer(
    sentence,
    return_tensors="pt",
    truncation=True,
    max_length=128
)

with torch.no_grad():
    outputs = model(**inputs)

prediction = outputs.logits.argmax(dim=-1).item()

label_map = {
    0: "Negative",
    1: "Positive"
}

print("Predicted sentiment:", label_map[prediction])
```

This example assumes the checkpoint has already been saved and downloaded successfully.

The example performs inference only; it does not retrain the model.

---

## 13. Evaluation and Model Comparison

The primary metrics used in this project are:

- Accuracy
- Macro F1

Additional analysis may include:

- Confusion matrices.
- Precision and recall.
- Class-specific F1.
- Sentence-length performance.
- Error inspection.

### Experiment Results

| Model | Evaluation Protocol | Accuracy | Macro F1 |
|-------|---------------------|---------:|---------:|
| TF-IDF + Logistic Regression | 5-fold internal CV | 92.63% | 0.9253 |
| TF-IDF + Logistic Regression | Official validation | 81.77% | 0.8168 |
| BERT Base | Internal validation | 94.88% | 0.9488 |

The official validation baseline was trained on the full official training split. The cross-validation result was calculated using internal folds.

Consequently, these results should be interpreted according to their respective evaluation protocols.

The BERT result will be added after a successful end-to-end training and evaluation run.

---

## 14. Future Improvements

Potential extensions include:

- Fine-tuning RoBERTa and DeBERTa.
- Experimenting with alternative learning rates.
- Learning-rate warmup and decay.
- Gradient clipping.
- Early stopping.
- Additional random seeds.
- More detailed error analysis.
- Comparing different tokenization strategies.
- Evaluating robustness to longer and more complex sentences.
- Comparing saved models on identical validation examples.
- Experimenting with probability-based ensembles.

Future changes should be evaluated systematically rather than assuming that greater model complexity will produce better results.

---

## 15. Project Status

| Component | Status |
|-----------|--------|
| Dataset loading | Completed |
| Exploratory data analysis | Completed |
| Group-aware validation | Completed |
| TF-IDF + Logistic Regression baseline | Completed |
| Baseline cross-validation | Completed |
| Official validation analysis | Completed |
| BERT tokenization pipeline | Implemented |
| Dynamic padding and DataLoaders | Implemented |
| BERT forward-pass testing | Completed |
| BERT backpropagation testing | Completed |
| Training and evaluation functions | Implemented |
| Best-model saving logic | Implemented |
| Full Kaggle GPU fine-tuning | Not yet verified |
| Fine-tuned BERT metrics | Not yet verified |
| Saved BERT model retrieval | Not yet verified |
| Independent saved-model inference | Not yet verified |

The project currently contains a completed traditional machine learning baseline and an implemented BERT fine-tuning pipeline.

Final transformer results and inference verification will be documented after successful GPU execution and artifact retrieval.

---

## 16. Technologies Used

**Language**
- Python

**Machine Learning**
- scikit-learn
- PyTorch

**NLP**
- Hugging Face Transformers
- Hugging Face Datasets
- BERT

**Data Analysis**
- pandas
- NumPy
- Matplotlib

**Development**
- VS Code
- Jupyter Notebook
- Git
- GitHub

**Compute**
- Apple Silicon MPS
- Kaggle GPU

---

## 17. References

1. [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/)
2. [SST-2 Dataset — Hugging Face](https://huggingface.co/datasets/stanfordnlp/sst2)
3. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
4. [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
5. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
6. [Kaggle API and CLI](https://github.com/Kaggle/kaggle-cli)

---

## 18. Acknowledgment

This project was developed as a practical study of machine learning
experimentation, NLP classification, transformer fine-tuning, and
reproducible model development.

The implementation draws on concepts from *Build a Large Language
Model (From Scratch)* by Sebastian Raschka, alongside the PyTorch,
Hugging Face, and scikit-learn ecosystems.
