# IMDB Sentiment Analysis: BERT vs. TF-IDF + SVM


A comparative study of sentiment classification on the IMDB movie reviews dataset, contrasting a fine-tuned **BERT** transformer against a classical **TF-IDF + Linear SVM** baseline. The project benchmarks accuracy, precision, recall, and F1-score across both approaches to quantify the trade-off between model complexity and performance.

---

## Table of Contents

- [Motivation](#motivation)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Results](#results)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)
- [License](#license)

---

## Motivation

Sentiment analysis is a cornerstone NLP task with applications in product reviews, social listening, and content moderation. This project answers a practical question: **when does fine-tuning a 110M-parameter transformer justify its cost over a well-tuned linear model?** By training both on identical splits of the IMDB dataset, the project isolates the performance contribution of deep contextual embeddings.

---

## Project Structure

```
project/
├── src/
│   ├── train_bert.py          # Fine-tune BERT on IMDB
│   ├── train_svm.py           # Train TF-IDF + SVM baseline
│   ├── predict.py             # Inference using trained BERT model
│   ├── inspect_tokenizer.py   # Tokenizer exploration utilities
│   └── check_gpu.py           # Verify CUDA availability
├── data/
│   └── test_review.txt        # Sample input for inference
├── reports/
│   ├── classification_report.txt
│   └── training_log.txt
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

> **Note:** The trained BERT model (~16 GB) and the full IMDB dataset are excluded from version control via `.gitignore`. See [Installation](#installation) for instructions on obtaining both.

---

## Requirements

- Python 3.8 or higher
- CUDA 11.7+ (optional, for GPU acceleration)
- 16 GB RAM minimum (32 GB recommended for BERT training)
- ~20 GB free disk space for model checkpoints

### Core dependencies

| Package        | Version    |
|----------------|------------|
| `transformers` | ≥ 4.30     |
| `datasets`     | ≥ 2.14     |
| `torch`        | ≥ 2.0      |
| `scikit-learn` | ≥ 1.3      |
| `pandas`       | ≥ 2.0      |
| `numpy`        | ≥ 1.24     |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU setup (optional)

Verify CUDA availability before training:

```bash
python src/check_gpu.py
```

If CUDA is not detected, install the correct PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Usage

### 1. Train the BERT model

```bash
python src/train_bert.py
```

Default configuration:

| Hyperparameter   | Value               |
|------------------|---------------------|
| Base model       | `bert-base-uncased` |
| Max sequence len | 256                 |
| Batch size       | 16                  |
| Learning rate    | 2e-5                |
| Epochs           | 3                   |
| Optimizer        | AdamW               |

The trained model is saved to `./bert-imdb-model/`.

### 2. Train the SVM baseline

```bash
python src/train_svm.py
```

Uses TF-IDF features (unigrams + bigrams, `max_features=50_000`) with a `LinearSVC` classifier.

### 3. Run inference

Edit `data/test_review.txt` with the review you want to classify, then run:

```bash
python src/predict.py
```

Example output:

```
Review: "The cinematography was stunning and the performances deeply moving."
Predicted sentiment: Positive (confidence: 0.97)
```

---

## Dataset

- **Source:** [Large Movie Review Dataset (IMDB)](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Size:** 50,000 reviews (25,000 train / 25,000 test), balanced across classes
- **Labels:** binary (positive / negative)
- **Preprocessing:** HTML tag removal, lowercasing (for SVM only — BERT uses its own tokenizer)

---

## Model Details

### BERT
The `bert-base-uncased` checkpoint (12 layers, 768 hidden units, 110M parameters) is fine-tuned end-to-end with a single classification head on the `[CLS]` token. Training uses cross-entropy loss with linear warmup over the first 10% of steps.

### TF-IDF + SVM
Reviews are vectorized using TF-IDF over word unigrams and bigrams, then classified with a linear Support Vector Machine. This baseline is lightweight (< 100 MB), trains in under a minute on CPU, and provides a strong reference point.

---

## Results

Evaluated on the 25,000-review test split:

| Model              | Accuracy | Precision | Recall | F1-score | Train time (GPU) |
|--------------------|:--------:|:---------:|:------:|:--------:|:----------------:|
| TF-IDF + LinearSVM |  0.884   |   0.886   | 0.882  |  0.884   | ~45 s (CPU)      |
| BERT (fine-tuned)  |  0.924   |   0.925   | 0.923  |  0.924   | ~35 min (T4)     |

**Observations:**
- BERT improves accuracy by ~4 points over the SVM baseline.
- The SVM trains ~50× faster and uses ~160× less disk space.
- For latency-sensitive applications, the SVM remains a defensible choice.

Full per-class metrics are available in [`reports/classification_report.txt`](reports/classification_report.txt).

---

## Limitations & Future Work

- **Sequence truncation:** BERT's 512-token limit truncates long reviews; a hierarchical or Longformer approach could help.
- **Domain specificity:** Trained on movie reviews — performance on other domains (products, tweets) is not guaranteed.
- **No hyperparameter search:** BERT hyperparameters were fixed at commonly cited defaults; Optuna-based search could yield further gains.
- **Potential extensions:** distillation to a smaller model (DistilBERT), calibration analysis, or multilingual evaluation via XLM-R.

