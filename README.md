# NLP Text Classification with BERT

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Transformers](https://img.shields.io/badge/Transformers-4.x-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A collection of text classification experiments built on `bert-base-uncased`, exploring how a single transformer backbone performs across different NLP tasks and how it compares to classical baselines.

This repository contains two self-contained sub-projects:

| Project | Task | Dataset | Classes |
|---|---|---|---|
| [`imdb-sentiment/`](./imdb-sentiment) | Sentiment analysis of long-form movie reviews | IMDB (50k reviews) | Positive / Negative |
| [`sarcasm-detection/`](./sarcasm-detection) | Sarcasm detection on short news headlines | News Headlines Dataset | Sarcastic / Not Sarcastic |

---

## Table of Contents

- [Why these two tasks?](#why-these-two-tasks)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Project 1 — IMDB Sentiment Analysis](#project-1--imdb-sentiment-analysis)
- [Project 2 — Sarcasm Detection](#project-2--sarcasm-detection)
- [Results Summary](#results-summary)
- [Key Takeaways](#key-takeaways)
- [References](#references)
- [License](#license)

---

## Why these two tasks?

Both are binary text classification, but they stress different aspects of a language model:

- **IMDB sentiment** rewards understanding of *long-range context* — a review's polarity often depends on phrases scattered across hundreds of tokens.
- **Sarcasm detection** rewards sensitivity to *tone, irony, and world knowledge* — a one-line headline like *"area man constantly mentioning he doesn't own a television"* is syntactically identical to sincere writing.

Using the same backbone (`bert-base-uncased`) across both lets us study where transformer pre-training transfers well and where it struggles.

---

> **Trained models are not committed.** The IMDB checkpoint (~16 GB) and the sarcasm checkpoint (~440 MB) are excluded via `.gitignore`. Re-train locally using the instructions in each sub-project.

---

## Quick Start

```bash
git clone https://github.com/<your-username>/nlp-bert-classification.git
cd nlp-bert-classification

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# verify GPU (optional, strongly recommended for training)
python imdb-sentiment/src/check_gpu.py
```

Then jump into whichever project you want to run — each has its own README with dataset, training, and inference instructions.

---

## Project 1 — IMDB Sentiment Analysis

Binary sentiment classification on the Stanford IMDB Large Movie Review Dataset, comparing a fine-tuned BERT model against a TF-IDF + Linear SVM baseline.

**Highlights**
- Fine-tunes `bert-base-uncased` for 3 epochs at `lr=2e-5`, `batch_size=16`, `max_len=256`.
- TF-IDF baseline uses unigrams + bigrams with `LinearSVC`.
- Evaluates on the 25,000-review held-out test split.

See [`imdb-sentiment/README.md`](./imdb-sentiment/README.md) for full details.

---

## Project 2 — Sarcasm Detection

Binary sarcasm classification on news headlines scraped from *The Onion* (sarcastic) and *HuffPost* (genuine).

**Highlights**
- Fine-tunes `bert-base-uncased` for 4 epochs at `lr=1e-5`, `batch_size=16`, `max_len=128`.
- Dataset: [News Headlines Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) by Rishabh Misra (~28k headlines).
- Includes **error analysis**: false positives and false negatives are written to `reports/` after training to aid qualitative review.

### Training

```bash
python sarcasm-detection/src/train_sarcasm.py
```

### Inference

Put the sentence to classify in `sarcasm-detection/data/test_text.txt`, then:

```bash
python sarcasm-detection/src/predict_sarcasm.py
```

Example output:

```
Input: "scientists confirm water is, in fact, wet"
Prediction: Sarcastic  (confidence: 0.93)
```


---

## Results Summary

| Project  | Model              | Accuracy | F1    | Notes |
|----------|--------------------|:--------:|:-----:|-------|
| IMDB     | TF-IDF + LinearSVM | 0.884    | 0.884 | CPU, <1 min train |
| IMDB     | BERT (fine-tuned)  | 0.924    | 0.924 | ~35 min on T4 |
| Sarcasm  | BERT (fine-tuned)  | *TBD*    | *TBD* | See `sarcasm-detection/reports/` |

> Replace the sarcasm row with your actual numbers from the classification report printed at the end of `train_sarcasm.py`.

---

## Key Takeaways

- **Baselines matter.** A linear SVM with TF-IDF lands within ~4 points of BERT on IMDB while training 50× faster — worth running before reaching for a transformer.
- **Short text is harder than it looks.** Headline-level sarcasm removes most of the contextual cues BERT can exploit in long reviews, and error analysis reveals systematic failure modes (literal-sounding jokes, news that reads like satire, etc.).
- **Error analysis is non-optional.** The `false_positives.txt` / `false_negatives.txt` artifacts in the sarcasm project are more informative than aggregate metrics for guiding the next iteration.

