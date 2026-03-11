# Fake News Detection with BERT and Explainability Analysis

## Overview
This project builds a fake news classification system using BERT fine-tuning and analyzes model behavior through explainability methods (SHAP) and systematic error analysis. The focus is not only on classification performance but on understanding **where and why the model fails**, which is directly relevant to real-world disinformation detection systems.

## Dataset
- **Source:** [GonzaloA/fake_news](https://huggingface.co/datasets/GonzaloA/fake_news)
- **Train:** 24,353 articles / **Test:** 8,117 articles
- **Labels:** 0 = Real, 1 = Fake
- **Used:** 10,000 train samples / 2,000 test samples (T4 GPU)

## Pipeline
1. **EDA** - Label distribution, text length analysis
2. **Baseline** - TF-IDF + Logistic Regression
3. **BERT Fine-tuning** - bert-base-uncased, 3 epochs
4. **Explainability** - SHAP PartitionExplainer
5. **Error Analysis** - Misclassification pattern analysis

## Results

### Model Comparison
| Model | Accuracy | F1 (Real) | F1 (Fake) |
|-------|----------|-----------|-----------|
| TF-IDF + Logistic Regression | 97% | 0.97 | 0.97 |
| BERT fine-tuned | 99% | 0.99 | 0.99 |

### Error Analysis
- Total misclassified: **21 out of 2,000 (1.1%)**
- False Positive (Real → Fake): 7
- False Negative (Fake → Real): 14

![Error Analysis](error_analysis.png)

## Key Findings

**1. Political content causes higher misclassification**
Keywords such as *trump*, *clinton*, *hillary*, and *election* appear heavily in both false positive and false negative cases. The model shows instability on politically charged content, likely reflecting biases in the training data.

**2. Fake news with professional vocabulary evades detection**
False negative cases frequently contain words like *pipeline*, *foundation*, *teneo*, and *jihad* — terms associated with formal or institutional language. Fake news written in a professional tone is harder to detect.

**3. Longer texts are harder to classify**
Misclassified samples averaged 3,093 characters vs 2,556 for correctly classified samples — a difference of 537 characters. Longer articles may contain more mixed signals that confuse the model.

**4. Lexical bias over contextual understanding**
SHAP analysis shows that the model relies heavily on specific words rather than contextual meaning. For example, a Reuters article about the Russian consulate was misclassified as fake, likely due to politically sensitive terms like *Moscow* and *Russian*.

## Relevance to Disinformation Research
These findings highlight that high overall accuracy does not guarantee reliable detection in practice. Disinformation that mimics professional writing or exploits politically sensitive topics can systematically evade detection. This has direct implications for platform-level disinformation monitoring, where false negatives carry significant social cost.

## Setup
```bash
pip install datasets transformers shap torch
```

## Usage
Open `fakenews_bert.ipynb` in Google Colab (GPU recommended) and run cells sequentially.

## Tech Stack
- Python, PyTorch
- HuggingFace Transformers (bert-base-uncased)
- SHAP
- scikit-learn
- pandas, matplotlib, seaborn
