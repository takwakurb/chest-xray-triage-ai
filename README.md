# AI-Assisted Chest X-Ray Triage

A machine learning system that flags chest X-rays for automated processing or defers them to a radiologist, using a confidence-based **selective prediction** framework. Final year project, BSc Computer Science, University of Birmingham.

## What it does

Most chest X-ray classifiers treat every prediction as final. This project instead asks: *when should a model's prediction be trusted, and when should it be deferred to a human?*

A DenseNet121 classifier is fine-tuned on the CheXpert-small dataset to detect 5 pathology labels. A second, lightweight model estimates how likely each prediction is to be wrong, using the classifier's own confidence as a signal. Cases the system is confident about are auto-accepted; everything else is routed to a radiologist — trading some coverage for a strong safety guarantee.

**At the selected operating point:**
| | Coverage | Accuracy | False Negatives |
|---|---|---|---|
| Unfiltered classifier | 100% | 79.7% | 8 |
| **Triage system** | **51%** | **92.2%** | **0** |

A random-selection baseline at the same coverage confirms the gain comes from informed selection, not just seeing fewer cases (79.7% accuracy, ~4.2 false negatives on average).

The classifier itself reaches a macro AUROC of **0.830**, outperforming a pretrained TorchXRayVision baseline on 4 of 5 labels.

## How it works

1. **Classifier** — DenseNet121 (ImageNet-pretrained, fine-tuned) produces per-label probabilities for Lung Opacity, Pleural Effusion, Edema, Cardiomegaly, and Pneumothorax.
2. **Confidence model** — A logistic regression model estimates the probability of a case-level error, using the classifier's own maximum predicted probability as the input signal.
3. **Selective prediction layer** — Cases below an error-probability threshold are accepted automatically; the rest are deferred to a radiologist.

Trained with a masked binary cross-entropy loss (to handle CheXpert's "uncertain" labels without introducing label noise) and patient-level splitting (to prevent data leakage between train/val/test).

## Repo structure

```
notebooks/
  tuned_model_final.ipynb            final DenseNet121 training run
  confidence_reliability_analysis.ipynb   confidence signal selection & modeling
  evaluation.ipynb                   full test set evaluation
demo.py                              Streamlit demo with Grad-CAM visualisation
```

## Running the demo

**Install dependencies:**
```bash
pip install streamlit torch torchvision pillow numpy joblib opencv-python pytorch-grad-cam
```

**Download model weights:**
Model weights aren't included in this repo (too large for git) — download them and place the contents into a `model/` folder in the repo root:
[Model weights](https://drive.google.com/drive/folders/1SPNnis7miP5vRAyb_WKsYBJD2Y9GR09w?usp=sharing)

**Sample test images** (optional, for trying out the demo):
[Sample images](https://drive.google.com/drive/folders/1cDM4dOTuWZxo_mCJXcAPw3a3N7r7k539?usp=sharing)

**Launch:**
```bash
streamlit run demo.py
```

## Tech stack

Python, PyTorch, Streamlit, scikit-learn — trained on Google Colab (T4 GPU), CheXpert-small accessed via KaggleHub.

## Key results & honest limitations

- Test set is small (n=202), so results should be read with wide confidence intervals in mind.
- The confidence model's AUROC drops from 0.830 (validation) to 0.705 (test) — a known generalisation gap, discussed in detail in the [full report](#).
- Pneumothorax was the least stable label (low prevalence, low AUPRC) and drives most false positives — flagged as a structural limitation rather than smoothed over.

Full methodology, evaluation, and discussion available in the accompanying project report.
