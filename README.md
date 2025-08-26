# The Laws of Anomaly: A Framework for Regression Model Selection

### Official Code and Data Repository for the Thesis by Priyanuj Boruah, IIT Madras

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open Science](https://img.shields.io/badge/Open%20Science-Reproducible-blue)](https://github.com/priyanuj/laws-of-anomaly-thesis)

**[📜 Read the Full Thesis (PDF)](./Thesis.pdf)**

---

This repository contains the complete codebase, datasets, and analysis scripts for the undergraduate thesis, *"The Laws of Anomaly: A New Framework for Regression Model Selection Based on a Large-Scale Empirical Study of Structural Data Challenges"*.

## Abstract

The prevalent "ensemble-first" strategy in machine learning, while effective, often overlooks specific data challenges where it is suboptimal. This research introduces the **Efficiency-Based Model Selection Framework (EMSF)**, a new methodology for aligning model architecture with a dataset's primary structural challenge. Through a benchmark of over 20 models across 100+ datasets, this work establishes three fundamental, data-driven laws of applied regression that provide a practical guide for practitioners to move beyond a one-size-fits-all approach.

---

## 📜 The Three Laws of Anomaly

The core findings of this research are summarized in three practical laws:

### I. The Law of Ensemble Dominance
> In the absence of specific, identifiable data challenges, the "ensemble-first" approach is a statistically sound starting point.

Our research provides extensive evidence that **Tree-Based Ensembles** (e.g., XGBoost, LightGBM, Random Forest) are the Point of Maximum Efficiency in **over 70%** of standard regression tasks.

### II. The Law of Anomaly Supremacy
> The most critical skill in model selection is not defaulting to the most powerful model, but identifying the data anomalies that demand a specialized one.

This research proves two critical anomalies where ensemble dominance breaks down:
* **The Wide Data Anomaly:** For high-dimensional data (many features, few samples), simple **K-Nearest Neighbors (KNN)** emerges as a surprisingly powerful and often superior model, especially for sensor-like data.
* **The Hidden Outlier Anomaly:** For datasets with hidden outliers or structural breaks, robust models like the **Huber Regressor** act as "silver bullets," succeeding with performance margins **exceeding 1500%** where all other models, including ensembles, fail catastrophically.

### III. The Law of Predictive Futility
> A comprehensive benchmark serves as a powerful diagnostic tool for the data itself.

The consistent, cross-family failure of models is a reliable indicator of a fundamental lack of predictive signal within a dataset's features. This provides a clear directive to **halt model tuning and return to feature engineering**.

---

## 📂 Repository Structure


.
├── 📜 Thesis.pdf
├── 📊 data/
│   ├── cohort_assignments.csv
│   └── regression_datasets/
│       ├── dataset_1.csv
│       ├── dataset_2.csv
│       └── ... (100+ datasets)
├── 🔬 scripts/
│   ├── 1_run_benchmarks.py
│   └── 2_statistical_analyzer.py
├── 📝 results/
│   └── regression_observations.txt
├── requirements.txt
└── README.md

* **Thesis.pdf:** The full text of the research paper.
* **data/:** Contains all 100+ datasets used in the study, categorized by cohort.
* **scripts/:** Contains the Python scripts to reproduce the analysis.
    * `1_run_benchmarks.py`: (Optional) The script to re-run the full PyCaret benchmark on all datasets.
    * `2_statistical_analyzer.py`: The script to parse the results and perform the statistical analysis.
* **results/:** The raw output file from the benchmark (`regression_observations.txt`).
* **requirements.txt:** A list of all necessary Python libraries.

---

## 🚀 How to Reproduce the Analysis

You can reproduce the statistical analysis from the final results file (`regression_observations.txt`).

### Prerequisites
* Python 3.8+
* Git

### 1. Clone the Repository
```bash
git clone [https://github.com/priyanuj/laws-of-anomaly-thesis.git](https://github.com/priyanuj/laws-of-anomaly-thesis.git)
cd laws-of-anomaly-thesis

2. Set Up a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install Dependencies
pip install -r requirements.txt

4. Run the Statistical Analysis
The analysis script will parse the raw results and print the statistical test outcomes to your terminal.

python scripts/2_statistical_analyzer.py

✍️ How to Cite This Work
If you use this framework, the datasets, or the findings from this research in your own work, please cite the original thesis:

@misc{boruah2025lawsofanomaly,
  author       = {Priyanuj Boruah},
  title        = {The Laws of Anomaly: A New Framework for Regression Model Selection Based on a Large-Scale Empirical Study of Structural Data Challenges},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{[https://github.com/priyanuj/laws-of-anomaly-thesis](https://github.com/priyanuj/laws-of-anomaly-thesis)}}
}
