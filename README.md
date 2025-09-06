# Predictive Healthcare Claim Denial Model

### A machine learning project focused on predicting healthcare claim denials. We analyzed various strategies to overcome severe class imbalance, ultimately using decision threshold tuning to successfully identify nearly 100% of denials. This demonstrates a workflow for optimizing model sensitivity to meet specific business goals.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Key Findings](#key-findings)
- [Top Predictive Features](#top-predictive-features)
- [How to Run This Project](#how-to-run-this-project)
- [Future Work](#future-work)

---

## Problem Statement

Healthcare claim denials are a major source of administrative cost and lost revenue for providers. A denied claim must be manually reworked, appealed, and resubmitted, delaying payment and increasing the risk of a write-off.

The goal of this project was to build a machine learning model that can **proactively identify high-risk claims before submission**. By flagging these claims for review by an expert, a healthcare organization can correct errors upfront, increasing the clean claim rate and accelerating the revenue cycle. The primary challenge was a severe class imbalance, as denied claims represent a small minority of the total claim volume.

---

## Dataset

This project uses the synthetic **Enhanced Health Insurance Claims Dataset**, which is publicly available on Kaggle. It was designed to simulate real-world claim scenarios and includes patient demographics, provider information, and clinical codes.

- **Source:** [Kaggle Dataset Page](https://www.kaggle.com/datasets/leandrenash/enhanced-health-insurance-claims-dataset)

---

## Project Workflow

The project followed a systematic, multi-experiment approach to address the core challenge of class imbalance.

1.  **Exploratory Data Analysis (EDA):** Initial analysis confirmed that denied claims made up a small fraction of the dataset, highlighting the need for specialized modeling techniques.

2.  **Feature Engineering:** All categorical features (e.g., `ProviderSpecialty`, `ProcedureCode`) were converted into a numerical format. One-Hot Encoding was used for features with few categories, while Frequency Encoding was used for high-cardinality features like clinical codes.

3.  **Modeling Experiments:** Four distinct strategies were tested:
    * **SMOTE Oversampling:** Attempted to balance the dataset by creating synthetic denial samples.
    * **Class Weighting:** Forced the models to place a higher penalty on misclassifying the rare "denial" class.
    * **L1 Regularization (Lasso):** Used to automatically perform feature selection and reduce noise from thousands of features.
    * **Optimal Threshold Tuning:** Mathematically calculated the probability cutoff that maximized the F1-Score, balancing precision and recall.

4.  **Evaluation:** All models were evaluated based on their **precision** and **recall** for the "Denied" class, as accuracy is a misleading metric for imbalanced datasets.

---

## Key Findings

The results clearly showed that standard modeling techniques were ineffective, while tuning the model's sensitivity was the key to success.

| Model                 | Precision | Recall  | F1-Score |
| --------------------- | --------- | ------- | -------- |
| LR (SMOTE)            | 0.183     | 0.036   | 0.061    |
| RF (Weighted)         | 0.286     | 0.007   | 0.013    |
| LR (L1 Regularized)   | 0.231     | 0.050   | 0.082    |
| **RF (Optimal Thr.)** | **0.337** | **0.997** | **0.504**|

> **Key Takeaway:** The first three strategies failed, yielding near-zero recall. The **Optimal Threshold Tuning** strategy was the only successful approach. By lowering the decision threshold from the default 50% to a mathematically calculated 12%, we were able to increase denial detection **recall to 99.7%**. This came at the cost of precision, confirming that the features in this dataset are not strong enough to cleanly separate the classes without a trade-off.

---

## Top Predictive Features

The feature importance analysis from our best model revealed that patient demographic and financial data were surprisingly more predictive than the clinical codes in this dataset.

**Top 5 Features:**
1.  `PatientIncome`
2.  `ClaimAmount`
3.  `PatientAge`
4.  `PatientGender_M`
5.  `ClaimSubmissionMethod_Paper`

---

## How to Run This Project

To replicate this analysis, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/brennansk1/Claim-Denial-Prediction.git](https://github.com/brennansk1/Claim-Denial-Prediction.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas scikit-learn imbalanced-learn matplotlib seaborn jupyter
    ```
3.  **Download the dataset** from the [Kaggle link](https://www.kaggle.com/datasets/leandrenash/enhanced-health-insurance-claims-dataset) and place it in the project directory.

4.  **Launch Jupyter Notebook** and open the main analysis notebook to run the project.

---

## Future Work

While this project successfully created a proof-of-concept, the next steps would focus on making the model more practical for business use:

1.  **Constrained Optimization:** Instead of maximizing F1-score, tune the threshold to meet a specific business goal, such as "maximize precision while maintaining at least 75% recall."
2.  **Cost-Based Optimization:** Assign dollar values to model errors (e.g., cost of a false positive vs. cost of a missed denial) and find the threshold that minimizes the overall financial impact.
3.  **Advanced Feature Engineering:** Create more powerful features, such as interactions between provider specialty and procedure codes, to improve the model's ability to separate classes without sacrificing as much precision.
