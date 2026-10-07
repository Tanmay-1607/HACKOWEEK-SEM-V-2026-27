# Week 9 – Week 10: Complete Scikit-Learn Workflow & Clustering

[![Scikit-Learn](https://img.shields.io/badge/Workflow-Scikit--Learn%20Pipelines-orange.svg)](https://scikit-learn.org/)
[![Evaluation](https://img.shields.io/badge/Evaluation-ROC--AUC%20%7C%20Cross--Validation-green.svg)](https://scikit-learn.org/stable/modules/model_evaluation.html)
[![Clustering](https://img.shields.io/badge/Clustering-K--Means%20%7C%20Hierarchical%20%7C%20DBSCAN-blue.svg)](https://scikit-learn.org/stable/modules/clustering.html)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-red.svg)](notebooks/week9_week10_ml_pipeline_clustering.ipynb)

## 👤 Student Information
- **Name:** Tanmay Sankulwar
- **PRN:** 24070521058
- **Program:** B.Tech Computer Science / Information Technology (5th Semester)
- **GitHub Profile:** [@Tanmay-1607](https://github.com/Tanmay-1607)
- **Repository:** [HACKOWEEK-SEM-V-2026-27](https://github.com/Tanmay-1607/HACKOWEEK-SEM-V-2026-27)

---

## 📌 Syllabus Overview (SIT-N Hack-o-Week 5th Semester)
- **Scikit-Learn Preprocessing Pipelines**:
  - Numerical feature imputation (`SimpleImputer` median) and standardization (`StandardScaler`).
  - Categorical feature encoding (`OneHotEncoder`) with `ColumnTransformer`.
  - Clean feature pipeline integration preventing data leakage during model training.
- **Model Evaluation & Diagnostics**:
  - Stratified Train/Test Split (80/20) preserving class distribution.
  - 5-Fold Cross-Validation (`cross_val_score`) for generalisation stability.
  - Confusion Matrix (TP, TN, FP, FN), Precision, Recall, Specificity, and F1-Score.
  - Receiver Operating Characteristic (ROC) curve and Area Under Curve (ROC-AUC) score.
- **Unsupervised Clustering**:
  - **K-Means Clustering**: Centroid optimization, inertia tracking, and silhouette analysis.
  - **Hierarchical Agglomerative Clustering**: Ward linkage and dendrogram cluster trees.
  - **DBSCAN**: Density-Based Spatial Clustering of Applications with Noise for outlier isolation.
  - 2D PCA visual projection of student behavioral cohorts (High Achievers, Balanced, At-Risk).

---

## 📓 Jupyter Notebook
The complete end-to-end coding walkthrough, confusion matrices, ROC curves, and clustering evaluations are in:
[`notebooks/week9_week10_ml_pipeline_clustering.ipynb`](notebooks/week9_week10_ml_pipeline_clustering.ipynb)

### Launch Notebook Locally:
```bash
cd "WEEK9 - WEEK10"
jupyter notebook notebooks/week9_week10_ml_pipeline_clustering.ipynb
```

---

## 🌐 Interactive ML Pipeline & Clustering Studio
An interactive web dashboard offering:
1. **⚡ Scikit-Learn Pipeline Visualizer**: Inspect the complete ColumnTransformer and Logistic Regression pipeline architecture, viewing hold-out metrics and cross-validation fold accuracies.
2. **📈 ROC-AUC & Evaluation**: Interactive Confusion Matrix breakdown and HTML5 Canvas ROC curve rendering with real-time AUC score display.
3. **🧬 Unsupervised Clustering Studio**: Interactively run K-Means, Hierarchical, or DBSCAN clustering; adjust cluster count $k$ or epsilon $\epsilon$; render a live 2D PCA scatter plot; and inspect student cohort demographic profiles.

---

## 🚀 How to Run

### 1. Requirements
```bash
pip install flask flask-cors scikit-learn pandas numpy jupyter
```

### 2. Launch Web Application
```bash
cd "WEEK9 - WEEK10"
python app.py
```
Open your browser and navigate to: **`http://localhost:5004`**

---

## 📂 Project Structure

```
WEEK9 - WEEK10/
├── app.py                      # Flask REST API server (port 5004)
├── workflow_engine.py          # Pipelines, CV, ROC-AUC, and clustering algorithms
├── README.md                   # Week 9-10 documentation
├── data/
│   └── kaggle_student_performance.csv  # Benchmark student dataset
├── notebooks/
│   └── week9_week10_ml_pipeline_clustering.ipynb # Comprehensive ML notebook
└── public/
    ├── index.html              # Interactive pipeline & clustering dashboard
    ├── css/
    │   └── styles.css          # Theme styling
    └── js/
        └── app.js              # Canvas ROC renderer, PCA scatter & clustering UI
```
