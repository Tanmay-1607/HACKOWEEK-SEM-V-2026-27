# Week 7 – Week 8: Supervised Machine Learning (Regression & Classification)

[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![Regression](https://img.shields.io/badge/Regression-Linear%20%7C%20Poly%20%7C%20Ridge%20%7C%20Lasso-blue.svg)](https://scikit-learn.org/stable/modules/linear_model.html)
[![Classification](https://img.shields.io/badge/Classification-Logistic%20%7C%20KNN-green.svg)](https://scikit-learn.org/stable/modules/neighbors.html)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-red.svg)](notebooks/week7_week8_regression_classification.ipynb)

## 👤 Student Information
- **Name:** Tanmay Sankulwar
- **PRN:** 24070521058
- **Program:** B.Tech Computer Science / Information Technology (5th Semester)
- **GitHub Profile:** [@Tanmay-1607](https://github.com/Tanmay-1607)
- **Repository:** [HACKOWEEK-SEM-V-2026-27](https://github.com/Tanmay-1607/HACKOWEEK-SEM-V-2026-27)

---

## 📌 Syllabus Overview (SIT-N Hack-o-Week 5th Semester)
- **Regression Algorithms**:
  - **Linear Regression (OLS)**: Baseline Ordinary Least Squares modeling of CGPA based on attendance and study habits.
  - **Polynomial Regression**: Degree-2 feature interactions capturing non-linear relationships.
  - **Ridge Regression ($L_2$ Regularization)**: Shrinkage penalty preventing multicollinearity and extreme coefficients.
  - **Lasso Regression ($L_1$ Regularization)**: Feature selection via sparse coefficient shrinkage.
- **Classification Algorithms**:
  - **Logistic Regression**: Sigmoid probability estimation for classifying students graduating with Academic Distinction (`CGPA >= 8.5`).
  - **K-Nearest Neighbors (KNN)**: Non-parametric instance-based voting using scaled Euclidean distance.

---

## 📓 Jupyter Notebook
The complete model comparisons, mathematical formulations, confusion matrices, and code walkthrough are in:
[`notebooks/week7_week8_regression_classification.ipynb`](notebooks/week7_week8_regression_classification.ipynb)

### Launch Notebook Locally:
```bash
cd "WEEK7 - WEEK8"
jupyter notebook notebooks/week7_week8_regression_classification.ipynb
```

---

## 🌐 Interactive Supervised ML Studio
An interactive web dashboard offering:
1. **📊 Model Benchmarks Scorecard**: Side-by-side comparative metrics ($R^2$, MSE, Accuracy, F1-Score) across all 6 models.
2. **🎯 Live Student Predictor**: Dynamic sliders for study hours, attendance rate, and exam marks to produce live regression predictions and distinction probabilities.
3. **🎛️ Real-Time Hyperparameter Tuning**: Adjust Ridge $\alpha$, Lasso $\alpha$, and KNN $k$ dynamically to observe instant updates to model weights and validation metrics.

---

## 🚀 How to Run

### 1. Requirements
```bash
pip install flask flask-cors scikit-learn pandas numpy jupyter
```

### 2. Launch the Web Studio
```bash
cd "WEEK7 - WEEK8"
python app.py
```
Then navigate to: **`http://localhost:5003`**

---

## 📂 Project Structure

```
WEEK7 - WEEK8/
├── app.py                      # Flask REST API server (port 5003)
├── ml_engine.py                # Model training, prediction & evaluation pipeline
├── README.md                   # Week 7-8 documentation
├── data/
│   └── kaggle_student_performance.csv  # Benchmark student dataset
├── notebooks/
│   └── week7_week8_regression_classification.ipynb # Complete ML notebook
└── public/
    ├── index.html              # Supervised ML interactive dashboard
    ├── css/
    │   └── styles.css          # Theme styling
    └── js/
        └── app.js              # Live predictions, benchmark cards & UI logic
```
