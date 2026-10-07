# Week 11 – Week 12: Dimensionality Reduction with PCA & t-SNE

[![PCA](https://img.shields.io/badge/Dimensionality%20Reduction-PCA-blue.svg)](https://scikit-learn.org/stable/modules/decomposition.html#pca)
[![t-SNE](https://img.shields.io/badge/Manifold%20Learning-t--SNE-green.svg)](https://scikit-learn.org/stable/modules/manifold.html#t-sne)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)](https://scikit-learn.org/)

## 👤 Student Information
- **Name:** Tanmay Sankulwar
- **PRN:** 24070521058
- **Program:** B.Tech Computer Science / Information Technology (5th Semester)
- **GitHub Profile:** [@Tanmay-1607](https://github.com/Tanmay-1607)
- **Repository:** [HACKOWEEK-SEM-V-2026-27](https://github.com/Tanmay-1607/HACKOWEEK-SEM-V-2026-27)

---

## 📌 Syllabus Overview (SIT-N Hack-o-Week 5th Semester)
This practical applies **Principal Component Analysis (PCA)** and **t-distributed Stochastic Neighbor Embedding (t-SNE)** to the multi-dimensional student academic dataset from Week 9–10.

### Features Analyzed:
- `CGPA`
- `AttendanceRate`
- `StudyHoursPerWeek`
- `MathScore`
- `ReadingScore`
- `WritingScore`

*All numeric features are standardized using `StandardScaler` prior to dimensionality reduction to prevent variables with larger numeric scales from dominating the variance axes.*

---

## 💡 Theoretical Concepts

### 1. Principal Component Analysis (PCA)
- **Linear Transformation**: Identifies orthogonal axes (principal components) maximizing the variance of projected data points.
- **Dimensionality Reduction**: Compresses multi-dimensional correlated variables into a compact set of uncorrelated components.
- **Explained Variance**: Quantifies the exact ratio of total information retained across each component.
- **Global Structure Preservation**: Retains large-scale pairwise distances and global geometric variance.

### 2. t-Distributed Stochastic Neighbor Embedding (t-SNE)
- **Non-Linear Manifold Learning**: Converts pairwise similarities into probabilities and maps them into low-dimensional space.
- **Local Neighborhood Preservation**: Clusters points that are close in high-dimensional feature space.
- **Exploratory Visualization**: Ideal for discovering clusters and manifold geometry in 2D projections.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Execute the Dimensionality Reduction Script
From the repository root:
```bash
python "WEEK11 - WEEK12/dimensionality_reduction.py"
```

The script prints the PCA explained variance ratios and saves visual figures:
- `pca_projection.png` — 2D scatter plot of students projected onto PC1 and PC2.
- `pca_explained_variance.png` — Component-wise and cumulative explained variance scree plot.
- `tsne_projection.png` — 2D t-SNE manifold projection.

---

## 📂 Project Structure

```
WEEK11 - WEEK12/
├── README.md                   # Week 11-12 documentation
└── dimensionality_reduction.py # PCA & t-SNE modeling and plotting script
```
