# Week 5 – Week 6: Mathematics for Machine Learning (Linear Algebra & Calculus)

[![Linear Algebra](https://img.shields.io/badge/Linear%20Algebra-Vectors%20%26%20Matrices-purple.svg)](https://en.wikipedia.org/wiki/Linear_algebra)
[![Calculus](https://img.shields.io/badge/Calculus-Gradients%20%26%20Chain%20Rule-blue.svg)](https://en.wikipedia.org/wiki/Differential_calculus)
[![PCA](https://img.shields.io/badge/Dimensionality%20Reduction-Eigen%20Decomposition-green.svg)](https://en.wikipedia.org/wiki/Principal_component_analysis)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)](notebooks/week5_week6_linear_algebra_calculus.ipynb)

## 👤 Student Information
- **Name:** Tanmay Sankulwar
- **PRN:** 24070521058
- **Program:** B.Tech Computer Science / Information Technology (5th Semester)
- **GitHub Profile:** [@Tanmay-1607](https://github.com/Tanmay-1607)
- **Repository:** [HACKOWEEK-SEM-V-2026-27](https://github.com/Tanmay-1607/HACKOWEEK-SEM-V-2026-27)

---

## 📌 Syllabus Overview (SIT-N Hack-o-Week 5th Semester)
- **Linear Algebra**:
  - Vectors, Euclidean norm, dot products, vector projections, and cosine similarity.
  - Matrices, matrix multiplication, inverse, transpose, and determinant computation.
  - Eigenvalues and eigenvectors: Intuition-level decomposition of student mark covariance matrix for PCA.
- **Calculus**:
  - Derivatives, gradient vectors $\nabla J$, and multivariable optimization.
  - Gradient descent implementation for fitting Study Hours vs CGPA on the Kaggle dataset.
  - Chain rule applied to computational graphs for neural network backpropagation intuition.

---

## 📓 Jupyter Notebook
The complete mathematical derivations, NumPy implementations, and visual step-by-step proofs are contained in:
[`notebooks/week5_week6_linear_algebra_calculus.ipynb`](notebooks/week5_week6_linear_algebra_calculus.ipynb)

### Launch Notebook Locally:
```bash
cd "WEEK5 - WEEK6"
jupyter notebook notebooks/week5_week6_linear_algebra_calculus.ipynb
```

---

## 🌐 Interactive Math for ML Studio
An interactive web application featuring 5 real-time math exploration modules:
1. **⚡ Vector Operations**: Computes Euclidean norm, dot product, angle in degrees, and cosine similarity between interactive vectors.
2. **📐 2D Matrix Visualizer**: Canvas rendering linear transformations ($2 \times 2$) on the unit square and calculating determinants (area scaling factor).
3. **🧬 Covariance Matrix & Eigenvalues (PCA)**: Interactive eigen-decomposition of Kaggle student academic marks (Math, Reading, Writing) to derive principal variance directions.
4. **📉 Gradient Descent Simulator**: Configurable learning rate $\alpha$ and epochs, rendering real-time MSE loss convergence curves.
5. **🔗 Chain Rule & Backpropagation Trace**: Visualizes forward pass and backward error propagation through a composite scalar function.

---

## 🚀 How to Run

### 1. Requirements
```bash
pip install flask flask-cors numpy pandas matplotlib jupyter
```

### 2. Launch the Application
```bash
cd "WEEK5 - WEEK6"
python app.py
```
Open your browser and navigate to: **`http://localhost:5002`**

---

## 📂 Project Structure

```
WEEK5 - WEEK6/
├── app.py                      # Flask server with API endpoints (port 5002)
├── math_engine.py              # Mathematical calculation backend algorithms
├── README.md                   # Week 5-6 documentation
├── data/
│   └── kaggle_student_performance.csv  # Benchmark student dataset
├── notebooks/
│   └── week5_week6_linear_algebra_calculus.ipynb # Step-by-step math notebook
└── public/
    ├── index.html              # Interactive canvas & simulation interface
    ├── css/
    │   └── styles.css          # Theme styling
    └── js/
        └── app.js              # Vector rendering, matrix math & chart logic
```
