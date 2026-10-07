# Week 3 – Week 4: Python Essentials, NumPy, Pandas & Data Visualization

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Array%20Broadcasting-013243.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Cleaning%20%26%20GroupBy-150458.svg)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Visualization-Matplotlib%20%26%20Seaborn-green.svg)](https://seaborn.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)](notebooks/week3_week4_python_numpy_pandas_viz.ipynb)

## 👤 Student Information
- **Name:** Tanmay Sankulwar
- **PRN:** 24070521058
- **Program:** B.Tech Computer Science / Information Technology (5th Semester)
- **GitHub Profile:** [@Tanmay-1607](https://github.com/Tanmay-1607)
- **Repository:** [HACKOWEEK-SEM-V-2026-27](https://github.com/Tanmay-1607/HACKOWEEK-SEM-V-2026-27)

---

## 📌 Syllabus Overview (SIT-N Hack-o-Week 5th Semester)
- **Python Essentials**: Functions, Object-Oriented Programming (OOP with `StudentRecord`), and List/Dict comprehensions.
- **NumPy**: Multidimensional arrays, broadcasting, and vectorized operations (z-score standardization, composite score generation).
- **Pandas**: DataFrames, data cleaning, relational merging with course enrollments (`pd.merge`), and multi-metric `groupby` aggregations.
- **Data Visualization**: Publication-quality plots using Matplotlib and Seaborn (KDE distributions, departmental boxplots, study hours vs CGPA regression, correlation heatmap).

---

## 📓 Jupyter Notebook
The complete coding walkthrough and visual outputs are contained in:
[`notebooks/week3_week4_python_numpy_pandas_viz.ipynb`](notebooks/week3_week4_python_numpy_pandas_viz.ipynb)

### Launch Notebook Locally:
```bash
cd "WEEK3 - WEEK4"
jupyter notebook notebooks/week3_week4_python_numpy_pandas_viz.ipynb
```

---

## 🌐 Interactive Web Dashboard
An interactive browser interface allowing users to:
1. **📊 Visual Gallery**: Explore all 4 generated Matplotlib/Seaborn visualization plots with interactive analysis summaries.
2. **🔍 Live Pandas GroupBy**: Run live server-side GroupBy queries (select dimension, metric, and aggregation function: Mean, Median, Min, Max, Count).
3. **🔗 Relational Data Explorer**: Inspect Students, Course Enrollments, and Merged Inner-Join tables with pagination and sorting.
4. **⚡ NumPy Vectorization & Array Broadcasting**: Live Z-score standardization calculator demonstrating broadcasting across student marks.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install flask flask-cors pandas numpy matplotlib seaborn jupyter
```

### 2. Generate Plots (Optional)
```bash
cd "WEEK3 - WEEK4"
python analysis.py
```

### 3. Launch Web Server
```bash
cd "WEEK3 - WEEK4"
python app.py
```
Navigate to: **`http://localhost:5001`**

---

## 📂 Project Structure

```
WEEK3 - WEEK4/
├── analysis.py                 # Standalone script generating visualization plots
├── app.py                      # Flask web server (port 5001)
├── README.md                   # Week 3-4 documentation
├── data/
│   ├── course_enrollments.csv  # Relational enrollment data
│   └── kaggle_student_performance.csv  # Primary student dataset
├── notebooks/
│   └── week3_week4_python_numpy_pandas_viz.ipynb # Jupyter notebook
└── public/
    ├── index.html              # Interactive dashboard UI
    ├── css/
    │   └── styles.css          # Theme and component styles
    ├── js/
    │   └── app.js              # Client-side dynamic interaction
    └── plots/                  # Generated visualization assets
        ├── cgpa_distribution.png
        ├── correlation_heatmap.png
        ├── department_boxplot.png
        └── study_hours_vs_cgpa.png
```
