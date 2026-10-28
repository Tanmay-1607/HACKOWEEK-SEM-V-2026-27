# Week 15 – Week 16: Neural Networks from Scratch & Model Deployment Basics

[![Deployment](https://img.shields.io/badge/Deployment-Pickle%20%7C%20Joblib-green.svg)](https://scikit-learn.org/stable/model_persistence.html)
[![NumPy](https://img.shields.io/badge/Neural_Net-Pure%20NumPy-blue.svg)](https://numpy.org/)
[![Optimizers](https://img.shields.io/badge/Optimizers-SGD%20%7C%20Adam-orange.svg)](https://arxiv.org/abs/1412.6980)
[![Notebook](https://img.shields.io/badge/Notebook-Jupyter-red.svg)](notebooks/week15_week16_neural_network_deployment.ipynb)

## 👤 Student Information
- **Name:** Tanmay Sankulwar
- **PRN:** 24070521058
- **Program:** B.Tech Computer Science / Information Technology (5th Semester)
- **GitHub Profile:** [@Tanmay-1607](https://github.com/Tanmay-1607)
- **Repository:** [HACKOWEEK-SEM-V-2026-27](https://github.com/Tanmay-1607/HACKOWEEK-SEM-V-2026-27)

---

## 📌 Syllabus Overview (SIT-N Hack-o-Week 5th Semester)

### 1. Model Deployment Basics (Saving models with `pickle` & `joblib`)
- **Serialization & Deserialization Protocols:**
  - `pickle`: Native Python object serialization protocol.
  - `joblib`: High-performance disk caching and serialization optimized for heavy NumPy multi-dimensional arrays and compressed machine learning pipelines.
- **Performance Benchmarks:**
  - Measuring write latency (save time in ms), read latency (deserialization in ms), and artifact footprint (KB) on disk.
  - End-to-end inference verification ensuring deserialized pipelines reproduce identical classification outputs.
- **Production REST Serving:**
  - Live prediction endpoint dynamically loading serialized artifacts from disk to compute real-time inferences.

### 2. Perceptrons, Activation Functions, Forward & Backward Propagation
- **The Perceptron (Rosenblatt, 1958):**
  - Linear decision boundary: $z = \mathbf{w}^T \mathbf{x} + b$.
  - Perceptron learning rule: $\mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat{y}) \mathbf{x}$.
  - Boolean logic gates evaluation: AND, OR, NAND (converge linearly) vs. XOR (linearly in-separable; proving the necessity of multi-layer architectures).
- **Activation Functions & Analytical Derivatives:**
  - **Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$, $\quad \sigma'(z) = \sigma(z)(1 - \sigma(z))$
  - **ReLU:** $f(z) = \max(0, z)$, $\quad f'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}$
  - **Leaky ReLU:** $f(z) = \max(\alpha z, z)$, $\quad f'(z) = \begin{cases} 1 & z > 0 \\ \alpha & z \le 0 \end{cases}$
  - **Tanh:** $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, $\quad \tanh'(z) = 1 - \tanh^2(z)$
- **Computational Graph & Chain Rule:**
  - Forward propagation with activation caching: $a^{[l]} = g^{[l]}(W^{[l]} a^{[l-1]} + b^{[l]})$.
  - Backward pass recursively evaluating gradients: $\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{1}{m} (a^{[l-1]})^T \delta^{[l]}$.

### 3. Loss Functions & Optimizers (SGD & Adam)
- **Loss Formulations:**
  - **Binary Cross-Entropy (Log Loss):** $\mathcal{L} = -\frac{1}{m} \sum \left[ y \ln(\hat{y}) + (1-y)\ln(1-\hat{y}) \right]$
  - **Mean Squared Error (MSE):** $\mathcal{L} = \frac{1}{2m} \sum (\hat{y} - y)^2$
- **Optimization Algorithms from Scratch:**
  - **Stochastic Gradient Descent (SGD):** $\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$
  - **SGD with Momentum:** $v_t = \beta v_{t-1} + (1-\beta) \nabla_\theta \mathcal{L}, \quad \theta \leftarrow \theta - \alpha v_t$
  - **Adam (Adaptive Moment Estimation):**
    $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
    $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
    $$\theta \leftarrow \theta - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### 4. Deep Neural Network from Scratch in NumPy
- Fully object-oriented, modular Multi-Layer Perceptron (MLP) written strictly in NumPy without high-level neural network wrappers.
- Configurable layer architecture, He/Xavier initializations, forward propagation, backpropagation with vectorised gradients, and adaptive optimizer updates.
- Evaluated on the Student Performance dataset to classify Distinction students with confusion matrix and ROC-AUC tracking.

---

## 📓 Jupyter Notebook
The complete end-to-end coding walkthrough is in:
[`notebooks/week15_week16_neural_network_deployment.ipynb`](notebooks/week15_week16_neural_network_deployment.ipynb)

To launch the notebook:
```bash
cd "WEEK 15 - WEEK 16"
jupyter notebook notebooks/week15_week16_neural_network_deployment.ipynb
```

---

## 🌐 Interactive Neural Networks & Deployment Studio
An interactive web dashboard has been developed allowing users to:
1. **🚀 Model Deployment**: Compare Pickle vs. Joblib serialization latency and file size, and submit live student profiles to trigger real-time predictions directly from saved artifacts.
2. **⚡ Perceptrons & Activations**: Test Perceptron convergence across AND, OR, NAND, and XOR logic gates, and explore interactive curves & analytical derivatives for Sigmoid, ReLU, Leaky ReLU, and Tanh.
3. **📈 Losses & Optimizers**: Compare training loss and validation accuracy curves between SGD, Momentum, and Adam over 100 epochs side-by-side.
4. **🔬 Neural Net from Scratch**: Dynamically build custom neural network architectures (neurons, activations, optimizers, learning rate, epochs), trigger pure NumPy training, and inspect real-time loss curves, accuracy, and confusion matrix.

### How to Run the Web Dashboard:
```bash
cd "WEEK 15 - WEEK 16"
python app.py
```
Then navigate to: **`http://localhost:5007`**

---

## 📂 Project Structure
```
WEEK 15 - WEEK 16/
├── app.py                      # Flask REST API server (port 5007)
├── neural_engine.py            # Pure NumPy NN, optimizers, perceptron, deployment benchmarks
├── README.md                   # Detailed documentation
├── data/
│   └── kaggle_student_performance.csv
├── notebooks/
│   └── week15_week16_neural_network_deployment.ipynb
├── saved_models/
│   ├── student_model.pkl       # Serialized Pickle pipeline artifact
│   └── student_model.joblib    # Serialized Joblib compressed artifact
└── public/
    ├── index.html              # Interactive dark glassmorphic web studio
    ├── css/
    │   └── styles.css          # Design system, styling tokens, glow animations
    └── js/
        └── app.js              # Real-time chart rendering & API client
```
