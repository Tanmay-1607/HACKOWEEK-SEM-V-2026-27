"""
Week 15 - Week 16: Neural Networks from Scratch & Model Deployment Basics
Author: Tanmay Sankulwar (PRN: 24070521058)
HACKOWEEK Semester V (2026-27) - SIT Nagpur

Features:
1. Model Deployment Basics (Pickle vs Joblib serialization, latency benchmarks, saved model inference)
2. Perceptrons & Activation Functions (AND, OR, NAND, XOR linearly separable logic gates, activation curves & derivatives)
3. Loss Functions & Optimizers from Scratch (SGD, Momentum, Adam math & convergence comparison)
4. Deep Neural Network from Scratch in NumPy (Arbitrary dense layers, forward pass, backpropagation, optimizer updates)
"""

import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'kaggle_student_performance.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

NUMERIC_FEATURES = [
    'Age', 'StudyHoursPerWeek', 'AttendanceRate',
    'MathScore', 'ReadingScore', 'WritingScore'
]
CATEGORICAL_FEATURES = ['Gender', 'Department']


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION & PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found at {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    df['Distinction'] = (df['CGPA'] >= 8.5).astype(int)
    return df


def get_preprocessed_data():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['Distinction'].values

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), NUMERIC_FEATURES),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), CATEGORICAL_FEATURES)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return {
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train': X_train_proc,
        'X_test': X_test_proc,
        'preprocessor': preprocessor,
        'feature_names': (
            NUMERIC_FEATURES +
            list(preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(CATEGORICAL_FEATURES))
        )
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL DEPLOYMENT BASICS: PICKLE VS JOBLIB BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

PICKLE_PATH = os.path.join(MODEL_DIR, 'student_model.pkl')
JOBLIB_PATH = os.path.join(MODEL_DIR, 'student_model.joblib')

def train_and_benchmark_deployment():
    data = get_preprocessed_data()
    X_train_raw = data['X_train_raw']
    y_train = data['y_train']
    X_test_raw = data['X_test_raw']
    y_test = data['y_test']

    # Production End-to-End Pipeline
    full_pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), NUMERIC_FEATURES),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), CATEGORICAL_FEATURES)
        ])),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42
        ))
    ])

    full_pipeline.fit(X_train_raw, y_train)
    y_pred = full_pipeline.predict(X_test_raw)
    test_acc = float(accuracy_score(y_test, y_pred))

    # Benchmark Pickle
    t0 = time.perf_counter()
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(full_pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_save_time_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    with open(PICKLE_PATH, 'rb') as f:
        loaded_pickle = pickle.load(f)
    pickle_load_time_ms = (time.perf_counter() - t0) * 1000.0
    pickle_size_bytes = os.path.getsize(PICKLE_PATH)

    # Benchmark Joblib
    t0 = time.perf_counter()
    joblib.dump(full_pipeline, JOBLIB_PATH, compress=3)
    joblib_save_time_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    loaded_joblib = joblib.load(JOBLIB_PATH)
    joblib_load_time_ms = (time.perf_counter() - t0) * 1000.0
    joblib_size_bytes = os.path.getsize(JOBLIB_PATH)

    # Verification: Ensure loaded models predict identical outputs
    p1 = loaded_pickle.predict(X_test_raw.iloc[:5])
    p2 = loaded_joblib.predict(X_test_raw.iloc[:5])
    verified = bool(np.array_equal(p1, p2))

    return {
        "status": "success",
        "model_architecture": "Pipeline [ColumnTransformer -> MLPClassifier(32, 16)]",
        "accuracy": round(test_acc * 100, 2),
        "verified": verified,
        "pickle": {
            "format": "Python Pickle (.pkl)",
            "save_time_ms": round(pickle_save_time_ms, 3),
            "load_time_ms": round(pickle_load_time_ms, 3),
            "size_kb": round(pickle_size_bytes / 1024, 2),
            "path": PICKLE_PATH
        },
        "joblib": {
            "format": "Joblib Compressed (.joblib)",
            "save_time_ms": round(joblib_save_time_ms, 3),
            "load_time_ms": round(joblib_load_time_ms, 3),
            "size_kb": round(joblib_size_bytes / 1024, 2),
            "path": JOBLIB_PATH
        },
        "verdict": "Joblib provides superior efficiency for large NumPy arrays and multi-layer ML pipelines through optimized serialization and compression."
    }


def predict_deployed_model(input_data: dict, backend: str = "joblib"):
    """
    Live Deployment Inference Endpoint:
    Loads saved artifact from disk, executes end-to-end preprocessing & prediction.
    """
    t0 = time.perf_counter()
    if backend == "pickle":
        if not os.path.exists(PICKLE_PATH):
            train_and_benchmark_deployment()
        with open(PICKLE_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        if not os.path.exists(JOBLIB_PATH):
            train_and_benchmark_deployment()
        model = joblib.load(JOBLIB_PATH)

    # Format input into single-row DataFrame
    input_df = pd.DataFrame([{
        'Age': float(input_data.get('Age', 20)),
        'StudyHoursPerWeek': float(input_data.get('StudyHoursPerWeek', 25)),
        'AttendanceRate': float(input_data.get('AttendanceRate', 85)),
        'MathScore': float(input_data.get('MathScore', 82)),
        'ReadingScore': float(input_data.get('ReadingScore', 80)),
        'WritingScore': float(input_data.get('WritingScore', 85)),
        'Gender': str(input_data.get('Gender', 'Female')),
        'Department': str(input_data.get('Department', 'Computer Science'))
    }])

    prob = model.predict_proba(input_df)[0]
    pred = int(model.predict(input_df)[0])
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "prediction": "Distinction (>= 8.5 CGPA)" if pred == 1 else "Regular (< 8.5 CGPA)",
        "prediction_code": pred,
        "probability_distinction": round(float(prob[1]) * 100, 2),
        "probability_regular": round(float(prob[0]) * 100, 2),
        "inference_latency_ms": round(latency_ms, 3),
        "backend_used": backend,
        "model_file": JOBLIB_PATH if backend == 'joblib' else PICKLE_PATH
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PERCEPTRON & LOGIC GATES (AND, OR, NAND, XOR)
# ═══════════════════════════════════════════════════════════════════════════════

class Perceptron:
    """
    Single-layer Perceptron (Rosenblatt, 1958)
    Formula: z = w1*x1 + w2*x2 + b
             y_hat = 1 if z >= 0 else 0
    Weight update rule: w = w + lr * (y - y_hat) * x
                        b = b + lr * (y - y_hat)
    """
    def __init__(self, lr=0.1, max_epochs=100):
        self.lr = lr
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = 0.0
        self.history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.history = []

        converged = False
        epoch = 0

        for epoch in range(1, self.max_epochs + 1):
            errors = 0
            for xi, target in zip(X, y):
                z = np.dot(xi, self.weights) + self.bias
                pred = 1 if z >= 0 else 0
                error = target - pred
                if error != 0:
                    update = self.lr * error
                    self.weights += update * xi
                    self.bias += update
                    errors += 1

            self.history.append({
                "epoch": epoch,
                "weights": [round(float(w), 3) for w in self.weights],
                "bias": round(float(self.bias), 3),
                "errors": errors
            })

            if errors == 0:
                converged = True
                break

        return converged, epoch

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)


def evaluate_logic_gates():
    gates = {
        "AND": {
            "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            "y": np.array([0, 0, 0, 1]),
            "description": "Linearly separable: True only when both inputs are 1."
        },
        "OR": {
            "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            "y": np.array([0, 1, 1, 1]),
            "description": "Linearly separable: True when at least one input is 1."
        },
        "NAND": {
            "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            "y": np.array([1, 1, 1, 0]),
            "description": "Linearly separable: False only when both inputs are 1."
        },
        "XOR": {
            "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            "y": np.array([0, 1, 1, 0]),
            "description": "Non-linearly separable: True when inputs differ. A single Perceptron CANNOT solve this (Minsky & Papert, 1969)!"
        }
    }

    results = {}
    for name, data in gates.items():
        clf = Perceptron(lr=0.1, max_epochs=50)
        converged, epochs = clf.fit(data["X"], data["y"])
        preds = clf.predict(data["X"]).tolist()
        acc = float(accuracy_score(data["y"], preds))

        results[name] = {
            "truth_table": [
                {"x1": int(row[0]), "x2": int(row[1]), "target": int(target), "pred": int(pred)}
                for row, target, pred in zip(data["X"], data["y"], preds)
            ],
            "converged": converged,
            "epochs": epochs,
            "accuracy": round(acc * 100, 1),
            "final_weights": [round(float(w), 3) for w in clf.weights],
            "final_bias": round(float(clf.bias), 3),
            "decision_boundary": f"{round(float(clf.weights[0]), 2)}*x1 + {round(float(clf.weights[1]), 2)}*x2 + ({round(float(clf.bias), 2)}) = 0",
            "history": clf.history[:15],
            "description": data["description"]
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ACTIVATION FUNCTIONS & DERIVATIVES
# ═══════════════════════════════════════════════════════════════════════════════

def get_activation_curves():
    """Generates activation function curves and their analytical derivatives across [-5, 5]."""
    z = np.linspace(-5.0, 5.0, 101)

    # Sigmoid & derivative
    sig = 1.0 / (1.0 + np.exp(-z))
    sig_prime = sig * (1.0 - sig)

    # ReLU & derivative
    relu = np.maximum(0.0, z)
    relu_prime = np.where(z > 0, 1.0, 0.0)

    # Leaky ReLU (alpha = 0.1 for clear visibility)
    alpha = 0.1
    lrelu = np.where(z > 0, z, alpha * z)
    lrelu_prime = np.where(z > 0, 1.0, alpha)

    # Tanh & derivative
    tanh = np.tanh(z)
    tanh_prime = 1.0 - np.power(tanh, 2)

    return {
        "z": [round(float(val), 2) for val in z],
        "functions": {
            "Sigmoid": {
                "formula": "σ(z) = 1 / (1 + e^(-z))",
                "derivative_formula": "σ'(z) = σ(z) * (1 - σ(z))",
                "range": "(0, 1)",
                "use_case": "Binary classification output layer, gating mechanisms.",
                "values": [round(float(v), 4) for v in sig],
                "derivatives": [round(float(d), 4) for d in sig_prime]
            },
            "ReLU": {
                "formula": "f(z) = max(0, z)",
                "derivative_formula": "f'(z) = 1 if z > 0 else 0",
                "range": "[0, ∞)",
                "use_case": "Standard hidden layer activation (computationally efficient, avoids vanishing gradient).",
                "values": [round(float(v), 4) for v in relu],
                "derivatives": [round(float(d), 4) for d in relu_prime]
            },
            "Leaky ReLU": {
                "formula": "f(z) = max(α*z, z), α=0.1",
                "derivative_formula": "f'(z) = 1 if z > 0 else α",
                "range": "(-∞, ∞)",
                "use_case": "Prevents dying ReLU problem by maintaining a small positive gradient for z < 0.",
                "values": [round(float(v), 4) for v in lrelu],
                "derivatives": [round(float(d), 4) for d in lrelu_prime]
            },
            "Tanh": {
                "formula": "tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))",
                "derivative_formula": "tanh'(z) = 1 - tanh^2(z)",
                "range": "(-1, 1)",
                "use_case": "Zero-centered hidden layers, RNNs, recurrent states.",
                "values": [round(float(v), 4) for v in tanh],
                "derivatives": [round(float(d), 4) for d in tanh_prime]
            }
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PURE NUMPY NEURAL NETWORK WITH BACKPROPAGATION & OPTIMIZERS
# ═══════════════════════════════════════════════════════════════════════════════

class ScratchNeuralNetwork:
    """
    Arbitrary Multi-Layer Perceptron built strictly with NumPy.
    Implements:
    - Forward Propagation with activation caching
    - Binary Cross-Entropy Loss
    - Backpropagation via Chain Rule
    - Optimizers: Vanilla SGD, SGD with Momentum, and Adam
    """
    def __init__(self, layer_sizes, activations=None, optimizer='adam', lr=0.01,
                 beta1=0.9, beta2=0.999, epsilon=1e-8, random_state=42):
        self.layer_sizes = layer_sizes
        self.activations = activations or ['relu'] * (len(layer_sizes) - 2) + ['sigmoid']
        self.optimizer_name = optimizer.lower()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.random_state = random_state

        np.random.seed(random_state)
        self.weights = []
        self.biases = []

        # He / Xavier Initialization
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            # He init for relu, Xavier for sigmoid/tanh
            act = self.activations[i]
            scale = np.sqrt(2.0 / n_in) if act in ['relu', 'leaky_relu'] else np.sqrt(1.0 / n_in)
            W = np.random.randn(n_in, n_out) * scale
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)

        # Optimizer State
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.s_w = [np.zeros_like(w) for w in self.weights]  # 2nd moment for Adam
        self.s_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0  # time step for Adam

    def _activate(self, z, kind):
        if kind == 'relu':
            return np.maximum(0.0, z)
        elif kind == 'leaky_relu':
            return np.where(z > 0, z, 0.01 * z)
        elif kind == 'tanh':
            return np.tanh(z)
        elif kind == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return z

    def _activate_grad(self, z, a, kind):
        if kind == 'relu':
            return np.where(z > 0, 1.0, 0.0)
        elif kind == 'leaky_relu':
            return np.where(z > 0, 1.0, 0.01)
        elif kind == 'tanh':
            return 1.0 - np.power(a, 2)
        elif kind == 'sigmoid':
            return a * (1.0 - a)
        return np.ones_like(z)

    def forward(self, X):
        a = X
        a_cache = [a]
        z_cache = []

        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self._activate(z, self.activations[i])
            z_cache.append(z)
            a_cache.append(a)

        return a, z_cache, a_cache

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
        loss = - (1.0 / m) * np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return float(loss)

    def backward(self, y_true, z_cache, a_cache):
        m = y_true.shape[0]
        num_layers = len(self.weights)
        grad_W = [None] * num_layers
        grad_b = [None] * num_layers

        # Output layer error
        a_last = a_cache[-1]
        z_last = z_cache[-1]

        # For Sigmoid + Binary Cross-Entropy, dL/dz = a - y
        if self.activations[-1] == 'sigmoid':
            delta = (a_last - y_true)
        else:
            delta = (a_last - y_true) * self._activate_grad(z_last, a_last, self.activations[-1])

        for l in reversed(range(num_layers)):
            a_prev = a_cache[l]
            grad_W[l] = (1.0 / m) * np.dot(a_prev.T, delta)
            grad_b[l] = (1.0 / m) * np.sum(delta, axis=0, keepdims=True)

            if l > 0:
                W_curr = self.weights[l]
                z_prev = z_cache[l - 1]
                a_prev = a_cache[l]
                delta = np.dot(delta, W_curr.T) * self._activate_grad(z_prev, a_prev, self.activations[l - 1])

        return grad_W, grad_b

    def update_params(self, grad_W, grad_b):
        self.t += 1
        num_layers = len(self.weights)

        for l in range(num_layers):
            gw = grad_W[l]
            gb = grad_b[l]

            if self.optimizer_name == 'sgd':
                self.weights[l] -= self.lr * gw
                self.biases[l] -= self.lr * gb

            elif self.optimizer_name == 'momentum':
                # Momentum: v = beta1 * v + (1 - beta1) * grad
                self.v_w[l] = self.beta1 * self.v_w[l] + (1.0 - self.beta1) * gw
                self.v_b[l] = self.beta1 * self.v_b[l] + (1.0 - self.beta1) * gb
                self.weights[l] -= self.lr * self.v_w[l]
                self.biases[l] -= self.lr * self.v_b[l]

            elif self.optimizer_name == 'adam':
                # Adam 1st moment
                self.v_w[l] = self.beta1 * self.v_w[l] + (1.0 - self.beta1) * gw
                self.v_b[l] = self.beta1 * self.v_b[l] + (1.0 - self.beta1) * gb
                # Adam 2nd moment
                self.s_w[l] = self.beta2 * self.s_w[l] + (1.0 - self.beta2) * (gw ** 2)
                self.s_b[l] = self.beta2 * self.s_b[l] + (1.0 - self.beta2) * (gb ** 2)
                # Bias correction
                v_w_corrected = self.v_w[l] / (1.0 - self.beta1 ** self.t)
                v_b_corrected = self.v_b[l] / (1.0 - self.beta1 ** self.t)
                s_w_corrected = self.s_w[l] / (1.0 - self.beta2 ** self.t)
                s_b_corrected = self.s_b[l] / (1.0 - self.beta2 ** self.t)
                # Parameter updates
                self.weights[l] -= self.lr * v_w_corrected / (np.sqrt(s_w_corrected) + self.eps)
                self.biases[l] -= self.lr * v_b_corrected / (np.sqrt(s_b_corrected) + self.eps)

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=150, batch_size=32):
        m = X_train.shape[0]
        y_train_2d = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        if y_val is not None:
            y_val_2d = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val

        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(m)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_2d[indices]

            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]

                preds, z_cache, a_cache = self.forward(xb)
                grad_W, grad_b = self.backward(yb, z_cache, a_cache)
                self.update_params(grad_W, grad_b)

            # Record metrics periodically
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                train_preds, _, _ = self.forward(X_train)
                train_loss = self.compute_loss(y_train_2d, train_preds)
                train_acc = float(accuracy_score(y_train_2d > 0.5, train_preds > 0.5))

                history["epoch"].append(epoch)
                history["train_loss"].append(round(train_loss, 4))
                history["train_acc"].append(round(train_acc * 100, 2))

                if X_val is not None and y_val is not None:
                    val_preds, _, _ = self.forward(X_val)
                    val_loss = self.compute_loss(y_val_2d, val_preds)
                    val_acc = float(accuracy_score(y_val_2d > 0.5, val_preds > 0.5))
                    history["val_loss"].append(round(val_loss, 4))
                    history["val_acc"].append(round(val_acc * 100, 2))

        return history

    def predict_proba(self, X):
        probs, _, _ = self.forward(X)
        return probs

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. OPTIMIZER BENCHMARK COMPARISON (SGD vs MOMENTUM vs ADAM)
# ═══════════════════════════════════════════════════════════════════════════════

def compare_optimizers():
    data = get_preprocessed_data()
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    n_features = X_train.shape[1]
    architecture = [n_features, 16, 8, 1]
    epochs = 100

    optimizers_config = [
        {"name": "SGD (Standard)", "key": "sgd", "lr": 0.05, "color": "#f59e0b"},
        {"name": "SGD + Momentum (beta=0.9)", "key": "momentum", "lr": 0.02, "color": "#38bdf8"},
        {"name": "Adam (Adaptive)", "key": "adam", "lr": 0.01, "color": "#10b981"}
    ]

    results = []

    for opt in optimizers_config:
        model = ScratchNeuralNetwork(
            layer_sizes=architecture,
            activations=['relu', 'relu', 'sigmoid'],
            optimizer=opt["key"],
            lr=opt["lr"],
            random_state=42
        )

        hist = model.fit(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            epochs=epochs,
            batch_size=32
        )

        # Final Evaluation
        y_pred = model.predict(X_test).flatten()
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        results.append({
            "name": opt["name"],
            "key": opt["key"],
            "lr": opt["lr"],
            "color": opt["color"],
            "final_accuracy": round(acc * 100, 2),
            "final_f1": round(f1, 4),
            "final_loss": hist["train_loss"][-1],
            "history": hist
        })

    return {
        "dataset": "Student Performance (Predicting Distinction >= 8.5 CGPA)",
        "samples": len(X_train) + len(X_test),
        "features": n_features,
        "epochs": epochs,
        "results": results
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. INTERACTIVE CUSTOM SCRATCH NETWORK TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_custom_scratch_network(params: dict):
    data = get_preprocessed_data()
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    hidden_units_1 = int(params.get('hidden_1', 16))
    hidden_units_2 = int(params.get('hidden_2', 8))
    activation = str(params.get('activation', 'relu')).lower()
    optimizer = str(params.get('optimizer', 'adam')).lower()
    lr = float(params.get('lr', 0.01))
    epochs = int(params.get('epochs', 100))
    batch_size = int(params.get('batch_size', 32))

    n_features = X_train.shape[1]
    layers = [n_features]
    if hidden_units_1 > 0:
        layers.append(hidden_units_1)
    if hidden_units_2 > 0:
        layers.append(hidden_units_2)
    layers.append(1)

    activations = [activation] * (len(layers) - 2) + ['sigmoid']

    nn = ScratchNeuralNetwork(
        layer_sizes=layers,
        activations=activations,
        optimizer=optimizer,
        lr=lr,
        random_state=42
    )

    t0 = time.perf_counter()
    history = nn.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=epochs,
        batch_size=batch_size
    )
    train_duration_ms = (time.perf_counter() - t0) * 1000.0

    y_pred = nn.predict(X_test).flatten()
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "architecture": f"Inputs ({n_features}) -> " + " -> ".join([f"Dense({u}, {activation})" for u in layers[1:-1]]) + " -> Output(1, sigmoid)",
        "layers": layers,
        "optimizer": optimizer.upper(),
        "lr": lr,
        "epochs": epochs,
        "train_duration_ms": round(train_duration_ms, 2),
        "metrics": {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm
        },
        "history": history
    }


if __name__ == '__main__':
    print("Testing neural_engine.py locally...")
    dep = train_and_benchmark_deployment()
    print("Deployment benchmark:", dep["pickle"], dep["joblib"])
    gates = evaluate_logic_gates()
    print("Logic Gates evaluated:", list(gates.keys()))
    opts = compare_optimizers()
    print("Optimizers evaluated:", [o["name"] for o in opts["results"]])
    print("Engine verified successfully!")
