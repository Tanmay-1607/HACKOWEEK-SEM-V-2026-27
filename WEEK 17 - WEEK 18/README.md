# Week 17 – Week 18: Deep Learning Architectures, Transfer Learning & Generative LLMs

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20%7C%20CUDA-red.svg)](https://pytorch.org/)
[![CNN](https://img.shields.io/badge/CNN-Conv2D%20%7C%20Pooling-blue.svg)](https://pytorch.org/docs/stable/nn.html)
[![Sequence](https://img.shields.io/badge/Sequence-RNN%20%7C%20LSTM%20%7C%20GRU-green.svg)](https://pytorch.org/docs/stable/nn.html#recurrent-layers)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![LLM](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Anthropic%20%7C%20Ollama-purple.svg)](https://ollama.com/)
[![Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)](notebooks/week17_week18_deep_learning_llms.ipynb)

## 👤 Student Information
- **Name:** Tanmay Sankulwar
- **PRN:** 24070521058
- **Program:** B.Tech Computer Science / Information Technology (5th Semester)
- **GitHub Profile:** [@Tanmay-1607](https://github.com/Tanmay-1607)
- **Repository:** [HACKOWEEK-SEM-V-2026-27](https://github.com/Tanmay-1607/HACKOWEEK-SEM-V-2026-27)

---

## 📌 Syllabus Overview (SIT-N Hack-o-Week 5th Semester)

### 1. Intro to PyTorch (Tensors, Autograd, Training Loop)
- Tensor creation, shapes, GPU device handling (`torch.cuda.is_available()`, device mapping).
- Computational graphs and Automatic Differentiation (`autograd` engine with `loss.backward()`).
- Complete PyTorch training loop (`optimizer.zero_grad()`, forward pass, loss calculation, backpropagation, and `optimizer.step()`).

### 2. Convolutional Neural Networks (CNNs)
- Spatial feature extraction using `nn.Conv2d` (kernels, padding, stride).
- Receptive fields, spatial downsampling via `nn.MaxPool2d(2, 2)`, and normalization via `nn.BatchNorm2d`.
- Layer-by-layer tensor transformations from single-channel 28x28 inputs to feature vectors and classification logits.

### 3. Recurrent Models (RNNs, LSTMs, GRUs)
- **Vanilla RNN:** Hidden state recurrence ($h_t = \tanh(W_{ih} x_t + W_{hh} h_{t-1} + b)$).
- **LSTM (Long Short-Term Memory):** Mitigating vanishing gradients via forget gate ($f_t$), input gate ($i_t$), candidate cell state ($\tilde{c}_t$), cell state update ($c_t$), and output gate ($o_t$).
- **GRU (Gated Recurrent Unit):** Streamlined gating architecture combining cell and hidden state with reset ($r_t$) and update ($z_t$) gates.
- Benchmark comparing parameter efficiency and MSE convergence on sequential series.

### 4. Regularization in Deep Learning
- **Dropout (`nn.Dropout(p)`):** Randomly zeroing activations during training to break co-adaptation.
- **Batch Normalization (`nn.BatchNorm1d`, `nn.BatchNorm2d`):** Stabilizing internal covariate shift.
- **Early Stopping:** Tracking validation loss across epochs with patience counters to eliminate the generalization gap and prevent overfitting.

### 5. Transfer Learning (Pretrained CNNs)
- Adapting deep architectures pretrained on ImageNet (ResNet-18 / MobileNetV2).
- Freezing convolutional base layers (`param.requires_grad = False`).
- Replacing the final classification head with custom dense layers for target domain tasks.
- Parameter savings: Training only 33,000 parameters out of 11.2 million (over 99.7% parameter savings).

### 6. Hugging Face `transformers` Library
- Pipeline architecture for Natural Language Processing (NLP).
- Sub-word tokenization breakdown (`[CLS]`, token strings, `[SEP]`, Input IDs, Attention Masks).
- Transformer classification head yielding Softmax sentiment probabilities.

### 7. LLM APIs & Open Models
- Unified prompt execution interface across:
  - Commercial APIs: OpenAI (`gpt-4o-mini`), Anthropic Claude (`claude-3-5-sonnet`).
  - Open-weight models: Ollama (`llama-3-8b-instruct`), Hugging Face Inference (`mistral-7b-instruct`).
  - Task modes: Text Generation, Text Summarization, and Technical Question Answering.

---

## 📓 Jupyter Notebook
The complete end-to-end coding walkthrough is in:
[`notebooks/week17_week18_deep_learning_llms.ipynb`](notebooks/week17_week18_deep_learning_llms.ipynb)

To launch the notebook:
```bash
cd "WEEK 17 - WEEK 18"
jupyter notebook notebooks/week17_week18_deep_learning_llms.ipynb
```

---

## 🌐 Interactive Deep Learning & LLMs Studio
An interactive web dashboard has been developed allowing users to:
1. **🔥 PyTorch Core**: Inspect hardware device status (NVIDIA RTX 4070 GPU detection), execute tensor operations, and visualize autograd differentiation.
2. **🖼️ CNN Classifier**: Inspect layer-by-layer tensor shape traces and track real-time training loss and validation accuracy curves.
3. **🔁 Sequence Modeling**: Compare Vanilla RNN vs. LSTM vs. GRU convergence and parameter efficiency on time-series forecasting.
4. **🛡️ DL Regularization**: Side-by-side comparison of unregularized overfitting vs. regularized models with early stopping.
5. **🚀 Transfer Learning**: Inspect ResNet-18 frozen feature extractors and fine-tuning strategies.
6. **🤗 Hugging Face Transformers**: Enter custom sentences to see sub-word tokenization, input IDs, attention masks, and sentiment classification.
7. **🤖 LLM Studio**: Test multi-provider prompt dispatch (OpenAI, Anthropic, Ollama, Hugging Face) for generation, summarization, and technical Q&A with token and latency metrics.

### How to Run the Web Dashboard:
```bash
cd "WEEK 17 - WEEK 18"
python app.py
```
Then navigate to: **`http://localhost:5008`**

---

## 📂 Project Structure
```
WEEK 17 - WEEK 18/
├── app.py                      # Flask REST server (port 5008)
├── pytorch_engine.py           # PyTorch deep learning & LLM execution engine
├── README.md                   # Comprehensive documentation
├── data/
│   └── kaggle_student_performance.csv
├── notebooks/
│   └── week17_week18_deep_learning_llms.ipynb
└── public/
    ├── index.html              # Dark glassmorphic studio interface (7 interactive tabs)
    ├── css/
    │   └── styles.css          # Design system & responsive layout
    └── js/
        └── app.js              # Native canvas chart visualizers & API client
```
