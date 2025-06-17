# Mini Autograd Engine

A lightweight, educational deep-learning framework built from scratch in Python and NumPy. It provides:

* **Automatic Differentiation**
  Forward/backward passes over a computation graph for arbitrary N-dimensional arrays, with broadcasting-aware gradients.
* **Neural-Network Modules**
  Fully-connected layers (`MLP`), activations (`ReLU`, `Softmax`, etc.), plus `Dropout` and `BatchNorm1d`.
* **Optimizers**
  SGD (with momentum) and Adam, including bias correction and optional L₂ weight decay.
* **End-to-End Training Demos**
  Train & evaluate on Digits dataset via simple script.

---

## In-Depth Project Overview

1. **Core Tensor Engine**

   * **Data + Gradient Storage**: Each `Tensor` wraps a NumPy `ndarray` (`.data`) and a matching `.grad` array.
   * **Graph Building**: Overloaded operators (`+`, `*`, `@`, activations, reductions) record parent pointers and a small backward function.
   * **Backward Pass**: Calling `.backward()` topologically sorts the graph and invokes each node’s gradient update, un-broadcasting as needed.

2. **Neural-Network API**

   * **`Module` Base Class**: Standardizes `parameters()` and `zero_grad()`.
   * **`Neuron` & `Layer`**: Compose lists of weights and biases into single neurons and layers with optional non-linearities.
   * **`MLP`**: Stacks layers into deep networks, with automatic vector-stacking of outputs for classification.
   * **Regularization**: Built-in `Dropout` and `BatchNorm1d` modules follow the same pattern.

3. **Optimizers**

   * **SGD**: Supports vanilla and momentum-based updates, with an optional L₂ (weight-decay) term.
   * **Adam**: Maintains first/second moment estimates, applies bias correction, and uses a small epsilon for stability.

4. **Comparison to PyTorch**
   While PyTorch offers GPU acceleration, fused kernels, JIT compilation, and a vast ecosystem, this mini-engine trades raw speed for **clarity**, **inspectability**, and **educational value**—making it ideal for learning how autodiff and neural nets work under the hood.

---

## Setup & Usage

1. **Clone & install**

   ```bash
   git clone https://github.com/<your_github_username>/mini_autograd_engine.git
   cd mini_autograd_engine
   python -m venv .venv
   source .venv/bin/activate       # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run tests**

   ```bash
   pytest -v
   ```

3. **Run demos**

   ```bash
   python -m demo.digits_tensor_demo
   ```

> **Digits demo results (10 epochs, 500 samples)**
>
> ```
> Epoch  1  TrainLoss 9.9027  ValAcc 40.83%  
> …  
> Epoch 10  TrainLoss 0.0926  ValAcc 85.28%  
> ```
