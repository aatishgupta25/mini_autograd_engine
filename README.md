# mini_autograd_engine

A simple, self-contained implementation of reverse-mode automatic differentiation (autograd), modeled after PyTorch’s `.backward()` system - but built entirely from scratch in under 250 lines of pure Python + NumPy.

---

## What This Is

This project builds a **computation graph–based autograd engine** using a custom `Value` class. It performs automatic differentiation of scalar expressions by tracking operations in a graph and computing gradients via the chain rule - just like deep learning frameworks such as PyTorch.

It also includes:

* Visualizations of the computation graph using Graphviz
* Tests for gradient correctness
* A comparison against PyTorch's built-in gradient engine

---

## Why This Project Matters

Most people use `.backward()` in PyTorch without understanding how it works. This project **builds that functionality from scratch**, helping demystify:

* How operation graphs are constructed dynamically
* How the chain rule is applied to propagate gradients backward

It's useful for:

* Learning backpropagation
* Building mental models of how neural networks learn

---

## How It Works

The engine revolves around a custom `Value` class that:

### 1. Wraps numerical data

Each `Value` stores:

* `data` — the scalar value
* `grad` — its gradient
* `_prev` — the parent nodes (for graph tracking)
* `_backward` — a function that computes local gradients

```python
a = Value(1.0)
b = Value(2.0)
c = a * b + b
```

### 2. Builds a computation graph dynamically

Every operation (`+`, `*`, `tanh`, `**`, etc.) creates a new `Value` node linked to its inputs, recording:
- The operation performed
- The path for computing derivatives

---

### 3. Computes gradients with `.backward()`

Calling `.backward()` on the output `Value`:
- Topologically sorts the graph
- Starts from the output node (sets `grad = 1`)
- Propagates gradients backward through all parents using each node’s `_backward()` function

---

### 4. Comparison to PyTorch

It includes a function to compute the same expression using PyTorch and prints both sets of gradients for comparison, validating correctness.

---

### 5. Generates graph visualizations

The engine can output `.dot` files visualizing the computation graph before and after backpropagation, which are rendered to `.svg` using Graphviz.

---

### 6. MLP Modules & Demo: Binary Classification with SVM-style Loss

Beyond the core autograd engine, this project includes a fully functional mini neural network stack with:

- **`Neuron`** — a single perceptron with optional ReLU activation  
- **`Layer`** — a dense layer of neurons  
- **`MLP`** — a multi-layer perceptron built by stacking layers

These classes use the custom `Value` type to support automatic differentiation and training with gradient descent — no PyTorch or TensorFlow involved.
In [`demo.ipynb`](demo.ipynb), we train an `MLP` with architecture `2 → 16 → 16 → 1` on a binary classification task using the `make_moons` dataset.

## Installation & Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Engine

```
python engine/engine.py
```

This will:

- Build a computation graph  
- Run a forward and backward pass  
- Generate two graph visualizations:  
  - [`graph_before_backprop.svg`](visualizations/graph_before_backprop.svg)  
  - [`graph_after_backprop.svg`](visualizations/graph_after_backprop.svg)  
- Print gradients from both your engine and PyTorch

## File Overview

```
mini_autograd_engine/
├── engine/
│   ├── engine.py             # Autograd Value class
│   ├── modules.py            # Neuron, Layer, MLP classes
├── tests/
│   └── tests.py              # Unit tests for Value class
├── visualizations/           # .svg files for computation graphs
├── demo.ipynb                # MLP demo on 2D classification
├── make_graph_for_viz.py # Graphviz visualizer
├── README.md
└── requirements.txt
```