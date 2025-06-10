# mini_autograd_engine

A simple, self-contained implementation of reverse-mode automatic differentiation (autograd), modeled after PyTorch’s `.backward()` system — but built entirely from scratch in under 200 lines of pure Python + NumPy.

---

## What This Is

This project builds a **computation graph–based autograd engine** using a custom `Value` class. It performs automatic differentiation of scalar expressions by tracking operations in a graph and computing gradients via the chain rule — just like deep learning frameworks such as PyTorch.

It also includes:

* Visualizations of the computation graph using Graphviz
* Tests for gradient correctness
* A comparison against PyTorch's built-in gradient engine

---

## Why This Project Matters

Most people use `.backward()` in PyTorch without understanding how it works. This project **builds that functionality from scratch**, helping demystify:

* How frameworks compute gradients
* How operation graphs are constructed dynamically
* How the chain rule is applied to propagate gradients backward

It's useful for:

* Learning backpropagation
* Showcasing deep technical skills to recruiters
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

```text
mini_autograd_engine/
├── engine/engine.py           # The full autograd engine
├── visualizations/            # .dot and .svg files for graph views
├── tests/tests.py             # Gradient validation tests
├── requirements.txt
└── README.md
```