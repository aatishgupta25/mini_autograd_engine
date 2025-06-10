# mini_autograd_engine

A lightweight, educational reimplementation of PyTorch's automatic differentiation engine from scratch — including support for forward and backward passes, tensor operations, and graph-based gradient computation.

## What This Is

This repository contains a **minimal autograd engine** written in Python that mimics the core ideas behind PyTorch's `torch.Tensor.backward()` functionality. It is designed to help developers and students understand how modern deep learning frameworks compute gradients automatically under the hood.

It includes:
- A core **engine** for tracking operations and computing gradients via reverse-mode automatic differentiation (aka backpropagation)
- **Visualizations** of computation graphs for intuitive understanding
- A set of **unit tests** to ensure correctness
- Clean code structure for **readability and educational value**

---

## Why This Matters

>  Most people use PyTorch or TensorFlow without knowing what happens behind `.backward()`.

This project demystifies that black box. It shows:

- How tensors track operations
- How gradient flow is computed in reverse through the graph
- How a neural network can be trained from just raw Python + NumPy logic

### Use cases:
- Reinforce understanding of backpropagation
- Build intuition for writing custom layers and loss functions in real frameworks
- Showcase ability to implement **core AI infrastructure** without dependencies

---

## 🗂️ Repo Structure

```text
mini_autograd_engine/
├── engine/                 # Core autograd engine (tensors, operations, gradients)
│   └── engine.py
├── visualizations/         # Graphviz visual outputs of computation graphs
├── tests/                  # Unit tests for core components
│   └── tests.py
├── requirements.txt        # Dependencies
└── README.md               # You're here!
