# ONNX Decomposition Project

This repository contains the source code and experimental results for the graduation project titled:  
**"Comparison and Enhancement of Fission Techniques in TorchInductor and Korch Compilers"**

The project focuses on applying operator-level decomposition (fission) techniques to ONNX models, analyzing their effect on runtime performance across different compiler backends and GPU architectures.

## Project Objectives

- Reverse-engineer decomposition patterns from PyTorch's TorchInductor backend.
- Apply decomposition to composite ONNX operators using ONNX-GraphSurgeon.
- Integrate these decompositions into the Korch compiler.
- Benchmark inference latency on:
  - Local GPU (GTX1650)
  - HPC cluster (A100 at UHeM)
- Evaluate backend-specific fusion behaviors in:
  - Korch
  - TensorRT
  - PyTorch

## Decomposed Operators

The following composite operators were decomposed into primitive ONNX ops:

- `LayerNormalization`
- `Sigmoid`
- `HardSigmoid`
- `HardSwish`
- `Tanh`
- `Dropout`
- `LeakyReLU`
- `Reciprocal`

## References

- [Korch Compiler (arXiv 2024)](https://arxiv.org/abs/2406.09465)
  
  *A GPU-centric kernel orchestration framework for ONNX models with profiling-based subgraph fusion.*

- [ONNX-GraphSurgeon (NVIDIA)](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)
  
  *A Python API for editing ONNX graphs, enabling custom transformations and graph surgery.*

- [TVM Meta-Schedule](https://tvm.apache.org/docs/arch/#tvm-meta-schedule)
  
  *Auto-tuning and scheduling module in Apache TVM for generating optimized kernels across hardware backends.*

- [TorchInductor (PyTorch)](https://docs.pytorch.org/docs/stable/torch.compiler.html)
  
  *A deep learning compiler backend in PyTorch that performs operator decomposition and generates fused kernels.*

