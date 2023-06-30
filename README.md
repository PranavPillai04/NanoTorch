# NanoTorch

An intuitive modular deep learning library that builds on some fundamental features from popular open-source libraries, like PyTorch and Tensorflow. Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

## Features

- Contains concise neural network building blocks:
  - Neurons
  - Linear Layers
  - Multi-layer Perceptrons (MLPs)
- Simple features for forwarding, backpropagation, and inference.
- Broadly applicable to different problem settings
  - NLP
  - Computer Vision
  - Anything else that can define an objective goal and a loss function to optimize

## SmartGrad

NanoTorch contains its own small autograd (reverse-mode automatic differentiation) engine with fundamental functionalities. Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

### Features

- Wraps data to record a representation of the calculations it is used for and the corresponding gradient function.
- Creates computational graphs so that gradients during backpropagation can be easily calculated.
- Includes support for 10+ mathematical operators, such as addition, multiplication, exponentiation, hyperbolic tangent, etc.
- Models the same underlying mechanism found in popular libraries like PyTorch, TensorFlow, etc.
