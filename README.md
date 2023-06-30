# SmartGrad

A small autograd (reverse-mode automatic differentiation) engine with fundamental functionalities. Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Features

- Wraps data to record a representation of the calculations it is used for and the corresponding gradient function.
- Creates computational graphs so that gradients during backpropagation can be easily calculated.
- Includes support for 10+ mathematical operators, such as addition, multiplication, exponentiation, hyperbolic tangent, etc.
- Models the same underlying mechanism found in popular libraries like PyTorch, TensorFlow, etc.
