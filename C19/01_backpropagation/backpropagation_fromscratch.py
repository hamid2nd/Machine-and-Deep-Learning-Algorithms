"""
C19 Foundations of Deep Learning — Backpropagation exercise.

Adapted from the Colab notebook (Backpropagation.ipynb, kept alongside for reference).

Run from the repo root:
    .venv/bin/python C19/01_backpropagation/backpropagation_fromscratch.py
"""

import math
import numpy as np


# =============================================================================
# Worked example: linear layer with sigmoid (example from lecture in code)
# =============================================================================
# NB: a linear layer = fully connected layer (each input has its own weight,
# used once), NOT a convolution (one small set of weights shared across positions).
# Here it's a single neuron: s3 = w0*x0 + w1*x1 + w2, with w2 acting as the bias.

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_forward(x0, w0, x1, w1, w2):
    s0 = w0 * x0
    s1 = w1 * x1
    s2 = s0 + s1
    s3 = s2 + w2
    L = sigmoid(s3)
    return L


def sigmoid_backward(x0, w0, x1, w1, w2, L):
    # "L" is just the output f, not a loss yet: dL = 1 starts the chain (df/df = 1)
    dL = 1.0
    # Sigmoid derivative: sigma'(s) = sigma(s)(1 - sigma(s)) -> reuses forward output L
    ds3 = dL * L * (1 - L)
    # Addition nodes: copy the upstream gradient unchanged to every input
    dw2 = ds3
    ds2 = ds3
    ds1 = ds2
    ds0 = ds2
    # Multiplication nodes: gradient = upstream * the OTHER input (held constant),
    # since d(w*x)/dw = x and d(w*x)/dx = w
    dw1 = ds1 * x1
    dx1 = ds1 * w1
    dw0 = ds0 * x0
    dx0 = ds0 * w0
    return dx0, dw0, dx1, dw1, dw2


def lecture_example():
    # Verify with example from the lecture
    x0 = -1.0
    w0 = 2.0
    x1 = -2.0
    w1 = -3.0
    w2 = -3.0

    L = sigmoid_forward(x0, w0, x1, w1, w2)
    print("Output:", L)

    dx0, dw0, dx1, dw1, dw2 = sigmoid_backward(x0, w0, x1, w1, w2, L)

    print("Gradients:")
    print("dx0:", dx0)
    print("dw0:", dw0)
    print("dx1:", dx1)
    print("dw1:", dw1)
    print("dw2:", dw2)


# =============================================================================
# Q2.1 / Q2.2: Two-layer NN with ReLU activation
# =============================================================================

# ReLU activation function
def relu(z):
    return np.maximum(0, z) #Very simple to implement ReLU activation function 


# Forward pass
def forward(x, W1, b1, W2, b2):
    # TODO: Implement forward pass
    # .T transposes a matrix; @ is matrix multiplication (same as np.dot here)
    z1 = W1.T @ x + b1  # (H, D) @ (D,) + (H,) -> (H,)
    h = relu(z1)
    z2 = W2.T @ h + b2  # (1, H) @ (H,) + scalar -> (1,)
    # z2 is the prediction; z1 and h are cached for the backward pass
    return z2, z1, h

# Backward pass
# Hint 1: remember that intermediate results from the forward pass are also needed in the backward computations
# Hint 2: vector-in / vector-out -> Derivative is Jacobian
def backward(x, W2, z1, h, pred, gt):
    # Q2.2: start from the loss, not dL = 1. For L = (y - y_hat)^2, dL/dy_hat = 2(y_hat - y)
    dL = 2 * (pred[0] - gt)  # pred is (1,) -> take the scalar
    # TODO: Implement backward pass and return the gradients with respect to all learnable parameters
    # Output layer: z2 = W2^T h + b2
    db2 = dL  # addition nodes just pass gradient through
    dW2 = dL * h[:, None]  # mult node: other input (h); (H,) -> (H, 1) to match W2
    dh = dL * W2[:, 0]  # mult node: other input (W2); (H, 1) -> (H,)
    # ReLU: gradient passes where z1 > 0, blocked (zeroed) where z1 <= 0
    dz1 = dh * (z1 > 0)  # (H,)
    # Hidden layer: z1 = W1^T x + b1
    db1 = dz1  # (H,)
    dW1 = np.outer(x, dz1)  # dW1[i, j] = x[i] * dz1[j] -> (D, H), matches W1
    return dW1, db1, dW2, db2


# Loss function (Q2.2)
def MSE_loss(pred, gt):
    # TODO: Implement the mean squared error
    loss = np.mean((gt-pred)**2)
    return loss


def two_layer_net():
    np.random.seed(42)

    D = 4
    H = 16
    K = 1

    # Set some random values for the parameters
    W1 = np.random.randn(D, H)
    b1 = np.random.randn(H)
    W2 = np.random.randn(H, 1)
    b2 = np.random.randn()

    # Input and target output (scalar)
    x = np.random.randn(D)
    y = np.random.randn()

    # TODO: Complete the forward and backward calls
    # Forward pass
    pred, z1, h = forward(x, W1, b1, W2, b2)
    # Backward pass
    dW1, db1, dW2, db2 = backward(x, W2, z1, h, pred, y)

    print("Prediction:", pred, "Target:", y, "Loss:", MSE_loss(pred, y))
    print("Gradients:")
    print("dW1:", dW1)
    print("db1:", db1)
    print("dW2:", dW2)
    print("db2:", db2)


# =============================================================================
# Q2.3: Numerical gradient check
# =============================================================================
# Not implemented — understanding only.
#
# Algorithm (my words): run multiple forward passes, each time adding/subtracting
# delta to a single variable, to approximate the gradients (dW1 etc.), then check
# these against the result from backpropagation.
#
# Notes:
# - For each single entry of each parameter: nudge +delta -> forward -> loss L+,
#   nudge -delta -> forward -> loss L-, restore it. Gradient ~ (L+ - L-) / (2*delta).
# - Differentiate the loss (not just pred), since backward now starts from the MSE.
# - Cost: 97 parameter entries x 2 = 194 forward passes. Backprop gets all of them
#   from 1 forward + 1 backward -> too slow to train with, but a good test.
# - Compare with np.max(np.abs(numerical - analytical)): ~1e-8 or smaller = correct,
#   ~1e-2 = bug.

def gradient_check():
    # TODO: Verify your implementation above numerically (Q2.3)
    delta = 1e-5


if __name__ == "__main__":
    print("=== Lecture example ===")
    lecture_example()

    print("\n=== Q2.1/2.2: Two-layer net ===")
    two_layer_net()

    print("\n=== Q2.3: Gradient check ===")
    gradient_check()
