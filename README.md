# RNN From Scratch with NumPy

A vanilla Recurrent Neural Network (RNN) implemented completely from scratch using Python and NumPy.

The goal of this project was to understand how RNNs work internally by implementing the mathematics, forward pass, Backpropagation Through Time (BPTT), training, and prediction without using deep-learning frameworks such as PyTorch or TensorFlow.

## Project Overview

This project implements a simple vanilla RNN that learns to predict the next element in a repeating sequence:

1 → 2 → 3 → 1 → 2 → 3 → ...

After training, the model learns:

1 → 2
2 → 3
3 → 1

## RNN Architecture

The model contains:

- One-hot encoded input
- Hidden state
- Recurrent hidden-to-hidden connection
- `tanh` activation
- Output layer
- Softmax
- Cross-entropy loss
- Backpropagation Through Time (BPTT)
- Gradient clipping
- Gradient descent

## Forward Pass

The hidden state is calculated using:

h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)

Where:

- `x_t` = current input
- `h_(t-1)` = previous hidden state
- `W_xh` = input-to-hidden weights
- `W_hh` = hidden-to-hidden weights
- `b_h` = hidden bias

The hidden state acts as the RNN's memory.

### Output Layer

o_t = W_hy h_t + b_y

The output scores are converted into probabilities using Softmax.

## Loss Function

The model uses cross-entropy loss:

L_t = -log(p_correct)

## Backpropagation Through Time

The model is trained using Backpropagation Through Time (BPTT).

Output error:

δ_o = p_t - y_true

Output gradients:

dW_hy = δ_o h_t^T

db_y = δ_o

Hidden-state gradient:

δ_h = (W_hy^T δ_o + W_hh^T δ_h_next) ⊙ (1 - h_t²)

Hidden-layer gradients:

dW_xh = δ_h x_t^T

dW_hh = δ_h h_(t-1)^T

db_h = δ_h

## Gradient Clipping

Gradient clipping is used to reduce the risk of exploding gradients.

## How to Run

1. Clone this repository:
```bash
git clone https://github.com/omarsamehabobaker619-bot/Rnn-From-Scratch.git
cd Rnn-From-Scratch
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
venv\Scripts\activate
```
On Mac/Linux:
```bash
source venv/bin/activate
```

3. Install the dependencies:
```bash
pip install numpy jupyter
```

4. Open `rnn.ipynb` in VS Code or Jupyter Notebook.

5. Run all cells from top to bottom.
