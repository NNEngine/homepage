We are in the age of AI and we all have heard about Neural Networks. Everyone is talking about them whether they are from AI field or not. And Sometimes it's  a little confusing to understand what actually they are.

So, here we are, In this article we are going to understand what Neural Netowks are and How they can be used for function mapping. And we will map sinusoidal functions with the help of Neural Networks.


So, Let's start with defining Neural Networks!

**Formal Definition**

> A Neural Network is a parameterized function $f(x, y)$: $R^n$ $->$ $R^m$ constructed as a composition of linear transformation and nonlinear activation functions where the parameters $\theta$ = $W_i$, $b_i$ are learned from data by minimizing a loss function using optimization algorithms such as gradient descent.

**Intuitive Definition**
> A neural network is a system of interconnected artificial neurons that:
> - Receives numerical inputs,
> - Performs weighted summation,
> - Applies nonlinear transformation,
> - Learns pattersn by adjusting weights based on error feedback.

From the formal definition of the neural networks, we can see that they are some functions. Or More precisely they are function approximators. When we are given some data, we try to find the relationship between inputs and outputs and that relationship is called approximated function as it's hard to find the exact function. And neural networks are the best in finding the relationship between inputs and outputs in a data or more precisely neural networks are used to find the approximate function that maps inputs to outputs.

Infact, Neural Networks can map any existing function, that's why they are called `Universal Function Approximators` and there is a theorem named **Universal Approximation Theorem (UAT)** given by Cybenko in 1989 which goes something like this.


> **A Single hidden layer neural network with a sufficient number of neurons can approximate any continuous function on a closed and bounded domain.**

<video autoplay loop muted playsinline controls style="display:block; margin: 0 auto; width:600px; max-width:100%;">
  <source src="https://github.com/user-attachments/assets/56f11459-2ddc-4832-9423-8bc50e92cb1d" type="video/mp4">
</video>


So all we need is one hidden layer to map any continuous function in a bounded function. But here is an issue, theoretically one hidden layer is enough but it might need a very large number of neurons which is inefficient. So, we will be having more than one layer in our neural network and it will help us with the following

- Heirarichical abstraction
- More efficient representation
- Better generalization

So, let's start with the mapping of sinusoidal function. We will conduct many experiments each one with a different goal

# Sinusoidal Function Mapping

We will be covering the following cases

- Sine Function mapping in the domain of [$\pi$, $\pi$]
- Sine function mapping in given domain [$a$, $b$]

## Sine Function mapping in the domain of [$\pi$, $\pi$]

Let's define the setup
- Input $\in$ [$\pi$, $\pi$]
- Output $\in$ $sin(Input)$
- Model
	- 1 input neuron
	- 35 neurons in hidden layer
	- 1 output neuron
	- with `Tanh` Activation Function
	- xavier uniform weight activation
	- MSELoss Function
	- SGD Optimizer
	- EPOCHS = 1000

```python
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go

# ---------------- GPU / CPU selection ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# ------------------------------------------------------

# Model
model = nn.Sequential(
    nn.Linear(1, 35),
    nn.Tanh(),
    nn.Linear(35, 1)
).to(device)  # send model to GPU

# ---- Proper Weight Initialization ----
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(m.bias)

model.apply(init_weights)

# Data
x = torch.linspace(-3.14, 3.14, 1000).unsqueeze(1).to(device)
y = torch.sin(x)

# Normalize
x = (x - x.mean()) / x.std()
y = (y - y.mean()) / y.std()

# Training settings
epochs = 1000
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.02)

# Store results for animation
predictions = []
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()

    predictions.append(output.detach().cpu().numpy().flatten())
    losses.append(loss.item())

```

Here is the Output



<video autoplay loop muted playsinline controls style="display:block; margin: 0 auto; width:600px; max-width:100%;">
  <source src="https://github.com/user-attachments/assets/a5cf3793-7fd2-4595-923c-fc0909419800" type="video/mp4">
</video>
