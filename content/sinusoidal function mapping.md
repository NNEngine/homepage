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

---

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


As you can see that the function is beautifully mapped by the Neural Network.

# Model Explanation
We have a 1D --> MLP --> 1D with architecture
```
x ∈ R
    ↓
Linear(1 → 35)
    ↓
tanh
    ↓
Linear(35 → 1)
    ↓
ŷ ∈ R
```

Because this is a finite-width tanh MLP, its representational class is:
> Smooth, continuous, bounded, low-to-moderate frequency function functions on compact intervals

And if we change the input range then model won't be able to learn the function.

### The phenomenon
The model fits sin(x) perfectly on [-$\pi$, $\pi$] but when the input range is expanded to (say [-5pi, 5pi] or [-10pi, 10pi], the model:
- losses precision
- fails to capture oscillations uniformly
- or fits only part of the range well

This happens even though sin(x) is the same function.

### Core reason (one sentence)
> Your model does not know that sin(x) is periodic. It only leans a local approximation over the training domain.

### What your model actually learns (not what you think)

Your tanh MLP is not learning “sin(x)”.

It is learning:
    f(x)≈sin(x)for x∈[a,b]

as a **smooth interpolant**, not as a symbolic or periodic rule.

Key implication
The learned function is:
- domain-dependent
- coordinate-dependent
- scale-dependent

Change the range → change the difficulty → change the learned mapping.

**Low frwquency over the interval**

Frequency of sin(x) is fixed, but effective frequency relative to domain size changes

| Domain      | Oscillations | Effective frequency |
| ----------- | ------------ | ------------------- |
| [-π, π]     | 1            | low                 |
| [-5π, 5π]   | 5            | 5× higher           |
| [-10π, 10π] | 10           | 10× higher          |

Your model is frequency-limited.

### The real culprit: spectral bias

Spectral bias (a.k.a. frequency bias)

>Neural networks trained with gradient descent learn low-frequency components first, and struggle with high-frequency components.

When you expand the domain:
- sin(x) contains more oscillations
- which means higher frequencies
- which your tanh MLP:
  - represents inefficiently
  - learns extremely slowly

This is why:
- center region looks okay
- outer regions degrade

## Solutions

### Solution 1 (BEST): Sinusoidal activations (SIREN)
replace
```
tanh → sin
```
Why it works:
- sin is a periodic basis
- model naturally represents oscillations
- frequency does NOT degrade with domain size

This directly fixes your problem.

You will see:
- perfect modelling even on large ranges
- stable gradients
- no center-only fitting

### Solution 2: Fourier feature mapping (very powerful)

Instead of feeding x, feed:

[sin(2kx), cos(2kx)]

This:
- lifts input into frequency space
- lets even tanh networks model oscillations

This is used in:
- NeRF
- signal modelling
- scientific ML

---

## Sine function mapping in given domain [$a$, $b$]


# Implementing SIREN

> SIREN gives your network a periodic inductive bias,
so it can model periodic / high-frequency functions much better.

It does NOT mean:
- “any input range works automatically”
- “extrapolation is solved”
- “all functions become easy”

## What SIREN CAN do

### Handle large input ranges for periodic functions

For functions like:
- sin(x)
- cos(x)
- sums of sinusoids
- wave equations
- oscillatory physics solutions

A SIREN can model:
```
x ∈ [-π, π]
x ∈ [-100π, 100π]
x ∈ ℝ   (in practice)
```

### Learn high frequencies early (no spectral bias)

Unlike tanh networks:
- SIREN does not strongly prefer low frequencies
- gradients for high-frequency components are not suppressed

This is why:
- sin(x) over many periods works
- sharp oscillations are captured uniformly

## What SIREN CANNOT do

### Arbitrary non-periodic extrapolation

If you train on:
```
x ∈ [-10, 10]
```

and test on:
```
x ∈ [10, 20]
```

SIREN will still:
- extrapolate poorly
- possibly oscillate wildly

Why?
- sin activations extrapolate periodically, not semantically

So SIREN improves interpolation, not logical extrapolation.


```python
#------------------------------------------MODEL-------------------------------------

class Sine(nn.Module):
    def __init__(self, omega_0=1.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)

class SirenModel(nn.Module):
    def __init__(self, omega_0=20.0):
        super().__init__()

        self.omega_0 = omega_0

        self.fc1 = nn.Linear(1, 35)
        self.act1 = Sine(omega_0)

        self.fc2 = nn.Linear(35, 35)
        self.act2 = Sine(1.0)

        self.fc3 = nn.Linear(35, 1)

        self.apply(self.init_weights)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                if m is self.fc1:
                    m.weight.uniform_(
                        -1 / m.in_features,
                         1 / m.in_features
                    )
                else:
                    bound = math.sqrt(6 / m.in_features) / self.omega_0
                    m.weight.uniform_(-bound, bound)

                nn.init.zeros_(m.bias)


#-------------------------------------------------Dataset--------------------------------

# With normalization

class SIRENDataset(Dataset):
    def __init__(self):
        x = torch.linspace(-10 * math.pi, 10 * math.pi, 10000).unsqueeze(1)
        y = torch.sin(x)

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

model = SirenModel().to(device)

dataset = SIRENDataset()
loader = DataLoader(dataset, batch_size = 256, shuffle = True)

# training settings

EPOCHS = 3000
criterion  = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Store results for animation
predictions = []
losses = []
pred_epochs = []   # store exact epoch numbers
grad_norms = [] # for stroing normalized gradients


for epoch in tqdm(range(EPOCHS), desc="Training", unit="epoch"):
    model.train()
    epoch_loss = 0.0
    epoch_grad_norm = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()

        # ---- gradient norm (L2) (totall removable)----
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        epoch_grad_norm += total_norm
        # --------------------------------------

        optimizer.step()
        epoch_loss += loss.item()

    # ---- store per-epoch values ----
    epoch_loss /= len(loader)
    epoch_grad_norm /= len(loader)

    losses.append(epoch_loss)
    grad_norms.append(epoch_grad_norm)

    # ---- store predictions for animation ----
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            x_all = dataset.x.to(device)
            y_pred = model(x_all)
            predictions.append(y_pred.cpu().numpy().flatten())
            pred_epochs.append(epoch)


    if epoch % 100 == 0:
        tqdm.write(
            f"Epoch {epoch} | Loss {epoch_loss:.6f} | GradNorm {epoch_grad_norm:.4f}"
        )

```
