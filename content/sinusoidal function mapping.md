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
