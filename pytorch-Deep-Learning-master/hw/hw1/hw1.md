# Homework 1: Backpropagation

### 1.1 Two Layer Neural Nets
Neural net architecture
$$
Linear_1 \rightarrow f \rightarrow Linear_2 \rightarrow g
$$

where $linear_{i}(x) = W^{(i)}x + b^{(i)}$ is the $i$-th affine transformation and $f,g$ are element wise nonlinear activation functions. When an input $x \in \mathbb{R}$ is fed to the network, $ \hat{y} \in \mathbb{R}^K$ is obtained as the output.

### 1.2 Regression Task
We would like to perform a regression task. We choose $f(.) = (.)^+ = ReLU(.)$ and $g$ to be the identitiy function. To train this network, we choose MSE loss function $\ell_{MSE}( \hat{y},y )= \| \hat{y}-y \|^2 $, where y is the target output.

a) Name and mathematically describe the 5 programming steps you wouldtake to train this model withPyTorchusing SGD on a single batch of data.
1. Feed forward to get the logits
2. Compute the loss (MSE)
3. Zero the gradients before running
4. Backward pass to compute the gradient
5. Update params

(b)  For a single data point $(x,y)$, write down all inputs and outputs for forwardpass of each layer. You can only use variable $x,y,W^{(1)},b^{(1)},W^{(2)},b^{(2)} $ in your answer. (note that $Linear_i(x)=W^{(i)}x+b^{(i)}$).

$ W^{(1)} x + b^{(1)} $
$ (W^{(1)} x + b^{(1)})^+ $
$ W^{(2)}(W^{(1)} x + b^{(1)})^+ + b^{(2)} $
$ W^{(2)}(W^{(1)} x + b^{(1)})^+ + b^{(2)} $

(c)  Write down the gradient calculated from the backward pass.  You can only use the following variables: $ x,y,W^{(1)},b^{(1)},W^{(2)},b^{(2)}, \frac{\delta \ell}{\delta \hat{y}}, \frac{\delta z_2}{\delta z_1}, \frac{\delta \hat{y}}{\delta z_3} $ in your answer, where $z_1,z_2,z_3,\hat{y}$ are the outputs of $Linear_1,f,Linear_2,g$.

(d)  Show us the elements of $\frac{\delta \ell}{\delta \hat{y}}, \frac{\delta z_2}{\delta z_1}, \frac{\delta \hat{y}}{\delta z_3}  $(be careful about the dimensional-ity)?