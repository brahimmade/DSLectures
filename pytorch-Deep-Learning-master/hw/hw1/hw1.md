# Homework 1: Backpropagation

### 1.1 Two Layer Neural Nets
Neural net architecture
$$
Linear_1 \rightarrow f \rightarrow Linear_2 \rightarrow g
$$

where $linear_{i}(x) = W^{(i)}x + b^{(i)}$ is the $i$-th affine transformation and $f,g$ are element wise nonlinear activation functions. When an input $x \in \mathbb{R}$ is fed to the network, $\hat{y} \in \mathbb{R}^K$ is obtained as the output.

### 1.2 Regression Task
We would like to perform a regression task. We choose $f(.) = (.)^+ = ReLU(.)$ and $g$ to be the identitiy function. To train this network, we choose MSE loss function $\ell_{MSE}( \hat{y},y )= \| \hat{y}-y \|^2$, where y is the target output.

a) Name and mathematically describe the 5 programming steps you wouldtake to train this model with PyTorch using SGD on a single batch of data.
1. Feed forward to get the logits
2. Compute the loss (MSE)
3. Zero the gradients before running
4. Backward pass to compute the gradient
5. Update params

(b)  For a single data point $(x,y)$, write down all inputs and outputs for forwardpass of each layer. You can only use variable $x,y,W^{(1)},b^{(1)},W^{(2)},b^{(2)}$ in your answer. (note that $Linear_i(x)=W^{(i)}x+b^{(i)}$).

| Layer      | Input                                          | Output                                         |
| ---------- | ---------------------------------------------- | ---------------------------------------------- |
| $Linear_1$ | $x$                                            | $W^{(1)\intercal} x + b^{(1)}$                          |
| $f$        | $W^{(1)\intercal} x + b^{(1)}$                 | $(W^{(1)\intercal} x + b^{(1)})^+$                      |
| $Linear_2$ | $(W^{(1)} x + b^{(1)})^+$                      | $W^{(2)\intercal}(W^{(1)\intercal} x + b^{(1)})^+ + b^{(2)}$     |
| $g$        | $W^{(2)\intercal}(W^{(1)\intercal} x + b^{(1)})^+ + b^{(2)}$     | $W^{(2)\intercal}(W^{(1)\intercal} x + b^{(1)})^+ + b^{(2)}$     |
| $Loss$     | $W^{(2)\intercal}(W^{(1)\intercal} x + b^{(1)})^+ + b^{(2)}$ , y | $W^{(2)\intercal}(W^{(1)\intercal} x + b^{(1)})^+ + b^{(2)} - y$ |


(c)  Write down the gradient calculated from the backward pass.  You can only use the following variables: $x,y,W^{(1)},b^{(1)},W^{(2)},b^{(2)}, \frac{\delta \ell}{\delta \hat{y}}, \frac{\delta z_2}{\delta z_1}, \frac{\delta \hat{y}}{\delta z_3}$ in your answer, where $z_1,z_2,z_3,\hat{y}$ are the outputs of $Linear_1,f,Linear_2,g$.

$$
\frac{\delta \ell}{W^{(2)}} = \frac{\delta \ell}{\delta \hat{y}}  \frac{\delta \hat{y}}{\delta W^{(2)}}
$$

$$
\frac{\delta \ell}{b^{(2)}} = \frac{\delta \ell}{\delta \hat{y}}  \frac{\delta \hat{y}}{\delta b^{(2)}}
$$

$$
\frac{\delta \ell}{W^{(1)}} = \frac{\delta \ell}{\delta \hat{y}}  \frac{\delta \hat{y}}{\delta z_3}   \frac{\delta z_2}{\delta z_1}  \frac{\delta z_2}{\delta W^{(1)}}
$$

$$
\frac{\delta \ell}{b^{(1)}} = \frac{\delta \ell}{\delta \hat{y}}  \frac{\delta \hat{y}}{\delta z_3}  \frac{\delta z_2}{\delta z_1} \frac{\delta z_2}{\delta b^{(1)}}
$$

$z_3$ = $z_2 * I$

(d)  Show us the elements of $\frac{\delta \ell}{\delta \hat{y}}, \frac{\delta z_2}{\delta z_1}, \frac{\delta \hat{y}}{\delta z_3}$ (be careful about the dimensional-ity)?

$$
\frac{\delta \ell}{\delta \hat{y}} = 2(W^{(2)\intercal}(W^{(1)\intercal} x + b^{(1)})^+ + b^{(2)} - y)
$$

$$
\frac{\delta z_2}{\delta z_1} = if \hspace{0.5em} z_1 > 0 \hspace{0.5em} , \hspace{0.5em} 1 
\newline else \hspace{0.5em} 0
$$

$$
\frac{\delta \hat{y}}{\delta z_3} = 1
$$

### 1.3 Classification Task

We would like to perform multi-class classification task, so we set bothf,g=σ,the logistic sigmoid function $σ(z)=(1+exp(−z))^{−1}$.

a)  If you want to train this network, what do you need to change in the equa-tions of (b), (c) and (d), assuming we are using the same MSE loss function.

(b)  Now you think you can do a better job by using a $Binary \ Cross \ Entropy$ (BCE) loss function $\ell_{BCE}(\hat{y},y)=\frac{1}{K}\sum_{i=1}^{K} −[y_i log(\hat{y_i})+(1−y_i) log(1−\hat{y_i})]$. What do you need to change in the equations of (b), (c) and (d)?

(c)  Things are getting better.  You realize that not all intermediate hidden activations need to be binary (or soft version of binary).  You decide to use $f(·)=(·)^+$ but keep $g$ as σ. Explain why this choice of $f$ can be beneficial for training a (deeper) network.