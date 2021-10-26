# Gaussian Process

- 数据集 $\text{Data}:\{(x_i,y_i)^{N}_{i=1}\},\ x_i\in\mathbb{R}^P,\ y_i\in\mathbb{R}$

- 线性回归 $\text{Model}:\left\{\begin{array}{**lr**}f(X)=X^\mathrm{T}W\\y=f(x)+\epsilon\end{array} \right.$

  $\epsilon\sim \mathcal{N}(0,\sigma^2)$

- Bayesian Method: $W\sim \text{dist}$

  - Inference: 根据$W$的prior和$\text{Data}$计算posterior $P(W|\text{Data})$

    由高斯自共轭性，先验和似然均为高斯分布则后验也为高斯分布

    $P(W|\text{Data})\propto \text{likelihood}\times\text{prior}$

  - Prediction: 根据$x^\star$计算$y^\star$
    $$
    \begin{align}
    P(y^\star|\text{Data},x^\star)&=\int_wP(y^\star|w,Data,x^\star)P(w|\text{Data},x^\star)\ dw \\
    &= \int_wP(y^\star|w,x^\star)P(w|\text{Data})\ dw
    \end{align}
    $$

- Bayesian linear regression + kernal trick = Gaussian process regression

## Bayesian linear regression

- 详细推导

  The standard linear regression model with Gaussian noise
  $$
  \left\{\begin{array}{**lr**}f(X)=X^\mathrm{T}W\\y=f(x)+\epsilon\end{array} \right.
  $$
  where $x$ is the input vector 
  $$
  X=
  \begin{bmatrix}
  x_{11}&\cdots&x_{1N}\\
  \vdots&\ddots&\vdots \\
  x_{p1}&\cdots&x_{pN}
  \end{bmatrix}_{p\times N}
  $$

  $w$ is a vector of weights (parameters) of the linear model. In the Bayesian formalism we need to specify a prior over the parameters, expressing our beliefs about the parameters before we look at the observations. We put a zero mean Gaussian prior with covariance matrix $\Sigma_p$ on the weights
  $$
  W\sim\mathcal{N}(0,{\Sigma_p}_{p\times p})
  $$
  $f$ is the function value.

  $y$ is the observed target value. 
  $$
  y=
  \begin{bmatrix}
  y_1\\
  y_2\\
  \vdots\\
  y_N
  \end{bmatrix}_{N\times 1}
  $$
  The probability density of the observations given the parameters, which is factored over cases in the training set (becaues of the independence assumption) to give
  $$
  \begin{align}
  p(y|X,W)&=\prod^n_{i=1}p(y_i|x_i,w)\\
  &=\prod_{i=1}^n\frac{1}{\sqrt{2\pi}\sigma_n}\exp(-\frac{(y_i-x_i^\mathrm{T}w)^2}{2\sigma_n^2})\\
  &=\frac{1}{(2\pi\sigma_n^2)^{n/2}}\exp(-\frac{1}{2\sigma^2_n}|y-X^\mathrm{T}W|^2)\\
  &=\mathcal{N}(X^\mathrm{T}W,\sigma_n^2I)
  \end{align}
  $$
  $\epsilon$ is the noise
  $$
  \epsilon \sim \mathcal{N}(0,\sigma^2)
  $$

  - Inference

    Inference in the Bayesian linear model is based on the posterior distribution over the weights, computed by Bayes's rule,
    $$
    \text{posterior}=\frac{\text{likelihood}\times \text{prior}}{\text{marginal likelihood}}
    $$

    $$
    \begin{align}
    p(W|y,X)&=\frac{p(y|W,X)p(W|X)}{p(y|X)} \\
    &=\frac{p(y|W,X)p(W)}{p(y|X)}
    \end{align}
    $$

    It can be derived from the Gaussian conjugacy
    $$
    p(W|X,y)\propto \exp{\left(-\frac{1}{2}(W-\bar{W})^\mathrm{T}(\frac{1}{\sigma^2_n}XX^\mathrm{T}+\Sigma^{-1}_p)(W-\bar{W})\right)}
    $$
    where $\bar{W}=\sigma^{-2}_n(\sigma^{-2}_nXX^\mathrm{T}+\Sigma_p^{-1})^{-1}Xy$, and we recognize the form of the posterior distribution as Gaussian with mean $\bar{W}$ and covariance matrix $A^{-1}$
    $$
    p(W|X,y)\sim \mathcal{N}(\bar{W}=\frac{1}{\sigma_n^2}A^{-1}Xy,A^{-1})
    $$
    where $A=\sigma^{-2}_nXX^\mathrm{T}+\Sigma_p^{-1}$.

  - Prediction
    $$
    \because f(X)=X^\mathrm{T}W,\ p(W|X,y)\sim \mathcal{N}(\mu_W,\Sigma_W), \ f(x_\star)=x_\star^\mathrm{T}W,\\
    \therefore f(x_\star)\sim \mathcal{N}(x_\star^\mathrm{T}\mu_W,x_\star^\mathrm{T}\Sigma_Wx_\star),\ 
    f(x_\star)+\epsilon\sim \mathcal{N}(x_\star^\mathrm{T}\mu_W,x_\star^\mathrm{T}\Sigma_Wx_\star+\sigma^2_nI)
    $$

    $$
    \begin{align}
    p(f_\star|x_\star,X,y)&=\int_Wp(f_\star|x_\star,W)p(W|X,y)\ dW \\
    &=\mathcal{N}(\frac{1}{\sigma_n^2}x_\star^\mathrm{T} A^{-1}Xy,x_\star^\mathrm{T}A^{-1}x_\star)
    \end{align}
    $$


### Projections of Inputs into Feature Space

A very simple idea to overcome the limited expressiveness of Bayesian linear regression is to first project the inputs into some high dimensional space using a set of basis functions and then apply the linear model in this space instead of directly on the inputs themselves. For example $\phi(x)=(1,x,x^2,x^3,\dots)^\mathrm{T}$ to implement polynomial regression.

The function $\phi(\mathrm{x})$ which maps a D-dimensional input vector $\mathrm{x}$ into an $N$ dimensional feature space. Let the matrix $\Phi(\mathrm{X})$ be the aggregation of columns $\phi(\mathrm{x})$ for all cases in the training set. Now the model is
$$
f(\mathrm{x})=\phi(\mathrm{x})^\mathrm{T}\mathrm{w}
$$
where the vector of parameters now has lenght $N$

The analysis for this model is analogous to the standard linear model, except that everywhere $\Phi(X)$ is substituted for $X$. Thus the predictive distribution becomes
$$
\begin{align}
f_\star|\mathrm{x}_\star,X,\mathrm{y}&\sim\mathcal{N}\left(\frac{1}{\sigma_n^2}\phi(\mathrm{x}_\star)^\mathrm{T} A^{-1}\Phi \mathrm{y},\phi(\mathrm{x}_\star)^\mathrm{T}A^{-1}\phi(\mathrm{x}_\star)\right) \\
f_\star|\mathrm{x}_\star,X,\mathrm{y}&\sim\mathcal{N}\left(
\phi_\star^\mathrm{T}\Sigma_p\Phi(K+\sigma_n^2I)^{-1}\mathrm{y},
\phi_\star^\mathrm{T}\Sigma_p\phi_\star-\phi_\star^\mathrm{T}\Sigma_p\Phi(K+\sigma^2_nI)^{-1}\Phi^\mathrm{T}\Sigma_p\phi_\star
\right)
\end{align}
$$
with $\Phi=\Phi(x)$ ,  $A=\sigma^{-2}_n\Phi\Phi^\mathrm{T}+\Sigma_p^{-1}$, $\phi(x_\star)=\phi_\star$ and defined $K=\Phi^\mathrm{T}\Sigma_p\Phi$.

The feature space always enters in the form of $\Phi^\mathrm{T}\Sigma_p\Phi$, $\phi_\star^\mathrm{T}\Sigma_p\Phi$, or $\phi_\star^\mathrm{T}\Sigma_p\phi_\star$; thus the entries of these matrices are invariably of the form $\phi(x)^\mathrm{T}\Sigma_p\phi(x^\prime)$ where $x$ and $x^\prime$ are in either the training or the test sets.

Define $k(x,x^\prime)=\phi(x)^\mathrm{T}\Sigma_p\phi(x^\prime)$, we call $k(\cdot,\cdot)$ a covariance function or kernal.
$$
\begin{align}
k(x,x^\prime)
&=\phi(x)^\mathrm{T}\Sigma_p\phi(x^\prime)\\
&=\phi(x)^\mathrm{T}(\Sigma_p^{1/2})^2\phi(x^\prime)\\
&=\left(\Sigma_p^{1/2}\phi(x)\right)^\mathrm{T}\Sigma_p^{1/2}\phi(x^\prime)\\
&=\psi(x)\cdot\psi(x^\prime)
\end{align}
$$

## Function-space View

- Definition: A Gaussian process is a collection of random variables, any finite number of which have a joint Gaussian distribution.

A Gaussian process is completely specified by its mean function and covariance function. We define mean function $m(x)$and the covariance function $k(x,x^\prime)$ of a real process $f(x)$ as
$$
\begin{align}
m(x)&=\mathbb{E}[f(x)]\\
k(x,x^\prime)&=\mathbb{E}[(f(x)-m(x))(f(x^\prime)-m(x^\prime))]
\end{align}
$$
write the Gaussian process as
$$
f(x)\sim\mathcal{GP}(m(x),k(x,x^\prime))
$$
A simple example of a Gaussian process can be obtained from our Bayesian linear regression model $f(x)=\phi(x)^\mathrm{T}\mathrm{w}$ with prior $\mathrm{w}\sim\mathcal{N}(0,\Sigma_p)$. We have for the mean and covariance
$$
\begin{align}
\mathbb{E}[f(x)]&=\phi(x)^\mathrm{T}\mathbb{E}[\mathrm{w}]=0\\
\mathbb{E}[f(x)f(x^\prime)]&=\phi(x)^\mathrm{T}\mathbb{E}[\mathrm{ww^T}]\phi(x^\prime)=\phi(x)^\mathrm{T}\Sigma_p\phi(x^\prime)
\end{align}
$$
The covariance function (squared exponential SE) specifies the covariance between pairs of random variables
$$
cov\left(f(x_p),f(x_q)\right)=k(x_p,x_q)=\exp(-\frac{1}{2}|x_p-x_q|^2)
$$
For this covariance function, we see that the covariance is almost unity between variables whose corresponding inputs are very close, and decreases as their distance in the input space increase.

### Prediction with Noise-free Observations

Initially, we will consider the simple special case where the observations are noise free, that is we know $\{(x_i,f_i)|i=1,\dots,n\}$. The joint distribution of the training outputs, $f$, and the test outputs $f_\ast$ according to the prior is
$$
\begin{bmatrix}
f\\
f_\ast
\end{bmatrix}\sim
\mathcal{N}\left(0,
\begin{bmatrix}
K(X,X)&K(X,X_\ast)\\
K(X_\ast,X)&K(X_\ast,X_\ast)
\end{bmatrix}
\right).
$$
To get the posterior distribution over functions we need to restrict this joint prior distribution to contain only those functions which agree with the observed data points.

![image-20211020212520980](D:\Master\program\study-notes\fig\Gaussian Processes\image-20211020212520980.png)

Conditioning the joint Gaussian prior distribution on the observations to give
$$
f_\ast|X_\ast,X,f\sim\mathcal{N}\left(K(X_\ast,X)K(X,X)^{-1}f, 
K(X_\ast,X_\ast)-K(X_\ast,X)K(X,X)^{-1}K(X,X_\ast)\right)
$$

### Prediction using Noisy Observations

It is typical for more realistic modelling situations that we do not have access to function values themselves, but only noisy versions thereof $y=f(x)+\epsilon$. Assuming additive independent indentically distributed Gaussian noise $\epsilon$ with variance $\sigma_n^2$, the prior on the noisy observations becomes
$$
\mathrm{cov}(y_p,y_q)=k(x_p,x_q)+\sigma_n^2\delta_{pq}
$$

$$
\mathrm{cov}(y)=K(X,X)+\sigma_n^2I
$$

where $\delta_{pq}$ is a Kronecker delta which is one if $p=q$ and zero otherwise.

Introducing the noise term in equation, we can write the joint distribution of the observed target values and the function values at the test locations under the prior as
$$
\begin{bmatrix}
y\\
f_\ast
\end{bmatrix}\sim
\mathcal{N}\left(0,
\begin{bmatrix}
K(X,X)+\sigma_n^2I&K(X,X_\ast)\\
K(X_\ast,X)&K(X_\ast,X_\ast)
\end{bmatrix}
\right).
$$
Deriving the conditional distribution corresponding to the equation of noise-free, we arrive at the key predictive equations of Gaussian process regression
$$
\begin{align}
f_\ast|X,y,X_\ast&\sim\mathcal{N}\left(\bar{f_\ast},\mathrm{cov}(f_\ast)\right),\qquad \text{where}\\
\bar{f}_\ast&\triangleq\mathbb{E}[f_\ast|X,y,X_\ast]=K(X_\ast,X)[K(X,X)+\sigma_n^2I]^{-1}y\\
\mathrm{cov}(f_\ast)&=K(X_\ast,X_\ast)-K(X_\ast,X)[K(X,X)+\sigma_n^2I]^{-1}K(X,X_\ast)
\end{align}
$$

# Deep Probabilistic Modeling with Gaussian Processes

[video](https://www.bilibili.com/video/BV1p441117qB?share_source=copy_web)

# [Introduction to Gaussian Processes](http://bridg.land/posts/gaussian-processes-1)

## What is a Gaussian Process and why would I use one?

Most modern techniques in machine learning tend to avoid this by  parameterising functions and then modeling these parameters (e.g. the  weights in linear regression). However GPs are nonparametric models that model the function directly. 

