# 高维随机变量$P(x_1,x_2,\dots,x_p)$

- Sum Rule: $P(x_1)=\int P(x_1,x_2)\,dx_2$
- Product Rule: $P(x_1,x_2)=P(x_1)P(x_2|x_1)=P(x_2)P(x_1|x_2)$
- Chain Rule: $P(x_1,x_2,\dots,x_p)=\prod^p_{i=1}P(x_i|x_1,x_2,\dots,x_{i-1})$
- Bayesian Rule: $P(x_2|x_1)=\frac{P(x_1,x_2)}{P(x_1)}=\frac{P(x_1,x_2)}{\int P(x_1,x_2)\,dx_2}=\frac{P(x_2)P(x_1|x_2)}{\int P(x_2)P(x_1|x_2)\,dx_2}$
- 若$x_1,x_2$互相独立，则$P(x_1,x_2)=P(x_1)P(x_2)$
- 一些推论
  - $P(X,Y,W)=P(X,Y|W)P(W)=P(Y|X,W)P(X,W)=P(Y|X,W)P(W|X)P(X)$
- 边缘概率$P(x_i)$
- 条件概率$P(x_j|x_i)$

# 共轭（conjugate）分布

- 定义：在贝叶斯统计中，如果后验分布与先验分布属于同类（分布形式相同），则先验分布与后验分布被称为**共轭分布**，而先验分布被称为似然函数的**共轭先验**。
- 高斯分布具有自共轭性：如果先验和似然都是高斯分布，则后验一定是高斯分布。

- $posterior\propto prior\times likelihood$
  - 高斯分布是共轭的
  - 例如对于bayesian线性回归模型，$posterior$后验：$P(w|data)\propto P(Y|X,W)P(W)$，

# 期望

- 离散变量$E(X)=\sum_{i=1}^nx_ip_i$

- 连续变量$E(X)=\int_{-\infty}^{+\infty}xf(x)\ dx$

- 若$C$为常数，$X,Y$为随机变量
  $$
  E(C)=C
  $$

  $$
  E(CX)=CE(X)
  $$

  $$
  E(X+Y)=E(X)+E(Y)
  $$

  若$X,Y$互相独立
  $$
  E(XY)=E(X)E(Y)
  $$
  

# 方差

- $$
  \begin{align}
  D(x)
  &=E\left[(x-E(x))^2\right]\\
  &=E(x^2)-E^2(x)
  \end{align}
  $$

  $$
  D(cx)=c^2D(x)
  $$

  $$
  D(x+c)=D(x)
  $$

  $$
  D(x\pm y)=D(x)+D(y)\pm 2Cov(x,y)
  $$

  若$X,Y$互相独立
  $$
  D(x\pm y)=D(x)+D(y)
  $$
  

# 协方差

$$
\begin{align}
Cov(x,y)
&=E[(x-E[x])(y-E[y])] \\
&=E[xy]-E[x]E[y]
\end{align}
$$

$$
Cov(x,x)=D(x)
$$

# Univariate Gaussian Properties

- Sum of Gaussian variables is also Gaussian
  $$
  y_i\sim\mathcal{N}(\mu_i,\sigma_i^2)
  $$

  $$
  \sum_{i=1}^ny_i\sim\mathcal{N}\left(\sum_{i=1}^n\mu_i,\sum_{i=1}^n\sigma_i^2\right)
  $$

- Scaling a Gaussian leads to a Gaussian
  $$
  y\sim\mathcal{N}(\mu,\sigma)
  $$

  $$
  wy\sim\mathcal{N}(w\mu,w^2\sigma^2)
  $$

# Multivariate Gaussian Properties

- If
  $$
  y=Wx+\epsilon
  $$

- Assume
  $$
  x\sim\mathcal{N}(\mu,C)
  $$

  $$
  \epsilon\sim\mathcal{N}(0,\Sigma)
  $$

- Then
  $$
  y\sim\mathcal{N}(W\mu,WCW^\mathrm{T}+\Sigma)
  $$
  If $\Sigma=\sigma^2\mathrm{I}$, this is Probabilistic Principal Component Analysis.

# Gaussian Identites

The multivariate Gaussian (or Normal) distribution has a joint probability density given by
$$
p(x|m,\Sigma)=(2\pi)^{-D/2}|\Sigma|^{-1/2}\exp\left(-\frac{1}{2}(x-m)^\mathrm{T}\Sigma^{-1}(x-m)\right)
$$
where $m$ is the mean vector (of length $D$) and $\Sigma$ is the (symmetric, positive definite) covariance matrix (of size $D\times D$). As a shorthand we write $x\sim\mathcal{N}(m,\Sigma)$.

Let $x$ and $y$ be the jointly Gaussian random vectors
$$
\begin{bmatrix}
x\\
y
\end{bmatrix}\sim\mathcal{N}\left(
\begin{bmatrix}
\mu_x\\
\mu_y
\end{bmatrix},
\begin{bmatrix}
A&C\\
C^\mathrm{T}&B
\end{bmatrix}
\right)=\mathcal{N}\left(
\begin{bmatrix}
\mu_x\\
\mu_y
\end{bmatrix},
\begin{bmatrix}
\tilde{A}&\tilde{C}\\
\tilde{C}^\mathrm{T}&\tilde{B}
\end{bmatrix}^{-1}
\right)
$$
then the marginal distribution of $x$ and the conditional distribution of $x$ given $y$ are $x\sim\mathcal{N}(\mu_x,A)$
$$
\begin{align}
x|y&\sim\mathcal{N}(\mu_x+CB^{-1}(y-\mu_y),A-CB^{-1}C^\mathrm{T})\\
x|y&\sim\mathcal{N}(\mu_x-\tilde{A}^{-1}\tilde{C}(y-\mu_y),\tilde{A}^{-})
\end{align}
$$
