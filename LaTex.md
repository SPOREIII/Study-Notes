# LaTex相关语法

## 字体

| 代码            | 符号          |
| --------------- | ------------- |
| `$\mathbb{R}$`  | $\mathbb{R}$  |
| `$\mathcal{R}$` | $\mathcal{R}$ |
| `$\mathscr{R}$` | $\mathscr{R}$ |



## 希腊字母

|    代码    |   符号   |    代码    |   符号   |
| :--------: | :------: | :--------: | :------: |
| `$\alpha$` | $\alpha$ | `$\Alpha$` | $\Alpha$ |
| `$\theta$` | $\theta$ | `$\Theta$` | $\Theta$ |
| `$\omega$` | $\omega$ | `$\Omega$` | $\Omega$ |

## 存在$\exists$与所有$\forall$符

|    代码     |   符号    |
| :---------: | :-------: |
| `$\forall$` | $\forall$ |
| `$\exists$` | $\exists$ |

## 字符

|    代码    |   符号   |
| :--------: | :------: |
|  `$\sim$`  |  $\sim$  |
| `$\ldots$` | $\ldots$ |

## 帽子

|      代码       |     符号      |
| :-------------: | :-----------: |
|  `$\tilde{A}$`  |  $\tilde{A}$  |
| `$\widehat{A}$` | $\widehat{A}$ |
|   `$\hat{A}$`   |   $\hat{A}$   |
|   `$\bar{N}$`   |   $\bar{N}$   |

## 括号

- 括号加大，加`\left`和`\right`

  `$$J_t=\mathbb{E}\left(\sum^\bar{N}_{k=0}{l_t(x(k),u(k),k)}\right)$$`

  $$J_t=\mathbb{E}\left(\sum^\bar{N}_{k=0}{l_t(x(k),u(k),k)}\right)$$

  `$$J_t=\mathbb{E}(\sum^\bar{N}_{k=0}{l_t(x(k),u(k),k)})$$`

  $$J_t=\mathbb{E}(\sum^\bar{N}_{k=0}{l_t(x(k),u(k),k)})$$

## 正体

- `\text`

  `$\text{Pr}(\bar{X}\in\bar{\mathcal{X}_j})\geq p_j\ \text{for \ all}\ j=1,...,n_{c_x}$`

   $\text{Pr}(\bar{X}\in\bar{\mathcal{X}_j})\geq p_j\ \text{for \ all}\ j=1,...,n_{c_x}$

- `\rm`

  `$\rm Pr(\bar{X}\in\bar{\mathcal{X}_j})\geq p_j\ for \ all\ j=1,...,n_{c_x}$`

  $\mathrm{Pr}(\bar{X}\in\bar{\mathcal{X}_j})\geq p_j\ \mathrm{for \ all}\ j=1,...,n_{c_x}$

## 运算符

|       代码       |      符号      |
| :--------------: | :------------: |
| `$\textgreater$` | $\textgreater$ |
|  `$\textless$`   |  $\textless$   |
|     `$\geq$`     |     $\geq$     |
|     `$\ge$`      |     $\ge$      |
|     `$\leq$`     |     $\leq$     |
|     `$\le$`      |     $\le$      |
|                  |                |

## 空格

|     代码      |     符号     |
| :-----------: | :----------: |
|    `$ab$`     |     $ab$     |
| `$a\qquad b$` | $a \qquad b$ |
| `$a\quad b$`  |  $a\quad b$  |
|   `$a\ b$`    |    $a\ b$    |
|   `$a\;b$`    |    $a\;b$    |
|   `$a\,b$`    |    $a\,b$    |
|   `$a\!b$`    |    $a\!b$    |

## 正下方下标

$$
\mathop{\mathrm{minimize}}\limits_{x}
$$

## 规划问题

- 例子1

  $$
  \begin{array}{cl}{\min } & {|\mathcal{S}|} \\ {\text { subject to }} & {\operatorname{tr}\left(\Sigma_{z_{k-1}}^{-1}\right) \geq R} \\ {} & {\mathcal{S} \subseteq\left[m_{x}\right] \operatorname{and} / \text { or } \mathcal{S} \subseteq\left[w_{x} m_{x}\right]} \\ {} & {s \notin\left[m_{x}^{\prime}\right]}\end{array}
  $$
  
- 例子2

  $$
  \begin{align*}
    &\max\quad z=\sum\limits_{i=1}^m c_i x_i\\
    & \begin{array}{r@{\quad}r@{}l@{\quad}l}
    s.t.&\sum\limits_{j=1}^m a_{ij} x_j&\leq b_i,  &i=1,2,3\ldots,n\\
     &x_j&\geq110,  &i=1,2,3\ldots,n  \\
     &x_j&\geq10,  &i=1,2,3\ldots,n  \\
     &x_j&\geq0,  &i=1,2,3\ldots,n  \\
    & x_j&\geq0,  &i=1,2,3\ldots,n  \\
    \end{array} 
    \end{align*}
  $$
  
- 例子3

  $$
  \begin{align*}
  	&J^\star_t=\mathop{\text{minimize}}\limits_{\{\pi_k\}}\quad \mathbb{E}\left(\sum^{\bar{N}}_{k=0}{l_t(x(k),u(k),k)}\right)\\
  	& \begin{array}{r@{\quad}r@{}l@{\quad}l}
  		\text{subject to}\quad
  		&x(k+1)&=f_t(x(k),u(k),k,w(k),\theta_t),&{}\\
  		&u(k)&=\pi_k(x(0),\ldots,x(k)),&{}\\
  		&\bar{W}&=[w(0),\ldots,w(\bar{N})]\sim\mathcal{Q}^{\bar{W}},&\theta_t\sim\mathcal{Q}^{\theta_t}\\
  		&\mathrm{Pr}(\bar{X}&=[x(0),\ldots,x(\bar{N})]\sim\bar{\mathcal{X}_j})\geq p_j&\text{for all}\ j=1,\ldots,n_{c_x}\\
  		&\mathrm{Pr}(\bar{U}&=[u(0),\ldots,u(\bar{N})]\sim\bar{\mathcal{U}_j})\geq p_j&\text{for all}\ j=1,\ldots,n_{c_u}\\
  	\end{array}
  \end{align*}
  $$
  

