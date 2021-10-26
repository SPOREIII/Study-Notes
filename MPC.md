# Model predictive control

# Nominal model

- [In practice, what is the difference between the “nominal plant model” and the “plant model”?](https://math.stackexchange.com/questions/3368737/in-practice-what-is-the-difference-between-the-nominal-plant-model-and-the-p)

  You design your control for the nominal model, i.e., for the model  where you assume that you know all the parameters, or where you use some averages of what the parameters should be.

  In practice, you use only this model, and no other model is  available. However, the real plant (for sure) differs from your model:  it can have different values of the parameters (parametric  uncertainties) or some unmodelled dynamics, e.g., the senor's inertia.  These uncertainties define how and why the real plant is not the same as the nominal one that you have used for control design, and why your  real signals are not as good as the simulations results. 

  At this moment, you ask: well, can I check in advance how does my  controller deal with all these uncertainties? Yes, to do so you need a  model of the plant that is different from the nominal one, e.g., it  considers possible saturations.  Then you test you nominal controller  (the one that you have designed for the nominal plant) for various plant models with different parameters and so on. It gives you some ideas  about the sensitivity of your controller with respect to uncertainties.  And finally, you apply your controller to the real plant and hope it  will work.

  To summarize, you use the nominal model for control design, you use  plant models to test your design against uncertainties, and you apply  your controller to a real plant.  

# Learning-Based Model Predictive Control: Toward Safe Learning in Control

## introduction

- This review provides an overview of these research efforts in the context of learning-based MPC.
- Online MPC represents the learning affects the MPC optimization problem to be solved at each sampling instance.
- A central question in the context of learning and control is the exploration-exploitation trade-off and the problem of dual control.
- A data-based optimization technique with several applications to MPC that has received substantial interest in recent years is scenario optimization.

## optimal control of uncertain systems

### The general stochastic optimal control problem

- formulate the system dynamic in discrete time as
  $$
  x(k+1)=f_t(x(k),u(k),k,w(k),\theta_t)
  $$
  $x(k)\in\mathbb{R}^{n_x}$ is the system state and $u(k)\in\mathbb{R}^{n_u}$ is the applied input at time $k$.

  The dynamics are subject to various sources of uncertainty, distinguish by using two categories:

  - $\theta_t\sim\mathcal{Q}^{\theta_t}$ is a random variable describing the parametric uncertainty of the system,  constant over time.
  - $w(k)$ describes  a sequence of random variables corresponding to disturbances or process noise in the system, often assumed to be independent and identically distributed (i.i.d.)

- The true problem therefore relates to the development of an optimal controller for a distribution of systems given by $\mathcal{Q}^{\theta_t}$ under random disturbances $w(k)$

- Throughout this review, we assume direct access to measurements of the system state $x(k)$ and neglect the problem of state estimation.

- The optimality of the controller is defined with respect to a cost or objective function.
  $$
  J_t=\mathbb{E}\left(\sum^\bar{N}_{k=0}{l_t(x(k),u(k),k)}\right)
  $$

- Expressing the state and input trajectories over the horizon as $\bar{X}=[x(0),...,x(\bar{N})]$ and $\bar{U}=[u(0),...,u(\bar{N})]$, respectively, one can compactly express such system constraints as $\text{Pr}(\bar{X}\in\bar{\mathcal{X}_j})\geq p_j\ \text{for \ all}\ j=1,...,n_{c_x}$, $\text{Pr}(\bar{U}\in\bar{\mathcal{U}_j})\geq p_j\ \text{for \ all}\ j=1,...,n_{c_u}$

- Stochastic optimal control problem can be formulated as 
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
  optimizing over a sequence of control laws $\{\pi_k\}$ (policy), which can make use of all information in the form of state measurements $x(k)$ up to time step $k$.

## Model Predictive Control

- MPC approximates the problem in stochastic optimal control problem by repeatedly solving a simplified version of the problem initialized at the currently measured state $x(k)$ over a shorted horizon $N$ in a receding-horizon fashion. 

- The prediction model can be formulated as $x_{i+1|k}=f(x_{i|k},u_{i|k},i+k,w_{i|k},\theta)$. As well as predicted states and inputs. Use the subscript $i|k$ to emphasize predictive quantities. $x_{i|k}$ is the $i$-step-ahead prediction of the state, initialized at $x_{0|k}=x(k)$.

- The prediction dynamics $f$ typically aims at approximating the true dynamics in system but often differs, e.g., for computational reasons or because a succinct description of the true dynamics is unavailable.

- Control formulations based on, e.g., a linear approximation can often yield tractable and powerful controllers.

- Considering the cost and constraints over a shortened prediction horizon $N$, in nominal MPC, the optimization can be performed over control sequences $U=[u_{0|k},\ldots,u_{N-1|k}]$ rather than policies, resulting in the constrained optimal control problem
  $$
  \begin{align*}
  &J^\star=\mathop{\text{minimize}}\limits_U\quad l_f(x_{N|k},u_{N|k},k+N)+\sum_{i=0}^{N-1}l(x_{i|k},u_{i|k},i+k)\\
  &\begin{array}{r@{\quad}r@{}l@{\quad}l}
  \text{subject to}\quad
  &x_{i+1|k}&=f(x_{i|k},u_{i|k},i+k),&{}\\
  &U&=[u_{0|k},\ldots,u_{N|k}]\in\mathcal{U}_j,&\text{for all}\ j=1,\ldots,n_{c_u}\\
  &X&=[x_{0|k},\ldots,x_{N|k}]\in\mathcal{X}_j,&\text{for all}\ j=1,\ldots,n_{c_x}\\
  &x_{N|k}&\in\mathcal{X}_f,&{}\\
  &x_{0|k}&=x(k).&{}
  \end{array}
  \end{align*}
  $$

  - The stage cost function $l$ does not necessarily coincide with the actual cost $l_t$, which may express complex objectives that can be sparse, nondifferentiable, or even unavailable as a precise mathematical expression, such that it is often ill suited for a solution using gradient-based numerical optimization.
  - To mitigate the effect of the shortened horizon, a particular cost $l_f(x_{N|k},u_{N|k},k+N)$ and constraint $\mathcal{X}_f$ on the last predicted state is imposed to approximate the cost and the effect of the constraints over the remainder of the possibly infinite control horizon $\bar{N}$. In many MPC formulations, these terminal components $l_f$ and $\mathcal{X}_f$ play a significant role in establishing properties of the closed-loop control system.
  - The MPC control law is then implicitly defined through the optimization problem as $\pi^{\text{MPC}}(x(k),k)=u^\star_{0|k}$, where $u^\star_{0|k}$ is the first element of the computed optimal control sequence $U^\star$.

## Learning-Based Model Predictive Control

- Learning-based model predictive control addresses the automated and data-driven generation or adaptation of elements of the MPC formulation such that the control performance with respect to the desired closed-loop system behavior is improved.
- The categories of learning-based MPC:
  - Learning the system dynamics: MPC relies heavily on suitable and sufficiently accurate model representations of the system dynamics.

  - Learning the controller design: focuses more on the employed cost function $\mathcal{l}$，the constraints $\mathcal{X}$, or the terminal components $\mathcal{l}_f$ and $\mathcal{X}_f$, such that the resulting closed-loop MPC controller behaves favorably with respect to the underlying task.

  - MPC for safe learning: derive safety guarantees for learning-based controllers, to decouple the optimization of the objective function $\mathcal{l}_t$ from the requirement  of constraint satisfaction.

## Leaning the system dynamics

System modeling represents the first step of an MPC design, which is traditionally addressed by deriving a parametric prediction model offline using physical principles and applying system identification techniques.

- Model and uncertainty descriptions are typically fixed offline and assumed available before controller design.

  Many learning-based MPC techniques make use of an explicit distinction between a nominal system model $f_n$ and an additive learned term $f_1$ accommodating uncertainty: $f(x,u,k,\theta,w)=f_n(x,u,k)+f_1(x,u,k,\theta,w)$.

- Most methods do not distinguish between the true system dynamics $f_t$ and prediction dynamics $f$--i.e., a typical assumption is that the true dynamics lie within the class of considered prediction dynamics, and given knowledge of the true parameter realization $\theta_t$, there is no model mismatch.

## Robust Models

Robust MPC schemes guarantee the satisfaction of closed-loop constraints for all possible realizations of the uncertain element $\theta_t$ and $w(k)$.

### Robust parametric models

Given a measured state and input trajectory $X=[x(0),...,x(k)]$, $U=[u(0),...,u(k-1)]$, parametric set-membership estimation aims at finding the set of possible parameter values $\theta$ for which the observed trajectories are consistent. This is formulated as a feasible parameter set:

$$
\Tau_{k}=\{\theta|\forall j=0,...,k\ \exists\mathcal{w}\in\mathcal{W}, such\ that\ x(j+1)=f(x(j),u(j),j,\theta,w)\}
$$

$$
J^\star_t=\mathop{\text{minimize}}\limits_{\{\pi_k\}} \quad\mathbb{E}\left(\sum^\bar{N}_{k=0}{l_t(x(k),u(k),k)}\right)
$$

