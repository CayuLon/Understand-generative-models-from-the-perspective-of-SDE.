## Flow matching

### 1. Six objects

**1）Conditional probability path**: $p_t(·|z)=\mathcal{N}(\alpha_t z, \beta_t^2\mathbf{I})$,  $p_0(·|z)=p_{init}$,  $p_1(·|z)=p_{data}$. ( $t\in[0, 1]$ and $z\sim p_{data}$, $p_{init} \sim \mathcal{N}(0,\mathbf{I})$, $\alpha_0=0, \beta_0=1,\alpha_1=1,\beta_1=0$)

**2）Marginal probability path**: $x\sim p_t$ ( forget z, and can be seen as the intermediate points or in the latent space ), and $p_t(x)=\int p_t(x|z)p_{data}(z) \,dz$, then $p_0(x)=p_{init}$, $p_1(x)=p_{data}$.

<img src="images/9f284b4e-4558-4062-9370-22b27ed290ad-17562097610292.png" alt="9f284b4e-4558-4062-9370-22b27ed290ad" style="zoom:50%;" />

**3）Conditional vector field**: 
```math
\underbrace{x_0 \sim p_{init}}_{init}, \color{red}{\underbrace{\frac{\,dx_t}{\,dt}=u_t^{tar}(x_t|z)}_{direction}} \color{black}{,then \Rightarrow \underbrace{x_t \sim p_t(·|z)}_{progress}}.
```
And we formulate as: 
<div align="center">

$$
u_t^{tar}(x_t|z)=(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t})z+\frac{\dot{\beta}_t}{\beta_t}x_t,
$$

</div>
which is the Interpolation of $z$ and $x_t$ .

**4) Marginal vector field**: $\color{red} {u_t^{tar}(x_t)=\int u_t^{tar}(x_t|z)p_t(z|x_t) \,dz =\int u_t^{tar}(x_t|z) \frac{p_t(x_t|z)p_{data}(z)}{p_t(x_t)} \,dz}$. So, we can derive: $\color{red}{\frac{\,dx_t}{\,dt}=u_t^{tar}(x_t)} \Rightarrow x_t \sim p_t(x_t)$. i.e. Condition VF is equivalent to marginal VF iff the definition of $u_t^{tar}(x_t)$ as shown above.

​						**Proof**:
​											 $\begin{split} \frac{\,d p_t(x_t)}{dt} &= \frac{d (\int p_t(x_t|z)p_{data}(z)dz)}{dt} \\&=\int(\frac{dp_t(x_t|z)}{dt})p_{data}(z)dz \\&= -\mathbf{div}\int p_t(x_t|z) u_t^{tar}(x_t|z)p_{data}(z)dz \\&= -\mathbf{div} \int u_t^{tar}(x_t|z) \frac{p_t(x_t|z) p_{data}(z)}{p_t(x_t)} dz \cdot p_t(x_t)\\&= -\mathbf{div} \; u_t^{tar}(x_t)\cdot p_t(x_t) \end{split}$

 						**Q.E.D**

**5) Conditional score function.** which terms as log-likelihood function. $\nabla_xlogp_t(x|z) = -\frac{x-\alpha_tz}{\beta_t^2}$ (Calculate according to the definition).

**6) Marginal score function.** 

​												$\begin{split} \nabla_xlogp_t(x) &= \frac{\nabla p_t(x)}{p_t(x)} \\&= \frac{\int \nabla p_t(x|z)p_{data}(z)\,dz}{p_t(x)} \\&= \int \nabla logp_t(x|z)\frac{p_t(x|z)p_{data}(z)}{p_t(x)}dz \end{split}$.

All in all, the six objects are summarized as follow:

<img src="images/wechat_2025-08-26_173514_920.png" alt="wechat_2025-08-26_173514_920" style="zoom: 25%;" />

<img src="images/wechat_2025-08-26_173658_261.png" alt="wechat_2025-08-26_173658_261" style="zoom: 25%;" />



### 2. Rectified flow
1.  x0, x1    -->  flow matching 建模noise到original data的流： $x_t = t x_1 + (1-t)x_0$ , $x_0$ 是noise；

   DDPM是从original data到noise的加噪：($x_t = \sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$), $x_0$ 是original data。

2. training: $\hat{V} = model(x_t, t)$,  $V = \frac{d}{dt}x_t = x_1 - x_0$, 每次迭代都要对每个 $x_1$ 随机取 $t\in[0,1]$ 和 $x_0$ 以得到 $x_t$

3. using ODE solver sampling:    $x_{t+h} = x_t + h * model(x_t, t)$,  $h$ 足够小



#### Euler method:

```python
x_0 = torch.randn(), x_t = x_0, dt = 1 / total_steps

for i in range(total_steps):

	V = model(x_t, i · dt)
	
	x_t = x_t + dt · V

return x_t
```



​	

​	

