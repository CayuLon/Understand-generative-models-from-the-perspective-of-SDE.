## Flow matching
### Rectified flow
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

