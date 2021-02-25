# WaveSim
A simulation of the wave equation in Python. In the beginning everything 
will be 1D. The Wave equation in 1D reads:
<br>
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\frac{1}{c^{2}}\frac{\partial^{2}u(x,t)}{\partial&space;t^{2}}&space;=&space;\frac{\partial^{2}u(x,t)}{\partial&space;x^{2}}." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\frac{1}{c^{2}}\frac{\partial^{2}u(x,t)}{\partial&space;t^{2}}&space;=&space;\frac{\partial^{2}u(x,t)}{\partial&space;x^{2}}." title="\large \frac{1}{c^{2}}\frac{\partial^{2}u(x,t)}{\partial t^{2}} = \frac{\partial^{2}u(x,t)}{\partial x^{2}}." /></a>
</p>
In the next step, this partial differential equation is discretized by using a
finite difference method
<br>
<br>
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;f''(x)\approx\frac{f(x&plus;\Delta&space;x)&space;&plus;&space;f(x-\Delta&space;x)&space;-&space;2f(x)}{\Delta&space;x^{2}}." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;f''(x)\approx\frac{f(x&plus;\Delta&space;x)&space;&plus;&space;f(x-\Delta&space;x)&space;-&space;2f(x)}{\Delta&space;x^{2}}." title="\large f''(x)\approx\frac{f(x+\Delta x) + f(x-\Delta x) - 2f(x)}{\Delta x^{2}}." /></a>
</p>
The above equation is a central finite difference. Using this expression 
simplifies the wave equation to
<br>
<br>
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;u(x,t&plus;\Delta&space;t)&space;&plus;&space;u(x,t&space;-&space;\Delta&space;t)&space;-&space;2&space;u(x,t)&space;=&space;\alpha&space;(u(x&plus;\Delta&space;x,t)&space;&plus;&space;u(x-\Delta&space;x)&space;-&space;2u(x,t)),&space;\,&space;\alpha&space;=&space;\frac{c^{2}\Delta&space;t^{2}}{\Delta&space;x^{2}}." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;u(x,t&plus;\Delta&space;t)&space;&plus;&space;u(x,t&space;-&space;\Delta&space;t)&space;-&space;2&space;u(x,t)&space;=&space;\alpha&space;(u(x&plus;\Delta&space;x,t)&space;&plus;&space;u(x-\Delta&space;x)&space;-&space;2u(x,t)),&space;\,&space;\alpha&space;=&space;\frac{c^{2}\Delta&space;t^{2}}{\Delta&space;x^{2}}." title="\large u(x,t+\Delta t) + u(x,t - \Delta t) - 2 u(x,t) = \alpha (u(x+\Delta x,t) + u(x-\Delta x) - 2u(x,t)), \, \alpha = \frac{c^{2}\Delta t^{2}}{\Delta x^{2}}." /></a>
</p>
Now this does not look particularly simple. But setting fully discretizing 
space and time results in the following equation for updating the equation
from one time step to another
<br>
<br>
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;u_{i,j&plus;1}=\alpha&space;(u_{i&plus;1,j}&space;&plus;&space;u_{i-1,j})&space;&plus;&space;2(1-\alpha)u_{i,j}&space;-&space;u_{i,j-1},&space;\,&space;u_{i,j}=u(i\Delta&space;x,j\Delta&space;t)." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;u_{i,j&plus;1}=\alpha&space;(u_{i&plus;1,j}&space;&plus;&space;u_{i-1,j})&space;&plus;&space;2(1-\alpha)u_{i,j}&space;-&space;u_{i,j-1},&space;\,&space;u_{i,j}=u(i\Delta&space;x,j\Delta&space;t)." title="\large u_{i,j+1}=\alpha (u_{i+1,j} + u_{i-1,j}) + 2(1-\alpha)u_{i,j} - u_{i,j-1}, \, u_{i,j}=u(i\Delta x,j\Delta t)." /></a>
</p>
The above equation is visualized in the image below, where the amplitude
of the i-th point at the (j+1)-th time step is given the point at the time
step and its neighbors. Further it relies on the state of the amplitudes at
the (j-1)-ths time step.
<p align="center"> 
<img src=/images/time_step_visualization.png>
</p>
The state of N amplitudes at time step j can be imagined as a vector
<br>
<br>
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\vec{u}_{j}=\left(u_{0,j},\ldots,u_{N-1,j}\right)." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\vec{u}_{j}=\left(u_{0,j},\ldots,u_{N-1,j}\right)." title="\large \vec{u}_{j}=\left(u_{0,j},\ldots,u_{N-1,j}\right)." /></a>
</p>
This transform the update equation to a linear equation involving an
update matrix, the j-th state and the (j-1)-th state
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\vec{u}_{j&plus;1}=T\vec{u}_{j}-u_{j-1},&space;\,&space;T=\begin{pmatrix}&space;0&space;&&space;0&space;&&space;\ldots&space;&&space;0&space;&&space;0\\&space;0&space;&&space;2\left(1-\alpha\right)&space;&&space;\alpha&space;&&space;0&space;&&space;0\\&space;\vdots&space;&&space;\alpha&space;&&space;\ddots&space;&&space;\alpha&space;&&space;\vdots\\&space;0&space;&&space;0&space;&&space;\alpha&space;&&space;2\left(1-\alpha\right)&space;&&space;0\\&space;0&space;&&space;0&space;&&space;\ldots&space;&&space;0&space;&&space;0&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\vec{u}_{j&plus;1}=T\vec{u}_{j}-u_{j-1},&space;\,&space;T=\begin{pmatrix}&space;0&space;&&space;0&space;&&space;\ldots&space;&&space;0&space;&&space;0\\&space;0&space;&&space;2\left(1-\alpha\right)&space;&&space;\alpha&space;&&space;0&space;&&space;0\\&space;\vdots&space;&&space;\alpha&space;&&space;\ddots&space;&&space;\alpha&space;&&space;\vdots\\&space;0&space;&&space;0&space;&&space;\alpha&space;&&space;2\left(1-\alpha\right)&space;&&space;0\\&space;0&space;&&space;0&space;&&space;\ldots&space;&&space;0&space;&&space;0&space;\end{pmatrix}" title="\vec{u}_{j+1}=T\vec{u}_{j}-u_{j-1}, \, T=\begin{pmatrix} 0 & 0 & \ldots & 0 & 0\\ 0 & 2\left(1-\alpha\right) & \alpha & 0 & 0\\ \vdots & \alpha & \ddots & \alpha & \vdots\\ 0 & 0 & \alpha & 2\left(1-\alpha\right) & 0\\ 0 & 0 & \ldots & 0 & 0 \end{pmatrix}" /></a>
</p>
