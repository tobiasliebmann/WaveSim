# WaveSim
A simulation of the wave equation in Python. In the beginning everything 
will be 1D. The Wave equation in 1D reads:
<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\frac{1}{c^{2}}\frac{\partial^{2}u(x,t)}{\partial&space;t^{2}}&space;=&space;\frac{\partial^{2}u(x,t)}{\partial&space;x^{2}}." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\frac{1}{c^{2}}\frac{\partial^{2}u(x,t)}{\partial&space;t^{2}}&space;=&space;\frac{\partial^{2}u(x,t)}{\partial&space;x^{2}}." title="\large \frac{1}{c^{2}}\frac{\partial^{2}u(x,t)}{\partial t^{2}} = \frac{\partial^{2}u(x,t)}{\partial x^{2}}." /></a>
<br>
In the next step, this partial differential equation is discretized by using a
finite difference method
<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;f''(x)\approx\frac{f(x&plus;\Delta&space;x)&space;&plus;&space;f(x-\Delta&space;x)&space;-&space;2f(x)}{\Delta&space;x^{2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;f''(x)\approx\frac{f(x&plus;\Delta&space;x)&space;&plus;&space;f(x-\Delta&space;x)&space;-&space;2f(x)}{\Delta&space;x^{2}}" title="\large f''(x)\approx\frac{f(x+\Delta x) + f(x-\Delta x) - 2f(x)}{\Delta x^{2}}." /></a>
<br>
The above equation is a central finite difference. Using this expression 
simplifies the wave equation to
<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;u(x,t&plus;\Delta&space;t)&space;&plus;&space;u(x,t&space;-&space;\Delta&space;t)&space;-&space;2&space;u(x,t)&space;=&space;\alpha&space;(u(x&plus;\Delta&space;x,t)&space;&plus;&space;u(x-\Delta&space;x)&space;-&space;2u(x,t)),&space;\,&space;\alpha&space;=&space;\frac{c^{2}\Delta&space;t^{2}}{\Delta&space;x^{2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;u(x,t&plus;\Delta&space;t)&space;&plus;&space;u(x,t&space;-&space;\Delta&space;t)&space;-&space;2&space;u(x,t)&space;=&space;\alpha&space;(u(x&plus;\Delta&space;x,t)&space;&plus;&space;u(x-\Delta&space;x)&space;-&space;2u(x,t)),&space;\,&space;\alpha&space;=&space;\frac{c^{2}\Delta&space;t^{2}}{\Delta&space;x^{2}}" title="\large u(x,t+\Delta t) + u(x,t - \Delta t) - 2 u(x,t) = \alpha (u(x+\Delta x,t) + u(x-\Delta x) - 2u(x,t)), \, \alpha = \frac{c^{2}\Delta t^{2}}{\Delta x^{2}}" /></a>
<br>
Now this does not look particularly simple. But setting fully discretizing 
space and time results in the following equation for updating the equation
from one time step to another
<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;u_{i,j&plus;1}=\alpha&space;(u_{i&plus;1,j}&space;&plus;&space;u_{i-1,j})&space;&plus;&space;2(1-\alpha)u_{i,j}&space;-&space;u_{i,j-1},&space;\,&space;u_{i,j}=u(i\Delta&space;x,j\Delta&space;t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;u_{i,j&plus;1}=\alpha&space;(u_{i&plus;1,j}&space;&plus;&space;u_{i-1,j})&space;&plus;&space;2(1-\alpha)u_{i,j}&space;-&space;u_{i,j-1},&space;\,&space;u_{i,j}=u(i\Delta&space;x,j\Delta&space;t)" title="\large u_{i,j+1}=\alpha (u_{i+1,j} + u_{i-1,j}) + 2(1-\alpha)u_{i,j} - u_{i,j-1}, \, u_{i,j}=u(i\Delta x,j\Delta t)." /></a>
<br>


