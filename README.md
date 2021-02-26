# Wave equation simulator

The Python code in this repository solves a 1D wave equation on a grid. 
In the future it will be updated for 2D wave equations. The topics which
will be discussed are:

- Discritization of the wave equation
- Choosing and implementation of a boundary condition
- Implementation of initial conditions
- Stability analysis
- Results
- Benchmark


## Discretizing the wave equation

The Wave equation in 1D reads
<br>
<br>
<p align="center">
<img src=/images/wave_equation.jpg>
</p>
In the next step, this partial differential equation is discretized by using a
finite difference method
<br>
<br>
<p align="center">
<img src=/images/second_derivative_approx.jpg>
</p>
The above equation is a central finite difference. Using this expression 
simplifies the wave equation to
<br>
<br>
<p align="center">
<img src=/images/discrete_wave_equation.jpg>
</p>
Now this does not look particularly simple. But setting fully discretizing 
space and time results in the following equation for updating the equation
from one time step to another
<br>
<br>
<p align="center">
<img src=/images/time_step_equation.jpg>
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
<img src=/images/time_step_vector.jpg>
</p>
This transforms the update equation to a linear equation involving an
update matrix T, the j-th state and the (j-1)-th state
<br>
<br>
<p align="center">
<img src=/images/time_step_linear_equation.jpg>
</p>
The frame of zeros in the matrix T correspond to the boundary condition of
fixed end points which was already quitely implemented without mentioning.
Further only the diagonal and the off-diagonal elements of T are populated. 
The discussion regarding the boundary condition will follow now.

## Boundary condition

As stated in the last chapter the chosen boundary condition are
fixed endpoints for all time steps. For a 1D grid consisting of N 
amplitudes and T time steps this corresponds to
<br>
<br>
<p align="center">
<img src=/images/boundary_condition.jpg>
</p>
The frame of zeros in the matrix T makes sure that the boundary condition
is always fulfilled.

## Initial conditions

The wave equation is second order partial derivative because of this two
initial conditions
<br>
<br>
<p align="center">
<img src=/images/initial_conditions.jpg>
</p>
These equations have to be discretized now. The first condition is 
fulfilled by setting
<br>
<br>
<p align="center">
<img src=/images/first_initial_condition.jpg>
</p>
This result can now be inserted in the basic update equation to calculate
the first time step. However, doing this causes a problem since the it
involves the amplitudes at time step -1 which are not defined at this
point. But, the amplitudes at this time step can be calculated using the
second initial condition. To do this the derivative is discritized via a
central finite difference
<br>
<br>
<p align="center">
<img src=/images/first_initial_condition.jpg>
</p>
