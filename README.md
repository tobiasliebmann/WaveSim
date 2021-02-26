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
<img src=/images/second_initial_condition.jpg>
</p>
This equation can now be rearranged to get an expression for the amplitudes
at the -1 time step
<br>
<br>
<p align="center">
<img src=/images/minus_first_time_step.jpg>
</p>
Using the derived expressions results in the following expression for the
amplitude of the first time step
<br>
<br>
<p align="center">
<img src=/images/first_time_step_amplitude.jpg>
</p>
Using the matrix T results in a rather handy expression for the amplitudes
of the first time step
<br>
<br>
<p align="center">
<img src=/images/first_time_step_matrix_equation.jpg>
</p>
This equation reflects the impact of the initial conditions on the
solution of the wave equation.

## Stability analysis

In this section the stability of the discrete scheme will be examined
using the Von-Neumann-stability formalism. This theory examines the 
stability of a discrete scheme by looking at the propagation of errors. 
The total error of the i-th amplitude at the j-th
time step of the scheme is therefore decomposed into a Fourier series
, where A is the so called amplification factor. The amplification factor 
is assumed to behave according to a power law in time
<br>
<br>
<p align="center">
<img src=/images/error_fourier.jpg>
</p>
A scheme is unstable if the modulus of the amplification factor is greater
than 1, |A| > 1. Further, the scheme diverges if the amplification factor of
one Fourier component is greater than 1. The errors propagate in the scheme
via the update equation. Plugging one Fourier component into the update
equation from the j-th to the (j+1)-th time step results in
<br>
<br>
<p align="center">
<img src=/images/amplification_factor_equation.jpg>
</p>
The possible solutions for A read
<br>
<br>
<p align="center">
<img src=/images/beta_equation.jpg>
</p>
This ultimately leads to the following Courant number
<br>
<br>
<p align="center">
<img src=/images/courant_number_equation.jpg>
</p>
As long as this condition is fulfilled the scheme is guaranteed to be stable.

## Results

A simulation using the values
<br>
<br>
<p align="center">
<img src=/images/grid_parameters.jpg>
</p>
and the initial conditions
<br>
<br>
<p align="center">
<img src=/images/wave_sim_initial_conditions.jpg>
</p>
resulted in the wave shown below
