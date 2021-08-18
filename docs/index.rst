.. WaveSim documentation master file, created by
   sphinx-quickstart on Wed Aug 18 17:25:56 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

WaveSim documentation
===================================
The Python code in this repository solves a 1D wave equation on a grid. Additionally, a 2D wave equation
simulator was added. The topics which will be discussed are:

- Discretization of the wave equation in 1D and 2D
- Choosing and implementation of a boundary condition
- Implementation of initial conditions
- Stability analysis
- Results
- Benchmark comparing 1D and 2D simulation
- Sources

1D wave simulator
-----------------
.. autoclass:: simulator.Numeric1DWaveSimulator
    :members:

2D wave simulator
-----------------
.. autoclass:: simulator.Numeric2DWaveSimulator
    :members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
