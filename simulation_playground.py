import simulator1D as sim1D

import Visualization as vis

import numpy as np


if __name__ == "__main__":
    # Initial amplitudes.
    a0 = np.array([0, -1, 1, -1, 1, -1, 1, -1, 1, 0])
    # Initial velocities of the amplitudes.
    v0 = np.array([0, 0, 1, 1, -1, 0, 0, 0, 0, 0])
    # Grid spacing.
    dx: float = 0.1
    # Spacing of the time steps.
    dt = 0.005
    # speed of sound.
    c = 10
    # Number of grid points.
    n = 10
    # Number of time steps.
    t = 10
    my_sim = sim1D.Numeric1DWaveSimulator(dx, dt, c, n, t, a0, v0)
    result = my_sim.run()
    print(result)
    visualizer = vis.Visualizer(result, n, dx)
