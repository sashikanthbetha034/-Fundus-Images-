import numpy as np
import time


def LEA(initsol, fname, xmin, xmax, Max_iter):
    start_time = time.time()

    # Initialize population
    Npop, dim = initsol.shape  # Population size and problem dimension
    X = initsol.copy()  # Current population positions
    velocity = np.zeros_like(X)  # Initialize velocity to zero

    # Evaluate fitness of initial population
    fitness = np.array([fname(ind) for ind in X])
    best_idx = np.argmin(fitness)
    g_best = X[best_idx]  # Global best solution
    bestfit = fitness[best_idx]

    # Initialize parameters
    a, c, f, s, w, e = 0.1, 0.1, 0.1, 0.1, 0.1, 0.1

    for t in range(Max_iter):
        for i in range(Npop):
            P = np.random.rand()
            if P <= 0.5:  # Global pollination
                velocity[i] = velocity[i] + a + c + f + s + w  # Update velocity
                X[i] = X[i] + velocity[i]  # Update position
            else:
                if np.linalg.norm(X[i] - g_best) <= 1.0:
                    velocity[i] = velocity[i] + a + c + f + s + w
                    X[i] = X[i] + velocity[i]
                else:
                    X[i] = X[i] + np.random.uniform(-0.1, 0.1, size=X[i].shape)  # Local adjustment

            X[i] = np.clip(X[i], xmin[i], xmax[i])

        fitness = np.array([fname(ind) for ind in X])
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < bestfit:
            g_best = X[current_best_idx]
            bestfit = fitness[current_best_idx]

        a, c, f, s, w, e = a + 0.01, c + 0.01, f + 0.01, s + 0.01, w + 0.01, e + 0.01

    bestsol = g_best
    time_taken = time.time() - start_time

    return bestfit, fitness, bestsol, time_taken
