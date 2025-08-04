import numpy as np
import time


def OOA(initsol, obj_fun, xmin, xmax, max_iter):
    lambda_const = 0.5
    start_time = time.time()

    Npop, dimensions = initsol.shape
    M = initsol.copy()  # Male population
    F = initsol.copy()  # Female population

    # Evaluate initial fitness for males and females
    OM = np.array([obj_fun(ind) for ind in M])
    OF = np.array([obj_fun(ind) for ind in F])

    best_fit_history = []  # Store best fitness over iterations

    for iter_count in range(max_iter):
        # Find the best male and female
        m_best = M[np.argmin(OM)]
        f_best = F[np.argmin(OF)]

        # Transition males
        for j in range(Npop):
            # Compute the inverse probability of transition
            p_mj = OM[j] / np.sum(OM)
            r = np.random.rand()

            if p_mj < r:
                # Guided update
                M[j] += (m_best - M[j]) * lambda_const
            else:
                # Random exploration
                M[j] = xmin[j] + (xmax[j] - xmin[j]) * np.random.rand(dimensions)

            # Bound constraints
            M[j] = np.clip(M[j], xmin[j], xmax[j])

            # Update fitness
            OM[j] = obj_fun(M[j])

        # Transition females
        for j in range(Npop):
            # Compute the inverse probability of transition
            p_fj = OF[j] / np.sum(OF)
            r = np.random.rand()

            if p_fj < r:
                # Guided update
                F[j] += (f_best - F[j]) * lambda_const
            else:
                # Random exploration
                F[j] = xmin[j] + (xmax[j] - xmin[j]) * np.random.rand(dimensions)

            # Bound constraints
            F[j] = np.clip(F[j], xmin[j], xmax[j])

            # Update fitness
            OF[j] = obj_fun(F[j])

        # Track the best solution and fitness
        current_best_fit = min(np.min(OM), np.min(OF))
        best_fit_history.append(current_best_fit)

    # Final results
    bestfit = np.min(best_fit_history)
    bestsol = M[np.argmin(OM)] if np.min(OM) < np.min(OF) else F[np.argmin(OF)]
    elapsed_time = time.time() - start_time

    return bestfit, best_fit_history, bestsol, elapsed_time
