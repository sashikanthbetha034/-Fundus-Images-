import numpy as np
from scipy.special import gamma
import time


def SOA(initsol, fobj, xmin, xmax, Max_iter):

    start_time = time.time()

    # Parameters
    Npop = initsol.shape[0]
    nD = initsol.shape[1]
    pos = initsol.copy()  # Initial positions

    fit = np.full(Npop, np.inf)
    newfit = np.full(Npop, np.inf)
    Curve = np.full(Max_iter, np.inf)

    # Evaluate fitness of initial solutions
    for i in range(Npop):
        fit[i] = fobj(pos[i, :])

    # Best solution initialization
    fvalbest = np.min(fit)
    xposbest = pos[np.argmin(fit), :]

    # Main loop
    for T in range(1, Max_iter + 1):
        newpos = pos.copy()
        WF = 0.1 - 0.05 * (T / Max_iter)  # Whale fall probability
        kk = (1 - 0.5 * T / Max_iter) * np.random.rand(Npop)  # Exploration or exploitation probability

        for i in range(Npop):
            if kk[i] > 0.5:  # Exploration phase
                r1, r2 = np.random.rand(), np.random.rand()
                RJ = np.random.randint(Npop)
                while RJ == i:
                    RJ = np.random.randint(Npop)

                if nD <= Npop / 5:
                    params = np.random.permutation(nD)[:2]
                    newpos[i, params[0]] = pos[i, params[0]] + (pos[RJ, params[0]] - pos[i, params[1]]) * (r1 + 1) * np.sin(r2 * 360)
                    newpos[i, params[1]] = pos[i, params[1]] + (pos[RJ, params[0]] - pos[i, params[1]]) * (r1 + 1) * np.cos(r2 * 360)
                else:
                    params = np.random.permutation(nD)
                    for j in range(nD // 2):
                        newpos[i, 2 * j] = pos[i, params[2 * j]] + (pos[RJ, params[0]] - pos[i, params[2 * j]]) * (r1 + 1) * np.sin(
                            r2 * 360)
                        newpos[i, 2 * j + 1] = pos[i, params[2 * j + 1]] + (pos[RJ, params[0]] - pos[i, params[2 * j + 1]]) * (
                                r1 + 1) * np.cos(r2 * 360)
            else:  # Exploitation phase
                r3, r4 = np.random.rand(), np.random.rand()
                C1 = 2 * r4 * (1 - T / Max_iter)
                RJ = np.random.randint(Npop)
                while RJ == i:
                    RJ = np.random.randint(Npop)

                alpha = 3 / 2
                sigma = (gamma(1 + alpha) * np.sin(np.pi * alpha / 2) / (gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))) ** (
                        1 / alpha)
                u = np.random.randn(nD) * sigma
                v = np.random.randn(nD)
                S = u / np.abs(v) ** (1 / alpha)
                KD = 0.05
                LevyFlight = KD * S
                newpos[i, :] = r3 * xposbest - r4 * pos[i, :] + C1 * LevyFlight * (pos[RJ, :] - pos[i, :])

            # Boundary handling
            Flag4ub = newpos[i, :] > xmax[0]
            Flag4lb = newpos[i, :] < xmin[0]
            newpos[i, :] = (newpos[i, :] * ~(Flag4ub | Flag4lb)) + xmax[0] * Flag4ub + xmin[0] * Flag4lb
            newfit[i] = fobj(newpos[i, :])

            if newfit[i] < fit[i]:
                pos[i, :] = newpos[i, :]
                fit[i] = newfit[i]

        for i in range(Npop):
            # Whale fall
            if kk[i] <= WF:
                RJ = np.random.randint(Npop)
                r5, r6, r7 = np.random.rand(), np.random.rand(), np.random.rand()
                C2 = 2 * Npop * WF
                stepsize2 = r7 * (xmax[0] - xmin[0]) * np.exp(-C2 * T / Max_iter)
                newpos[i, :] = (r5 * pos[i, :] - r6 * pos[RJ, :]) + stepsize2

                # Boundary handling
                Flag4ub = newpos[i, :] > xmax[0]
                Flag4lb = newpos[i, :] < xmin[0]
                newpos[i, :] = (newpos[i, :] * ~(Flag4ub | Flag4lb)) + xmax[0] * Flag4ub + xmin[0] * Flag4lb
                newfit[i] = fobj(newpos[i, :])

                if newfit[i] < fit[i]:
                    pos[i, :] = newpos[i, :]
                    fit[i] = newfit[i]

        # Update global best
        fval = np.min(fit)
        index = np.argmin(fit)
        if fval < fvalbest:
            fvalbest = fval
            xposbest = pos[index, :]

        Curve[T - 1] = fvalbest

    end_time = time.time()
    execution_time = end_time - start_time

    return fvalbest, Curve, xposbest, execution_time