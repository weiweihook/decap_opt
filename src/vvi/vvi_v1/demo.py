#-- annealing method demo

import numpy as np
from scipy.optimize import dual_annealing

func = lambda x: np.sum(x*x - 10*np.cos(2*np.pi*x)) + 10*np.size(x)

lw = [-5.12] * 10

up = [5.12] * 10

sol = dual_annealing(func, bounds=list(zip(lw, up)))

print(sol)
