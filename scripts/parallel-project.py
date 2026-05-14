#!.venv/bin/python
#!.venv/bin/python

import itertools
import pandas as pd
import numpy as np

grid_size  = [5, 10, 20, 100] # NxN grids size
threads    = [1, 2, 5, 10, 20, 40] # 1..20 to test only with physical cores, 40 to test with logical threads
                                   # Remember to set correct OpenMP flags to alloc to right cores.
episodes = [100, 1000] # needed?
obstacles = [1, 5, 20] # needed?
#There is more factors? There is need to change alpha, gamma and things like that?

replications = 10
seed = 67

combinations = list(itertools.product(grid_size, threads, episodes, obstacles))
rng = np.random.default_rng(seed)

rows = []
for block in range(1, replications + 1):
    for i in rng.permutation(len(combinations)):
        grid, threads, eps, obstacles = combinations[i]
        rows.append({"grid_size": int(grid), "threads": int(threads),
                     "episodes": int(eps), "obstacles": int(obstacles),
                     "Blocks": block})

df = pd.DataFrame(rows, columns=["grid_size", "threads", "episodes", "obstacles", "Blocks"])
df.to_csv("parallel_project.csv", index=False)
print(df.head(100).to_string())
