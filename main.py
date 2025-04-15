import numpy as np
from src.chore_allocation import (
    pEF1_fPO_three_agent_chore_allocation,
    pEF1_fPO_ILP_chore_allocation,
)


np.random.seed(0)

# m, N = 6, 3
# D = np.random.randint(1, 6, size=(m, N)).astype(float)
# D[1,1], D[5,1]= 3,1


m, n = 10, 3
D = np.random.randint(1, 6, size=(m, n)).astype(float)

X = pEF1_fPO_three_agent_chore_allocation(m, n, D)

print(X)

X = pEF1_fPO_ILP_chore_allocation(m, n, D)

print(X)
