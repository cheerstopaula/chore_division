import numpy as np
from src.chore_allocation import (
    pEF1_fPO_three_agent_chore_allocation,
    pEF1_fPO_ILP_chore_allocation,
    fPO,
    EF1,
    EF_violations,
)


import gurobipy as gp
from gurobipy import GRB

# exit()
np.random.seed(0)

m, n = 6, 3
D = np.random.randint(1, 6, size=(m, n)).astype(float)
D[1, 1], D[5, 1] = 3, 1


m, n = 50, 6
D = np.random.randint(1, 6, size=(m, n)).astype(float)

# X = pEF1_fPO_three_agent_chore_allocation(m, n, D)

# print(X)
# print("Allocation is fPO:",fPO(X,D))
# print("Allocation is EF1:", EF1(X, D))
# print("Envy violation:", EF_violations(X, D))


X = pEF1_fPO_ILP_chore_allocation(m, n, D)

print(X)
print("Allocation is fPO:",fPO(X,D))
print("Allocation is EF1:", EF1(X, D))
print("Envy violation:", EF_violations(X, D))

