import numpy as np 
from src.chore_allocation import pEF1_fPO_three_agent_chore_allocation



np.random.seed(0)

m, N = 6, 3
D = np.random.randint(1, 6, size=(m, N)).astype(float)
D[1,1], D[5,1]= 3,1

X=pEF1_fPO_three_agent_chore_allocation(m,N,D)

print(X)