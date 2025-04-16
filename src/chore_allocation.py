import numpy as np
from gurobipy import Model, GRB, quicksum


def initialize_allocation(m, N, D_og):
    X = np.zeros((m, N), dtype=int)
    agent_i = np.random.randint(0, N)
    X[:, agent_i] = 1
    D = D_og.copy()
    p = D[:, agent_i % m]
    return X, p


def compute_alphas(D, p):
    alpha_matrix = D / p[:, np.newaxis]
    alpha = np.min(alpha_matrix, axis=0)
    return alpha_matrix, alpha


def compute_p_x(X, p):
    px_matrix = X * p[:, np.newaxis]
    p_x = np.sum(px_matrix, axis=0)
    p_1x = np.sum(px_matrix, axis=0) - np.max(px_matrix, axis=0)
    return p_x, p_1x


def determine_earners(p_x, p_1x):
    b = np.argmax(p_1x)
    l = np.argmin(p_x)
    h = 3 - l - b
    return b, h, l


def pEF1(p_x, p_1x, b, l):
    if b == l:
        return True
    if p_1x[b] <= p_x[l]:
        return True
    return False


def find_MPB_sets(alpha_matrix, alpha):
    return [
        list(np.where(alpha_matrix[:, j] == alpha[j])[0])
        for j in range(alpha_matrix.shape[1])
    ]


def find_transferable_chore(X, MPBs, i, k):
    valid_indices = [j for j in MPBs[k] if X[j, i] == 1]
    return valid_indices[0] if valid_indices else None


def transfer_chore(X, p, chore, i, k):
    X[chore, i] = 0
    X[chore, k] = 1
    p_x, p_1x = compute_p_x(X, p)
    return X, p_x, p_1x


def update_prices_1(X, p, D, m, alpha, alpha_matrix, b, h, l):

    beta = 0
    X_b = [j for j in range(m) if X[j, b] == 1]
    X_h = [j for j in range(m) if X[j, h] == 1]
    X_l = [j for j in range(m) if X[j, l] == 1]

    for j in X_b:
        candidate1 = alpha[l] / (D[j, l] / p[j])
        candidate2 = alpha[h] / (D[j, h] / p[j])
        if max(candidate1, candidate2) > beta:
            beta = max(candidate1, candidate2)

    bundle = list(set([*X_l, *X_h]))

    for j in bundle:
        p[j] = beta * p[j]
    p_x, p_1x = compute_p_x(X, p)
    alpha_matrix, alpha = compute_alphas(D, p)

    MPBs = find_MPB_sets(alpha_matrix, alpha)

    return p, p_x, p_1x, alpha_matrix, alpha, MPBs


def update_prices_2(X, p, D, m, alpha, alpha_matrix, b, h, l):

    beta = 0
    X_b = [j for j in range(m) if X[j, b] == 1]
    X_h = [j for j in range(m) if X[j, h] == 1]
    X_l = [j for j in range(m) if X[j, l] == 1]

    bundle = list(set([*X_b, *X_h]))

    for j in bundle:
        candidate = alpha[l] / (D[j, l] / p[j])
        if candidate > beta:
            beta = candidate

    for j in X_l:
        p[j] = beta * p[j]
    p_x, p_1x = compute_p_x(X, p)
    alpha_matrix, alpha = compute_alphas(D, p)

    MPBs = find_MPB_sets(alpha_matrix, alpha)

    return p, p_x, p_1x, alpha_matrix, alpha, MPBs


def pEF1_fPO_three_agent_chore_allocation(m, N, D):
    X, p = initialize_allocation(m, N, D)
    alpha_matrix, alpha = compute_alphas(D, p)
    p_x, p_1x = compute_p_x(X, p)
    b, h, l = determine_earners(p_x, p_1x)
    MPBs = find_MPB_sets(alpha_matrix, alpha)

    while not pEF1(p_x, p_1x, b, l):

        chore = find_transferable_chore(X, MPBs, b, l)

        if chore is not None:
            X, p_x, p_1x = transfer_chore(X, p, chore, b, l)
            b, h, l = determine_earners(p_x, p_1x)

        elif find_transferable_chore(X, MPBs, h, l) is not None:
            chore = find_transferable_chore(X, MPBs, h, l)
            if p_x[h] - p[chore] > p_x[l]:
                X, p_x, p_1x = transfer_chore(X, p, chore, h, l)
                b, h, l = determine_earners(p_x, p_1x)

            elif find_transferable_chore(X, MPBs, b, h) is not None:
                chore = find_transferable_chore(X, MPBs, b, h)
                X, p_x, p_1x = transfer_chore(X, p, chore, b, h)
                b, h, l = determine_earners(p_x, p_1x)

            else:
                p, p_x, p_1x, alpha_matrix, alpha, MPBs = update_prices_1(
                    X, p, D, m, alpha, alpha_matrix, b, h, l
                )

        else:
            p, p_x, p_1x, alpha_matrix, alpha, MPBs = update_prices_2(
                X, p, D, m, alpha, alpha_matrix, b, h, l
            )

    return X


def pEF1_fPO_ILP_chore_allocation(m, n, D):

    ##### Define ILP model
    model = Model("ILP")
    ## Define variables
    # Main variable: allocation X
    x = {}
    for i in range(n):
        for j in range(m):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # Aux variable: weights w for fPO constraint
    epsilon = 1e-6
    w = {}
    for i in range(n):
        w[i] = model.addVar(
            lb=epsilon, ub=1 - epsilon, vtype=GRB.CONTINUOUS, name=f"w_{i}"
        )

    # Aux variable: guessing chore to remove from bundle for EF-1 constaints
    z = {}
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            for j in range(m):
                z[i, k, j] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}_{j}")

    model.update()

    ## Define Constraints
    # Constraint 1: Every chore must be assigned to exactly one agent (complete allocation)
    for j in range(m):
        model.addConstr(quicksum(x[i, j] for i in range(n)) == 1)

    # Constraint 2: auxiliary weights need to add up to 1
    model.addConstr(quicksum(w[i] for i in range(n)) == 1)

    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            # Constraint 4, 5 and 6: Allocation must be EF-1
            model.addConstr(quicksum(z[i, k, j] for j in range(m)) <= 1)
            model.addConstr(
                quicksum(D[j][i] * (x[i, j] - z[i, k, j]) for j in range(m))
                <= quicksum(D[j][k] * x[k, j] for j in range(m))
            )

            for j in range(m):
                model.addConstr(z[i, k, j] <= x[i, j])

                # Constraint 3: allocation must be fPO
                model.addConstr((x[i, j] == 1) >> (w[i] * D[j][i] <= w[k] * D[j][k]))

    # model.setObjective(quicksum(x[i,j] for i in range(n)), GRB.MINIMIZE) #define objective here
    model.optimize()

    X = np.array([[x[i, j].X for i in range(n)] for j in range(m)])

    return np.round(X).astype(int)


def EF1(X, D):
    m, n = X.shape
    D = np.array(D)
    V = X.T @ D

    for i in range(n):
        max_removal = max([D[j, i] for j in range(m) if X[j, i] == 1])
        for k in range(n):
            if i == k:
                continue
            if V[i, i] - max_removal > V[k, i]:
                return False

    return True
