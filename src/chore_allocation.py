import numpy as np


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

def update_prices_1(X, p, D, m, alpha, alpha_matrix,b, h, l):

    beta=0
    X_b = [j for j in range(m) if X[j,b]==1]
    X_h = [j for j in range(m) if X[j,h]==1]
    X_l = [j for j in range(m) if X[j,l]==1]

    for j in X_b:
        candidate1 = alpha[l]/(D[j,l]/p[j])
        candidate2 = alpha[h]/(D[j,h]/p[j])
        if max(candidate1,candidate2)>beta:
            beta = max(candidate1,candidate2)

    bundle= list(set([*X_l, *X_h]))

    for j in bundle:
        p[j]=beta*p[j]
    p_x, p_1x = compute_p_x(X, p)
    alpha_matrix, alpha = compute_alphas(D, p)  

    MPBs = find_MPB_sets(alpha_matrix, alpha)

    return p, p_x, p_1x, alpha_matrix, alpha, MPBs


def update_prices_2(X, p, D, m, alpha, alpha_matrix,b, h, l):

    beta=0
    X_b = [j for j in range(m) if X[j,b]==1]
    X_h = [j for j in range(m) if X[j,h]==1]
    X_l = [j for j in range(m) if X[j,l]==1]

    bundle= list(set([*X_b, *X_h]))

    for j in bundle:
        candidate = alpha[l]/(D[j,l]/p[j])
        if candidate>beta:
            beta = candidate

    for j in X_l:
        p[j]=beta*p[j]
    p_x, p_1x = compute_p_x(X, p)
    alpha_matrix, alpha = compute_alphas(D, p)  

    MPBs = find_MPB_sets(alpha_matrix, alpha)

    return p, p_x, p_1x, alpha_matrix, alpha, MPBs


def pEF1_fPO_three_agent_chore_allocation(m,N,D):
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
            chore =find_transferable_chore(X, MPBs, h, l)
            if p_x[h] - p[chore]> p_x[l]:
                X, p_x, p_1x = transfer_chore(X, p, chore, h, l)
                b, h, l = determine_earners(p_x, p_1x)
            
            elif find_transferable_chore(X, MPBs, b, h) is not None:
                chore = find_transferable_chore(X, MPBs, b, h)
                X, p_x, p_1x = transfer_chore(X, p, chore, b, h)
                b, h, l = determine_earners(p_x, p_1x)
            
            else:
                p, p_x, p_1x, alpha_matrix, alpha, MPBs = update_prices_1(X, p, D, m, alpha,alpha_matrix, b, h, l)
        
        else:
            p, p_x, p_1x, alpha_matrix, alpha, MPBs = update_prices_2(X, p, D, m, alpha,alpha_matrix, b, h, l)
        
    return X


