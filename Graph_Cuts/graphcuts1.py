import numpy as np
def createGraph(X,sigma_I,sigma_S,r):
    # X is the input image of nxmx3
    n,m,k = X.shape
    W = np.zeros((n*m,n*m))
    D = np.zeros_like(W)
    for i in range(n*m):
        for j in range(n*m):
            pi, qi = i/n ,i%n
            pj, qj  = j/n ,j%n
            dist_term = ((pi - pj)**2 + (qi - qj)**2)/sigma_S
            if (dist_term < r):
                feature_term = (np.linalg.norm(X[pi][qi] - X[pj][qj])**2)/sigma_I
                W[i][j] = np.exp(-feature_term) * np.exp(-dist_term)
            else:
                W[i][j] = 0
        D[i][i] = np.sum(W[i])
    return W,D
