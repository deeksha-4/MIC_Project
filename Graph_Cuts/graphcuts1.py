import numpy as np
import matplotlib.pyplot as plt
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
def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    ### START CODE HERE ###
    for i in range(X.shape[0]):
        J = np.dot((X[i] - centroids[0]),(X[i] - centroids[0]))
        for j in range(K):
            if (J > (np.dot((X[i] - centroids[j]),(X[i] - centroids[j])))):
                J = np.dot((X[i] - centroids[j]),(X[i] - centroids[j]))
                idx[i] = j
            
     ### END CODE HERE ###
    return idx
def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    
    ### START CODE HERE ###
    count = np.zeros(K)
    for i in range(m):
        for j in range(K):
            if (idx[i] == j):
                centroids[j] += X[i]
                count[j] += 1
    for i in range(K):
        centroids[i] /= count[i]
    ### END CODE HERE ## 
    return centroids
def run_kMeans(X, initial_centroids, max_iters=10):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):
        
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)            
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx

def twoStageSegmentation(X,k1,sigma_I,sigma_S,r):
    N,M,F = X.shape
    n,m = N/k1**0.5, M/k1**0.5
    for i in range(k1**0.5):
        for j in range(k1**0.5):
            W,D =createGraph(X[i:i + n][j:j + m],sigma_I,sigma_S,r)
            eigVal,eigVec = np.linalg.eig(D - W)

            run_kMeans(eigVec,)

    pass