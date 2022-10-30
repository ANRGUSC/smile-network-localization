from process_data import fake_dataset
import torch.optim as optim
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import det
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from scipy.sparse.linalg import svds
from scipy.linalg import eigh, svd, qr, solve, lstsq

def reduce_rank(X,k=4,make_sym=False):
    if make_sym:
        X=(X+X.T)/2
    U, S, V = svds(X,k)
    X_ = U.dot(np.diag(S)).dot(V)
    return X_

def eig_(X):
    eigenValues, eigenVectors = np.linalg.eig(X)
    idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues, eigenVectors)

def denoise_via_SVD(euclidean_matrix,k=4,fill_diag=False,take_sqrt=False):
    x = euclidean_matrix
    new_x = torch.Tensor(reduce_rank(x,k))
    if fill_diag:
        new_x.fill_diagonal_(0)
    if take_sqrt:
        new_x = new_x**0.5
    return new_x

def solve_direct(noisy_distance_matrix, anchor_locs, mode="None", dont_square=False):
    x = noisy_distance_matrix
    num_nodes = x.shape[0]
    num_anchors = anchor_locs.shape[0]
    M = np.zeros(x.shape)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if dont_square:
                M[i][j] = (x[0][j] + x[i][0] - x[i][j])/2
            else:
                M[i][j] = (x[0][j]**2 + x[i][0]**2 - x[i][j]**2)/2

    q, v = eig_(M) #np.linalg.eig(M)
    # print("rank of M:",np.linalg.matrix_rank(M, tol=0.001))
    # print("q:",np.round(q,2))
    # plt.plot(q.real)
    # plt.show()

    locs = np.zeros((num_nodes,2))
    # locs[:,0] = np.sqrt(q[0]).real*v[:,0].real
    # locs[:,1] = np.sqrt(q[1]).real*v[:,1].real

    locs[:,0] = np.sqrt(q[0])*v[:,0]
    locs[:,1] = np.sqrt(q[1])*v[:,1]

    # print(anchor_locs)
    # print(locs)

    if mode == "None":
        pass

    if mode == "Kabsch":
        A = anchor_locs.numpy()
        B = locs[:num_anchors]
        n, m = A.shape
        EA = np.mean(A, axis=0)
        EB = np.mean(B, axis=0)
        VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)
        H = ((A - EA).T @ (B - EB)) / n
        U, D, VT = np.linalg.svd(H)

        # TRY d = 1
        d = 1
        S_pos = np.diag([1] * (m - 1) + [d])
        c_pos = VarA / np.trace(np.diag(D) @ S_pos)

        # TRY d = 1
        d = -1
        S_neg = np.diag([1] * (m - 1) + [d])
        c_neg = VarA / np.trace(np.diag(D) @ S_neg)

        if abs(c_pos - 1) < abs(c_neg - 1):
            S, c = S_pos, c_pos
        else:
            S, c = S_neg, c_neg

        R = U @ S @ VT
        t = EA - c * R @ EB
        locs = np.array([t + c * R @ b for b in locs])

    return torch.Tensor(locs)

if __name__=="__main__":
    print("executing mds.py")
    seed_ = 0
    np.random.seed(seed_)
    torch.manual_seed(seed_)

    num_nodes = 500
    num_anchors = 50
    data_loader, num_nodes, noisy_distance_matrix, true_k1 = fake_dataset(num_nodes, num_anchors, threshold=1.2, p_nLOS=10)

    for batch in data_loader:
        anchor_locs = batch.y[batch.anchors]
        noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)
        rank_reduced = denoise_via_SVD(noisy_distance_matrix**2, k=4, fill_diag=False, take_sqrt=False)
        # rank_reduced = (rank_reduced+rank_reduced.T)/2
        rank_reduced_decomposition = solve_direct(rank_reduced, anchor_locs, mode="Kabsch", dont_square=True)
        rank_reduced_decomposition_error = np.mean(np.linalg.norm((rank_reduced_decomposition[batch.nodes].detach().numpy() - batch.y[batch.nodes].detach().numpy()), axis=1))
    print("RMSE:",rank_reduced_decomposition_error)
