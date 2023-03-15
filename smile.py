from process_data import matrix_from_locs, fake_dataset
import torch.optim as optim
import torch
import torch_geometric
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import eigh, svd, qr, solve, lstsq
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import eigsh

def S_from_D(D, k1):
    D_ = D.flatten()
    S_ = torch.zeros_like(D_)
    s = torch.argsort(D_,descending=True)
    sk1 = s[:k1]
    S_[sk1] = 1
    return S_.reshape(D.shape)

def f(D, X, Y, lam, mu):
    A = torch.linalg.norm(D-X-Y, ord="fro")**2
    # A = torch.sum((D-X-Y)**2)
    B = lam*torch.linalg.norm(X)**2
    # B = lam*torch.sum(X**2)
    C = mu*torch.linalg.norm(Y)**2
    # C = mu*torch.sum(Y**2)
    # print("f(D,X,Y)=",A+B+C)
    return A+B+C

def check_sparsity(X):
    count_pos = torch.sum(X>1e-9)
    return count_pos, X.shape[0]*X.shape[1]

def check_rank(X):
    return np.linalg.matrix_rank(X)

def reduce_rank(X,k=4):
    if X.shape[0] == k:
        U, S, V = np.linalg.svd(X,k)
    else:
        U, S, V = svds(X,k)
    X_ = U.dot(np.diag(S)).dot(V)
    return torch.Tensor(X_)

def solve_sparse_problem(D, X, mu, k1):
    D_ = D-X
    S = S_from_D(D_,k1)
    Y = S*D_/(1+mu)
    return Y

def solve_rank_problem(D, Y, lam, k0):
    D_ = D-Y
    return torch.Tensor(reduce_rank(D_/(1+lam), k0))
    # D_k0 = torch.Tensor(reduce_rank(D_,k0))
    # X = 1/(1+lam)*D_k0
    # return X

def constrain_X(X):
    X[X<0] = 0
    return (X+X.T)/2

def constrain_Y(Y):
    return (Y+Y.T)/2

def separate_dataset(measured, k0, k1, lam=0.1, mu=0.1, eps=0.001, X=None, Y=None, constrain_solution=False):
    D = measured**2
    if X is None:
        X = torch.zeros_like(D)
    if Y is None:
        Y = torch.zeros_like(D)
    fi = f(D, X, Y, lam, mu)
    for iter in range(100):
        Y = solve_sparse_problem(D, X, mu, k1)
        X = solve_rank_problem(D, Y, lam, k0)
        if constrain_solution:
            X = constrain_X(X)
            Y = constrain_Y(Y)
        ff = f(D, X, Y, lam, mu)
        if (fi-ff)/fi <= eps:
            return X, Y, ff
        fi = ff
    return X, Y, ff

def separate_dataset_multiple_inits(measured, k0, k1, n_init=10, lam=0.1, mu=0.1, eps=0.001, constrain_solution=False):
    best_X, best_Y, best_ff = separate_dataset(measured, k0, k1, lam=lam, mu=mu, eps=eps, constrain_solution=constrain_solution)
    for init in range(1,n_init):
        init_X = reduce_rank(torch.rand(measured.shape)*torch.max(measured))
        X, Y, ff = separate_dataset(measured, k0, k1, lam=lam, mu=mu, eps=eps, X=init_X, constrain_solution=constrain_solution)
        if ff < best_ff:
            best_X, best_Y, best_ff = X, Y, ff
    return best_X, best_Y, best_ff

def separate_dataset_find_k1(measured, k0, k1_init=0, step_size=1, n_init=1, lam=0.1, mu=0.1, eps=0.001, eps_k1=0.01, plot=False, constrain_solution=False):
    if check_rank(measured**2) == k0:
        print("already low rank")
        return measured**2, torch.zeros_like(measured), 0, 0

    num_edges = int(measured.shape[0]*measured.shape[1])
    step_size = int(num_edges*step_size/100)
    k1 = k1_init
    X, Y, fi = separate_dataset_multiple_inits(measured, k0, k1, n_init=n_init, lam=lam, mu=mu, eps=eps, constrain_solution=constrain_solution)
    for iter in range(100):
        k1 += step_size
        X, Y, ff = separate_dataset_multiple_inits(measured, k0, int(k1), n_init=n_init, lam=lam, mu=mu, eps=eps, constrain_solution=constrain_solution)
        if (fi-ff)/fi <= eps_k1:
            return X, Y, ff, k1
        fi = ff
    return X, Y, ff, k1

def barycenter_weights(distance_matrix, indices, reg=1e-5, dont_square=False):
    n_samples, n_neighbors = indices.shape
    B = np.empty((n_samples, n_neighbors))
    v = np.ones(n_neighbors)
    D = distance_matrix.numpy()

    if not dont_square:
        D = D**2

    for i, ind in enumerate(indices):
        row = D[i,ind]
        m1 = np.outer(np.ones(n_neighbors), row)
        col = D[ind,i]
        m2 = np.outer(col, np.ones(n_neighbors))
        C = (m1+m2-D[ind][:,ind])/2
        # print(C)

        # C = np.empty((n_neighbors, n_neighbors))
        # for j in range(n_neighbors):
        #     for k in range(n_neighbors):
        #         C[j][k] = (D[i][ind[k]] + D[ind[j]][i] - D[ind[j]][ind[k]])/2
        # print(C)

        trace = np.trace(C)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        C.flat[:: n_neighbors + 1] += R
        try:
            w = solve(C, v, assume_a='pos')
        except np.linalg.LinAlgError:
            # print('in barycenter_weights, matrix C is singular -> use least squares')
            perfect = False
            w, res, rnk, s = lstsq(C, v)
        B[i, :] = w / np.sum(w)
    return B

def weight_to_mat(weights, indices):
    n_samples, n_neighbors = indices.shape
    mat = np.zeros((n_samples, n_samples))
    for i, ind in enumerate(indices):
        mat[i][ind] = weights[i]
    return mat

def neighbors(distance_matrix, n_neighbors):
    distance_matrix.fill_diagonal_(-1000) # hacky, makes sure that you are not one of your own nearest neighbors
    indices = np.argsort(distance_matrix.numpy(), axis=1)
    return indices[:,1:n_neighbors+1]

def solve_like_LLE(num_nodes,num_anchors,n_neighbors,anchor_locs,noisy_distance_matrix,dont_square=False,anchors_as_neighbors=False, return_indices=False):
    if anchors_as_neighbors:
        indices = np.vstack([np.linspace(0,n_neighbors-1,n_neighbors,dtype=int)]*num_nodes)
    else:
        indices = neighbors(noisy_distance_matrix, n_neighbors)
    start = time.time()
    res = barycenter_weights(noisy_distance_matrix, indices, reg=1e-3,dont_square=dont_square)
    # print(f"{time.time()-start} to find weight mat")
    mat = weight_to_mat(res, indices)
    I_minus_W = np.eye(num_nodes)-mat
    RHS = I_minus_W[:,:num_anchors]
    RHS = RHS.dot(anchor_locs)
    LHS = -1*I_minus_W[:,num_anchors:]
    start = time.time()
    node_locs, res, rnk, s = lstsq(LHS, RHS)
    # print("RES:",res)
    # print(f"{time.time()-start} to find locs")
    pred = np.vstack((anchor_locs,node_locs))
    if return_indices:
        return torch.Tensor(pred), indices
    return torch.Tensor(pred)

if __name__=="__main__":
    print("executing smile.py")
    seed_ = 0
    np.random.seed(seed_)
    torch.manual_seed(seed_)

    num_nodes = 500
    num_anchors = 50
    data_loader, num_nodes, noisy_distance_matrix, true_k1 = fake_dataset(num_nodes, num_anchors, threshold=1.2, p_nLOS=10)

    n_neighbors = 50
    for batch in data_loader: # all loaded into one batch
        anchor_locs = batch.y[batch.anchors]
        noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)
        start = time.time()
        X, Y, ff, k1 = separate_dataset_find_k1(noisy_distance_matrix, k0=4, k1_init=0, step_size=1, n_init=1, lam=0.01, mu=0.1, eps=0.001, eps_k1=0.1, plot=False, constrain_solution=False)
        novel_pred, indices = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, X, dont_square=True, anchors_as_neighbors=False, return_indices=True)
        novel_solve_time = time.time()-start
        novel_error = np.mean(np.linalg.norm((novel_pred[batch.nodes].detach().numpy() - batch.y[batch.nodes].detach().numpy()), axis=1))
    print(f"...done in {novel_solve_time} secs")

    print("RMSE:", novel_error)
