import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from torch_geometric.data import Data
import torch_geometric
from torch_geometric.loader import DataLoader
import scipy.sparse as sp

from scipy.io import loadmat
from sklearn.linear_model import LinearRegression

pdist = torch.nn.PairwiseDistance(p=2)

def matrix_from_locs(locs):
    num_nodes = locs.shape[0]
    distance_matrix = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            d = pdist(locs[i].unsqueeze(0), locs[j].unsqueeze(0))
            distance_matrix[i][j] = d
    return distance_matrix

def fake_dataset(num_nodes, num_anchors, threshold=1.0, p_nLOS=10, std=0.1, nLOS_max=10, noise_floor_dist=None):
    # nodes is total nodes, including anchors
    true_locs = torch.rand((num_nodes,2))*5
    distance_matrix = torch.zeros((num_nodes, num_nodes))
    nLOS_noise = torch.zeros((num_nodes, num_nodes))
    p_nLOS = p_nLOS/100

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i < j:
                d = pdist(true_locs[i].unsqueeze(0), true_locs[j].unsqueeze(0))
                distance_matrix[i][j] = d
                distance_matrix[j][i] = d

                if np.random.random() < p_nLOS:
                    uniform_noise = torch.rand(())*nLOS_max
                    nLOS_noise[i][j] = uniform_noise
                    nLOS_noise[j][i] = uniform_noise

    noise = np.random.normal(loc=0.0,scale=std,size=(num_nodes,num_nodes))
    noise = torch.Tensor(noise)
    noise.fill_diagonal_(0)

    # p_nLOS = p_nLOS/100
    # nLOS = np.random.choice([0, 1], size=(num_nodes,num_nodes), p=[1-p_nLOS, p_nLOS])
    # nLOS = torch.Tensor(nLOS)
    # nLOS.fill_diagonal_(0)
    # nLOS_noise = torch.rand((num_nodes,num_nodes))*nLOS_max

    true_k1 = np.count_nonzero(nLOS_noise.numpy())
    # print(true_k1/(num_nodes*(num_nodes-1)))

    noisy_distance_matrix = distance_matrix + noise + nLOS_noise

    if noise_floor_dist:
        max_dist = torch.max(distance_matrix)
        print("distances over", noise_floor_dist, "are measured as", max_dist)
        # turn distances above a threshold into noise floor distances
        extra_k1 = np.count_nonzero(noisy_distance_matrix>noise_floor_dist)
        noisy_distance_matrix[noisy_distance_matrix>noise_floor_dist] = max_dist
        print("original k1:", true_k1, "new k1:", true_k1+extra_k1)
        true_k1 += extra_k1

    adjacency_matrix = (noisy_distance_matrix<threshold).float()
    thresholded_noisy_distance_matrix  = noisy_distance_matrix.clone()
    thresholded_noisy_distance_matrix[thresholded_noisy_distance_matrix>threshold] = 0.0

    def normalize(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))+1e-9
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return torch.Tensor(mx)

    features = normalize(thresholded_noisy_distance_matrix)
    normalized_adjacency_matrix = normalize(adjacency_matrix)

    # features = normalize_tensor(thresholded_noisy_distance_matrix)
    # normalized_adjacency_matrix = normalize_tensor(adjacency_matrix)

    anchor_mask = torch.zeros(num_nodes).bool()
    node_mask = torch.zeros(num_nodes).bool()
    for a in range(num_anchors):
        anchor_mask[a] = True
    for n in range(num_anchors,num_nodes):
        node_mask[n] = True
    data = Data(x=features, adj=normalized_adjacency_matrix, y=true_locs, anchors=anchor_mask, nodes=node_mask)
    return DataLoader([data]), num_nodes, noisy_distance_matrix, true_k1

if __name__=="__main__":
    print("executing process_dataset.py")
    seed_ = 0
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    data_loader, num_nodes, noisy_distance_matrix, true_k1 = fake_dataset(500, 50, threshold=1.2, p_nLOS=10)
    print("noisy_distance_matrix:",noisy_distance_matrix.shape)
