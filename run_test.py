from mds import *
from smile import *
from gcn import *

if __name__=="__main__":
    print("executing run_test.py")
    seed_ = 0
    np.random.seed(seed_)
    torch.manual_seed(seed_)

    # DATA PARAMS
    num_nodes = 500
    num_anchors = 50
    p_nLOS = 10
    std = 0.3
    noise_floor_dist = 5.0

    # GCN PARAMS
    threshold = 1.2
    nhid = 2000
    nout = 2
    dropout = 0.5
    lr = 0.01
    weight_decay = 0
    num_epochs = 200

    # NOVEL PARAMS
    n_neighbors = 50
    k0 = 4
    lam = 0.01
    mu = 0.1
    eps = 0.001
    n_init = 1
    k1_init = 0
    step_size = 1
    eps_k1 = 0.01
    constrain_solution=False


    data_loader, num_nodes, noisy_distance_matrix, true_k1 = fake_dataset(num_nodes, num_anchors, threshold=threshold, p_nLOS=p_nLOS, std=std, noise_floor_dist=noise_floor_dist)
    loss_fn = torch.nn.MSELoss()

    print("GCN")
    model = GCN(nfeat=num_nodes, nhid=nhid, nout=nout, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gcn_train_time = train_GCN(model, optimizer, loss_fn, data_loader, num_epochs=num_epochs)
    gcn_pred, gcn_error, gcn_predict_time = test_GCN(model, loss_fn, data_loader)
    gcn_total_time = gcn_train_time + gcn_predict_time
    print("RMSE:",gcn_error)

    for batch in data_loader:
        anchor_locs = batch.y[batch.anchors]
        noisy_distance_matrix = torch.Tensor(noisy_distance_matrix)

        print("Decomposition")
        decomposition = solve_direct(noisy_distance_matrix, anchor_locs, mode="Kabsch")
        decomposition_error = torch.sqrt(loss_fn(decomposition[batch.nodes], batch.y[batch.nodes])).item()
        print("RMSE:",decomposition_error)

        print("Reduce rank + Decomposition")
        rank_reduced = denoise_via_SVD(noisy_distance_matrix**2, k=k0, fill_diag=False, take_sqrt=False)
        # rank_reduced = (rank_reduced+rank_reduced.T)/2
        rank_reduced_decomposition = solve_direct(rank_reduced, anchor_locs, mode="Kabsch", dont_square=True)
        rank_reduced_decomposition_error = torch.sqrt(loss_fn(rank_reduced_decomposition[batch.nodes], batch.y[batch.nodes])).item()
        print("RMSE:",rank_reduced_decomposition_error)

        print("Sparse inference + Decomposition")
        # approximate sparsity:
        # X, Y, ff, k1 = separate_dataset_find_k1(noisy_distance_matrix**2, k0, k1_init=int(k1_init), step_size=step_size, n_init=n_init, lam=lam, mu=mu, eps=eps, eps_k1=eps_k1, plot=False)
        # assume known sparsity:
        X, Y, ff = separate_dataset(noisy_distance_matrix, k0, k1=25000, lam=lam, mu=mu, eps=eps)
        x_decomposition = solve_direct(X, anchor_locs, mode="Kabsch", dont_square=True)
        x_decomposition_error = torch.sqrt(loss_fn(x_decomposition[batch.nodes], batch.y[batch.nodes])).item()
        print("RMSE:",x_decomposition_error)

        print("LLE")
        lle = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, noisy_distance_matrix, dont_square=False, anchors_as_neighbors=False, return_indices=False)
        lle_error = torch.sqrt(loss_fn(lle[batch.nodes], batch.y[batch.nodes])).item()
        print("RMSE:",lle_error)

        print("Reduce rank + LLE")
        rank_reduced_lle = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, rank_reduced, dont_square=True, anchors_as_neighbors=False, return_indices=False)
        rank_reduced_lle_error = torch.sqrt(loss_fn(rank_reduced_lle[batch.nodes], batch.y[batch.nodes])).item()
        print("RMSE:",rank_reduced_lle_error)

        print("SMILE")
        smile = solve_like_LLE(num_nodes, num_anchors, n_neighbors, anchor_locs, torch.Tensor(X), dont_square=True, anchors_as_neighbors=False, return_indices=False)
        novel_error = torch.sqrt(loss_fn(smile[batch.nodes], batch.y[batch.nodes])).item()
        print("RMSE:",novel_error)
