import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from .losses import bpr_loss
from .evaluation import get_metrics
from .utils import sample_mini_batch

def train(model, train_sparse_edge_index, train_edge_index, val_edge_index, val_sparse_edge_index, ITERATIONS, ITERS_PER_EVAL, ITERS_PER_LR_DECAY, K, LAMBDA, BATCH_SIZE, LR):
    """Training loop

    Args:
        model (LighGCN): lightgcn model
        train_sparse_edge_index (SparseTensor): sparse adjacency matrix for training
        train_edge_index (torch.Tensor): 2 by N list of edges for training
        val_edge_index (torch.Tensor): 2 by N list of edges for validation
        val_sparse_edge_index (SparseTensor): sparse adjacency matrix for validation
        ITERATIONS (int): total iterations for training
        ITERS_PER_EVAL (int): number of iterations per evaluation
        ITERS_PER_LR_DECAY (int): number of iterations per lr decay
        K (int): top k items to compute metrics on
        LAMBDA (float): lambda for bpr loss
        BATCH_SIZE (int): minibatch size
        LR (float): learning rate

    Returns:
        list: train_losses, val_losses
    """
    device = torch.device('cpu')
    print(f"Using device {device}.")

    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    edge_index = train_edge_index.to(device)
    train_edge_index = train_edge_index.to(device)
    train_sparse_edge_index = train_sparse_edge_index.to(device)

    val_edge_index = val_edge_index.to(device)
    val_sparse_edge_index = val_sparse_edge_index.to(device)

    train_losses = []
    val_losses = []

    for iter in range(ITERATIONS):
        # forward propagation
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
            train_sparse_edge_index)

        # mini batching
        user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(
            BATCH_SIZE, train_edge_index)
        user_indices, pos_item_indices, neg_item_indices = user_indices.to(
            device), pos_item_indices.to(device), neg_item_indices.to(device)
        users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
        pos_items_emb_final, pos_items_emb_0 = items_emb_final[
            pos_item_indices], items_emb_0[pos_item_indices]
        neg_items_emb_final, neg_items_emb_0 = items_emb_final[
            neg_item_indices], items_emb_0[neg_item_indices]

        # loss computation
        train_loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final,
                              pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if iter % ITERS_PER_EVAL == 0:
            model.eval()
            val_loss, recall, precision, ndcg = evaluation(
                model, val_edge_index, val_sparse_edge_index, [train_edge_index], K, LAMBDA)
            print(f"[Iteration {iter}/{ITERATIONS}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss, 5)}, val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)}, val_ndcg@{K}: {round(ndcg, 5)}")
            train_losses.append(train_loss.item())
            val_losses.append(val_loss)
            model.train()

        if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
            scheduler.step()

    return train_losses, val_losses
