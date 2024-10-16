# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# # Install required packages.
# %%capture
# import torch
# import os
# os.environ['TORCH'] = torch.__version__
# !pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}.html
# !pip install git+https://github.com/pyg-team/pytorch_geometric.git

# import required modules
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim, Tensor

from torch_sparse import SparseTensor, matmul

from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj

"""Base implementation of code inspired by https://arxiv.org/abs/2002.02126


https://github.com/gusye1234/LightGCN-PyTorch
"""

# load user and movie nodes
def load_node_csv(path, index_col):
    """Loads csv containing node information

    Args:
        path (str): path to csv file
        index_col (str): column name of index column

    Returns:
        dict: mapping of csv row to node id
    """
    df = pd.read_csv(path, index_col=index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    return mapping


user_mapping = load_node_csv(rating_path, index_col='userId')
movie_mapping = load_node_csv(movie_path, index_col='movieId')

# load edges between users and movies
def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, link_index_col, rating_threshold=4):
    """Loads csv containing edges between users and items

    Args:
        path (str): path to csv file
        src_index_col (str): column name of users
        src_mapping (dict): mapping between row number and user id
        dst_index_col (str): column name of items
        dst_mapping (dict): mapping between row number and item id
        link_index_col (str): column name of user item interaction
        rating_threshold (int, optional): Threshold to determine positivity of edge. Defaults to 4.

    Returns:
        torch.Tensor: 2 by N matrix containing the node ids of N user-item edges
    """
    df = pd.read_csv(path)
    edge_index = None
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) >= rating_threshold


    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])

    return torch.tensor(edge_index)


edge_index = load_edge_csv(
    rating_path,
    src_index_col='userId',
    src_mapping=user_mapping,
    dst_index_col='movieId',
    dst_mapping=movie_mapping,
    link_index_col='rating',
    rating_threshold=4,
)

# split the edges of the graph using a 60/20/20 train/validation/test split
num_users, num_movies = len(user_mapping), len(movie_mapping)
num_interactions = edge_index.shape[1]
all_indices = [i for i in range(num_interactions)]

train_indices, test_indices = train_test_split(
    all_indices, test_size=0.4, random_state=1)
val_indices, test_indices = train_test_split(
    test_indices, test_size=0.5, random_state=1)

train_edge_index = edge_index[:, train_indices]
val_edge_index = edge_index[:, val_indices]
test_edge_index = edge_index[:, test_indices]
eval_edge_index = torch.cat((val_edge_index, test_edge_index), dim=1)

train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1], sparse_sizes=(
    num_users + num_movies, num_users + num_movies))
val_sparse_edge_index = SparseTensor(row=val_edge_index[0], col=val_edge_index[1], sparse_sizes=(
    num_users + num_movies, num_users + num_movies))
test_sparse_edge_index = SparseTensor(row=test_edge_index[0], col=test_edge_index[1], sparse_sizes=(
    num_users + num_movies, num_users + num_movies))
eval_sparse_edge_index = SparseTensor(row=eval_edge_index[0], col=eval_edge_index[1], sparse_sizes=(
    num_users + num_movies, num_users + num_movies))

# function which random samples a mini-batch of positive and negative samples
def sample_mini_batch(batch_size, edge_index):
    """Randomly samples indices of a minibatch given an adjacency matrix

    Args:
        batch_size (int): minibatch size
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        tuple: user indices, positive item indices, negative item indices
    """
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    indices = random.choices(
        [i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices

# defines LightGCN model
class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """

    def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False):
        """Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.K = embedding_dim, K
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_dim) # e_u^0
        self.items_emb = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.embedding_dim) # e_i^0

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: SparseTensor):
        """Forward propagation of LightGCN Model.

        Args:
            edge_index (SparseTensor): adjacency matrix

        Returns:
            tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
        """
        # compute \tilde{A}: symmetrically normalized adjacency matrix
        edge_index_norm = gcn_norm(
            edge_index, add_self_loops=self.add_self_loops)

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight]) # E^0
        embs = [emb_0]
        emb_k = emb_0

        # multi-scale diffusion
        for i in range(self.K):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1) # E^K

        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items]) # splits into e_u^K and e_i^K

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)

model = LightGCN(num_users, num_movies)

def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, lambda_val):
    """Bayesian Personalized Ranking Loss as described in https://arxiv.org/abs/1205.2618

    Args:
        users_emb_final (torch.Tensor): e_u_k
        users_emb_0 (torch.Tensor): e_u_0
        pos_items_emb_final (torch.Tensor): positive e_i_k
        pos_items_emb_0 (torch.Tensor): positive e_i_0
        neg_items_emb_final (torch.Tensor): negative e_i_k
        neg_items_emb_0 (torch.Tensor): negative e_i_0
        lambda_val (float): lambda value for regularization loss term

    Returns:
        torch.Tensor: scalar bpr loss value
    """
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) +
                             pos_items_emb_0.norm(2).pow(2) +
                             neg_items_emb_0.norm(2).pow(2)) # L2 loss

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1) # predicted scores of positive samples
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1) # predicted scores of negative samples

    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss

    return loss

# helper function to get N_u
def get_user_positive_items(edge_index):
    """Generates dictionary of positive items for each user

    Args:
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        dict: dictionary of positive items for each user
    """
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items

# computes recall@K and precision@K
def RecallPrecision_ATk(groundTruth, r, k):
    """Computers recall @ k and precision @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (intg): determines the top k items to compute precision and recall on

    Returns:
        tuple: recall @ k, precision @ k
    """
    num_correct_pred = torch.sum(r, dim=-1)  # number of correctly predicted items per user
    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor([len(groundTruth[i])
                                  for i in range(len(groundTruth))])
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()

# computes NDCG@K
def NDCGatK_r(groundTruth, r, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute ndcg on

    Returns:
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()

# wrapper function to get evaluation metrics
def get_metrics(model, edge_index, exclude_edge_indices, k):
    """Computes the evaluation metrics: recall, precision, and ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on

    Returns:
        tuple: recall @ k, precision @ k, ndcg @ k
    """
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight

    # get ratings between every user and item - shape is num users x num movies
    rating = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        # gets all the positive items for each user from the edge index
        user_pos_items = get_user_positive_items(exclude_edge_index)
        # get coordinates of all edges to exclude
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        # set ratings of excluded edges to large negative value
        rating[exclude_users, exclude_items] = -(1 << 10)

    # get the top k recommended items for each user
    _, top_K_items = torch.topk(rating, k=k)

    # get all unique users in evaluated split
    users = edge_index[0].unique()

    test_user_pos_items = get_user_positive_items(edge_index)

    # convert test user pos items dictionary into a list
    test_user_pos_items_list = [
        test_user_pos_items[user.item()] for user in users]

    # determine the correctness of topk predictions
    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)

    return recall, precision, ndcg

# wrapper function to evaluate model
def evaluation(model, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val):
    """Evaluates model loss and metrics including recall, precision, ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        sparse_edge_index (sparseTensor): sparse adjacency matrix for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on
        lambda_val (float): determines lambda for bpr loss

    Returns:
        tuple: bpr loss, recall @ k, precision @ k, ndcg @ k
    """
    # get embeddings
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
        sparse_edge_index)
    edges = structured_negative_sampling(
        edge_index, contains_neg_self_loops=False)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[
        pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[
        neg_item_indices], items_emb_0[neg_item_indices]

    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0,
                    neg_items_emb_final, neg_items_emb_0, lambda_val).item()

    recall, precision, ndcg = get_metrics(
        model, edge_index, exclude_edge_indices, k)

    return loss, recall, precision, ndcg

"""# Training

"""

# define contants
ITERATIONS = 10000
BATCH_SIZE = 1024
LR = 1e-3
ITERS_PER_EVAL = 200
ITERS_PER_LR_DECAY = 200
K = 20
LAMBDA = 1e-6

# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}.")


model = model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

edge_index = edge_index.to(device)
train_edge_index = train_edge_index.to(device)
train_sparse_edge_index = train_sparse_edge_index.to(device)

val_edge_index = val_edge_index.to(device)
val_sparse_edge_index = val_sparse_edge_index.to(device)

# training loop
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

iters = [iter * ITERS_PER_EVAL for iter in range(len(train_losses))]
plt.plot(iters, train_losses, label='train')
plt.plot(iters, val_losses, label='validation')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('training and validation loss curves')
plt.legend()
plt.show()

# evaluate on test set
model.eval()
test_edge_index = test_edge_index.to(device)
test_sparse_edge_index = test_sparse_edge_index.to(device)

test_loss, test_recall, test_precision, test_ndcg = evaluation(
            model, test_edge_index, test_sparse_edge_index, [train_edge_index, val_edge_index], K, LAMBDA)

print(f"[test_loss: {round(test_loss, 5)}, test_recall@{K}: {round(test_recall, 5)}, test_precision@{K}: {round(test_precision, 5)}, test_ndcg@{K}: {round(test_ndcg, 5)}")

"""# LightGCNxGPT"""

model.eval()
df = pd.read_csv(movie_path)
movieid_title = pd.Series(df.title.values,index=df.movieId).to_dict()
movieid_genres = pd.Series(df.genres.values,index=df.movieId).to_dict()

user_pos_items = get_user_positive_items(edge_index)

movie_titles = pd.Series(df.title.values, index=df.movieId).to_dict() #map movies to titles
movie_genres = pd.Series(df.genres.values, index=df.movieId).to_dict() #map movies to genres

recall_model_sum = 0
precision_model_sum = 0
precision_final_sum = 0
recall_final_sum = 0
precision_total_sum = 0
recall_total_sum = 0
recall_rem_sum = 0
precision_rem_sum = 0
num_users_with_hr_movies = 0
accuracy_final_sum = 0
accuracy_model_sum = 0
occurences = 0
doing_better_recall = 0
doing_better_precision = 0
user_count_instances = 0
recall_lightgcn_sum = 0
precision_lightgcn_sum = 0

user_train_pos_items = get_user_positive_items(train_edge_index)
user_test_pos_items = get_user_positive_items(test_edge_index)
user_val_pos_items = get_user_positive_items(val_edge_index)


for user in test_user_indices:
  topK = 20
  # user = user_mapping[user_id] #map the user
  if user >= 0:
    print("===================================")
    print("user", user)
    e_u = model.users_emb.weight[user]
    scores = model.items_emb.weight @ e_u
    #pos_items = user_test_pos_items[user] + user_val_pos_items[user]

    values, indices = torch.topk(scores, k=len(user_pos_items[user]) + topK)

    user_instance = 0

    hrated_movies = [index.cpu().item() for index in indices if index in user_train_pos_items[user]] #movies that were rated highly by the user and model knows
    hrated_movie_ids = [list(movie_mapping.keys())[list(movie_mapping.values()).index(movie)] for movie in hrated_movies] #get the movie ids
    hrated_titles = [movieid_title[id] for id in hrated_movie_ids]
    hrated_genres = [movieid_genres[id] for id in hrated_movie_ids]

    # rec_ids_real = [index.cpu().item() for index in indices] #movies predicted by the user -> raw ids
    rec_ids_real = [index.cpu().item() for index in indices if index not in user_train_pos_items[user]]
    print("movie ids predicted", rec_ids_real)
    topk_movies_rec = rec_ids_real[:topK] #top movies recommended by model
    print("top k movies rec", topk_movies_rec)

    rec_map = [list(movie_mapping.keys())[list(movie_mapping.values()).index(movie)] for movie in topk_movies_rec] #movie ids recommended by model
    rec_titles = [movie_titles[id] for id in rec_map]
    rec_genres = [movie_genres[id] for id in rec_map]
    gr_user_pos_items = get_user_positive_items(edge_index)
    ground_truth = test_user_pos_items[user] #ground truth for the user, all highly rated movies by the user

    def calculate_recall_precision(gr, recommended_movies):
          # print("ground truth", gr)
          # print("recommended movies", recommended_movies)
          liked_set = set(gr)
          recommended_set = set(recommended_movies)
          true_positives = len(liked_set.intersection(recommended_set))
          false_negatives = len(liked_set.difference(recommended_set))
          false_positives = len(recommended_set.difference(liked_set))
          recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
          precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
          total_predictions = len(recommended_movies)
          return recall, precision

    titles_liked_json = json.dumps(hrated_titles)
    titles_rec_json = json.dumps(rec_titles)
    genres_liked_json = json.dumps(hrated_genres)
    print("recommended titles by LightGCN Model", titles_rec_json)
    print("liked movies", titles_liked_json)

    prompt = "Based on the user's history , can you generate a user profile and provide details of their {age: , gender, genres liked:, genres dislike:, favourite directors:, country;}. where information is not provided, make an educated guess."
    prompt += "can you recommend movies that you predict the user will rate highly by profiling them based on the movies they like and other users likes? Base these recommendations on your learnt knowledge of movies similar users are liking and online movies rating sources. return it as a list in this format movies_reco = [" ", " "]. ensure there are quotes and commas."
    prompt += "this is the movies the user liked"
    prompt += titles_liked_json
    prompt += "this is the genres of movies the user liked"
    prompt += genres_liked_json
    prompt += "Please remove any movies from the model's predictions that you think does not fit the user profile and preferences, from those towards the end of the list as the recommendations have been ranked by the model already, it is perfoming with an recall of 10%. return it as a list in this format movies_removed = [" ", " "]. ensure there are quotes and commas."
    prompt += "this is what the model predicted"
    prompt += titles_rec_json
    model_choice = "gpt-3.5-turbo"

    def generate_text(prompt, model=model_choice):
          try:
              response = openai.ChatCompletion.create(
                  model=model,
                  messages=[
                      {"role": "system", "content": "You are a hardworking assistant."},
                      {"role": "user", "content": prompt}
                  ]
              )
              movies_removed = response['choices'][0]['message']['content'].strip()
              return movies_removed
          except Exception as e:
              return f"An error occurred: {e}"
    generated_text = generate_text(prompt)
    print("GPT Prompt:")
    print(generated_text)
    # print("=======================")

    def extract_movie_lists(generated_text):
          # Define regular expressions to match the lists
          removed_pattern = r"movies_removed\s*=\s*(\[.*?\])"
          reco_pattern = r"movies_reco\s*=\s*(\[.*?\])"

          # Find matches for the patterns in the generated_text
          removed_match = re.search(removed_pattern, generated_text)
          reco_match = re.search(reco_pattern, generated_text)

          try:
              movies_removed = eval(removed_match.group(1)) if removed_match else None
          except (SyntaxError, NameError):
              print("Error evaluating movies_removed. Check the formatting.")
              movies_removed = None

          try:
              movies_reco = eval(reco_match.group(1)) if reco_match else None
          except (SyntaxError, NameError):
              print("Error evaluating movies_reco. Check the formatting.")
              movies_reco = None

          return movies_removed, movies_reco

    movies_removed, movies_reco = extract_movie_lists(generated_text)

    if movies_removed is not None and movies_reco is not None:
        movie_map_rem = [movie_id for movie_id, title in movie_titles.items() if movies_removed is not None and title in movies_removed]

        movie_real_rem = [
    list(movie_mapping.values())[  # Retrieve the keys (movie IDs) of the movie_mapping dictionary
        list(movie_mapping.keys()).index(movie)  # Get the index of the current movie ID in the values of movie_mapping
    ] for movie in movie_map_rem  # Iterate over each movie ID in topk_movies_rec
]
        rec_movie_ids_filtered = [movie_id for movie_id in topk_movies_rec if movie_id not in movie_real_rem] #removing movie ids not recommended

        movie_names_reco = [title for title in movies_reco if any(fuzz.partial_ratio(title, movie_title) >= 80 for movie_title in movie_titles.values())]
        print("movie names recommended by GPT", movie_names_reco)
        movie_map_reco = [movie_id for movie_id, title in movie_titles.items() if movie_names_reco is not None and title in movie_names_reco]

        movie_real_reco = [
    list(movie_mapping.values())[list(movie_mapping.keys()).index(movie)] for movie in movie_map_reco]

        recall_model, precision_model = calculate_recall_precision(ground_truth, topk_movies_rec)
        print("========================================================")
        print("Recall of the LightGCN Model: ", recall_model)
        print("Precision of the LightGCN Model: ", precision_model)
        print("========================================================")
        precision_model_sum += precision_model
        recall_model_sum += recall_model
        print("Sum of Recall of LightGCN Model: ", recall_model_sum)
        print("Sum of Precision of LightGCN Model: ", precision_model_sum)

        # rec_movie_ids_total = topk_movies_rec + movie_real_reco
        rec_movie_ids_total = list(set(topk_movies_rec + movie_real_reco)) #ensures only unique ids from both
        rec_movie_ids_final = list(set(rec_movie_ids_filtered + movie_real_reco))
        if len(rec_movie_ids_final) >= topK:
          rec_movie_ids_final = rec_movie_ids_final[:topK]
          # print("after length of movie ids", len(rec_movie_ids_final))
        else:
          # print("length less than", len(rec_movie_ids_final))
          remaining = topK - len(rec_movie_ids_final)
          rec_movie_ids_final = rec_movie_ids_final + movie_real_rem[:remaining]
          # print("length after adding", len(rec_movie_ids_final))
        # Initialize counter

        # Iterate through each id in movie_names_reco
        for movie in movie_real_reco:
            # Check if the id exists in titles_liked_json
            if movie in hrated_movies:
              # Check if the id exists in rec_movie_ids_final
                if movie in rec_movie_ids_final:
                    occurences += 1
                    print("occurences", occurences)
                    user_instance = 1
        if user_instance == 1:
          user_count_instances += 1
          print("user instances", user_count_instances)

        if len(rec_ids_real) > len(rec_movie_ids_total):
          cut_off_compare = len(rec_movie_ids_total)
        else:
          cut_off_compare = len(rec_ids_real)

        recall_lightgcn, precision_lightgcn = calculate_recall_precision(ground_truth, rec_ids_real[:cut_off_compare])
        recall_lightgcn_sum += recall_lightgcn
        precision_lightgcn_sum += precision_lightgcn
        print("Sum of Recall of LightGCN Recommendations: ", recall_lightgcn_sum)
        print("Sum of Precision of LightGCN Recommendations: ", precision_lightgcn_sum)

        print("Sum of Recall after adding GPT Recommendations: ", recall_total_sum)
        print("Sum of Precision after adding GPT Recommendations: ", precision_total_sum)
        recall_total, precision_total = calculate_recall_precision(ground_truth, rec_movie_ids_total[:cut_off_compare])
        precision_total_sum += precision_total
        recall_total_sum += recall_total
        print("Sum of Recall after adding GPT Recommendations: ", recall_total_sum)
        print("Sum of Precision after adding GPT Recommendations: ", precision_total_sum)

        recall_rem, precision_rem = calculate_recall_precision(ground_truth, rec_movie_ids_filtered)
        recall_rem_sum += recall_rem
        precision_rem_sum += precision_rem
        print("Sum of Recall after removing GPT Removal Suggestions: ", recall_rem_sum)
        print("Sum of Precision after removing GPT Removal Suggestions: ", precision_rem_sum)

        recall_final, precision_final = calculate_recall_precision(ground_truth, rec_movie_ids_final)
        precision_final_sum += precision_final
        recall_final_sum += recall_final
        print("========================================================")
        print("Recall of the GPTGCN Model: ", recall_final)
        print("Precision of the GPTGCN Model: ", precision_final)
        print("========================================================")
        print("Sum of Recall of GPTGCN Model: ", recall_final_sum)
        print("Sum of Precision of GPTGCN Model: ", precision_final_sum)
        if recall_final >= recall_model:
          doing_better_recall += 1
          print("doing better than lightgcn", doing_better_recall)
        if precision_final >= precision_model:
          doing_better_precision += 1
          print("doing better than lightgcn", doing_better_precision)

        num_users_with_hr_movies += 1
        #time.sleep(5)
        print("num of users", num_users_with_hr_movies)
        # print("precision_final:", precision_final_sum)
        # print("recall_final:", recall_final_sum)
        # print("accuracy final:", accuracy_final_sum)
        # print("precision_total:", precision_total_sum)
        # print("recall_total:", recall_total_sum)
        # print("recall rem:", recall_rem_sum)
        # print("precision rem:", precision_rem_sum)
        # print("total_ndcg @", topK, "is", total_ndcg)
        print("==================================================")

# Calculate averages
if num_users_with_hr_movies > 0:
    avg_precision_model = precision_model_sum / num_users_with_hr_movies
    avg_recall_model = recall_model_sum / num_users_with_hr_movies
    avg_precision_final = precision_final_sum / num_users_with_hr_movies
    avg_recall_final = recall_final_sum / num_users_with_hr_movies
    avg_recall_total = recall_total_sum / num_users_with_hr_movies
    avg_precision_total = precision_total_sum / num_users_with_hr_movies
    avg_recall_rem = recall_rem_sum / num_users_with_hr_movies
    avg_precision_rem = precision_rem_sum / num_users_with_hr_movies
    # avg_accuracy_final = accuracy_final_sum / num_users_with_hr_movies
    # avg_accuracy_model = accuracy_model_sum / num_users_with_hr_movies
    avg_lightgcn_recall = recall_lightgcn_sum / num_users_with_hr_movies
    avg_lightgcn_precision = precision_lightgcn_sum / num_users_with_hr_movies
    # avg_ndcg = total_ndcg / num_users_with_hr_movies

else:
    avg_precision_final = 0
    avg_precision_total = 0
    avg_precision_model = 0
    avg_recall_model = 0
    avg_precision_total = 0
    avg_recall_total = 0
    avg_recall_rem = 0
    avg_precision_rem = 0
    # avg_accuracy = 0
    # avg_ndcg = 0
print("=======================================================")
print("Num of users counted", num_users_with_hr_movies)
print("=======================================================")
print("LightGCN Model")
print("Average Recall of LightGCN Model: ", avg_recall_model)
print("Average Precision of LightGCN Model: ", avg_precision_model)
print("=======================================================")
print("GPT GCN Model")
print("Average Recall of GPT GCN Model: ", avg_recall_final)
print("Average Precision of GPT GCN Model: ", avg_precision_final)
print("=======================================================")
print("Other Metrics")
print("Average Recall including all GPT Recommendations: ", avg_recall_total)
print("Average Precision including all GPT Recommendations: ", avg_precision_total)
print("Average Recall removing all GPT Suggestions: ", avg_recall_rem)
print("Average Precision removing all GPT Suggestions: ", avg_precision_rem)
# Print the total number of occurrences
print("Total occurrences in titles_liked_json:", occurences)
print("Total Number of user instances", user_count_instances)
print("Number of times GPTGNN is better than LightGCN (recall):", doing_better_recall)
print("Number of times GPTGNN is better than LightGCN (precision):", doing_better_precision)
