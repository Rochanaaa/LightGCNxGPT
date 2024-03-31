from model import LightGCN
from training import train
from evaluation import evaluation
from recommendation import make_recommendations
import torch
from torch_geometric.utils import structured_negative_sampling
import pandas as pd
from torch_geometric.data import download_url, extract_zip
from torch_geometric.typing import Adj
from torch_sparse import SparseTensor
import numpy as np

# Install required packages.
import torch
import os
os.environ['TORCH'] = torch.__version__
# pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
# pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
# pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}.html
# pip install git+https://github.com/pyg-team/pytorch_geometric.git
# pip install openai==0.28
# pip install fuzzywuzzy

# import required modules
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim, Tensor
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from evaluation_metrics import RecallPrecision_ATk, NDCGatK_r
from losses import bpr_loss
from evaluation import get_metrics, get_user_positive_items
from training import train
from recommendation import make_recommendations
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# download the dataset
url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
extract_zip(download_url(url, '.'), '.')

movie_path = './ml-latest-small/movies.csv'
rating_path = './ml-latest-small/ratings.csv'

user_mapping = load_node_csv(rating_path, index_col='userId')
movie_mapping = load_node_csv(movie_path, index_col='movieId')

edge_index = load_edge_csv(
    rating_path,
    src_index_col='userId',
    src_mapping=user_mapping,
    dst_index_col='movieId',
    dst_mapping=movie_mapping,
    link_index_col='rating',
    rating_threshold=4,
)

num_users, num_movies = len(user_mapping), len(movie_mapping)
num_interactions = edge_index.shape[1]

train_indices, test_indices = train_test_split(
    np.arange(num_interactions), test_size=0.2)
train_edge_index = edge_index[:, train_indices]
test_edge_index = edge_index[:, test_indices]

print('Data Loaded')

val_indices = np.random.choice(
    train_indices, size=int(0.1 * len(train_indices)), replace=False)
val_edge_index = train_edge_index[:, val_indices]
train_edge_index = train_edge_index[:, np.setdiff1d(
    train_indices, val_indices)]

print('Data split into train, test, and validation sets.')

train_sparse_edge_index = SparseTensor.from_edge_index(train_edge_index)
val_sparse_edge_index = SparseTensor.from_edge_index(val_edge_index)
test_sparse_edge_index = SparseTensor.from_edge_index(test_edge_index)

print('Sparse tensors created.')

ITERATIONS = 5000
ITERS_PER_EVAL = 100
ITERS_PER_LR_DECAY = 1000
K = 10
LAMBDA = 1e-5
BATCH_SIZE = 512
LR = 1e-3

model = LightGCN(num_users, num_movies, embedding_dim=64, K=3)
train_losses, val_losses = train(model, train_sparse_edge_index, train_edge_index, val_edge_index,
                                 val_sparse_edge_index, ITERATIONS, ITERS_PER_EVAL, ITERS_PER_LR_DECAY, K, LAMBDA, BATCH_SIZE, LR)

print('Training complete.')

test_recall, test_precision, test_ndcg = get_metrics(
    model, test_edge_index, [train_edge_index], K)

print(f'Test Recall@{K}: {test_recall}')
print(f'Test Precision@{K}: {test_precision}')
print(f'Test NDCG@{K}: {test_ndcg}')

print('Recommendations:')
make_recommendations(model, test_edge_index, movie_path)
