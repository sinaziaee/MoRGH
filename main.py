import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.functional import dropout
from sklearn.metrics import auc
from model import Model
from train_test_utils import create_movie_features, find_genres_columns, plot_losses, train_test_model
from sklearn.metrics import accuracy_score, f1_score


import torch
os.environ['TORCH'] = torch.__version__
# print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch_geometric.nn import SAGEConv, to_hetero, GCNConv
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

bert_movie = pd.read_csv('movie_graph/bert_100k.csv')
genres_movie = pd.read_csv('movie_graph/genres2_100k.csv')
jakard_movie = pd.read_csv('movie_graph/genres_100k.csv')
lda_movie = pd.read_csv('movie_graph/lda_100k.csv')

jakard_df = jakard_movie[jakard_movie['genres_sim'] >= 0.95]
m1_jakard_list = []
m2_jakard_list = []
movie_jakard_list = []
score_jakard_list = []
for row in jakard_df.values:
    m1 = int(row[0])
    m2 = int(row[1])
    score = float(row[2])
    m1_jakard_list.append(m1)
    m2_jakard_list.append(m2)
    movie_jakard_list.append([m1, m2])
    score_jakard_list.append(score)

bert_df = bert_movie[bert_movie['bert_sim'] >= 0.95]

m1_bert_list = []
m2_bert_list = []
movie_bert_list = []
score_bert_list = []
for row in bert_df.values:
    m1 = int(row[0])
    m2 = int(row[1])
    score = float(row[2])
    m1_bert_list.append(m1)
    m2_bert_list.append(m2)
    movie_bert_list.append([m1, m2])
    score_bert_list.append(score * 10)

lda_df = lda_movie[lda_movie['lda_sim'] >= 0.8]

m1_lda_list = []
m2_lda_list = []
score_lda_list = []
movie_lda_list = []
for row in lda_df.values:
    m1 = int(row[0])
    m2 = int(row[1])
    score = float(row[2])
    m1_lda_list.append(m1)
    m2_lda_list.append(m2)
    movie_lda_list.append([m1, m2])
    score_lda_list.append(score)

genre_df = genres_movie[genres_movie['genres_sim'] >= 3.5]
m1_genres_list = []
m2_genres_list = []
movie_genre_list = []
score_genre_list = []
for row in genre_df.values:
    m1 = int(row[0])
    m2 = int(row[1])
    score = float(row[2])
    m1_lda_list.append(m1)
    m2_lda_list.append(m2)
    movie_genre_list.append([m1, m2])
    score_genre_list.append(score)
    
movies_df = pd.read_csv('ml-latest-small/movies.csv')
rate_df = pd.read_csv('ml-latest-small/ratings.csv')

movies_df = pd.read_csv('ml-latest-small/movies.csv')
rate_df = pd.read_csv('ml-latest-small/ratings.csv')

# finding all the titles of movies genres
genres_title_list = []
for row in movies_df['genres'].values:
    genres = str(row).split('|')
    for genre in genres:
        if genre not in genres_title_list:
            genres_title_list.append(genre)

# movies_features, movie_dict = find_genres_columns(movies_df, genres_title_list)
movies_features, movie_dict = create_movie_features(movies_df)

movie_bert_list = np.array(movie_bert_list).T
score_bert_list = np.array(score_bert_list)
movie_bert_list = torch.from_numpy(movie_bert_list)
score_bert_list = torch.from_numpy(score_bert_list)

movie_lda_list = np.array(movie_lda_list).T
score_lda_list = np.array(score_lda_list)
movie_lda_list = torch.from_numpy(movie_lda_list)
score_lda_list = torch.from_numpy(score_lda_list)

movie_genre_list = np.array(movie_genre_list).T
score_genre_list = np.array(score_genre_list)
movie_genre_list = torch.from_numpy(movie_genre_list)
score_genre_list = torch.from_numpy(score_genre_list)

user_ids = rate_df['userId'].unique()

user_dict = {}
count = 0
for id in user_ids:
    if id not in list(user_dict.keys()):
        user_dict[id] = count
        count += 1
rate_list = []
edge_list = []

for row in rate_df.values:
    user = int(row[0])
    user = user_dict[user]
    movie = int(row[1])
    if movie not in list(movie_dict.keys()):
        continue
    movie = movie_dict[movie]
    rate = int(row[2]*2)
    rate_list.append(rate)
    edge_list.append([user, movie])

edge_list = np.array(edge_list)
rate_list = np.array(rate_list)

edge_list = torch.from_numpy(edge_list).T
rate_list = torch.from_numpy(rate_list)

data = HeteroData()
data['user'].num_nodes = len(user_ids)
data['movie'].x = movies_features
data['user', 'rates', 'movie'].edge_index = edge_list
data['user', 'rates', 'movie'].edge_label = rate_list
data['movie', 'edge', 'movie'].edge_index = movie_bert_list
data['movie', 'edge', 'movie'].edge_label = score_bert_list
    
    
data['user'].x = torch.eye(data['user'].num_nodes, device=device)
del data['user'].num_nodes
data = T.ToUndirected()(data)
del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.
data = data.to(device)
# Perform a link-level split into training, validation, and test edges:
train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[('user', 'rates', 'movie')],
    rev_edge_types=[('movie', 'rev_rates', 'user')])(data)

loss_list, model = train_test_model(train_data, val_data, test_data, data=data, lr=0.005, epochs_num=101, print_logs=True)

preds = model(test_data.x_dict, test_data.edge_index_dict,
                    test_data['user', 'movie'].edge_label_index)
preds = preds.cpu().detach().numpy()

# plotting results
plot_losses(loss_list[:100], min_tresh=0.7, max_tresh=1.8)

# statistic results

true_ratings = test_data['user', 'movie'].edge_label.cpu().detach().numpy()
# predicted_ratings = preds
predicted_ratings = np.round(preds)

# Calculate accuracy
accuracy = accuracy_score(true_ratings, predicted_ratings)

# Calculate F1 score (macro-averaged)
f1 = f1_score(true_ratings, predicted_ratings, average='macro')

print(f"Accuracy: {accuracy:0.3f}")
print(f"Macro-averaged F1 Score: {f1:0.3f}")

def ndcg_at_k(ground_truth, predicted_ratings, k):
    # Calculate ideal DCG (Discounted Cumulative Gain) at k
    ideal_dcg = 0.0
    for i in range(k):
        if len(ground_truth) > i and ground_truth[i] > 0:
            ideal_dcg += 1 / np.log2(i + 2)  # Assuming relevance scores are binary (1 or 0)

    # Calculate DCG@k for the predicted rankings
    dcg_at_k = 0.0
    for i in range(min(k, len(predicted_ratings))):
        if predicted_ratings[i] > 0:
            dcg_at_k += predicted_ratings[i] / np.log2(i + 2)

    # Calculate NDCG@k
    ndcg = dcg_at_k / (ideal_dcg + 1e-6)  # Add a small epsilon to avoid division by zero

    return ndcg

ndcg = ndcg_at_k(true_ratings, predicted_ratings, 5)
print(ndcg)
