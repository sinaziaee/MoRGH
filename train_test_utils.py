import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fill genres columns
def find_genres_columns(current_genres, genres_title_list):
    temp_list = []
    for genre in genres_title_list:
        if genre in current_genres:
            temp_list.append(1)
        else:
            temp_list.append(0)
    temp_list = np.array(temp_list)
    return temp_list

def create_movie_features(movies_df):
        # find all genres and their names
    genres_title_list = []
    for row in movies_df.values:
        genres = row[2].split("|")
        for genre in genres:
            if genre not in genres_title_list:
                genres_title_list.append(genre)

    # splitting the genres of movies from their string 
    temp = movies_df.copy(deep=True)
    year_list = []
    name_list = []
    genres_list = []
    for row in temp.values:
        id = row[0]
        name = str(row[1]).strip()
        name_inx = str(name).find('(')
        year = name[-5:-1]
        try:
            year = int(year)
        except Exception as e:
            year = -1
        genres = row[2].split('|')
        genres = find_genres_columns(genres, genres_title_list)
        genres_list.append(genres)
        name = name[:name_inx-1].replace('(', '').replace(')', '')
        year_list.append(year)
        name_list.append(name)
    genres_list = np.array(genres_list)
    year_list = np.array(year_list)
    name_list = np.array(name_list)
    temp['title'] = name_list
    temp['year'] = year_list
    temp['genres'] = list(genres_list)
    for i, genre in enumerate(genres_title_list):
        temp[genre] = genres_list[:, i]
    if '(no genres listed)' in list(temp.columns):
        temp = temp[temp['(no genres listed)'] == 0]
        del temp['(no genres listed)']
    temp.head()
    movies_df = temp
    del movies_df['genres']

    # creating dictionaries for movies
    movie_dict = dict()
    movie_name_dict = dict()
    count = 0
    movies_features = []
    for row in movies_df.values:
        movie_dict[row[0]] = count
        movie_name_dict[row[1]] = row[1]
        count+=1
        movies_features.append(np.array(row[3:]))
    # movies_features =np.array(movies_features, dtype=np.int64)
    # movies_features =np.array(movies_features)
    movies_features =np.array(movies_features, dtype=np.float32)
    print(movies_features.shape)
    movies_features = torch.from_numpy(movies_features)

    return movies_features, movie_dict

def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()

def train_test_model(train_data=None, val_data=None, test_data=None, data=None, lr = 0.01, epochs_num = 101, print_logs = True):
    model = Model(hidden_channels=64, data=data).to(device)

    with torch.no_grad():
        model.encoder(train_data.x_dict, train_data.edge_index_dict)

    weight = torch.bincount(train_data['user', 'movie'].edge_label)
    weight = weight.max() / weight

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train():
        model.train()
        optimizer.zero_grad()
        pred = model(train_data.x_dict, train_data.edge_index_dict,
                        train_data['user', 'movie'].edge_label_index)
        target = train_data['user', 'movie'].edge_label
        loss = weighted_mse_loss(pred, target, weight)
        loss.backward()
        optimizer.step()
        return float(loss)


    @torch.no_grad()
    def test(data):
        model.eval()
        pred = model(data.x_dict, data.edge_index_dict,
                    data['user', 'movie'].edge_label_index)
        pred = pred.clamp(min=0, max=10)
        target = data['user', 'movie'].edge_label.float()
        rmse = F.mse_loss(pred / 2, target / 2).sqrt()
        return float(rmse)

    loss_list = []
    for epoch in range(0, epochs_num):
        loss = train()
        train_rmse = test(train_data) 
        val_rmse = test(val_data) 
        test_rmse = test(test_data)
        loss_list.append([train_rmse, val_rmse, test_rmse])
        if epoch % 10 == 0 and print_logs == True:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
    loss_list = np.array(loss_list)
    return loss_list, model

def plot_losses(loss_list, min_tresh, max_tresh):
    plt.figure(figsize=(12, 10))
    plt.plot(loss_list[:, 0], label='train')
    plt.plot(loss_list[:, 1], label='val')
    plt.plot(loss_list[:, 2], label='test')
    plt.legend()
    plt.ylim(min_tresh, max_tresh)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()