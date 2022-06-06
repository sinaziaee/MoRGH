import numpy as np
import pandas as pd
import torch

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
    movies_features =np.array(movies_features)
    movies_features =np.array(movies_features, dtype=np.float32)
    print(movies_features.shape)
    movies_features = torch.from_numpy(movies_features)

    return movies_features