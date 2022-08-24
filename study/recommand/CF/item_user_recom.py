#  이웃기반 협업 필터링 방식으로 추천 시스템 구현 ###

import pandas as pd
import numpy as np
import re
import os
from math import sqrt
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# 코사인 유사도 계산 함수
def cossim_matrix(a, b):
    cossim_values = cosine_similarity(a.values, b.values)
    cossim_df = pd.DataFrame(data=cossim_values, columns = a.index.values, index=a.index)

    return cossim_df

# 성능 평가
def evaluate(test_df, prediction_result_df):
  groups_with_movie_ids = test_df.groupby(by='movieId')
  groups_with_user_ids = test_df.groupby(by='userId')
  intersection_movie_ids = sorted(list(set(list(prediction_result_df.columns)).intersection(set(list(groups_with_movie_ids.indices.keys())))))
  intersection_user_ids = sorted(list(set(list(prediction_result_df.index)).intersection(set(groups_with_user_ids.indices.keys()))))

  print(len(intersection_movie_ids))
  print(len(intersection_user_ids))

  compressed_prediction_df = prediction_result_df.loc[intersection_user_ids][intersection_movie_ids]
  # compressed_prediction_df

  # test_df에 대해서 RMSE 계산
  grouped = test_df.groupby(by='userId')
  result_df = pd.DataFrame(columns=['rmse'])
  for userId, group in tqdm(grouped):
      if userId in intersection_user_ids:
          pred_ratings = compressed_prediction_df.loc[userId][compressed_prediction_df.loc[userId].index.intersection(list(group['movieId'].values))]
          pred_ratings = pred_ratings.to_frame(name='rating').reset_index().rename(columns={'index':'movieId','rating':'pred_rating'})
          actual_ratings = group[['rating', 'movieId']].rename(columns={'rating':'actual_rating'})

          final_df = pd.merge(actual_ratings, pred_ratings, how='inner', on=['movieId'])
          final_df = final_df.round(4) # 반올림

          # if not final_df.empty:
          #     rmse = sqrt(mean_squared_error(final_df['rating_actual'], final_df['rating_pred']))
          #     result_df.loc[userId] = rmse
          #     # print(userId, rmse)
    
  return final_df



# 데이터 불러오기
path = '/home/yc/recommend_system/datasets/movielens'
ratings_df = pd.read_csv(os.path.join(path, 'ratings.csv'), encoding='utf-8')
print(ratings_df.shape)
print(ratings_df.head())

# 데이터 나누기
train_df, test_df = train_test_split(ratings_df, test_size = 0.2, random_state=1234)


print('Sparse Matrix 만들기')
# Sparse Matrix = (User + Movie)
# Sparse Matrix 속에 있는 NaN값을 0으로 해도 될까??(재미 없어서 빵점을 준 것이 아니라 안봐서 평가할 수가 없는 것일 뿐)
sparse_matrix = train_df.groupby('movieId').apply(lambda x: pd.Series(x['rating'].values, index=x['userId'])).unstack()
sparse_matrix.index.name = 'movieId'
print(sparse_matrix)


print('=========Item Based Recommender System=========')
# Item Based 추천 시스템 계산
item_sparse_matrix = sparse_matrix.fillna(0)
item_sparse_matrix.shape
 # 행이 Item, 열이 User

item_cossim_df = cossim_matrix(item_sparse_matrix, item_sparse_matrix)
item_cossim_df

# train_df에 포함된 userId를 계산에 반영한다
userId_grouped = train_df.groupby('userId')
# movieId: 8938개, userId: 610개
item_prediction_result_df = pd.DataFrame(index=list(userId_grouped.indices.keys()), columns=item_sparse_matrix.index)


for userId, group in tqdm(userId_grouped):
    # user가 rating한 movieId * 전체 movieId
    user_sim = item_cossim_df.loc[group['movieId']]
    # user가 rating한 movieId * 1
    user_rating = group['rating']
    # 전체 movieId * 1
    sim_sum = user_sim.sum(axis=0)

    # userId의 전체 rating predictions (8938 * 1)
    pred_ratings = np.matmul(user_sim.T.to_numpy(), user_rating) / (sim_sum+1)
    item_prediction_result_df.loc[userId] = pred_ratings
    
    
print('=========User Based Recommender System=========')
user_sparse_matrix = sparse_matrix.fillna(0).transpose() # 반대로 만들기
user_cossim_df = cossim_matrix(user_sparse_matrix, user_sparse_matrix)
user_cossim_df

movieId_grouped = train_df.groupby('movieId')
user_prediction_result_df = pd.DataFrame(index=list(movieId_grouped.indices.keys()), columns=user_sparse_matrix.index)
user_prediction_result_df

for movieId, group in tqdm(movieId_grouped):
    user_sim = user_cossim_df.loc[group['userId']]
    user_rating = group['rating']
    sim_sum = user_sim.sum(axis=0)

    pred_ratings = np.matmul(user_sim.T.to_numpy(), user_rating) / (sim_sum+1)
    user_prediction_result_df.loc[movieId] = pred_ratings
    
    
# 전체 user가 모든 movieId에 매긴 평점
print(item_prediction_result_df.head())
print(user_prediction_result_df.transpose().head())

user_prediction_result_df = user_prediction_result_df.transpose()


print('===Item, User 성능 비교===')

print('User 기반 성능')
evaluate(test_df, user_prediction_result_df)

print('Item 기반 성능')
evaluate(test_df, item_prediction_result_df)



print('===RMSE 성능으로 비교!!===')
user_result_df = evaluate(test_df, user_prediction_result_df)
print(f'user 기반 성능 : {user_result_df}')
print(f"RMSE: {sqrt(mean_squared_error(user_result_df['actual_rating'].values, user_result_df['pred_rating'].values))}")

item_result_df = evaluate(test_df, item_prediction_result_df)
print(f'item 기반 성능 : {item_result_df}')
print(f"RMSE: {sqrt(mean_squared_error(item_result_df['actual_rating'].values, item_result_df['pred_rating'].values))}")