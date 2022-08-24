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

#SVD로 Matrix Factorization 만들기
def get_svd(s_matrix, k=300):
    u, s, vh = np.linalg.svd(s_matrix.transpose())
    S = s[:k] * np.identity(k, np.float)
    T = u[:,:k]
    Dt = vh[:k,:]
    
    item_factors = np.transpose(np.matmul(S, Dt))
    user_factors = np.transpose(T)
    
    return item_factors, user_factors


def evaluate(test_df, prediction_result_df):
    groups_with_movie_ids = test_df.groupby(by='movieId')
    groups_with_user_ids = test_df.groupby(by='userId')
    intersection_movie_ids = sorted(list(set(list(prediction_result_df.columns)).intersection(set(list(groups_with_movie_ids.indices.keys())))))
    intersection_user_ids = sorted(list(set(list(prediction_result_df.index)).intersection(set(groups_with_user_ids.indices.keys()))))

    print(len(intersection_movie_ids))
    print(len(intersection_user_ids))

    compressed_prediction_df = prediction_result_df.loc[intersection_user_ids][intersection_movie_ids]

    # test_df에 대해서 RMSE 계산
    grouped = test_df.groupby(by='userId')
    rmse_df = pd.DataFrame(columns=['rmse'])
    for userId, group in tqdm(grouped):
        if userId in intersection_user_ids:
            pred_ratings = compressed_prediction_df.loc[userId][compressed_prediction_df.loc[userId].index.intersection(list(group['movieId'].values))]
            pred_ratings = pred_ratings.to_frame(name='rating').reset_index().rename(columns={'index':'movieId','rating':'pred_rating'})
            actual_ratings = group[['rating', 'movieId']].rename(columns={'rating':'actual_rating'})

            final_df = pd.merge(actual_ratings, pred_ratings, how='inner', on=['movieId'])
            final_df = final_df.round(4) # 반올림

            if not final_df.empty:
                rmse = sqrt(mean_squared_error(final_df['actual_rating'], final_df['pred_rating']))
                rmse_df.loc[userId] = rmse

    return final_df, rmse_df

def find_best_k(sparse_matrix, maximum_k=100):
    print("\n최적의 k값 찾기!!")
    k_candidates = np.arange(50, maximum_k, 10)
    final_df = pd.DataFrame(columns=['rmse'], index=k_candidates)
    for k in tqdm(k_candidates):
        item_factors, user_factors = get_svd(sparse_matrix, k)
        each_results_df = pd.DataFrame(np.matmul(item_factors, user_factors),
                                    columns=sparse_matrix.columns.values, index=sparse_matrix.index.values)
        each_results_df = each_results_df.transpose()
        
        result_df, _ = evaluate(test_df, each_results_df)
        each_rmse = sqrt(mean_squared_error(result_df['actual_rating'].values, result_df['pred_rating'].values))

        final_df.loc[k]['rmse'] = each_rmse

    return final_df



# 데이터 불러오기
path = '/home/yc/recommend_system/datasets/movielens'
ratings_df = pd.read_csv(os.path.join(path, 'ratings.csv'), encoding='utf-8')

# 데이터 나누기
train_df, test_df = train_test_split(ratings_df, test_size = 0.2, random_state=1234)


# rating matrix를 만들어야 함
print('Sparse Matrix 만들기')
# Sparse Matrix = (User + Movie)
# Sparse Matrix 속에 있는 NaN값을 0으로 해도 될까??(재미 없어서 빵점을 준 것이 아니라 안봐서 평가할 수가 없는 것일 뿐)
sparse_matrix = train_df.groupby('movieId').apply(lambda x: pd.Series(x['rating'].values, index=x['userId'])).unstack()
sparse_matrix.index.name = 'movieId'
print(sparse_matrix)


# model, item 기반의 차이가 중요 (뭘 쓸지)
sparse_matrix_withmovie = sparse_matrix.apply(lambda x: x.fillna(x.mean()), axis=1)
sparse_matrix_withuser = sparse_matrix.apply(lambda x: x.fillna(x.mean()), axis=0)


# movie로 prediction 만들기
item_factors, user_factors = get_svd(sparse_matrix_withmovie)
prediction_result_df = pd.DataFrame(np.matmul(item_factors, user_factors),
                                    columns = sparse_matrix_withmovie.columns.values, 
                                    index = sparse_matrix_withmovie.index.values)
movie_prediction_result_df = prediction_result_df.transpose()

print(item_factors.shape)
print(user_factors.shape)
print(movie_prediction_result_df.head())


# user로 prediction 만들기
item_factors, user_factors = get_svd(sparse_matrix_withuser)
prediction_result_df = pd.DataFrame(np.matmul(item_factors, user_factors),
                                    columns = sparse_matrix_withuser.columns.values, 
                                    index = sparse_matrix_withuser.index.values)
user_prediction_result_df = prediction_result_df.transpose()

print(item_factors.shape)
print(user_factors.shape)
print(user_prediction_result_df.head())


print('RMSE로 user matrix 성능 검사')
result_df, _ = evaluate(test_df, user_prediction_result_df)
print(result_df)
print("For user matrix")
print(f"RMSE: {sqrt(mean_squared_error(result_df['actual_rating'].values, result_df['pred_rating'].values))}")


print('RMSE로 movie matrix 성능 검사')
result_df, _ = evaluate(test_df, movie_prediction_result_df)
print(result_df)
print("For movie matrix")
print(f"RMSE: {sqrt(mean_squared_error(result_df['actual_rating'].values, result_df['pred_rating'].values))}")


print('============================================================추가 작업============================================================')
print('movie를 대상으로 최적의 k값 찾기 ==> k값 200으로 시작 (시간 오래 걸림)')
res = find_best_k(sparse_matrix_withmovie, 200)
print(res)