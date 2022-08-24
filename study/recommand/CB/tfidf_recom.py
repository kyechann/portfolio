#  tfidf 방식으로 추천 시스템 구현 ###

import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# 코사인 유사도 계산 함수
def cos_sim_matrix(a, b):
    cos_sim = cosine_similarity(a, b)
    result_df = pd.DataFrame(data=cos_sim, index=[a.index])

    return result_df


path = '/home/yc/recommand_system/datasets/movielens'

ratings_df = pd.read_csv(os.path.join(path, 'ratings.csv'), encoding='utf-8')
movies_df = pd.read_csv(os.path.join(path, 'movies.csv'), index_col = 'movieId', encoding='utf-8')
tags_df = pd.read_csv(os.path.join(path, 'tags.csv'), encoding='utf-8')

# 전체적인 개수 및 장르 파악
total_count = len(movies_df.index)
total_genres = list(set([genre for sublist in list(map(lambda x: x.split('|'), movies_df['genres'])) for genre in sublist]))

print(f'전체 영화 수 : {total_count}')
print(f'전체 영화 장르 : {total_genres}')


# 장르 별 개수 파악
genre_count = dict.fromkeys(total_genres)
for each_genre_list in movies_df['genres']:
    for genre in each_genre_list.split('|'):
        if genre_count[genre] == None:
            genre_count[genre] = 1
        else:
            genre_count[genre] = genre_count[genre]+1
            
            
for each_genre in genre_count:
    genre_count[each_genre] = np.log10(total_count/genre_count[each_genre])

# movieId에 맞는 장르 별 가중치 계산
genre_representation = pd.DataFrame(columns=sorted(total_genres), index=movies_df.index)
for index, each_row in tqdm(movies_df.iterrows()):
    dict_temp = {i: genre_count[i] for i in each_row['genres'].split('|')}
    row_to_add = pd.DataFrame(dict_temp, index=[index])
    genre_representation.update(row_to_add)

print('========================장르 관련 처리 완료=========================')

# 유일한 tag값 확인
tag_column = list(map(lambda x: x.split(','), tags_df['tag']))
unique_tags = list(set(list(map(lambda x: x.strip(), list([tag for sublist in tag_column for tag in sublist])))))

total_movie_count = len(set(tags_df['movieId']))
tag_count_dict = dict.fromkeys(unique_tags)

for each_movie_tag_list in tags_df['tag']:
    for tag in each_movie_tag_list.split(","):
        if tag_count_dict[tag.strip()] == None:
            tag_count_dict[tag.strip()] = 1
        else:
            tag_count_dict[tag.strip()] += 1

tag_idf = dict()
for each_tag in tag_count_dict:
    tag_idf[each_tag] = np.log10(total_movie_count / tag_count_dict[each_tag])


# movieId의 tag값의 가중치 계산
tag_representation = pd.DataFrame(columns=sorted(unique_tags), index=list(set(tags_df['movieId'])))
for name, group in tqdm(tags_df.groupby(by='movieId')):
    temp_list = list(map(lambda x: x.split(','), list(group['tag'])))
    temp_tag_list = list(set(list(map(lambda x: x.strip(), list([tag for sublist in temp_list for tag in sublist])))))

    dict_temp = {i: tag_idf[i.strip()] for i in temp_tag_list}
    row_to_add = pd.DataFrame(dict_temp, index=[group['movieId'].values[0]])
    tag_representation.update(row_to_add)

tag_representation = tag_representation.sort_index(0)

print('========================tag 관련 처리 완료=========================')


# genre와 tag represention을 합친다.
# NaN 값은 0으로 치환
movie_representation = pd.concat([genre_representation, tag_representation], axis=1).fillna(0) 
print(movie_representation.shape)


print('========================합치기 완료=========================')


# 유사도 진행
cs_df = cos_sim_matrix(movie_representation, movie_representation)
cs_df.head()

print('========================TF-IDF 추천 시스템 성능 평가=========================')

# 데이터 나누기
train_df, test_df = train_test_split(ratings_df, test_size = 0.2, random_state=1234)

# test에 사용할 userId(train_df에 포함)
test_userids = list(set(test_df.userId.values))

result_df = pd.DataFrame()
for user_id in tqdm(test_userids):
    user_record_df = train_df.loc[train_df.userId == int(user_id), :]
    
    user_sim_df = cs_df.loc[user_record_df['movieId']]  # (n, 9742); n은 userId가 평점을 매긴 영화 수
    user_rating_df = user_record_df[['rating']]  # (n, 1)
    sim_sum = np.sum(user_sim_df.T.to_numpy(), -1)  # (9742, 1)
    # print("user_id=", i, user_record_df.shape, user_sim_df.T.shape, user_rating_df.shape, sim_sum.shape)

    prediction = np.matmul(user_sim_df.T.to_numpy(), user_rating_df.to_numpy()).flatten() / (sim_sum+1) # (9742, 1)

    prediction_df = pd.DataFrame(prediction, index=cs_df.index).reset_index()
    prediction_df.columns = ['movieId', 'pred_rating']    
    prediction_df = prediction_df[['movieId', 'pred_rating']][prediction_df.movieId.isin(test_df[test_df.userId == user_id]['movieId'].values)]

    temp_df = prediction_df.merge(test_df[test_df.userId == user_id], on='movieId')
    result_df = pd.concat([result_df, temp_df], axis=0)


# movieId와 영화 이름을 한번에 보기 위한 작업
movies_df.reset_index(inplace=True)
results_df = pd.merge(result_df, movies_df, how = 'left', on = 'movieId')
results_df.drop(columns = ['genres'], inplace=True)
results_df.columns = ['movieId', 'pred_rating', 'userId', 'rating', 'timestamp', 'title']

print(results_df.head(10))

# RMSE로 성능 평가
mse = mean_squared_error(y_true=results_df['rating'].values, y_pred=results_df['pred_rating'].values)
rmse = np.sqrt(mse)

print(f' 추천 성능 평가 --> mse : {mse}, rmse : {rmse}')