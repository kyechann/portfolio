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
from matrix_factorization import BaselineModel, KernelMF, train_update_test_split

# 데이터 불러오기
path = '/home/yc/recommend_system/datasets/movielens'
ratings_df = pd.read_csv(os.path.join(path, 'ratings.csv'), encoding='utf-8')

# 데이터 분할
train_df, test_df = train_test_split(ratings_df, test_size = 0.2, random_state=1234)


# 라이브러리를 만든 사람이 원하는 형태로 바꿔주기
new_train_df = train_df
new_train_df = new_train_df.rename(columns={"userId": "user_id", "movieId": "item_id"})
new_train_df.head()


# dataset -> train valid test -> train valid : 실제 학습할때 , test : 학습된 모델의 성능을 평가할 때 사용
(
    X_train_initial,
    y_train_initial,
    X_train_update,
    y_train_update,
    X_test_update,
    y_test_update,
) = train_update_test_split(new_train_df, frac_new_users=0.2)


# X_train_initial만 훈련
matrix_fact = KernelMF(n_epochs=20, n_factors=100, verbose=1, lr=0.001, reg=0.005)
matrix_fact.fit(X_train_initial, y_train_initial)


# X_trian_update에 있는 새로운 user를 추가해서 더 확인하기
matrix_fact.update_users(
    X_train_update, y_train_update, lr=0.001, n_epochs=20, verbose=1
)

# 예상값, RMSE 성능 확인
pred = matrix_fact.predict(X_test_update)
rmse = mean_squared_error(y_test_update, pred, squared=False)
print(f"\nTest RMSE: {rmse:.4f}")


print('200번 user가 좋아할 만한 것을 추천하는 것')
# Get recommendations
user = 200
items_known = X_train_initial.query("user_id == @user")["item_id"]
matrix_fact.recommend(user=user, items_known=items_known)


# SGD로 구현하기
print('SGD로 구현하기!!!')
baseline_model = BaselineModel(method='sgd', n_epochs = 20, reg = 0.005, lr = 0.01, verbose=1)
baseline_model.fit(X_train_initial, y_train_initial)

pred = baseline_model.predict(X_test_update)
rmse = mean_squared_error(y_test_update, pred, squared = False)

print(f'\nTest RMSE: {rmse:.4f}')
print('   ')
print('   ')

# 추가로 업데이트 진행
print('추가로 업데이트 진행')
baseline_model.update_users(X_train_update, y_train_update, n_epochs=20, lr=0.001, verbose=1)
pred = baseline_model.predict(X_test_update)
rmse = mean_squared_error(y_test_update, pred, squared = False)

print(f'\nTest RMSE: {rmse:.4f}')
print('   ')
print('   ')

# ALS로 구현하기
print('ALS로 구현하기!!!')
baseline_model = BaselineModel(method='als', n_epochs = 20, reg = 0.5, verbose=1)
baseline_model.fit(X_train_initial, y_train_initial)

pred = baseline_model.predict(X_test_update)
rmse = mean_squared_error(y_test_update, pred, squared = False)

print(f'\nTest RMSE: {rmse:.4f}')

