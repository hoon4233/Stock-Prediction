# setting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

import FinanceDataReader as fdr

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# %matplotlib inline
warnings.filterwarnings('ignore')

# plt.rcParams['font.family'] = 'NanumGothic'



# 기업코드로 주가 정보 가져오기
## 삼성전자 주식코드: 005930
STOCK_CODE = '005930'
stock = fdr.DataReader(STOCK_CODE)

# 데이터 전처리

scaler = MinMaxScaler()
## 스케일을 적용할 column을 정의합니다.
scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
## 스케일 후 columns
scaled = scaler.fit_transform(stock[scale_cols])

df = pd.DataFrame(scaled, columns=scale_cols)
print('End preprocessing !')

# train, test set으로 분할
## random state 인자는 shuffle 시 seed 값을 의미
x_train, x_test, y_train, y_test = train_test_split(df.drop('Close', 1), df['Close'], test_size=0.2, random_state=0, shuffle=False)
## train, test set의 shape 
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
print('End train_test_split !')


# tf를 이용한 시퀀스 데이터셋 구성
def windowed_dataset(series, window_size, batch_size, shuffle):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)

## hyperparameter 
WINDOW_SIZE=20
BATCH_SIZE=32

train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)


# 모델


model = Sequential([
    # 1차원 feature map 생성
    Conv1D(filters=32, kernel_size=5,
           padding="causal",
           activation="relu",
           input_shape=[WINDOW_SIZE, 1]),
    # LSTM
    LSTM(16, activation='tanh'),
    Dense(16, activation="relu"),
    Dense(1),
])

# Sequence 학습에 비교적 좋은 퍼포먼스를 내는 Huber()를 사용합니다.
loss = Huber()
optimizer = Adam(0.0005)
model.compile(loss=Huber(), optimizer=optimizer, metrics=['mse'])

# earlystopping은 10번 epoch통안 val_loss 개선이 없다면 학습을 멈춥니다.
earlystopping = EarlyStopping(monitor='val_loss', patience=10)
# val_loss 기준 체크포인터도 생성합니다.
filename = os.path.join('tmp', 'ckeckpointer.ckpt')
checkpoint = ModelCheckpoint(filename, 
                             save_weights_only=True, 
                             save_best_only=True, 
                             monitor='val_loss', 
                             verbose=1)

history = model.fit(train_data, 
                    validation_data=(test_data), 
                    epochs=50, 
                    callbacks=[checkpoint, earlystopping])

print('End fitting !')

# 모델 사용하기
## weight 불러오기
model.load_weights(filename)
print('Load weights !')
## 예측 시작
pred = model.predict(test_data)
print('End prediction !')



np_y_test = np.asarray(y_test)[20:]
result_sum = 0
# factor = 10000
factor = 1
how_many = len(pred)
print('len_ori : %d, len_pred : %d'%(len(np_y_test), len(pred)))
print("======= difference ======= ")
for i in range(how_many):
    tmp_sum = pred[i]-np_y_test[i]
    # result_sum += int(abs(tmp_sum * factor))
    result_sum += abs(tmp_sum * factor)
    print('idx : %d, pred - ori : %f'%(i+1, tmp_sum ))
print('result_sum : %f, mean(result_sum) : %f'%(result_sum, result_sum / how_many))

# 예측 데이터 시각화
# plt.figure(figsize=(12, 9))
# plt.plot(np.asarray(y_test)[20:], label='actual')
# plt.plot(pred, label='prediction')
# plt.legend()
# plt.show()