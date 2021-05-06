import os
import pandas as pd
import numpy as np

import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

TEST_SIZE = 200
WINDOW_SIZE = 20
test_feature, test_label = None, None

BATCH_SIZE = 16
FILE_PATH = './model/models'
# FILE_PATH = './models'

scaler = MinMaxScaler()



### 기업코드에 해당하는 주식 가져와서 원하는 cols 만 남기기
# 기업코드는 '숫자'(문자열), cols 는 ['컬럼명1', '컬럼명2'...](리스트)로

def load_stock_and_preprocessing(code, cols):
    global scaler
    stock = fdr.DataReader(code)
    # scaler = MinMaxScaler()
    scaled = scaler.fit_transform(stock[cols])
    df = pd.DataFrame(scaled, columns=cols)
    print('End load_stock_and_preprocessing !\n')

    return df


### train, test dataset 만들기
# TEST_SIZE ==  과거로부터 200일 전 데이터는 학습으로, 200일 후 데이터는 테스트로 사용하겠다.
# WINDOW_SIZE == 며칠치 데이터를 이용해 다음날 데이터를 예측할 것인지

def make_dataset(data, label, window_size):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

def make_train_test_set(df, feature_cols, label_cols):
    global TEST_SIZE, WINDOW_SIZE, test_feature, test_label
    train, test = df[:-TEST_SIZE], df[-TEST_SIZE:]

    train_feature, train_label = train[feature_cols], train[label_cols]
    test_feature, test_label = test[feature_cols], test[label_cols]

    train_feature, train_label = make_dataset(train_feature, train_label, WINDOW_SIZE)
    x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2, random_state=0, shuffle=True)

    test_feature, test_label = make_dataset(test_feature, test_label, WINDOW_SIZE)

    print('IN make_train_test_set ')
    print('x_train shape, x_valid shpae :',x_train.shape, x_valid.shape)
    print('y_train shape, y_valid shpae :',y_train.shape, y_valid.shape)
    print('test_feature shape, test_label shpae :',test_feature.shape, test_label.shape)

    print('\nEnd make_train_test_set !\n')

    return x_train, y_train, x_valid, y_valid


def make_data_for_tomorrow(code):
    global WINDOW_SIZE
    cols = ['Close']  # 일단은 종가만 이용해서 예측할거니까 종가만 받아옴
    df = load_stock_and_preprocessing(code, cols)
    data = np.array([df[-WINDOW_SIZE:]])

    print('data shape :',data.shape)

    print('\nEnd make_data_for_tomorrow !\n')

    return data

    

### 모델 만들기
def add_layer(window_size, num_cols):
    model = Sequential()
    model.add(LSTM(16, 
                input_shape=(window_size, num_cols), 
                activation='relu', 
                return_sequences=False)  # many to many 문제 아니면 보통 false로 한다고 함
            )
    model.add(Dense(1))

    return model

def learning(model, code, train_input, train_output, valid_input, valid_output):
    global BATCH_SIZE, FILE_PATH

    loss = Huber()
    optimizer = Adam(0.0005)
    model.compile(loss=loss, optimizer=optimizer, metrics=['mse'])
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    filename = os.path.join(FILE_PATH, code+'.h5')
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    history = model.fit(train_input, train_output, 
                        epochs=200, 
                        batch_size=BATCH_SIZE,
                        validation_data=(valid_input, valid_output), 
                        callbacks=[early_stop, checkpoint])

    print('End Learning !\n')


def make_model(code):
    global test_feature, test_label
    cols = ['Close']  # 일단은 종가만 이용해서 예측할거니까 종가만 받아옴
    df = load_stock_and_preprocessing(code, cols)

    feature_cols, label_cols = ['Close'], ['Close'] # 일단은 종가만 이용해서 종가르 예측하니까 종가만 남겨줌
    train_input, train_output, valid_input, valid_output = make_train_test_set(df, feature_cols, label_cols)

    how_many_days, num_cols = 20, 1 # 20일 데이터로 예측하니까 20, num_cols는 지금 종가만 input으로 들어가니까
    model = add_layer(how_many_days, num_cols)
    learning(model, code, train_input, train_output, valid_input, valid_output)

    print('End make_moodel !\n')
    check_bias_variance(code)



### 모델 사용하기 
def load_model(filename):
    how_many_days, num_cols = 20, 1
    model = add_layer(how_many_days, num_cols)

    # filename = os.path.join('models', code+'.h5')
    model.load_weights(filename)

    print('End load_moodel !\n')
    return model

def prediction(code, data):
    global test_feature  # 테스트코드에서만 사용
    global FILE_PATH, scaler
    
    filename = os.path.join(FILE_PATH, code+'.h5')
    
    if os.path.exists(filename):
        print('There is model')
        model = load_model(filename)
        # 그냥 predict은 performance 체크할때, 아래 inverse 해주는건 실제 가격으로 바꿔서
        # pred = model.predict(data)
        pred = scaler.inverse_transform(model.predict(data)) 
        print('End prediction !\n')
        return pred

    else :
        print('There is no model')
        print('Create model and then start predicting')
        make_model(code)

        # data = test_feature  # 이줄은 테스트 코드에서만 사용

        model = load_model(filename)
        # pred = model.predict(data)
        pred = scaler.inverse_transform(model.predict(data)) 
        print('End prediction !\n')
        return pred



def check_performance(ori, pred):
    ori = np.asarray(scaler.inverse_transform(ori))
    result_sum = 0
    factor = 1
    how_many = len(pred)
    benefit = 0
    print('len_ori : %d, len_pred : %d'%(len(ori), len(pred)))
    print("======= difference ======= ")
    for i in range(how_many):
        tmp_sum = pred[i]-ori[i]
        if tmp_sum < 0 :
            benefit += 1
        result_sum += abs(tmp_sum * factor)
        print('idx : %d, pred - ori : %f'%(i+1, tmp_sum ))
    print('result_sum : %f, mean(result_sum) : %f'%(result_sum, result_sum / how_many))
    print('benefit : %d'%(benefit))

    print('\n End prediction !\n')

def check_bias_variance(code):
    global TEST_SIZE, WINDOW_SIZE

    df = load_stock_and_preprocessing(code, ['Close'])
    train,test = df[:-TEST_SIZE], df[-TEST_SIZE:]

    feature_cols, label_cols = ['Close'], ['Close']

    train_feature, train_label = train[feature_cols], train[label_cols]
    test_feature, test_label = test[feature_cols], test[label_cols]

    train_feature, train_label = make_dataset(train_feature, train_label, WINDOW_SIZE)
    test_feature, test_label = make_dataset(test_feature, test_label, WINDOW_SIZE)

    print(code, 'Start check_bias_variance !')

    print('Test about train_set')
    pred = prediction(code, train_feature)
    check_performance(train_label, pred)

    print('Test about test_set')
    pred = prediction(code, test_feature)
    check_performance(test_label, pred)

def make_data_for_tomorrow(code):
    global WINDOW_SIZE
    cols = ['Close']  # 일단은 종가만 이용해서 예측할거니까 종가만 받아옴
    df = load_stock_and_preprocessing(code, cols)
    data = np.array([df[-WINDOW_SIZE:]])

    print('data shape :',data.shape)

    print('\nEnd make_data_for_tomorrow !\n')

    return data


# 테스트 코드
if __name__ == '__main__' :
    make_model('005930')