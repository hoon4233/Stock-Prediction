import os
import pandas as pd
import numpy as np
import tensorflow as tf

import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

HOW_MANY_INFO = 550
scaler = MinMaxScaler()

WINDOW_SIZE=20
BATCH_SIZE=32

TEST_SIZE = 200

FILE_PATH = './model/models'


def load_stock_and_preprocessing(code:str, cols:list):
    global HOW_MANY_INFO, scaler

    stock = fdr.DataReader(code)
    try :
        stock = stock[-HOW_MANY_INFO:]
    except :
        print('Not enough INFO')
        exit(0)

    scaled = scaler.fit_transform(stock[cols])

    df = pd.DataFrame(scaled, columns=cols)

    print('End load_stock_and_preprocessing !\n')

    return df

def windowed_dataset(series, window_size, batch_size, shuffle):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)

def make_data_set(df):
    global TEST_SIZE
    
    print('IN make_train_test_set ')

    train,test = df[:-TEST_SIZE], df[-TEST_SIZE:]

    train_feature, train_label = train['Close'], train['Close']
    test_feature, test_label = test['Close'], test['Close']

    x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2, random_state=0, shuffle=False)

    train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
    valid_data = windowed_dataset(y_valid, WINDOW_SIZE, BATCH_SIZE, False)

    test_data = windowed_dataset(test_label, WINDOW_SIZE, BATCH_SIZE, False)

    for data in train_data.take(1):
        print('train_data X shape( BATCH_SIZE, WINDOW_SIZE, feature ) :',data[0].shape ) 
        print('train_data Y shape( BATCH_SIZE, WINDOW_SIZE, feature ) :',data[1].shape )
    for data in valid_data.take(1):
        print('valid_data X shape( BATCH_SIZE, WINDOW_SIZE, feature ) :',data[0].shape ) 
        print('valid_data Y shape( BATCH_SIZE, WINDOW_SIZE, feature ) :',data[1].shape )

    for data in test_data.take(1):
        print('test_data X shape( BATCH_SIZE, WINDOW_SIZE, feature ) :',data[0].shape ) 
        print('test_data Y shape( BATCH_SIZE, WINDOW_SIZE, feature ) :',data[1].shape )

    print('\nEnd make_train_test_set !\n')

    return train_data, valid_data, test_data

def add_layer():
    global WINDOW_SIZE

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

    return model

def learning(model, code, train_data, valid_data):
    global FILE_PATH
    loss = Huber()
    optimizer = Adam(0.0005)
    model.compile(loss=loss, optimizer=optimizer, metrics=['mse'])
    earlystopping = EarlyStopping(monitor='val_loss', patience=10)
    foldername = FILE_PATH+'/'+code+'/'
    filename = os.path.join(foldername, code+'.ckpt')

    checkpoint = ModelCheckpoint(filename, 
                             save_weights_only=True, 
                             save_best_only=True, 
                             monitor='val_loss', 
                             verbose=1)

    print('IN learning, print model', model) # 디버깅 위해 작성 (add_layer 함수 제대로 동작하나)
    for data in train_data.take(1):
        print('train_data X shape( BATCH_SIZE, WINDOW_SIZE, feature ) :',data[0].shape ) 
        print('train_data Y shape( BATCH_SIZE, WINDOW_SIZE, feature ) :',data[1].shape )
    for data in valid_data.take(1):
        print('valid_data X shape( BATCH_SIZE, WINDOW_SIZE, feature ) :',data[0].shape ) 
        print('valid_data Y shape( BATCH_SIZE, WINDOW_SIZE, feature ) :',data[1].shape )

    history = model.fit(train_data, 
                    validation_data=(valid_data), 
                    epochs=50, 
                    callbacks=[checkpoint, earlystopping])

    print('End learning !')


def make_model(code):
    cols = ['Close']
    df = load_stock_and_preprocessing(code, cols)

    train_data, valid_data, test_data = make_data_set(df)

    model = add_layer()
    print('IN make_model, print model', model) # 디버깅 위해 작성 (add_layer 함수 제대로 동작하나)
    learning(model, code, train_data, valid_data)

    print('End make_model !\n')

    check_bias_variance(code)


def load_model(filename):
    model = add_layer()
    model.load_weights(filename)

    print('End load_moodel !\n')
    return model

def prediction(code, data):
    global FILE_PATH

    foldername = FILE_PATH+'/'+code+'/'
    filename = os.path.join(foldername, code+'.ckpt')

    if os.path.exists(foldername):
        print('There is model')
        model = load_model(filename)
        # pred = scaler.inverse_transform(model.predict(data)) 
        pred = model.predict(data)
        print('End prediction !\n')
        return pred

    else :
        print('There is no model')
        print('Create model and then start predicting')
        make_model(code)

        model = load_model(filename)
        # pred = scaler.inverse_transform(model.predict(data)) 
        pred = model.predict(data)
        print('End prediction !\n')
        return pred

    # model = load_model(filename)
    # # pred = scaler.inverse_transform(model.predict(data)) 
    # pred = model.predict(data)
    # print('End prediction !\n')
    # return pred



def check_performance(ori, pred):
    global WINDOW_SIZE
    # ori = np.asarray(scaler.inverse_transform(ori))[WINDOW_SIZE:]
    ori = np.asarray(ori)[WINDOW_SIZE:]
    result_sum = 0
    factor = 1
    how_many = len(pred)
    benefit = 0
    print('len_ori : %d, len_pred : %d'%(len(ori), len(pred)))
    print("======= difference ======= ")
    for i in range(how_many):
        tmp_sum = pred[i]-ori[i]
        if tmp_sum < 0 : # 실제 값보단 낮지만 내가 가지고 있던 돈보다 큰 액수이면 내가 스스로 팔면 된다 -> 어쨋든 이득
            benefit += 1
        result_sum += abs(tmp_sum * factor)
        print('idx : %d, pred, ori : %f %f'%(i+1, pred[i], ori[i] ))
        print('idx : %d, pred - ori : %f'%(i+1, tmp_sum ))
    print('result_sum : %f, mean(result_sum) : %f'%(result_sum, result_sum / how_many))
    print('benefit : %d'%(benefit))

    print('\n End check_performance !\n')


def check_bias_variance(code):
    global TEST_SIZE

    df = load_stock_and_preprocessing(code, ['Close'])
    train,test = df[:-TEST_SIZE], df[-TEST_SIZE:]
    train_feature, train_label = train['Close'], train['Close']
    test_feature, test_label = test['Close'], test['Close']
    
    train_data = windowed_dataset(train_label, WINDOW_SIZE, BATCH_SIZE, False)
    test_data = windowed_dataset(test_label, WINDOW_SIZE, BATCH_SIZE, False)

    print(code, 'Start check_bias_variance !')

    print('Test about train_set')
    pred = prediction(code, train_data)
    check_performance(train_label, pred)

    print('Test about test_set')
    pred = prediction(code, test_data)
    check_performance(test_label, pred)