from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle

import pandas as pd
from keras.callbacks import ModelCheckpoint


def time_series_to_window(file_path, window_size=60, window_stride=1):
    "Arguments: -Path to time series csv file"
    "           -Size of sliding window"
    "           -Stride of sliding windows"
    "Returns:   -Array of dimensions:"
    " n          num_windows, window_size, num_features"

    data = pd.read_csv(file_path)
    n_rows, _ = data.shape

    features = ['leftear_x', 'leftear_y', 'rightear_x', 'rightear_y', 'nose_x', 'nose_y', 'lefthand_x', 'lefthand_y',
                'righthand_x', 'righthand_y', ]
    #
    x_features = ['leftear_x', 'rightear_x', 'nose_x', 'lefthand_x', 'righthand_x']
    y_features = ['leftear_y', 'rightear_y', 'nose_y', 'lefthand_y', 'righthand_y']

    x_data = data[x_features]
    data['leftear_x'] = x_data['leftear_x'] - x_data.mean(axis=1)
    data['rightear_x'] = x_data['rightear_x'] - x_data.mean(axis=1)
    data['nose_x'] = x_data['nose_x'] - x_data.mean(axis=1)
    data['lefthand_x'] = x_data['lefthand_x'] - x_data.mean(axis=1)
    data['righthand_x'] = x_data['righthand_x'] - x_data.mean(axis=1)

    y_data = data[y_features]
    data['leftear_y'] = y_data['leftear_y'] - y_data.mean(axis=1)
    data['rightear_y'] = y_data['rightear_y'] - y_data.mean(axis=1)
    data['nose_y'] = y_data['nose_y'] - y_data.mean(axis=1)
    data['lefthand_y'] = y_data['lefthand_y'] - y_data.mean(axis=1)
    data['righthand_y'] = y_data['righthand_y'] - y_data.mean(axis=1)

    stack = np.stack(
        data[features].iloc[i:i + window_size] for i in range(0, n_rows - (window_size - 1), window_stride))

    return stack


def get_all_windows(time_series_path, window_size=60, window_stride=1):
    videos_list = list(pd.read_csv('/home/pn/Desktop/post_deeplabcut/batch1/best_list.csv').iloc[:, 1])

    first_run = True
    for vid in videos_list:
        print(str(vid) + "/" + str(videos_list[-1:]), end="\r")
        if first_run:
            all_windows = time_series_to_window(time_series_path + str(vid) + '.csv', window_size, window_stride)
            first_run = False
            continue
        windows = time_series_to_window(time_series_path + str(vid) + '.csv', window_size, window_stride)
        all_windows = np.concatenate((all_windows, windows), axis=0)

    return all_windows

def train_test_split(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices], data[test_indices]

def normalise(data):
    x_max = data.max()
    x_min = data.min()
    normalised_data = (data - x_min)*1.0/(x_max - x_min)
    return normalised_data



window_size, window_stride = 60, 1

print('Reading data into memory...')
all_windows = get_all_windows('/home/pn/Desktop/post_deeplabcut/batch1/interpolated/', window_size, window_stride)
print('..done... now preprocessing..')

input_img = Input(shape=(window_size, 10, 1))

x_train, x_test = train_test_split(all_windows, 0.1)
x_train = normalise(x_train)
x_test = normalise(x_test)

x_train = np.reshape(x_train, (len(x_train), 60, 10, 1))    # adapt this if using 'channels_first' image data format
x_test = np.reshape(x_test, (len(x_test), 60, 10, 1))       # adapt this if using 'channels_first' image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8), i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 4), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#filepath = "weights.best.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]

print('Trainig model..')
autoencoder.fit(x_train, x_train, epochs=2500, batch_size=128, shuffle=True, validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='conv_autoencoder')], verbose=2)