import os
import numpy as np
import keras
import time
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import model_loader
import data_process

pixels = 256
batch_size = 64
epochs = 100
fold_num = 10
num_gpu = 4
model_name = 'cnn'  # 'inceptionResNetV2' or 'cnn' or 'baseline'
seed = 394

my_dict, chunk_num = data_process.data_preprosess_chunks(pixels, model_name, fold_num, seed)
print("my_dict size is: ", len(my_dict))
print("chunk_num: ", chunk_num)

for j in range(1, chunk_num + 1):
    x_name = 'chunk_x_' + str(j)
    y_name = 'chunk_y_' + str(j)
    idx_name = 'chunk_idx_' + str(j)
    print("Shape of the " + str(j) + "th chunk for x:   ", len(my_dict[x_name]))
    print("Shape of the " + str(j) + "th chunk for y:   ", len(my_dict[y_name]))
    print("Shape of the " + str(j) + "th chunk for idx: ", len(my_dict[idx_name]))
print("my_dict.keys()", my_dict.keys())
print("list(my_dict)[0]", list(my_dict)[0])
dict_list = np.arange(0, len(my_dict))
print("dict_list: ", dict_list)

fold_corrcoef = []
fold_mae_test = []
folds_mae_train = []
fold_predicted_seconds = []
fold_test_Y = []

for i in range(chunk_num):
    start_time = time.time()
    print("This is the %d fold: " % i)

    # Assign the ith dict item to testing data
    test_X = np.array(my_dict.get('chunk_x_' + str(i+1)))
    test_Y = np.array(my_dict.get('chunk_y_' + str(i+1)))
    test_idx = np.array(my_dict.get('chunk_idx_' + str(i+1)))
    test_Y_temp = np.concatenate(test_Y)
    test_Y_temp = test_Y_temp[np.where(test_Y_temp!=0)]

    # Assign the remaining dict items to training data
    dict_choice = np.arange(chunk_num)
    rest = dict_choice[(dict_choice != i)]
    print("rest: ", rest)

    train_X = np.concatenate((np.array(my_dict.get('chunk_x_' + str(rest[0] + 1))),
                              np.array(my_dict.get('chunk_x_' + str(rest[1] + 1))),
                              np.array(my_dict.get('chunk_x_' + str(rest[2] + 1))),
                              np.array(my_dict.get('chunk_x_' + str(rest[3] + 1))),
                              np.array(my_dict.get('chunk_x_' + str(rest[4] + 1))),
                              np.array(my_dict.get('chunk_x_' + str(rest[5] + 1))),
                              np.array(my_dict.get('chunk_x_' + str(rest[6] + 1))),
                              np.array(my_dict.get('chunk_x_' + str(rest[7] + 1))),
                              np.array(my_dict.get('chunk_x_' + str(rest[8] + 1)))), axis=0)
    train_Y = np.concatenate((np.array(my_dict.get('chunk_y_' + str(rest[0] + 1))),
                              np.array(my_dict.get('chunk_y_' + str(rest[1] + 1))),
                              np.array(my_dict.get('chunk_y_' + str(rest[2] + 1))),
                              np.array(my_dict.get('chunk_y_' + str(rest[3] + 1))),
                              np.array(my_dict.get('chunk_y_' + str(rest[4] + 1))),
                              np.array(my_dict.get('chunk_y_' + str(rest[5] + 1))),
                              np.array(my_dict.get('chunk_y_' + str(rest[6] + 1))),
                              np.array(my_dict.get('chunk_y_' + str(rest[7] + 1))),
                              np.array(my_dict.get('chunk_y_' + str(rest[8] + 1)))), axis=0)
    train_idx = np.concatenate((np.array(my_dict.get('chunk_idx_' + str(rest[0] + 1))),
                                np.array(my_dict.get('chunk_idx_' + str(rest[1] + 1))),
                                np.array(my_dict.get('chunk_idx_' + str(rest[2] + 1))),
                                np.array(my_dict.get('chunk_idx_' + str(rest[3] + 1))),
                                np.array(my_dict.get('chunk_idx_' + str(rest[4] + 1))),
                                np.array(my_dict.get('chunk_idx_' + str(rest[5] + 1))),
                                np.array(my_dict.get('chunk_idx_' + str(rest[6] + 1))),
                                np.array(my_dict.get('chunk_idx_' + str(rest[7] + 1))),
                                np.array(my_dict.get('chunk_idx_' + str(rest[8] + 1)))), axis=0)
    print("train_X shape: {}, train_Y shape: {}, train_idx shape: {}".format(train_X.shape, train_Y.shape, train_idx.shape))
    print("test_X shape: {}, test_Y shape: {}".format(test_X.shape, test_Y.shape))

    if model_name == 'cnn':
        my_model = model_loader.cnn_model(pixels)
    elif model_name == 'baseline':
        my_model = model_loader.baseline_model(pixels)
    elif model_name == 'inceptionResNetV2':
        my_model = model_loader.inceptionResNetV2(pixels)

        for k, layer in enumerate(my_model.layers):
            print(k, layer.name, layer.trainable)
        my_model.summary()

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional Xception layers
        for layer in my_model.layers[:780]:
            layer.trainable = False
        for layer in my_model.layers[780:]:
            layer.trainable = True

        my_model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['mae'])
        history = my_model.fit(train_X, train_Y, batch_size=batch_size, epochs=20, verbose=1)
        print("Done with top layers training.")

        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        for layer in my_model.layers:
            layer.trainable = True

    if num_gpu > 1:
        my_model_gpu = multi_gpu_model(my_model, gpus=num_gpu)
    elif num_gpu <= 1:
        my_model_gpu = my_model

    my_model_gpu.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['mae'])
    my_model_gpu.summary()

    # checkpoint
    filepath = 'multi_gpu_model_fold' + str(i) + '.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1, save_best_only=True, mode='min')
    callbacks = [checkpoint]

    history = my_model_gpu.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, verbose=1,
                               validation_data=(test_X, test_Y), callbacks=callbacks)

    mae_train = history.history['val_mean_absolute_error']
    folds_mae_train.append(mae_train)

    # Reload the best model
    del my_model
    del my_model_gpu

    my_model_gpu = load_model('multi_gpu_model_fold' + str(i) + '.h5')

    # Extract my_model trained on multi gpu to the single gpu model
    if num_gpu > 1:
        my_model = my_model_gpu.layers[-2]  # get single GPU model
    elif num_gpu <= 1:
        my_model = my_model_gpu
    my_model.save(os.getcwd() + '/single_gpu_model_fold' + str(i) + '.h5')

    my_model_gpu.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['mae'])

    predicted_seconds = my_model_gpu.predict(test_X)
    predicted_seconds_1d = np.concatenate(predicted_seconds)
    test_Y_1d = np.concatenate(test_Y)

    cc = np.corrcoef(predicted_seconds_1d, test_Y_1d)[0, 1]
    mae = my_model_gpu.evaluate(test_X, test_Y)[1] / 3600.

    fold_corrcoef.append(cc)
    fold_mae_test.append(mae)
    fold_predicted_seconds.append(predicted_seconds_1d)
    fold_test_Y.append(test_Y_1d)

    K.clear_session()
    print("--- %s seconds ---" % (time.time() - start_time))

np.save('folds_mae_train.npy', folds_mae_train)
dat = [fold_corrcoef, fold_mae_test, fold_predicted_seconds, fold_test_Y]
np.savez('cme_results.npz', *dat)
