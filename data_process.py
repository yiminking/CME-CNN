import os
from skimage.color import rgb2gray
from keras.preprocessing.image import img_to_array, load_img
from datetime import datetime
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from keras import backend as K


def load_data(data_directory, pixels):
    directories = [d for d in sorted(os.listdir(data_directory))  # use sorted otherwise listdir not in order!
                   if os.path.isdir(os.path.join(data_directory, d))]

    # This will be used when split data into chunks (or 80% training and 20% testing data)
    _directories_size = np.shape(directories)[0]
    print("directories/events size: ", _directories_size)

    labels = []
    images = []
    indices = []
    all_img_names = []
    FMT_arr = '%Y-%m-%d %H:%M:%S'
    FMT_cur = '%Y-%m-%d %H_%M_%S'
    for idx, d in enumerate(directories):
        label_directory = os.path.join(data_directory, d)

        # Get arrival time
        txt_name = [os.path.join(label_directory, f)
                    for f in os.listdir(label_directory)
                    if f.endswith(".txt")]
        with open(txt_name[0], 'r') as f:
            arrival_time = f.read()
        arrival_time = arrival_time[:10] + " " + arrival_time[11:19]

        # Load images
        image_names = [f for f in os.listdir(label_directory) if f.endswith(".png")]
        all_img_names.append(image_names)
        for f in image_names:
            image = load_img(os.path.join(label_directory, f), target_size=(pixels, pixels))
            image = img_to_array(image)
            images.append(image)
            current_time = f[3:13] + " " + f[14:22]
            time_interval = datetime.strptime(arrival_time, FMT_arr) - datetime.strptime(current_time, FMT_cur)
            time_interval = time_interval.total_seconds()
            labels.append([time_interval])
            indices.append(idx)

    return np.array(images), labels, indices, _directories_size, all_img_names


def data_preprosess_chunks(pixels, model_name, fold_num, _seed):
    print("model_name: ", model_name)

    data_directory = os.path.join(os.getcwd() + '/data/CME_NN')

    data_x, data_y, data_idx, directories_size, all_img_names = load_data(data_directory, pixels)
    print("original data_x shape: ", np.array(data_x).shape)
    print("original data_y shape: ", np.array(data_y).shape)

    # Collect all images names in time order
    all_img_names = np.array([np.array(xi) for xi in all_img_names])
    all_img_names = np.concatenate(all_img_names)

    # Pre-process data
    if model_name == 'cnn':
        data_x = data_x / 255.
        data_x = rgb2gray(np.array(data_x))
        data_x = data_x.astype('float32')
        print("data_x data type after rgb2gray: ", data_x.dtype)
        data_x = np.expand_dims(data_x, 3)
    elif model_name == 'baseline':
        data_x = data_x / 255.
        data_x = rgb2gray(np.array(data_x))
        data_x = data_x.astype('float32')
        print("data_x data type after rgb2gray: ", data_x.dtype)
        data_x = data_x.reshape(-1, pixels * pixels)

    # Split data into 10 folds, so each chunk have ~10% of the data
    # (1/fold_num) represents 10% of the data for each fold
    data_size = data_x.shape[0]
    print("data size: ", data_size)
    chunk_size = int(np.floor(data_size * (1/fold_num)))-1
    print("Chunk size: ", chunk_size)

    events = np.arange(directories_size)
    random.Random(_seed).shuffle(events)
    print("events: ", events)

    # Split all the 223 events (1122 images) into 5 or 10 chunks, images contained by any one event won't be separated!
    _chunk_num = 1
    # With defaultdict(list), the dictionary my_dict initializes to be an empty list on first access,
    # so I don't have to initialize my_dict[_x_name] with an empty list first before appending to it.
    _my_dict = defaultdict(list)  # dictionary
    shuffled_img_names = []
    for event in events:
        _x_name = 'chunk_x_' + str(_chunk_num)
        _y_name = 'chunk_y_' + str(_chunk_num)
        _idx_name = 'chunk_idx_' + str(_chunk_num)
        for i, e in enumerate(data_idx):
            if e == event:
                _my_dict[_x_name].append(data_x[i])
                _my_dict[_y_name].append(data_y[i])
                _my_dict[_idx_name].append(e)
                shuffled_img_names.append(all_img_names[i])
        if np.shape(_my_dict[_x_name])[0] >= chunk_size:
            _chunk_num += 1
        if _chunk_num > fold_num:
            _chunk_num -= 1
            break

    # This is the actual test data file order for k-fold cross validation
    shuffled_img_names = np.array(shuffled_img_names)
    fname = 'shuffled_img_names.npy'
    if not os.path.isfile(fname):
        np.save(fname, shuffled_img_names)

    return _my_dict, _chunk_num


# Visualize the first conv layer feature maps
def model_to_visualize(model, img_to_visualize, which_image_to_print):
    os.makedirs(os.getcwd() + '/figs_cnn_' + str(which_image_to_print), exist_ok=True)

    layer = model.get_layer(index=0)
    print("layer for model: ", layer.name)
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])

    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)

    n_filter = convolutions.shape[2]
    if n_filter == 32:
        row = 4
        col = 8
    elif n_filter == 64:
        row = 8
        col = 8
    elif n_filter == 128:
        row = 8
        col = 16
    elif n_filter == 256:
        row = 16
        col = 16

    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(col, row))
    for j in range(n_filter):
        ax = fig.add_subplot(row, col, j + 1)
        ax.imshow(convolutions[:, :, j], cmap='gray')
        ax.axis('off')
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.savefig(os.getcwd() + '/figs_cnn_' + str(which_image_to_print) + '/' + layer.name + '.png')


def weights_to_visualize(layer, which_image_to_print):
    print("layer for weights: ", layer.name)
    layer_weights = layer.get_weights()
    layer_weights = np.array(layer_weights)
    print("layer_weights shape: {}, data type: {}".format(layer_weights.shape, layer_weights[0].dtype))
    layer_weights = layer_weights[0]

    layer_shape = layer_weights.shape
    if layer_shape[2] == 1:  # if it is the first conv layer whose format is (x,x,1,x)

        if layer_shape[3] == 32:
            row = 4
            col = 8
        elif layer_shape[3] == 64:
            row = 8
            col = 8
        elif layer_shape[3] == 128:
            row = 8
            col = 16
        elif layer_shape[3] == 256:
            row = 16
            col = 16

        layer_weights = np.squeeze(layer_weights)
        print("layer_weights shape: ", layer_weights.shape)

        # Visualization of each filter of the layer
        fig = plt.figure(figsize=(col, row))
        for i in range(layer_shape[3]):
            ax = fig.add_subplot(row, col, i + 1)
            ax.imshow(layer_weights[:, :, i], cmap='gray', interpolation='none')
            ax.axis('off')
        # fig.suptitle("Weights: " + layer.name)
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig(os.getcwd() + '/figs_cnn_' + str(which_image_to_print) + '/weights_' + layer.name + '.png')
    else:
        print("not first layer_shape: ", layer_shape)
        n = layer_shape[2] * layer_shape[3]
        n = int(np.ceil(np.sqrt(n)))
        print("n: ", n)

        fig = plt.figure(figsize=(24, 15))
        for i in range(layer_shape[3]):
            for j in range(layer_shape[2]):
                ax = fig.add_subplot(n, n, i*32+j+1)
                single_layer_weights = layer_weights[:, :, j, i]
                ax.imshow(single_layer_weights, cmap='gray', interpolation='none')
                ax.axis('off')
        fig.suptitle("Weights: " + layer.name)
        plt.savefig(os.getcwd() + '/figs_cnn_' + str(which_image_to_print) + '/weights_' + layer.name + '.png')
