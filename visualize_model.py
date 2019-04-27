import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.models import load_model
import data_process

pixels = 1024
fold_num = 10
model_name = 'cnn'  # 'inceptionResNetV2' or 'cnn' or 'baseline'
seed = 394
print("seed: ", seed)
which_image_to_print = 5

my_dict, chunk_num = data_process.data_preprosess_chunks(pixels, model_name, fold_num, seed)
print("my_dict size is: ", len(my_dict))
print("chunk_num: ", chunk_num)

for i in range(9, 10):
    # Assign the ith dict item to testing data
    test_X = np.array(my_dict.get('chunk_x_' + str(i+1)))
    test_Y = np.array(my_dict.get('chunk_y_' + str(i+1)))
    test_idx = np.array(my_dict.get('chunk_idx_' + str(i+1)))
    print("test_X shape", test_X.shape)
    print("test_Y shape", test_Y.shape)
    print("test_idx shape", test_idx.shape)

    # Load the saved model
    my_model = load_model(os.getcwd() + '/single_gpu_model_fold9.h5')

    # Visualize the first layer of model
    img_to_visualize = np.expand_dims(test_X[which_image_to_print], axis=0)
    data_process.model_to_visualize(my_model, img_to_visualize, which_image_to_print)

    # print the original figure
    plt.figure()
    plt.imshow(np.squeeze(test_X[which_image_to_print]), cmap='gray')
    plt.axis('off')
    plt.title("Original figure")
    plt.gca().set_aspect('equal')
    plt.savefig(os.getcwd() + '/figs_cnn_' + str(which_image_to_print) + '/original_figure.png')

    # Visualize 3 conv layers of cnn
    data_process.weights_to_visualize(my_model.get_layer(index=0), which_image_to_print)
    data_process.weights_to_visualize(my_model.get_layer(index=4), which_image_to_print)
    # data_process.weights_to_visualize(my_model.get_layer(index=8), which_image_to_print)