from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.optimizers import RMSprop, SGD

import numpy as np

import argparse
import os
from os import path
import time
import matplotlib.image as img
import matplotlib.pyplot as plt
from scipy import misc

from utils_mod import Settings
from utils_mod import ElapsedTimer
from densemapnet import DenseMapNet

from skimage import io

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7 #try various numbers here
set_session(tf.Session(config=config))

DATA_PATH = './dataset/test_image/'
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540
IMAGE_CHAN = 3

def load_images_from_folder(folder):
    all_images = []
    image_list = os.listdir(folder)
    image_list.sort(key=lambda x: int(os.path.splitext(x)[0]))
    for image_path in image_list:
        img = io.imread(os.path.join(folder,image_path))
        idx = np.arange(0, 3, 1)
        img = img[:, :, idx]
        img = img.reshape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHAN])
        all_images.append(img.astype(np.float32)/255)
    return np.array(all_images)

def predict_images(settings, image, filepath):
    size = [image.shape[0], image.shape[1]]
    if settings.otanh:
        image += 1.0
        image =  np.clip(image, 0.0, 2.0)
        image *= (255*0.5)
    else:
        image =  np.clip(image, 0.0, 1.0)
        image *= 255

    image = image.astype(np.uint8)
    image = np.reshape(image, size)
    misc.imsave(filepath, image)
    return image

def setting_with_args():
    parser = argparse.ArgumentParser()
    help_ = "Load checkpoint hdf5 file of model trained weights"
    parser.add_argument("-w",
                        "--weights",
                        help=help_)
    parser.add_argument("-d",
                        "--dataset",
                        help="Name of dataset to load")
    parser.add_argument("-n",
                        "--num_dataset",
                        type=int,
                        help="Number of dataset file splits to load")
    help_ = "No training. Just prediction based on test data. Must load weights."
    parser.add_argument("-p",
                        "--predict",
                        action='store_true',
                        help=help_)
    help_ = "Generate images during prediction. Images are stored images/"
    parser.add_argument("-i",
                        "--images",
                        action='store_true',
                        help=help_)
    help_ = "No training. EPE benchmarking on test set. Must load weights."
    parser.add_argument("-t",
                        "--notrain",
                        action='store_true',
                        help=help_)
    help_ = "Use hyperbolic tan in the output layer"
    parser.add_argument("-o",
                        "--otanh",
                        action='store_true',
                        help=help_)
    help_ = "Best EPE"
    parser.add_argument("-e",
                        "--epe",
                        type=float,
                        help=help_)
    help_ = "No padding"
    parser.add_argument("-a",
                        "--nopadding",
                        action='store_true',
                        help=help_)
    help_ = "Mask images for sparse data"
    parser.add_argument("-m",
                        "--mask",
                        action='store_true',
                        help=help_)
    
    args = parser.parse_args()
    settings = Settings()
    settings.model_weights = args.weights
    settings.dataset = args.dataset
    settings.num_dataset = args.num_dataset
    settings.predict = args.predict
    settings.images = args.images
    settings.notrain = args.notrain
    settings.otanh = args.otanh
    settings.epe = args.epe
    settings.nopadding = args.nopadding
    settings.mask = args.mask
    return settings

if __name__ == '__main__':
    settings = setting_with_args()
    
    #[nsamples, y_dim, x_dim, channel]
    test_lx = load_images_from_folder(DATA_PATH+'left')
    test_rx = load_images_from_folder(DATA_PATH+'right')
    settings.channels = test_lx.shape[3]
    settings.xdim = test_lx.shape[2]
    settings.ydim = test_lx.shape[1]

    densemapnet = DenseMapNet(settings=settings)
    densemapnet_model = densemapnet.build_model()
    i = 0
    indexes = np.arange(i, i + 1)
    left_images = test_lx[indexes, :, :, : ]
    right_images = test_rx[indexes, :, :, : ]
    predicted = densemapnet_model.predict([left_images, right_images])

    plt.subplot(1, 3, 1)
    plt.imshow(left_images[0])
    plt.subplot(1, 3, 2)
    plt.imshow(right_images[0])
    plt.subplot(1, 3, 3)
    predict_int = predict_images(settings, predicted[0], 'predict.png')
    plt.imshow(predict_int)
    plt.show()