# Author: Jingwei Guo
# Date: 1-3-2018

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.utils.np_utils import to_categorical
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
import logging
import hickle
import os


# shuffle data
def Shuffle(x, sd):
    np.random.seed(sd)
    np.random.shuffle(x)
    return x


# labels to the categorical form
def categorical(labels, nb_class):
    # standardize the type of labels
    labels = np.array(labels)
    labels = labels.astype(np.int)
    # create categorical space
    categorical_labels = np.zeros(shape=[len(labels), nb_class], dtype=np.int)
    for idx, lb in enumerate(labels):
        categorical_labels[idx, lb] = 1

    return categorical_labels


def create_logging(file_path):
    # start a logging file
    # set log-config for printing into .log
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        filename=file_path,
                        filemode="a")
    # set StreamHandler for printing into screen
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        # logging.info("Allocating folder: " + folder_path)


# load images fitting the input of resNet50
def Load_Data_v1(filenames):
    images = np.ndarray(shape=(len(filenames), 224, 224, 3))
    for k in range(0, len(filenames)):
        img = image.load_img(filenames[k], target_size=[224, 224])
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        img = np.squeeze(img)
        images[k] = img
    return images


def Load_Data_v2(file_paths):
    # create space for saving images
    images = np.ndarray(shape=[len(file_paths), 224, 224, 3])
    for idx, fp in enumerate(file_paths):
        # load img
        img = Image.open(fp)
        # resize img
        resized_img = img.resize((224, 224))
        # img to array
        array_img = np.asarray(resized_img)
        array_img = preprocess_input(array_img)
        images[idx] = array_img
    return images


def save_image(img_array, file_path):
    """
    Writes the image-array to disk
    """
    if len(img_array.shape) != 3:
        img_array = np.expand_dims(img_array, axis=2)
        img_array = np.repeat(img_array, 3, axis=2)
    misc.imsave(file_path, img_array.astype(np.uint8))
    logging.info("Save image to " + file_path)


def save_binary_image(binary_img_array, file_path):
    """
    Writes the binary image-array to disk
    """
    logging.info("Save binary image")
    save_image(255 * np.array(binary_img_array, dtype=np.int), file_path)


# load hickle
def load_hickle(path):
    print('Loaded ' + path + '..')
    return hickle.load(path)


# save data in hickle
def save_hickle(data, path):
    print('Saved ' + path + '..')
    hickle.dump(data, path)


# load pickle
def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print('Loaded %s..' % path)
        return file


# save data in pickle
def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('Saved %s..' % path)


# sole plot-graph
def plot_sole_graph(data, data_name, indexes, step, title, xlabel_name, ylabel_name, saving_filepath):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(indexes + 1, data[indexes], linewidth='2', label=data_name)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    # add extra info
    plt.title(title)
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)
    ax.set_xticks(range(0, len(indexes) + 1, step))
    # show and save
    plt.savefig(saving_filepath)


# multiple plot-graph
def plot_multiple_graph(dict, indexes, step, ordered_key, title, xlabel_name, ylabel_name, saving_filepath):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for each_key in ordered_key:
        cur_mAPs = dict[each_key]
        ax.plot(indexes + 1, cur_mAPs[indexes], linewidth='2', label=each_key)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
    # add extra info
    plt.title(title)
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)
    ax.set_xticks(range(0, len(indexes) + 1, step))
    # show and save
    plt.savefig(saving_filepath)

