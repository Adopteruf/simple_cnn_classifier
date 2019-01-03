# Author: Jingwei Guo
# Date: 1-3-2018

from model_classifier import model_classifier
from solver_classifier import solver_classifier
from util_cla import *
import numpy as np

# set cur model name
model_name = ""
is_trained = True

# set invariant path
pre_trained_cnn_path = ""
cc_data_index_path = ""
cc_result_path = ""

# set results saving path
model_folder_path = cc_result_path + model_name + "/"
ckpt_folder_path = model_folder_path + model_name + "_ckpt/"
pred_folder_path = model_folder_path + model_name + "_pred/"


def _model_initializer():
    # create folders
    create_folder(folder_path=cc_result_path)
    create_folder(folder_path=model_folder_path)
    create_folder(folder_path=ckpt_folder_path)
    create_folder(folder_path=pred_folder_path)
    # create logging
    create_logging(file_path=model_folder_path + model_name + ".log")


def tmp_visualization():
    return 0


def main():
    # initialize
    _model_initializer()

    # load data
    train_data_dict = load_pickle(cc_data_index_path + "train_data_indexes.pkl")
    test_data_dict = load_pickle(cc_data_index_path + "test_data_indexes.pkl")

    # train list to array
    train_data_dict["filepath"] = np.array(train_data_dict["filepath"])
    train_data_dict["label"] = np.array(train_data_dict["label"])
    # test list to array
    test_data_dict["filepath"] = np.array(test_data_dict["filepath"])
    test_data_dict["label"] = np.array(test_data_dict["label"])

    # set input-params
    nb_epoch = 50
    init_model = pre_trained_cnn_path + "resnet_v1_50.ckpt"
    test_models = np.array([ckpt_folder_path + model_name + "__epoch_" + str(i + 1) + ".ckpt" for i in np.arange(nb_epoch)])
    pre_trained_model = ""
    pre_norm_centers_path = ""
    pred_IDs_path = ""


    # build model
    model = model_classifier(model_name=model_name,
                             log_net=False,
                             input_shape=[224, 224, 3],
                             dim_feature=2048,
                             nb_class=24,
                             LAMDA=0.6,
                             cost_name="plus_center_loss")

    # build solver
    solver = solver_classifier(model=model,
                               log_solver=False,
                               restore=False,
                               store_test=True,
                               init_model=init_model,
                               pre_trained_model=pre_trained_model,
                               test_models=test_models,
                               pre_norm_centers_path=pre_norm_centers_path,
                               pred_IDs_path=pred_IDs_path,
                               train_data_dict=train_data_dict,
                               test_data_dict=test_data_dict,
                               nb_epoch=nb_epoch,
                               batch_size=32,
                               decay_epochs=5,
                               dr=0.95,
                               lr=0.003,
                               momentum=0.01,
                               center_update_rate=0.9,
                               keep_prob=0.5,
                               ckpt_folder_path=ckpt_folder_path,
                               pred_folder_path=pred_folder_path,
                               op_name="SGD",
                               personalized_lr = 0.001,
                               personalized_op_name="SGD")


    # normal train and test
    solver.train()
    solver.test()


if __name__ == "__main__":
    main()

