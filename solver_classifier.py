# Author: Jingwei Guo
# Date: 1-3-2018

from __future__ import division, print_function, absolute_import
from util_cla import *
import tensorflow as tf
import numpy as np
import logging
import time
import os


def _load_chromosome_file_name_and_path(img_fp):
    # training images
    img_file_names = os.listdir(img_fp)
    valid_img_file_paths = []
    valid_img_file_names = []
    for fn in img_file_names:
        if not fn.endswith(".tif"):
            continue
        valid_img_file_names.append(fn)
        valid_img_file_paths.append(img_fp + fn)
    # list to array
    valid_img_file_names = np.array(valid_img_file_names)
    valid_img_file_paths = np.array(valid_img_file_paths)
    return valid_img_file_names, valid_img_file_paths


def precision_rate(gt_labels, pred_labels):
    return np.mean(gt_labels == pred_labels)


def losses_visualization(loss_dict, nb_epoch, model_name, pred_folder_path):
    main_name = model_name + "__" + str(nb_epoch) + "_epochs__"
    for cur_name in loss_dict.keys():
        cur_loss = loss_dict[cur_name]
        plot_sole_graph(data=cur_loss,
                        data_name=cur_name,
                        indexes=np.arange(nb_epoch),
                        step=10,
                        title=cur_name,
                        xlabel_name="epoch",
                        ylabel_name="loss_value",
                        saving_filepath=pred_folder_path + main_name + cur_name + ".jpg")


def LRs_visualization(LRs, model_name, pred_folder_path):
    nb_epoch = len(LRs)
    file_name = model_name + "__" + str(nb_epoch) + "_epochs__learning_rate"
    plot_sole_graph(data=LRs,
                    data_name="learning_rate",
                    indexes=np.arange(nb_epoch),
                    step=10,
                    title=file_name,
                    xlabel_name="epoch",
                    ylabel_name="lr_value",
                    saving_filepath=pred_folder_path + file_name + ".jpg")


class solver_classifier(object):
    def __init__(self, model, log_solver, restore, store_test, **kwargs):
        # model
        self.model = model
        self.model_name = self.model.model_name
        self.log_solver = log_solver
        self.restore = restore
        self.store_test = store_test
        self.init_model = kwargs.pop('init_model')
        self.pre_trained_model = kwargs.pop('pre_trained_model')
        self.test_models = kwargs.pop('test_models')

        # prediction
        self.pre_norm_centers_path = kwargs.pop("pre_norm_centers_path")
        self.pred_IDs_path = kwargs.pop("pred_IDs_path")

        # data
        self.train_data_dict = kwargs.pop('train_data_dict')
        self.test_data_dict = kwargs.pop('test_data_dict')

        # learning params
        self.nb_epoch = kwargs.pop('nb_epoch', 100)
        self.batch_size = kwargs.pop('batch_size', 64)
        self.nb_train = len(self.train_data_dict["filepath"])
        self.nb_test = len(self.test_data_dict["filepath"])
        self.nb_train_iter = np.int(np.ceil(np.float(self.nb_train) / np.float(self.batch_size)))
        self.nb_test_iter = np.int(np.ceil(np.float(self.nb_test) / np.float(self.batch_size)))
        self.decay_epochs = kwargs.pop("decay_epochs", 2)
        self.decay_steps = self.decay_epochs * self.nb_train_iter
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.dr = kwargs.pop("dr")
        self.lr = kwargs.pop('lr')
        self.momentum = kwargs.pop("momentum")
        self.center_update_rate = kwargs.pop('center_update_rate')
        self.keep_prob = kwargs.pop("keep_prob")

        # key-path
        self.ckpt_folder_path = kwargs.pop("ckpt_folder_path")
        self.pred_folder_path = kwargs.pop("pred_folder_path")

        # optimizer
        self.op_name = kwargs.pop("op_name", "adam")
        self.lr_node, self.train_op = self._optimizer(lr=self.lr, op_name=self.op_name, Loss=self.model.Loss)


    def _optimizer(self, lr, op_name, Loss):
        """
        Build the training-optimizer
        """
        # create placeholder
        # create training optimizer
        if op_name == "momentum":
            lr_node = tf.train.exponential_decay(learning_rate=lr,
                                                 global_step=self.global_step,
                                                 decay_steps=self.decay_steps,
                                                 decay_rate=self.dr,
                                                 staircase=True)
            train_op = tf.train.MomentumOptimizer(learning_rate=lr_node, momentum=self.momentum
                                                  ).minimize(Loss, global_step=self.global_step)
        elif op_name == "adam":
            lr_node = tf.train.exponential_decay(learning_rate=lr,
                                                 global_step=self.global_step,
                                                 decay_steps=self.decay_steps,
                                                 decay_rate=self.dr,
                                                 staircase=True)
            train_op = tf.train.AdamOptimizer(learning_rate=lr_node).minimize(Loss,
                                                                              global_step=self.global_step)
        elif op_name == "SGD":
            lr_node = tf.train.exponential_decay(learning_rate=lr,
                                                 global_step=self.global_step,
                                                 decay_steps=self.decay_steps,
                                                 decay_rate=self.dr,
                                                 staircase=True)
            train_op = tf.train.GradientDescentOptimizer(learning_rate=lr_node).minimize(Loss,
                                                                                         global_step=self.global_step)
        else:
            raise ValueError("Unknown optimizer name: " + op_name)
        return lr_node, train_op


    def _initializer_train(self):
        # create folders
        if not os.path.exists(self.pred_folder_path):
            logging.info("Allocating " + self.pred_folder_path)
            os.mkdir(self.pred_folder_path)
        if not os.path.exists(self.ckpt_folder_path):
            logging.info("Allocating " + self.ckpt_folder_path)
            os.mkdir(self.ckpt_folder_path)
        # log all the setting
        if self.log_solver:
            logging.info("Solver Information" + "\r\n" +
                         "restore: " + str(self.restore) + "\r\n" +
                         "store_test: " + str(self.store_test) + "\r\n" +
                         "init_model: " + self.init_model + "\r\n" +
                         "pre_trained_model: " + str(self.pre_trained_model) + "\r\n" +
                         "test_models: " + str(self.test_models) + "\r\n" +
                         "pre_norm_centers_path: " + str(self.pre_norm_centers_path) + "\r\n" +
                         "pred_IDs_path: " + str(self.pred_IDs_path) + "\r\n" +
                         "nb_epoch: " + str(self.nb_epoch) + "\r\n" +
                         "batch_size: " + str(self.batch_size) + "\r\n" +
                         "dr: " + str(self.dr) + "\r\n" +
                         "lr: " + str(self.lr) + "\r\n" +
                         "momentum: " + str(self.momentum) + "\r\n" +
                         "center_update_rate: " + str(self.center_update_rate) + "\r\n" +
                         "keep_prob: " + str(self.keep_prob) + "\r\n" +
                         "ckpt_folder_path: " + self.ckpt_folder_path + "\r\n" +
                         "pred_folder_path: " + self.pred_folder_path + "\r\n" +
                         "op_name: " + self.op_name + "\r\n" +
                         "nb_train: " + str(self.nb_train) + "\r\n" +
                         "nb_test: " + str(self.nb_test) + "\r\n" +
                         "nb_train_iter: " + str(self.nb_train_iter) + "\r\n" +
                         "nb_test_iter: " + str(self.nb_test_iter) + "\r\n")


    def _output(self, sess, image_filepaths, nb_image, nb_iter):
        # create output space
        features = np.ndarray(shape=[nb_image, self.model.dim_feature])
        norm_features = np.ndarray(shape=[nb_image, self.model.dim_feature])
        prediction = np.ndarray(shape=[nb_image, self.model.nb_class])
        # start batch-iteration
        start_idx = 0
        for i in np.arange(nb_iter):
            end_idx = start_idx + self.batch_size
            if start_idx >= nb_image:
                break
            if end_idx > nb_image:
                end_idx = nb_image
            cur_indexes = np.arange(start_idx, end_idx)
            feed_dict = {self.model.is_solo_dim: len(cur_indexes) == 1,
                         self.model.x: Load_Data_v1(image_filepaths[cur_indexes])}
            features[cur_indexes], norm_features[cur_indexes], prediction[cur_indexes] = sess.run([self.model.features,
                                                                                                   self.model.norm_features,
                                                                                                   self.model.predictor],
                                                                                                  feed_dict=feed_dict)
            start_idx = end_idx

        return features, norm_features, prediction


    def _cross_entropy_loss_train(self, sess, train_filepaths, gt_categorical_labels, keep_prob):
        # loss record
        losses = np.ndarray(shape=[self.nb_epoch, ], dtype=np.float)
        LRs = np.ndarray(shape=[self.nb_epoch, ], dtype=np.float)
        # epoch training
        for cur_epoch in np.arange(self.nb_epoch):
            # shuffle training-data
            train_filepaths = Shuffle(x=train_filepaths, sd=cur_epoch)
            gt_categorical_labels = Shuffle(x=gt_categorical_labels, sd=cur_epoch)
            # batch-training in one epoch
            cur_loss = 0
            cur_lr = self.lr
            start_idx = 0
            start_time = time.time()
            for cur_step in np.arange(self.nb_train_iter):
                print("iteration: " + str(cur_step + 1) + "/" + str(self.nb_train_iter))
                end_idx = start_idx + self.batch_size
                if start_idx >= self.nb_train:
                    break
                if end_idx > self.nb_train:
                    end_idx = self.nb_train
                cur_indexes = np.arange(start_idx, end_idx)
                feed_dict = {self.model.is_solo_dim: len(cur_indexes) == 1,
                             self.model.x: Load_Data_v1(train_filepaths[cur_indexes]),
                             self.model.y_categorical: gt_categorical_labels[cur_indexes],
                             self.model.keep_prob: keep_prob}
                bl, cur_lr, cur_global_step, _ = sess.run(
                    [self.model.Loss, self.lr_node, self.global_step, self.train_op],
                    feed_dict=feed_dict)
                cur_loss += bl
                print("current global_step: " + str(cur_global_step))
            cur_loss = np.float(cur_loss) / np.float(self.nb_train_iter)
            losses[cur_epoch] = cur_loss
            LRs[cur_epoch] = cur_lr
            # save model
            self.model.save(sess, self.model.vars,
                            self.ckpt_folder_path + self.model_name + "__epoch_" + str(cur_epoch + 1) + ".ckpt")
            logging.info("Epoch " + str(cur_epoch + 1) +
                         ", Learning Rate: " + str(cur_lr) +
                         ", Average Loss: " + str(cur_loss) +
                         ", Cost Time: " + str(time.time() - start_time))
        # save loss
        logging.info("Store losses and Visualize losses")
        save_pickle(losses, self.pred_folder_path + self.model_name + "__losses.pkl")
        save_pickle(LRs, self.pred_folder_path + self.model_name + "__LRs.pkl")
        # visualize epoch-losses
        losses_visualization(loss_dict={"loss": losses},
                             nb_epoch=self.nb_epoch,
                             model_name=self.model_name,
                             pred_folder_path=self.pred_folder_path)
        # learning rate visualization
        LRs_visualization(LRs=LRs, model_name=self.model_name, pred_folder_path=self.pred_folder_path)


    def _plus_center_loss_train(self, sess, restore, train_filepaths, gt_labels, gt_categorical_labels, keep_prob):
        logging.info("Training Initialization")
        if restore:
            norm_centers = load_hickle(self.pre_norm_centers_path)
            pred_IDs = load_pickle(self.pred_IDs_path)
        else:
            # training-initialization
            _, norm_features, _ = self._output(sess=sess,
                                               image_filepaths=train_filepaths,
                                               nb_image=self.nb_train,
                                               nb_iter=self.nb_train_iter)
            norm_centers = np.ndarray(shape=[self.model.nb_class, self.model.dim_feature])
            for i in np.arange(self.model.nb_class):
                norm_centers[i] = np.mean(norm_features[gt_labels == i])
            pred_IDs = gt_labels
        # loss record
        losses = np.ndarray(shape=[self.nb_epoch, ], dtype=np.float)
        softmax_losses = np.ndarray(shape=[self.nb_epoch, ], dtype=np.float)
        center_losses = np.ndarray(shape=[self.nb_epoch, ], dtype=np.float)
        LRs = np.ndarray(shape=[self.nb_epoch, ], dtype=np.float)
        # epoch training
        for cur_epoch in np.arange(self.nb_epoch):
            # shuffle training data
            train_filepaths = Shuffle(x=train_filepaths, sd=cur_epoch)
            gt_labels = Shuffle(x=gt_labels, sd=cur_epoch)
            gt_categorical_labels = Shuffle(x=gt_categorical_labels, sd=cur_epoch)
            # batch-training in one epoch
            cur_loss = 0
            cur_sl = 0
            cur_cl = 0
            cur_lr = self.lr
            start_idx = 0
            start_time = time.time()
            for cur_step in np.arange(self.nb_train_iter):
                print("iteration: " + str(cur_step + 1) + "/" + str(self.nb_train_iter))
                end_idx = start_idx + self.batch_size
                if start_idx >= self.nb_train:
                    break
                if end_idx > self.nb_train:
                    end_idx = self.nb_train
                cur_indexes = np.arange(start_idx, end_idx)
                feed_dict = {self.model.is_solo_dim: len(cur_indexes) == 1,
                             self.model.x: Load_Data_v1(train_filepaths[cur_indexes]),
                             self.model.y_categorical: gt_categorical_labels[cur_indexes],
                             self.model.batch_norm_fc: norm_centers[pred_IDs[cur_indexes]],
                             self.model.keep_prob: keep_prob}
                bl, bsl, bcl, cur_lr, cur_global_step, _ = sess.run([self.model.Loss,
                                                                     self.model.cross_entropy,
                                                                     self.model.center_loss,
                                                                     self.lr_node,
                                                                     self.global_step,
                                                                     self.train_op],
                                                                    feed_dict=feed_dict)
                cur_loss += bl
                cur_sl += bsl
                cur_cl += bcl
                start_idx = end_idx
                print("current global step: " + str(cur_global_step))
            cur_loss = np.float(cur_loss) / np.float(self.nb_train_iter)
            cur_sl = np.float(cur_sl) / np.float(self.nb_train_iter)
            cur_cl = np.float(cur_cl) / np.float(self.nb_train_iter)
            losses[cur_epoch] = cur_loss
            softmax_losses[cur_epoch] = cur_sl
            center_losses[cur_epoch] = cur_cl
            LRs[cur_epoch] = cur_lr
            logging.info("Epoch " + str(cur_epoch + 1) +
                         ", Average Loss: " + str(cur_loss) +
                         ", Cross Entropy: " + str(cur_sl) +
                         ", Center Loss: " + str(cur_cl) +
                         ", Learning Rate: " + str(cur_lr) +
                         ", Cost Time: " + str(time.time() - start_time))
            # save model by epoch
            self.model.save(sess, self.model.vars,
                            self.ckpt_folder_path + self.model_name + "__epoch_" + str(cur_epoch + 1) + ".ckpt")
            logging.info("Center Updating")
            # generate info required by epoch training
            _, norm_features, prediction = self._output(sess=sess,
                                                        image_filepaths=train_filepaths,
                                                        nb_image=self.nb_train,
                                                        nb_iter=self.nb_train_iter)
            # prediction to pred_IDs
            pred_IDs = np.argmax(prediction, axis=1)
            # update norm_centers
            for cur_id in np.unique(pred_IDs):
                norm_centers[cur_id] += self.center_update_rate * (
                        np.mean(norm_features[pred_IDs == cur_id]) - norm_centers[cur_id])
            logging.info("Predicted Result Saving")
            # save the predicted result
            save_pickle(pred_IDs,
                        self.pred_folder_path + self.model_name + "__epoch_" + str(cur_epoch + 1) + "__pred_IDs.pkl")
            save_hickle(norm_centers,
                        self.pred_folder_path + self.model_name + "__epoch_" + str(
                            cur_epoch + 1) + "__norm_centers.hkl")
        logging.info("Store losses and Visualize losses")
        # save losses and LRs after training
        save_pickle(losses, self.pred_folder_path + self.model_name + "__losses.pkl")
        save_pickle(softmax_losses, self.pred_folder_path + self.model_name + "__softmax_losses.pkl")
        save_pickle(center_losses, self.pred_folder_path + self.model_name + "__center_losses.pkl")
        save_pickle(LRs, self.pred_folder_path + self.model_name + "__LRs.pkl")
        # visualize epoch-losses
        losses_visualization(loss_dict={"loss": losses,
                                        "softmax_loss": softmax_losses,
                                        "center_loss": center_losses},
                             nb_epoch=self.nb_epoch,
                             model_name=self.model_name,
                             pred_folder_path=self.pred_folder_path)
        # learning rate visualization
        LRs_visualization(LRs=LRs, model_name=self.model_name, pred_folder_path=self.pred_folder_path)


    def train(self):
        # training initialization including folder creating and log setting
        self._initializer_train()

        # load training data
        train_filepaths = self.train_data_dict["filepath"]
        gt_labels = self.train_data_dict["label"]
        gt_categorical_labels = categorical(gt_labels, self.model.nb_class)

        # config setting
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        # session start
        with tf.Session(config=config) as sess:
            logging.info("Start training session")
            sess.run(tf.global_variables_initializer())
            # restore or not
            if self.restore:
                self.model.load(sess, self.model.vars, self.pre_trained_model)
            else:
                self.model.load(sess, self.model.cnn_vars, self.init_model)
            if self.model.cost_name == "cross_entropy":
                logging.info("Cross Entropy Loss Training")
                self._cross_entropy_loss_train(sess,
                                               train_filepaths,
                                               gt_categorical_labels,
                                               self.keep_prob)
            elif self.model.cost_name == "plus_center_loss":
                logging.info("Plus Center Loss Training")
                self._plus_center_loss_train(sess,
                                             self.restore,
                                             train_filepaths,
                                             gt_labels,
                                             gt_categorical_labels,
                                             self.keep_prob)
            else:
                logging.info("Unknown cost name: " + self.model.cost_name)
                raise ValueError("Unknown cost name: " + self.model.cost_name)


    def test(self):
        # load testing data
        test_filepaths = self.test_data_dict["filepath"]
        gt_labels = self.test_data_dict["label"]
        # config setting
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        # space for APs
        APs = np.ndarray(shape=[len(self.test_models), ], dtype=np.float)
        # testing session start
        with tf.Session(config=config) as sess:
            logging.info("Start testing session")
            sess.run(tf.global_variables_initializer())
            for i, cur_test_model in enumerate(self.test_models):
                self.model.load(sess, self.model.vars, cur_test_model)
                _, _, cur_prediction = self._output(sess=sess,
                                                    image_filepaths=test_filepaths,
                                                    nb_image=self.nb_test,
                                                    nb_iter=self.nb_test_iter)
                # prediction to pred_IDs
                cur_pred_IDs = np.argmax(cur_prediction, axis=1)
                cur_AP = precision_rate(gt_labels, cur_pred_IDs)
                APs[i] = cur_AP
                logging.info("Average Precision of " + cur_test_model + ": " + str(100 * cur_AP))
        # save APs
        logging.info("Testing Result Saving")
        save_pickle(APs, self.pred_folder_path + self.model_name + "__APs.pkl")
        # visualize testing result
        logging.info("Testing Result Visualization")
        plot_sole_graph(data=APs,
                        data_name="epoch_APs",
                        indexes=np.arange(len(APs)),
                        step=10,
                        title=str(self.nb_epoch) + "_epochs__AP",
                        xlabel_name="epoch",
                        ylabel_name="AP",
                        saving_filepath=self.pred_folder_path + self.model_name + "__APs.jpg")

