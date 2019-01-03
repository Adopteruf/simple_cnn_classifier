# Author: Jingwei Guo
# Date: 1-3-2018

from __future__ import division, print_function, absolute_import
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets import vgg
from tensorflow.contrib import slim
import tensorflow as tf
import logging


# resnet_50 model using right ar_scope is necessary
def resnet_50(input_image):
    arg_scope = resnet_v1.resnet_arg_scope()
    with slim.arg_scope(arg_scope):
        features, _ = resnet_v1.resnet_v1_50(input_image)
        # feature flatten
        features = tf.squeeze(features)
    return features


# vgg_16 model using right ar_scope is necessary
def VGG_16(input_image):
    arg_scope = vgg.vgg_arg_scope()
    with slim.arg_scope(arg_scope):
        features, _ = vgg.vgg_16(input_image)
        # feature flatten
        features = tf.reshape(features, shape=[1, -1])
        features = tf.squeeze(features)
    return features


# feature extractor using resnet50
def feature_extractor(images, is_solo_dim):
    features = resnet_50(images)
    # reshape features if len(shape) == 1
    def expand_dim(): return tf.expand_dims(features, axis=0)
    def original(): return features
    features = tf.cond(is_solo_dim, expand_dim, original)
    # normalization
    features_norm = tf.nn.l2_normalize(features, dim=-1)
    cnn_variables = tf.global_variables()

    return features, features_norm, cnn_variables


# cnn classifier
def cnn(is_solo_dim, images, dim_feature, nb_class, weight_init, const_init):
    # cnn-feature_extraction-output
    features, norm_features, cnn_variables = feature_extractor(images, is_solo_dim)
    # FC-layer
    with tf.variable_scope('softmax-layer'):
        w = tf.get_variable(name='ws',
                            shape=[dim_feature, nb_class],
                            initializer=weight_init)
        b = tf.get_variable(name='bs',
                            shape=[nb_class, ],
                            initializer=const_init)
    # build logits
    variables = tf.global_variables()
    logits = tf.matmul(features, w) + b

    return features, norm_features, cnn_variables, variables, logits


# one layer classifier with input feature
def personalized_classifier(features, dim_feature, nb_class, weight_init, const_init):
    # fc-layer
    with tf.variable_scope('personalized-classifier-layer'):
        w = tf.get_variable(name='ws',
                            shape=[dim_feature, nb_class],
                            initializer=weight_init)
        b = tf.get_variable(name='bs',
                            shape=[nb_class, ],
                            initializer=const_init)
    # build logits
    variables = [w, b]
    logits = tf.matmul(features, w) + b

    return variables, logits


class model_classifier(object):
    def __init__(self, model_name, log_net, **kwargs):
        tf.reset_default_graph()

        # set net-flag
        self.model_name = model_name
        self.log_net = log_net

        # set invariant params
        self.input_shape = kwargs.pop("input_shape")
        self.dim_feature = kwargs.pop("dim_feature")
        self.nb_class = kwargs.pop("nb_class")
        self.LAMDA = kwargs.pop("LAMDA")
        self.cost_name = kwargs.pop("cost_name", "cross_entropy")

        # placeholders
        self.is_solo_dim = tf.placeholder(tf.bool, shape=None, name="is_solo_dim")
        self.x = tf.placeholder(tf.float32, shape=[None] + list(self.input_shape), name="x")
        self.y_categorical = tf.placeholder(tf.float32, shape=[None, self.nb_class], name="y_categorical")
        self.batch_norm_fc = tf.placeholder(tf.float32, shape=[None, self.dim_feature], name="batch_norm_fc")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_prob")
        # log all the setting
        if self.log_net:
            logging.info("\r\n" +
                         "Classifier-Net information" + "\r\n" +
                         "model_name: " + self.model_name + "\r\n" +
                         "input_shape: " + str(self.input_shape) + "\r\n" +
                         "dim_feature: " + str(self.dim_feature) + "\r\n" +
                         "nb_class: " + str(self.nb_class) + "\r\n" +
                         "LAMDA: " + str(self.LAMDA) + "\r\n" +
                         "cost_name: " + self.cost_name + "\r\n")

        # build params-initializer
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)

        # build model
        self.features, self.norm_features, self.cnn_vars, self.vars, logits = cnn(is_solo_dim=self.is_solo_dim,
                                                                                  images=self.x,
                                                                                  dim_feature=self.dim_feature,
                                                                                  nb_class=self.nb_class,
                                                                                  weight_init=self.weight_initializer,
                                                                                  const_init=self.const_initializer)
        # build usual cost
        self.cross_entropy, self.center_loss, self.Loss = self._get_cost(logits=logits, cost_name=self.cost_name, kp=self.keep_prob, y_categorical=self.y_categorical)
        # build result
        with tf.name_scope("result"):
            self.predictor = tf.nn.softmax(logits)


    def _get_cost(self, logits, cost_name, kp, y_categorical):
        # dropout layer
        logits_dropout = tf.nn.dropout(logits, keep_prob=kp)
        with tf.name_scope("cost"):
            # cross entropy
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_dropout,
                                                                                   labels=y_categorical))
            # center loss
            center_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.norm_features - self.batch_norm_fc), axis=1))
            # build loss function
            if cost_name == "cross_entropy":
                Loss = cross_entropy
            elif cost_name == "plus_center_loss":
                Loss = cross_entropy + 0.5 * self.LAMDA * center_loss
            else:
                raise ValueError("Unknown cost name: " + cost_name)

        return cross_entropy, center_loss, Loss

    def save(self, sess, vars, model_path):
        saver = tf.train.Saver(vars, max_to_keep=200)
        saver.save(sess, model_path)
        logging.info("Model saved into file: " + model_path)

    def load(self, sess, vars, model_path):
        saver = tf.train.Saver(vars, max_to_keep=200)
        saver.restore(sess, model_path)
        logging.info("Model restored from file: " + model_path)

