#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-convnet.py

import argparse
import os
import tensorflow as tf
import sys

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils import summary
from tensorpack.tfutils.varreplace import remap_variables
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.utils.stats import RatioCounter

from dorefa import get_dorefa
"""
MNIST ConvNet example.
about 0.6% validation error after 30 epochs.
"""

IMAGE_SIZE = 28
# Network parameters
n_hidden_layers = 3
n_units = 4096

BITW = 1
BITA = 1
BITG = 32

class Model(ModelDesc):
    # See tutorial at https://tensorpack.readthedocs.io/tutorial/training-interface.html#with-modeldesc-and-trainconfig
    def inputs(self):
        """
        Define all the inputs (with type, shape, name) that the graph will need.
        """
        return [tf.TensorSpec((None, IMAGE_SIZE, IMAGE_SIZE), tf.float32, 'input'),
                tf.TensorSpec((None,), tf.int32, 'label')]

    def build_graph(self, image, label):
        """This function should build the model which takes the input variables (defined above)
        and return cost at the end."""

        is_training = get_current_tower_context().is_training

        fw, fa, fg = get_dorefa(BITW, BITA, BITG)

        # monkey-patch tf.get_variable to apply fw
        def binarize_weight(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or 'fc0' in name or 'fc_out' in name:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)

        def nonlin(x):
            if BITA == 32:
                return tf.nn.relu(x)
            #FIXMEreturn tf.clip_by_value(x, 0.0, 1.0)
            return tf.clip_by_value(x, -1.0, 1.0)

        def activate(x):
            return fa(nonlin(x))

        # The context manager `argscope` sets the default option for all the layers under
        # this context. Here we use 32 channel convolution with shape 3x3
        # See tutorial at https://tensorpack.readthedocs.io/tutorial/symbolic.html
        with remap_variables(binarize_weight), \
                argscope(FullyConnected, use_bias=False), \
                argscope(BatchNorm, momentum=0.1, epsilon=1e-4):
            # LinearWrap is just a syntax sugar.
            # See tutorial at https://tensorpack.readthedocs.io/tutorial/symbolic.html
            logits = (LinearWrap(image)
                      .Dropout('dropout_in', rate=0.2 if is_training else 0.0)
                      # hidden 0
                      .FullyConnected('fc0', n_units)
                      .BatchNorm('bn0')
                      .apply(activate)
                      .Dropout('dropout_hidden0', rate=0.5 if is_training else 0.0)
                      # hidden 1
                      .FullyConnected('fc1', n_units)
                      .BatchNorm('bn1')
                      .apply(activate)
                      .Dropout('dropout_hidden1', rate=0.5 if is_training else 0.0)
                      # hidden 2
                      .FullyConnected('fc2', n_units)
                      .BatchNorm('bn2')
                      .apply(activate)
                      .Dropout('dropout_hidden2', rate=0.5 if is_training else 0.0)
                      # output layer
                      .FullyConnected('fc_out', 10, activation=tf.identity)())

        # a vector of length B with loss of each sample
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss

        correct = tf.cast(tf.nn.in_top_k(predictions=logits, targets=label, k=1), tf.float32, name='correct')
        accuracy = tf.reduce_mean(correct, name='accuracy')

def get_data():
    # We don't need any fancy data loading for this simple example.
    # See dataflow tutorial at https://tensorpack.readthedocs.io/tutorial/dataflow.html
    train = BatchData(dataset.Mnist('train'), 100)
    test = BatchData(dataset.Mnist('test'), 200, remainder=True)

    train = PrintData(train)

    return train, test

def run(model, sess_init):
    pred_config = PredictConfig(
            model=model,
            session_init=sess_init,
            input_names=['input', 'label'],
            output_names=['correct']
            )
    dataset_train, dataset_test = get_data()
    predictor = SimpleDatasetPredictor(pred_config, dataset_test)
    acc1, acc5 = RatioCounter(), RatioCounter()
    for top1 in predictor.get_result():
        batch_size = top1[0].shape[0]
        acc1.feed(top1[0].sum(), batch_size)

    print("Top1: {}".format(acc1.ratio))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the physical ids of GPUs to use')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--dorefa',
                        help='number of bits for W,A,G, separated by comma. Defaults to \'1,2,4\'',
                        default='1,2,4')
    args = parser.parse_args()

    BITW, BITA, BITG = map(int, args.dorefa.split(','))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    run(Model(), SaverRestore(args.load))
    sys.exit()
