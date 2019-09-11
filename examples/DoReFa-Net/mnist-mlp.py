#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-convnet.py

import argparse
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils import summary
from tensorpack.tfutils.varreplace import remap_variables
from tensorpack.utils.gpu import get_num_gpu

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
                argscope(BatchNorm, momentum=0.9, epsilon=1e-4):
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

        # This will monitor training error & accuracy (in a moving average fashion). The value will be automatically
        # 1. written to tensosrboard
        # 2. written to stat.json
        # 3. printed after each epoch
        # You can also just call `tf.summary.scalar`. But moving summary has some other benefits.
        # See tutorial at https://tensorpack.readthedocs.io/tutorial/summary.html
        train_error = tf.reduce_mean(1 - correct, name='train_error')
        summary.add_moving_summary(train_error, accuracy)

        # Use a regex to find parameters to apply weight decay.
        # Here we apply a weight decay on all W (weight matrix) of all fc layers
        # If you don't like regex, you can certainly define the cost in any other methods.
        wd_cost = tf.multiply(1e-5,
                              regularize_cost('fc.*/W', tf.nn.l2_loss),
                              name='regularize_loss')
        total_cost = tf.add_n([wd_cost, cost], name='total_cost')
        summary.add_moving_summary(cost, wd_cost, total_cost)

        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary(('.*/W', ['histogram', 'rms']))
        # the function should return the total cost to be optimized
        return total_cost

    def optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=468 * 10,
            decay_rate=0.3, staircase=True, name='learning_rate')
        # This will also put the summary in tensorboard, stat.json and print in terminal,
        # but this time without moving average
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)


def get_data():
    # We don't need any fancy data loading for this simple example.
    # See dataflow tutorial at https://tensorpack.readthedocs.io/tutorial/dataflow.html
    train = BatchData(dataset.Mnist('train'), 128)
    test = BatchData(dataset.Mnist('test'), 256, remainder=True)

    train = PrintData(train)

    return train, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the physical ids of GPUs to use')
    parser.add_argument('--dorefa',
                        help='number of bits for W,A,G, separated by comma. Defaults to \'1,2,4\'',
                        default='1,2,4')
    args = parser.parse_args()

    BITW, BITA, BITG = map(int, args.dorefa.split(','))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # automatically setup the directory train_log/mnist-convnet for logging
    logger.auto_set_dir()

    dataset_train, dataset_test = get_data()

    # How many iterations you want in each epoch.
    # This len(data) is the default value.
    steps_per_epoch = len(dataset_train)

    # get the config which contains everything necessary in a training
    config = TrainConfig(
        model=Model(),
        # The input source for training. FeedInput is slow, this is just for demo purpose.
        # In practice it's best to use QueueInput or others.
        # See tutorial at https://tensorpack.readthedocs.io/tutorial/extend/input-source.html
        data=FeedInput(dataset_train),
        # We use a few simple callbacks in this demo.
        # See tutorial at https://tensorpack.readthedocs.io/tutorial/callback.html
        callbacks=[
            ModelSaver(),   # save the model after every epoch
            InferenceRunner(    # run inference(for validation) after every epoch
                dataset_test,   # the DataFlow instance used for validation
                ScalarStats(    # produce `val_accuracy` and `val_cross_entropy_loss`
                    ['cross_entropy_loss', 'accuracy'], prefix='val')),
            # MaxSaver needs to come after InferenceRunner to obtain its score
            MaxSaver('val_accuracy'),  # save the model with highest accuracy
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=1000,
    )
    # Use a simple trainer in this demo.
    # More trainers with multi-gpu or distributed functionalities are available.
    # See tutorial at https://tensorpack.readthedocs.io/tutorial/trainer.html
    num_gpu = get_num_gpu()
    trainer = SimpleTrainer() if num_gpu <= 1 \
        else SyncMultiGPUTrainerParameterServer(num_gpu)
    launch_train_with_config(config, trainer)
    #FIXMElaunch_train_with_config(config, SyncMultiGPUTrainer(args.gpu))
