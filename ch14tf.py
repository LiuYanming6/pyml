#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:57:55 2018

@author: liuyanming@live.cn
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

g = tf.Graph()

'''1. '''
with g.as_default():
    t1 = tf.constant(np.pi)
    t2 = tf.constant([1, 2, 3, 4])
    t3 = tf.constant([[1, 2], [3, 4]])

    r1 = tf.rank(t1)
    r2 = tf.rank(t2)
    r3 = tf.rank(t3)

    s1 = t1.get_shape()
    s2 = t2.get_shape()
    s3 = t3.get_shape()
    print('Shapes:', s1, s2, s3)

with tf.Session(graph=g) as sess:
    print('Ranks:',
          r1.eval(),    # scalar
          r2.eval(),    # matrix
          r3.eval())

'''2. '''
# add nodes to the graph
with g.as_default():
    a = tf.constant(1, name='a')
    b = tf.constant(2, name='b')
    c = tf.constant(3, name='c')

    z = 2 * (a - b) + c

# launch the graph
with tf.Session(graph=g) as sess:
    print('2 * (a - b) + c =>', sess.run(z))

print('3333333333333')
g = tf.Graph()
## defining placeholders
with g.as_default():
    tf_a = tf.placeholder(tf.int32, shape=[],
                          name='tf_a')
    tf_b = tf.placeholder(tf.int32, shape=[],
                          name='tf_b')
    tf_c = tf.placeholder(tf.int32, shape=[],
                          name='tf_c')
    r1 = tf_a - tf_b
    r2 = 2 * r1
    z = r2 + tf_c
## Feeding placeholders with data
with tf.Session(graph=g) as sess:
    feed = {tf_a: 1,
            tf_b: 10,
            tf_c: 3}
    print('z:',
          sess.run(z, feed_dict=feed))

## lauch the previus graph
with tf.Session(graph=g) as sess:
    feed = {tf_a:1, tf_b:2}
    print('r1:', sess.run(r1,feed_dict=feed))
    print('r2:', sess.run(r2,feed_dict=feed))

    feed = {tf_a: 1,
            tf_b: 2,
            tf_c: 3}
    print('r1:', sess.run(r1,feed_dict=feed))
    print('r2:', sess.run(r2,feed_dict=feed))


print('Defining placeholders for data arrays with varying batchsize')
g = tf.Graph()
with g.as_default():
    tf_x = tf.placeholder(tf.float32, shape=[None, 2], name='tf_x')
    x_mean = tf.reduce_mean(tf_x, axis=0, name='mean')

np.random.seed(0)
np.set_printoptions(precision=2)

with tf.Session(graph=g) as sess:
    x1 = np.random.uniform(low=0, high=1, size=(5, 2))
    print('Feeding data with shape', x1.shape)
    print('Result:', sess.run(x_mean, feed_dict={tf_x: x1}))

    x2 = np.random.uniform(low=1, high=2, size=(10, 2))
    print('Feeding data with shape', x2.shape)
    print('Result:', sess.run(x_mean, feed_dict={tf_x: x2}))





print('Variables in TensorFlow')
g1 = tf.Graph()

## Defining Variables
with g1.as_default():
    w = tf.Variable(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), name='w')
    print(w)

## Initializing variables
with tf.Session(graph=g1) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w))


g2 = tf.Graph()
with g2.as_default():
    w1 = tf.Variable(1, name='w1')
    init_op = tf.global_variables_initializer()
    w2 = tf.Variable(2, name='w2')  # not initialized

with tf.Session(graph=g2) as sess:
    sess.run(init_op)
    print('w1:', sess.run(w1))
    try:
        print('w2:', sess.run(w2))
    except tf.errors.FailedPreconditionError as e:
        print('catch error:', e)




print('Variable scope')
g = tf.Graph()

with g.as_default():
    with tf.variable_scope('net_A'):
        with tf.variable_scope('layer-1'):
            w1 = tf.Variable(tf.random_normal(
                shape=(10,4)), name='weights')#<net_A/layer-1/weights
        with tf.variable_scope('layer-2'):
            w2 = tf.Variable(tf.random_normal(
                shape=(20,10)), name='weights')
    with tf.variable_scope('net_B'):
        with tf.variable_scope('layer-1'):
            w3 = tf.Variable(tf.random_normal(
                shape=(10,4)), name='weights')

    print(w1)
    print(w2)
    print(w3)


print('Reusing variables')
##########################
#### Helper functions ####
##########################
def build_classifier(data, labels, n_classes=2):
    data_shape = data.get_shape().as_list()
    weights = tf.get_variable(name='weights',
                              shape=(data_shape[1], n_classes),
                              dtype=tf.float32)
    bias = tf.get_variable(name='bias',
                           initializer=tf.zeros(shape=n_classes))
    print(weights)
    print(bias)
    logiits = tf.add(tf.matmul(data, weights),
                     bias,
                     name='logiits')
    print(logiits)
    return logiits, tf.nn.softmax(logiits)




