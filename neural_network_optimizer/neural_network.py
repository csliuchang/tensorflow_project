# coding:utf-8
# 两层简单神经网络（全连接）

import tensorflow as tf

x = tf.constant([[0.7, 0.5]])
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))  # stddev 标准差 ，random正态分布随机数，seed随机种子
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("the result is ", sess.run(y), end="")
