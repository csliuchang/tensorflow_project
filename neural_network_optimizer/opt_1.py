# coding utf-8

import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
seed = 23455
COST = 9
PROFIT = 1

rng = np.random.RandomState(seed)

X = rng.rand(32, 2)
Y = [[x1+x2+(rng.rand()/10.0-0.05)] for (x1, x2) in X]

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))

y = tf.matmul(x, w1)

loss = tf.reduce_sum(tf.where(tf.greater(y,y_), (y - y_)*COST, (y_ - y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
# 训练模型
    STEPS = 20000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print("the loss after %d training steps is %g" % (i, total_loss))

    print("\n")
    print("w1:\n", sess.run(w1)) 