# coding utf-8
# 两层简单神经网络（全连接）

import tensorflow as tf

# 定义输入和参数
# 用 placeholder实现输入定义 （sess.run中喂一组数据）
x = tf.placeholder(tf.float32, shape=[None, 2])
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 通过placeholder占位 feed_dict多组数据进神经网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("the result is: \n", sess.run(y, feed_dict={x:[[0.7, 0.5],[0.2,0.3],[0.3,0.4],[0.4,0.5]]}))
    print(sess.run(w2))
