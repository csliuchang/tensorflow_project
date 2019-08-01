# coding utf-8
import tensorflow as tf
"""
w1: 32位浮点变量，初始值为0.0 这个代码就是不断更新w1参数， 优化w1参数， 滑动平均做了一个w1的影子
global_step: 定义num_updates(NN的迭代轮数）， 初始值为0， 不可被优化（训练）
MOVING_AVERAGE_DECAY: 实例化滑动平均类，给删减率为0.99，当前轮数global_step
"""
w1 = tf.Variable(0, dtype=tf.float32)

global_step = tf.Variable(0, dtype=tf.float32)

MOVING_AVERAGE_DECAY = 0.99

"""
em.apply后的括号里是更新列表，每次运行sess.run(ema_op)时， 对更新列表中的元素求滑动平均值
在实际应用中会使用tf.trainable_variables()自动将所有待训练的参数汇总为列表
ema_op = ema.apply(w1)
"""
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)


ema_op = ema.apply(tf.trainable_variables())

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(tf.assign(w1, 1))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))