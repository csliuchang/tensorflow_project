# coding utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward
# from .mnist_backward import MODEL_SAVE_PATH


def restore_model(testPicArr):
    """
    " 创建一个默认图，在改图中执行以下操作"
    args:

        MOVING_AVERAGE_DECAY: 用于控制模型更新的速度，训练过程中会对每一个变量维护一个影子变量，这个影子变量的初始值
                              就是相应变量的初始值，每次变量更新时，影子变量就是随之更新。
        preValue： axis返回每一行最大值的位置索引,得到概率最大的预测值
        variables_to_restore： 通过使用variables_to_restore函数，可以使在加载模型的时候将影子变量直接映射到变量的本身，
                               所以我们在获取变量的滑动平均值的时候只需要获取到变量的本身值而不需要去获取影子变量。
    """
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        MODEL_SAVE_PATH = "./model/"
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1


def pre_pic(picName):
    """
    ANTIALIAS: 抗锯齿
    convert('L')： 变为灰度图
    threshold ： 阈值
    """
    img = Image.open(picName)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                    im_arr[i][j] = 0   # 纯黑色0
            else: im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)

    return img_ready


def application():
    # testNum = input("input the number of test pictures:")
    # for i in range(testNum):
        # testPic = raw_input("the path of test picture:")
        # testPicArr = pre_pic('./1.png')
        # preValue = restore_model(testPicArr)
        # print("The prediction number is", preValue)
        # testPicArr = pre_pic('./2.png')
    preValue = restore_model(pre_pic(raw_input("the path of test picture :")))
    print("The prediction number is ", preValue)


def main():
    application()


if __name__ == "__main__":
    try:
        raw_input  # Python 2
    except NameError:
        raw_input = input  # Python 3
    main()