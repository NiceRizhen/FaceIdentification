# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf

#input the root path of the pic, return all pics in the folder
def Read(dict):
    data = []
    label = []

    for file in os.listdir(dict):

        data.append(dict + file)
        if 'p' in file:
            label.append(0)
        else:
            label.append(1)

    return data, label


def allDataGet():
    p_dict = 'D:\pyworkplace\FaceIdentification\p\\'
    z_dict = 'D:\pyworkplace\FaceIdentification\z\\'

    p_data, p_label = Read(p_dict)
    z_data, z_label = Read(z_dict)

    data = np.hstack((p_data, z_data))
    label = np.hstack((p_label, z_label))

    temp = np.array([data, label])
    temp = np.transpose(temp)
    np.random.shuffle(temp)

    data = list(temp[:, 0])
    label = list(temp[:, 1])

    return data, label


#输入一张图片，返回其中人脸区域
def img_get_face(img):

def get_batch(image, label, image_w, image_h, batch_size, capicity):
    tf.cast(image, tf.float32)
    tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label], shuffle=False)

    label = input_queue[1]

    image_content = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_content, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, image_w, image_h)
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size = batch_size,
                                              num_threads=32,
                                              capacity= capicity)

    image_batch = tf.cast(image_batch, tf.float32)
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch

