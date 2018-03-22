# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import cv2

#读取整个dict下的所有文件，得到文件路径
def Read(dict):
    data = []
    label = []

    for file in os.listdir(dict):

        data.append(dict + file)
        if 'pp' in dict:
            label.append(0)
        else:
            label.append(1)

    return data, label


def allDataGet():
    p_dict = 'D:\pyworkplace\FaceIdentification\pp\\'
    z_dict = 'D:\pyworkplace\FaceIdentification\zz\\'

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

# def get_image_label(data, label, image_w, image_h) :
#
#     input_queue = tf.train.slice_input_producer([data, label], shuffle=False)
#     image_content = tf.read_file(input_queue[0])
#     image = tf.image.decode_png(image_content, channels=3)
#     image = tf.image.resize_image_with_crop_or_pad(image, image_h, image_w)
#     image = tf.image.per_image_standardization(image)
#
#     return image , input_queue[1]



def get_batch(image, label, image_w, image_h, batch_size, capacity):
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
                                              capacity= capacity)

    image_batch = tf.cast(image_batch, tf.float32)
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch

#
# def image_get_face(filepath):
#     #使用opencv训练好的人脸检测数据集，分辨率高的图片识别率较高
#     face_cascade = cv2.CascadeClassifier(
#         r'D:\pyworkplace\FaceIdentification\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
#
#     image = cv2.imread(filepath)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.15, 5)
#
#     for (x, y, w, h) in faces:
#         gray =  gray[x:x+w, y:y+h]   #返回人脸区域
#         gray = cv2.resize(gray, [64, 64])
#         return tf.Variable(image)
#
#     return None

