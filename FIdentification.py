# -*- coding: utf-8 -*-

import cv2
import glob
import tensorflow

# p_dict = 'D:\pyworkplace\FaceIdentification\p'
# z_dict = 'D:\pyworkplace\FaceIdentification\z'
#
# #input the root path of the pic, return all pics in the folder
# def picRead(filepath):
#
#     flag = 0  #to name the img
#     picData = {}
#
#     img_path = glob.glob(filepath + '\*.png')
#
#     for path in img_path:
#         img = cv2.imread(path)
#
#         picData[filepath + str(flag)] = img
#         flag = flag + 1
#
#     return picData

face_cascade = cv2.CascadeClassifier(r'D:\pyworkplace\FaceIdentification\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')

for i in range(15):
    if i is 0:
        continue
    image = cv2.imread('D:\pyworkplace\FaceIdentification\p\p'+str(i)+'.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.15, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (122, 122, 0), 2)

    cv2.imshow("Faces found", image)
    cv2.waitKey(2000)

