from __future__ import division
import cv2
import numpy as np
import math
import os


################################################################
# Datasets have the output image and input image in one files
#       jpeg file [256, 512]
# To make it easier to read into our algorithm and compare
#   we just split the images into their own files
# A = one on left , B = one on the right
################################################################
def long_slice(dataset_path, size):
    A_path = dataset_path + '/A_train_only/'
    B_path = dataset_path + '/B_train_only/'
    train_path = dataset_path + '/train/'
    size+=1
    for i in range(1,size):
        if i %50 == 0:
            print ('Splitting image' + str(i))
        img = cv2.imread(train_path+str(i)+'.jpg')
        imageWidth = 512
        imageHeight = 256
        left = 0
        upper = 0

        crop_img = img[0:256, 0:256]
        crpB = img[0:256,256:512]

        cv2.imwrite(A_path+str(i)+'.jpg',crop_img)
        cv2.imwrite(B_path+str(i)+'.jpg',crpB)






if __name__ == '__main__':
    #slice_size is the max height of the slices in pixels
    long_slice('pix2pix-tensorflow/datasets/cityscapes', 1096)
