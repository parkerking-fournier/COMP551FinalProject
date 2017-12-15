import pickle
import cv2
import numpy as np
import sys



#############################################################################
# Self explanatory converts all the output images from your
# The output of pix2pix-tf saves the images in a folder which you specifiy
# It saves the input, output, and then the target
# in the form filename-inputs.png, filename-ouputs.png, filename-targets.png
# The training for the last part of our algorithm is the output of pix2pix
# We want to retrieve them and covrt to numpy arrays that we can read in output_image.py
#############################################################################
def imtonp(filepath, outfile):
    im = cv2.imread(filepath)
    print type(im)
    print im.shape
    im.dump(outfile)


def convert_outputs():
    trainingset_size = 400
    dataset_path = 'pix2pix-tensorflow/datasets/facades'
    train_output_path = dataset_path + '/' + 'train_pred'+'/'
    outfile_path = dataset_path + '/oL/'

    for i in range(1,trainingset_size+1):
        imtonp(train_output_path + str(i) + '-outputs.png', outfile_path + str(i) + '.pkl' )



if __name__=='__main__':
    convert_outputs()
