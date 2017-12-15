from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle
import cv2
import sys

########################################################################
#This file takes in the output of the compNN algorithm and returns the
    # a png file of the output image that was predicted by compNN
########################################################################

##This gets all the output images for the facades dataset
def get_ims():
    x = np.zeros((400, 256, 256, 3))
    trainingset_size = 401
    dataset_path = 'pix2pix-tensorflow/datasets/facades'
    for i in range(1, 401):
        x[i-1] = cv2.imread("datasets/facades/train_pred/" + str(i)+ "-targets.png")
    return x

#Reads in data from a previous output layer if you want to start in the middle of a run
#Note how long it takes so we always want to maintain that option
def load_precomputed(layer, testCase=1):
    if layer == -1:
        file = "datasets/facades/compNN/inp/" + str(testCase) + ".pkl"
    else:
        file = "datasets/facades/compNN/hL" + str(layer) + "/" + str(testCase) + ".pkl"
    with open(file, 'rb') as f:
    	data = pickle.load(f)

    return data

######Returns a variable x[num_samples][n][n][filters] for all outputs
        #current example training set
#####Also returns the computed layer by our KNN
def get_next_layer(prev_output):
    n = 256
    x = get_ims()
    next_layer = np.zeros((n, n, 3))
    print x.shape
    print next_layer.shape
    for i in range(n):
        for j in range(n):
            next_layer[i][j] = x[prev_output[i][j][2]-1][prev_output[i][j][0]][prev_output[i][j][1]]
    return next_layer


def main(argv):
    if (len(argv)!= 2):
        print "The function takes two arguments <test_image> <output>"
        sys.exit()
    im = get_next_layer(load_precomputed(14,int(argv[0])))
    #cv2.show('image', im)
    cv2.imwrite(argv[1],im)


if __name__=='__main__':
    main(sys.argv[1:])
