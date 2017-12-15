import cv2
import pickle
import numpy as np
import sys

#############################################################################
# Self explanatory converts all the input images to a numpy array
# This numpy array is then taken as input into the compNN algorithm
#############################################################################
def imtonp(filepath):
    im = cv2.imread(filepath)
    return im

def parse_train_encoder(instance):

	#make correct file path for hidden layers
	data = imtonp('pix2pix-tensorflow/datasets/facades/B_train_only/' + str(instance) + '.jpg')
	n = data.shape[0]

    #Convert it in a funky way which is explained in compNN.py
    #The dimensions are [n_next_layer, n_next_layer, 4, 4, filters]
    #This makes it sort through all the filters at this layer and make a nearest neighbor predictions
	neighbors = np.zeros((n/2,n/2,4,4,data.shape[2]))
	data = np.pad(data,((1,1),(1,1), (0,0)),mode='constant')
	for i in range(0,n,2):
		for j in range(0,n,2):
			neighbors[i/2][j/2] = data[i:i+4, j:j+4,:]
	return neighbors


def convert_outputs():
    trainingset_size = 400
    for instance in range(1,trainingset_size+1):
        print 'evaluating ' + str(instance)
        neighbors = parse_train_encoder(instance)
        file = "pix2pix-tensorflow/datasets/facades/internal_outputs/inp/" + str(instance) + ".pkl"
        neighbors.dump(file)

if __name__=='__main__':
    convert_outputs()
