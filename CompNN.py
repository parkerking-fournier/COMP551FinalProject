from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle
import cv2

########################################################################
####################HELPER FUNCTIONS USED BY ALL########################
########################################################################

#After each layer we save the output
#They are saved in a special folder called compNN
#Each layer inside of that has its own individual folder
#test examples are given by number
def save_output(test, layer_number,output):
    #input layer
    if layer_number == -1:
        file = "datasets/facades/compNN/inp/" + str(test) + ".pkl"
    else:
        file = "datasets/facades/compNN/hL" + str(layer_number) + "/" + str(test) + ".pkl"
    output.dump(file)

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

##Reads the training data to compute the training data
##Also needed to generate the layer from the information in our output variable from previous layer
def read_next(filters, n, maxes):
    x = np.zeros((train_examples, n, n, filters))

    for instance in range(0, train_examples):
        file = "datasets/facades/internal_outputs/hL" + str(layer) + "/" + str(instance+1) + ".pkl"
    	#open .pkl file
    	with open(file, 'rb') as f:
    		x[instance] = pickle.load(f)

    return x

######Returns a variable x[num_samples][n][n][filters] for all outputs
        #current example training set
#####Also returns the computed layer by our KNN
def get_next_layer(prev_output, layer_number,train_samples, filters):
    n = prev_output.shape[0]
    print prev_output.shape
    #print prev_output[30]
    x = read_next(layer_number, train_samples, filters, n)
    next_layer = np.zeros((n, n, filters))
    print x.shape
    print next_layer.shape
    for i in range(n):
        for j in range(n):
            next_layer[i][j] = x[prev_output[i][j][2]][prev_output[i][j][0]][prev_output[i][j][1]]
    return next_layer, x
########################################################################
#########################END OF HELPER FUNCTIONS########################
########################################################################






########################################################################
####################BEGINNING OF ENCODER FUNCTIONS######################
########################################################################

#####generates a training set based on the input layer
#####This can then be used to predict the next layer
def convert_train_encoder(x, n,filters, train_samples):
    next_dim_squared = n*n/4

    ###make an numpy array to store our training samples and their labels
    x_train = np.zeros((train_samples.shape[0] * next_dim_squared, 16*filters))
    y_train = np.zeros((train_samples.shape[0]* next_dim_sqared, 3))
    #####pos and next_pos store the range which the current instance takes up in the training set####
    # next_pos - pos = n*n
    pos = 0
    next_pos = next_dim_squared
    for i in range(train_samples.shape[0]):
        print "reading training instance: " + str(i + 1) #print so the user can follow along
        x_train[pos:next_pos], y_train[pos:next_pos] = read_dataset_file(x[i], layer_number,train_samples[i], filters,n)
        pos+= n_squared
        next_pos+=n_squared

    return x_train, y_train

#####submethod which converts for just one example in the array
def convert_encoder(x_i, instance, filters,n):
    x_i = np.pad(x_i,((1,1),(1,1), (0,0)),mode='constant')
    converted = np.zeros((n*n/4, 16*filters))
    y_instance = np.zeros((n*n/4, 3), dtype=np.int)
    pos = 0
    for i in range(0,n,2):
        for j in range(0,n,2):
            phi = data[i:i+4, j:j+4,:]
            converted[pos] = x_i[i:i+4, j:j+4,:].reshape(1, 16 *filters)
            y_instance[pos][0] = i
            y_instance[pos][1] = j
            y_instance[pos][2] = instance

            pos+=1

    return converted, y_instance


def compress(prev_output, layer_number,n, filters, train_samples, test_number):
    layer_in, x = get_next_layer(prev_output, layer_number,train_size,filters)
    #parse_train_encoder(x, )
    X_train, y_train = convert_train_encoder(x, n, filters, train_samples)
    x=None

    layer_in = np.pad(layer_in,((1,1),(1,1), (0,0)),mode='constant')
    output= np.zeros((n/2, n/2,3),dtype=np.int)
    neigh = KNeighborsClassifier(n_neighbors=1,n_jobs=2)
    neigh.fit(X_train,y_train)
    reshape_size = 4*4*layer_in.shape[2]
    for i in range(0,n,2):
        print "\tstarting row " + str(i)
        for j in range(0,n,2):
            phi = layer_in[i:i+4, j:j+4,:]
            #print phi.shape
            phi.reshape(1,reshape_size)
            #print phi.shape
            output[i/2][j/2] = neigh.predict(phi.reshape(1,reshape_size))
        print output[i/2]

    save_output(test_number,layer_number,output)
    return output



########################################################################
#########################END OF ENCODER FUNCTIONS#######################
########################################################################



########################################################################
####################     DEALING WITH THE INPUT   ######################
########################################################################
def get_next_layer(prev_output, layer_number,train_samples, filters):
    n = prev_output.shape[0]
    print prev_output.shape
    #print prev_output[30]
    x = read_next(layer_number, train_samples, filters, n)
    next_layer = np.zeros((n, n, filters))
    print x.shape
    print next_layer.shape
    for i in range(n):
        for j in range(n):
            next_layer[i][j] = x[prev_output[i][j][2]][prev_output[i][j][0]][prev_output[i][j][1]]
    return next_layer, x

###get the next layer given the input image
def compress_input(test_image, train_size):
    X_train = np.zeros((train_size * test_image.shape[0]/2 *test_image.shape[0]/2,  4 * 4 * test_image.shape[2]))
    y_train = np.zeros((train_size * test_image.shape[0]/2 *test_image.shape[0]/2,3),dtype=np.int)


    for i in range(train_size):
        retrieve_input(X_train, y_train, i+1)


    n=test_image.shape[0]
    test_image = np.pad(test_image,((1,1),(1,1), (0,0)),mode='constant')
    output= np.zeros((n/2, n/2,3),dtype=np.int)
    neigh = KNeighborsClassifier(n_neighbors=1,n_jobs=2)
    neigh.fit(X_train,y_train)
    reshape_size = 4*4*test_image.shape[2]
    for i in range(0,n,2):
        print "\tstarting row " + str(i)
        for j in range(0,n,2):
            phi = test_image[i:i+4,j:j+4,:]
            output[i/2][j/2] = neigh.predict(phi.reshape(1,reshape_size))
    saveoutput(train_size,-1,output)
    X_train = None
    y_train = None
    return output
########################################################################
####################        END OF INPUT FUNC     ######################
########################################################################


#######################################################################
##################    START OF 1x1 FUNCS     ##########################
#######################################################################
#Getting the nearest encoding here leads for an increased computation time, because we have fewer example images we have to focus on
#The authors wrote that they used a similar metric to decrease computation time as well
#We are not sure if this is the exact way that they calculated, but what we do is take the top 40 and pass through the decoding steps


#This Method reads in the
def read_onebyone(instance, filters, layer=7):
    file = "datasets/facades/internal_outputs/hL" + str(layer) + "/" +str(instance)+ ".pkl"

    with open(file, 'rb') as f:
    	data = pickle.load(f)
    data.reshape(512)
    return data


def generate_onebyone(num_train, filters, layer_number=7):
    x_train = np.zeros((num_train, filters))
    y_train = np.zeros((num_train,1))
    for i in range(0, num_train):
        print "reading_training instance" + str(i)
        x_train[i] = read_onebyone( i+1, filters)
        y_train[i] = np.array(i)
    return x_train, y_train

def read_test(test_sample, layer=7):
    file = "datasets/facades/val_internal_outputs/hL" + str(layer) + "/" +str(test_sample)+ ".pkl"
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data.reshape((1,512))


def OneXOne(size_train, size_split, test_sample layer_number=7):

    X,y=generate_onebyone(size_train, 512)
    print X.shape
    neigh = KNeighborsClassifier(n_neighbors=1,n_jobs=2)
    neigh.fit(X, y)
    test = read_test(test_sample)
    print test.shape
    print y.shape
    ###predicts the probabilities of each
    ###we want to the ge the 40 most likely
    probs = neigh.predict_proba(test)

    maxes = getKBiggest(probs.reshape(size_train),size_split)

    return maxes

####Returns the indicies of the K Biggest elements
def getKBiggest(arr,k):

    maxes = np.argsort(arr)[(arr.shape[0] - k):arr.shape[0]]

    for i in range(k):
        maxes[i] +=1

    return maxes
#######################################################################
######################END OF 1x1 Functions ############################
#######################################################################




#######################################################################
#######################   START OF DECODER FUNCTIONS   ################
#######################################################################
# The same function is used for decoder layers 1-7
# The only differences is that the output of the 7th should be the images
# There exists a separate script which generates the output form the image

#generateTrainData retuns the X and Y training Examples
# X = [train_size * n *n, filter_size * filter_size * filters]
    # where n is the size of each layer
# Y = [train_size * n *n, 3]
    # The three gives the x and y position in the next layer as well as the training example
    # This way we can easily retrieve its value at the next layer
#It iterates through all the training examples of interest and returns in the form x_train and y_train for labels
def generateTrainData(x, n,filters, train_samples):
    n_squared = n*n

    ###make an numpy array to store our training samples and their labels
    x_train = np.zeros((train_samples.shape[0] * n_squared, 4*filters))
    y_train = np.zeros((train_samples.shape[0]* n_squared, 3))
    #####pos and next_pos store the range which the current instance takes up in the training set####
    # next_pos - pos = n*n
    pos = 0
    next_pos = n_squared
    for i in range(train_samples.shape[0]):
        x_train[pos:next_pos], y_train[pos:next_pos] = convert_totrain(x[i],train_samples[i], filters,n)
        pos+= n_squared
        next_pos+=n_squared

    return x_train, y_train



def convert_totrain(x_i, instance, filters,n):
    x_i = np.pad(x_i,((1,1),(1,1), (0,0)),mode='constant')
    converted = np.zeros((n*n, 4*filters))
    y_instance = np.zeros((n*n, 3), dtype=np.int)
    pos = 0
    for i in range(n):
        for j in range(n):
            phi = x_i[i:i+2, j:j+2,:]
            converted[pos] = x_i[i:i+2, j:j+2,:].reshape(1, 4 *filters)
            y_instance[pos][0] = i
            y_instance[pos][1] = j
            y_instance[pos][2] = instance

            pos+=1

    return converted, y_instance

######this is the deconvolution filter applied via compositional nearest neighbors
def decompress(inp_layer, layer, n, filters, train_examples, test_number):
    print "Layer " + str(layer)
    x,input_layer = get_next_layer(inp_layer, filters, n, train_examples)

    ###add padding of 1 around the layer
    input_layer = np.pad(input_layer,((1,1),(1,1), (0,0)),mode='constant')
    #generate our training data
    X_train, y_train = generateTrainData(x, n, filters, train_examples)
    output_layer = np.zeros((n*2, n*2, 3), dtype = np.int)

    ##get the nearest neighbor
    #n_jobs=2 by default because our machine had two processors
    neigh = KNeighborsClassifier(n_neighbors=1,n_jobs=2)
    print "\tfitting the data"
    neigh.fit(X_train,y_train)
    print "\tDone fitting the data"
    ##iterate over the test example with stride 1
    for i in range(n):
        print "\t\tlooking at pixel row " + str(i) #pr
        for j in range(n):
            phi= input_layer[i:i+2, j:j+2,:] ##phi is the filter that is run over the test example
            y_pred = neigh.predict(phi.reshape(1, 2*2*filters)) ###get our predictions
            x = 2*i
            y = 2*j
            output_layer[x][y] = y_pred[0]
            y_pred[0][0] +=1
            output_layer[x+1][y] = y_pred[0]
            y_pred[0][1] += 1
            output_layer[x+1][y+1] = y_pred[0]
            y_pred[0][0] -=1
            output_layer[x][y+1] = y_pred[0]
    save_output(1, 14, output_layer)
    return output_layer
#######################################################################
################   END OF DECODER FUNCTIONS    ########################
#######################################################################


def main():
    inp = compress_input(test_image, 400)
    #encoders
    #compress(prev_output, layer_number,n, filters, train_samples, test_number)
    hidden0 = compress(inp, 0, 128,64, 400, 10)
    hidden1 = compress(hidden0, 1, 64,128,400, 10)
    hidden2 = compress(hidden1, 2, 32,256, 400, 10)
    hidden3 = compress(hidden2, 3, 16,512,400, 10)
    hidden4 = compress(hidden3, 4, 8,512,400, 10)
    hidden5 = compress(hidden4, 5, 4,512,400, 10)
    hidden6 = compress(hidden5, 6, 2,512,400, 10)

    #OneXOne(size_train, size_split, test_sample layer_number=7)
    maxes= = OneXOne(hidden6, 50,10)
    #returns the array of the top scoring functions in reverse order
    #len of that top scoring is size_split
    #from here on out only look at those

    #decoder functions
    #decompress(inp_layer,layer, n, filters, test_number, train_examples)
    hidden8 = decompress(maxes[maxes.shape[0]-1],8, 2, 512, maxes,10)
    hidden9 = decompress(hidden8,9, 4, 512, maxes,10)
    hidden10 = decompress(hidden9,10, 8, 512, maxes,10)
    hidden11 = decompress(hidden10,11, 16, 512, maxes,10)
    hidden12 = decompress(hidden11,12, 32, 512, maxes,10)
    hidden13 = decompress(hidden12,13, 64, 256, maxes,10)
    hidden14 = decompress(hidden13,14, 128, 128, maxes,10)

    ###Note that this program only stores an array of the positions in the training set of the solution
    ###You need to call python output_image.py to get the solution

main()
