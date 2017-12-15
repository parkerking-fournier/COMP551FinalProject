# COMP551 Final Project: A Reproducability Study

This project looked into reproducing a paper for the 2018 ICLR Contest. The group looked to show that Convolutional Neural Nets perform as compositional Nearest Neighbors. They showed very interesting results and we tried to reproduce them albeit uncessfully. If interested you can read more in the paper they submitted to the contest found here. 

    https://openreview.net/pdf/a2fc659e794dc6e3cc0cb0d7598a70c1046fd8a9.pdf
    
The group in the paper looked at multiple datasets, but we due to computationally constraints on our approach we only looked at the Facades dataset which can be found here.

    http://cmp.felk.cvut.cz/~tylecr1/facade/

The group then compared the predictions of two groupdbreaking conditional adverserial Networks (CNN architecture) and did a pixel wise approach to recreating that. The two  that were looked at were pix2pix (Isola et. al) and SegNet (Badrinarayanan et al). We only focused on pix2pix for our recreation. 

The first thing that needs to be done is setting up tensorflow 0.12.1 so that you can run pix2pix algorithm. To do that follow the instructions at

    https://www.tensorflow.org/versions/r0.12/get_started/os_setup

Note that this is the directions for installing the older version of tensorflow 0.12.1. We used pretrained models provided which were done in an older version of tensorflow. Then you need to download a tensorflow based package for pix2pix from the following github link. You can also clone this using git clone with the link.

    https://github.com/affinelayer/pix2pix-tensorflow
    
Additonally this link has the pretrained model for pix2pix. Download the facades_BtoA model which we used to train our network.

Now getting into the bulk of our code. Once you have downloaded the algorithm you can run our edited script to print out the outputs of the various layers for each of the training exampels which will be used by compNN.py. Thus run the following script 

    Python pix2pix_output.py  \
        --mode test \
        --output_dir facades_test \
        --input_dir /path/to/your/dataset \
        --checkpoint /path/to/your/pretrainedmodel/facades_BtoA

Next we are going to do some processing of our data to make sure its ready for the compNN algorithm. 
First the datasets come of the form of a jpeg file [256, 512]. We just split the images into their own files where A = one on left , B = one on the right. Run the script (note you may have to go inside and change the script (sorry)

        python split_train_images.py
        
Then finally to get the test images and training images in the form that you want run 

        python parse_input_image.py
        python parse_output_image.py
 
Then you are ready to run the compNN.py algorithm
Note it will take a long time.

        python compNN.py
 
 Enjoy!
