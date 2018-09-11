'''
This is the code to implement U-net for training radial k-space image with ground truth image.

Author: Peidong He

Date: 9/5/2018

'''


import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
import math
import matplotlib as plt
import matplotlib.pyplot

tf.reset_default_graph

print('tensorflow version'+tf.__version__)

number_of_samples=20

num_channels = 2
img_size=256
num_classes = img_size * img_size


GTimage = sio.loadmat('//Users/pdhe/Documents/DATA2/y_train.mat')
image2 = sio.loadmat('//Users/pdhe/Documents/DATA2/x_train_image2.mat')

GTimage=GTimage['y_train']
image2=image2['x_trai_i2']

# x_train_ACR = np.zeros([256, 256, number_of_samples],dtype=np.complex_)
# y_train_ACR = np.zeros([256, 256, number_of_samples],dtype=np.complex_)
#
#
# for i in range(120,131):
#     j=0
#     GTimage=sio.loadmat('//Users/pdhe/Documents/DATA/ImRiD_axial'+str(i)+'_channel1_GTimage.mat')
#     image2 = sio.loadmat('//Users/pdhe/Documents/DATA/Image2_ImRiD_axial'+str(i)+'_channel1_image.mat')
#     x_train_ACR[:,:,j]=image2['image2']
#     y_train_ACR[:,:,j]=GTimage['comb_slice']
#     j+=1



#x_train = np.concatenate((x_train_ACR,x_train_ADNI),axis=2)
#y_train = np.concatenate((y_train_ACR,y_train_ADNI),axis=2)
# x_test = np.concatenate((x_train_ACR,x_train_ADNI),axis=2)
# y_test = np.concatenate((y_train_ACR,y_train_ADNI),axis=2)




# only for now, will be changed later

# x_train = x_train_ACR
# y_train = y_train_ACR
# x_test = x_train_ACR
# y_test = y_train_ACR
#
# x_train_real = x_train.real
# x_train_imag = x_train.imag
#
# x_test_real = x_test.real
# x_test_imag = x_test.imag
#
# y_train_real = y_train.real
# y_train_imag = y_train.imag
#
# y_test_real = y_test.real
# y_test_imag = y_test.imag
#
# x_train=np.zeros([x_train_real.shape[0],x_train_real.shape[1],x_train_real.shape[2],2])
# x_test=np.zeros([x_test_imag.shape[0],x_test_imag.shape[1],x_test_imag.shape[2],2])
#
# y_train=np.zeros([x_train_real.shape[0],x_train_real.shape[1],x_train_real.shape[2],2])
# y_test=np.zeros([x_test_imag.shape[0],x_test_imag.shape[1],x_test_imag.shape[2],2])



# x_train = np.stack((image2[:,:,0:200].real,image2[:,:,0:200].imag),axis=3)
# y_train = np.stack((GTimage[:,:,0:200].real,GTimage[:,:,0:200].imag),axis=3)
# x_test = np.stack((image2[:,:,200:].real,image2[:,:,200:].imag),axis=3)
# y_test = np.stack((GTimage[:,:,200:].real,GTimage[:,:,200:].imag),axis=3)

x_train = np.stack((image2[:,:,0:200].real,image2[:,:,0:200].imag),axis=3)
y_train = np.stack((GTimage[:,:,0:200].real,GTimage[:,:,0:200].imag),axis=3)
x_test = np.stack((image2[:,:,200:].real,image2[:,:,200:].imag),axis=3)
y_test = np.stack((GTimage[:,:,200:].real,GTimage[:,:,200:].imag),axis=3)

X_test = x_test.reshape(x_test.shape[2],-1).T
Y_test = y_test.reshape(y_test.shape[2],-1).T
X_train = x_train.reshape(x_train.shape[2],-1).T
Y_train = y_train.reshape(y_train.shape[2],-1).T

print("Shape of the X_train", X_train.shape)
print("Shape of the Y_train", Y_train.shape)

print("Shape of the X_test", X_test.shape)
print("Shape of the Y_test", Y_test.shape)

def new_weights(shape, name): # function used to initialize weights in each layer of network
    return tf.get_variable(name, shape,dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())# Returns an initializer
                                                                    # performing "Xavier" initialization for weights.

def new_biases(length): # function used to initialize biases in each layer of network
    return tf.Variable(tf.constant(0.05,dtype=tf.float32, shape=[length]),name='B')


def new_conv_layer(input,  # The previous layer.
                   num_input_channels,  # Num. channels in prev. layer.
                   filter_size,  # Width and height of each filter.
                   num_filters,  # Number of filters.
                   use_pooling=True, name='W'):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters] # for example [5,5,1,64] if 5x5 filter size and
                                                                        # 64 filters

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape, name=name) # function new_weights is defined above

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # tensorflow function to define a 2d convolution layer.
    layer = tf.nn.conv2d(input=input, #dimensions?
                         filter=weights,
                         strides=[1, 1, 1, 1], # one pixel at a time when convolve
                         padding='SAME')

    # Add the biases to the results of the convolution. ???
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer, # 4D tensor
                               ksize=[1, 2, 2, 1], # size of the window[image_number, x-axis, y-axis, input-channel]
                               strides=[1, 2, 2, 1], # steps of moving the window
                               padding='SAME') #smallest possible padding to achieve the desired output size.

    # Batch Normalization. of the layer???
    cnn_bn = tf.contrib.layers.batch_norm(layer, # input
                                          data_format='NHWC', #The normalization is over all but the last dimension if
                                          # data_format is NHWC and the second dimension if data_format is NCHW.
                                          # Matching the "cnn" tensor which has shape (?, 480, 640, 128).
                                          center=True, # Add offset of beta to normalized tensor
                                          scale=True, # if true multiply by gamma
                                          is_training=True) # use moving mean and moving variance if in training

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(cnn_bn)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    return layer, weights


def new_deconv_layer(input,  # The previous layer.
                     num_output_channels,  # Num. channels in prev. layer.
                     filter_size,# Width and height of each filter. (5x5)
                     img_size, # out shape image dimensions
                     num_filters,  # Number of filters as input has. (64)
                     stride, name='W'):  # Use 2x2 max-pooling.


    shape = [filter_size, filter_size, num_output_channels, num_filters] # 3x3x1x64
    # shape = [filter_size, filter_size, num_filters, num_output_channels] # 3x3x1x64

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape, name=name) # why we need a name here ? for the tensorboard visualization
    print('weights shape',weights.shape)
    # Create new biases, one for each filter.
    biases = new_biases(length=num_output_channels) # shape[3]
    outShape = tf.stack([tf.shape(x_input)[0], img_size, img_size, num_output_channels])#[? 64 64 1]
    print('outshape',outShape)
    # there is no x_image being defined in this particular function.
    layer = tf.nn.conv2d_transpose(input, # value
                                   filter=weights, # filter has the same dimension as the value
                                   strides=stride,#[1, 2, 2, 1],
                                   output_shape=outShape,
                                   padding='SAME')
    # typically call transpose as deconvolution in NN community, but they are actually different
    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases
    layer.set_shape([None, img_size, img_size, num_output_channels]) #[? 64 64 1]
    print("layer shape", layer.shape)
    # Use pooling to down-sample the image resolution?

    # Batch Normalization.
    cnn_bn = tf.contrib.layers.batch_norm(layer,
                                          data_format='NHWC', # NCHW faster than NHWC in GPU
                                          #NHWC (N, Height, width, channel) is the TensorFlow default and NCHW is
                                          # the optimal format to use for NVIDIA cuDNN.
                                          # Matching the "cnn" tensor which has shape (?, 480, 640, 128).
                                          center=True, #
                                          scale=True, # if next layer is relu, scale is not needed, it will be done by next layer
                                          is_training=True)

    layer = tf.nn.relu(cnn_bn)

    return layer, weights

def maxpooling(layer,name):
    outlayer = tf.nn.max_pool(value=layer,  # 4D tensor
                                  ksize=[1, 2, 2, 1],  # size of the window[image_number, x-axis, y-axis, input-channel]
                                  strides=[1, 2, 2, 1],  # steps of moving the window
                                  padding='SAME', name=name)
    return outlayer


y_true = tf.placeholder(tf.float32, shape=[None, num_classes*num_channels], name='y_true')

x = tf.placeholder(tf.float32, shape=[None, num_classes*num_channels ], name='x') # 256*256*2 basically

x_input = tf.reshape(x,[-1, img_size,img_size,num_channels])
# layer 1 - two convolution
conv_layer11, layer_weight11 = new_conv_layer(input=x_input,
                                           num_input_channels=2,
                                           filter_size=3,
                                           num_filters=64,
                                           use_pooling=False,
                                           name='conv_layer1_1')

conv_layer12, layer_weight12 = new_conv_layer(input=conv_layer11,
                                           num_input_channels=64,
                                           filter_size=3,
                                           num_filters=64,
                                           use_pooling=False,
                                           name='conv_layer1_2')

# max pooling
conv_layer13 = maxpooling(conv_layer12,name='conv_layer1_3')

# layer 2 - two convolution
conv_layer21, layer_weight21 = new_conv_layer(input=conv_layer13,
                                           num_input_channels=64,
                                           filter_size=3,
                                           num_filters=128,
                                           use_pooling=False,
                                           name='conv_layer2_1')

conv_layer22, layer_weight22 = new_conv_layer(input=conv_layer21,
                                           num_input_channels=128,
                                           filter_size=3,
                                           num_filters=128,
                                           use_pooling=False,
                                           name='conv_layer2_2')

# max pooling

conv_layer23 = maxpooling(conv_layer22,name='conv_layer2_3')


# layer 3 - two convolution
conv_layer31, layer_weight31 = new_conv_layer(input=conv_layer23,
                                           num_input_channels=128,
                                           filter_size=3,
                                           num_filters=256,
                                           use_pooling=False,
                                           name='conv_layer3_1')

conv_layer32, layer_weight32 = new_conv_layer(input=conv_layer31,
                                           num_input_channels=256,
                                           filter_size=3,
                                           num_filters=256,
                                           use_pooling=False,
                                           name='conv_layer3_2')


# max pooling

conv_layer33 = maxpooling(conv_layer32,name='conv_layer3_3')

# layer 4 - two convolution
conv_layer41, layer_weight41 = new_conv_layer(input=conv_layer33,
                                           num_input_channels=256,
                                           filter_size=3,
                                           num_filters=512,
                                           use_pooling=False,
                                           name='conv_layer4_1')

conv_layer42, layer_weight42 = new_conv_layer(input=conv_layer41,
                                           num_input_channels=512,
                                           filter_size=3,
                                           num_filters=512,
                                           use_pooling=False,
                                           name='conv_layer4_2')

# max pooling

conv_layer43 = maxpooling(conv_layer42,name='conv_layer4_3')

# layer 5 - two convolution
conv_layer51, layer_weight51 = new_conv_layer(input=conv_layer43,
                                           num_input_channels=512,
                                           filter_size=3,
                                           num_filters=1024,
                                           use_pooling=False,
                                           name='conv_layer5_1')

conv_layer52, layer_weight52 = new_conv_layer(input=conv_layer51,
                                           num_input_channels=1024,
                                           filter_size=3,
                                           num_filters=1024,
                                           use_pooling=False,
                                           name='conv_layer5_2')
# up-conv
deconv_layer1, deconv_weight1=new_deconv_layer(input=conv_layer52,
                                               img_size=32,
                                               num_output_channels=512,
                                               filter_size=3, # or 3?
                                               num_filters=1024,
                                               stride=[1,2,2,1],
                                               name='deconv_layer_1')
# combine tensor from end of layer 4
conv_layer61=tf.concat([conv_layer42,deconv_layer1],axis=3,name='Tensor6_1')

# layer 6 - two convolution
conv_layer62, layer_weight62 = new_conv_layer(input=conv_layer61,
                                           num_input_channels=1024,
                                           filter_size=3,
                                           num_filters=512,
                                           use_pooling=False,
                                           name='conv_layer6_2')

conv_layer63, layer_weight63 = new_conv_layer(input=conv_layer62,
                                           num_input_channels=512,
                                           filter_size=3,
                                           num_filters=512,
                                           use_pooling=False,
                                           name='conv_layer6_3')
# up-conv
deconv_layer2, deconv_weight2=new_deconv_layer(input=conv_layer63,
                                               img_size=64,
                                               num_output_channels=256,
                                               filter_size=3,
                                               num_filters=512,
                                               stride=[1,2,2,1],
                                               name='deconv_layer_2')
# combine tensor from the end of layer3
conv_layer71=tf.concat([conv_layer32,deconv_layer2],axis=3,name='Tensor_7_1')

# layer 7 - two convolution
conv_layer72, layer_weight72 = new_conv_layer(input=conv_layer71,
                                           num_input_channels=512,
                                           filter_size=3,
                                           num_filters=256,
                                           use_pooling=False,
                                           name='conv_layer7_2')

conv_layer73, layer_weight73 = new_conv_layer(input=conv_layer72,
                                           num_input_channels=256,
                                           filter_size=3,
                                           num_filters=256,
                                           use_pooling=False,
                                           name='conv_layer7_3')
# up-conv
deconv_layer3, deconv_weight3=new_deconv_layer(input=conv_layer73,
                                               img_size=128,
                                               num_output_channels=128,
                                               filter_size=3,
                                               num_filters=256,
                                               stride=[1,2,2,1],
                                               name='deconv_layer_3')
# combine tensor from the end of layer 2
conv_layer81=tf.concat([conv_layer22,deconv_layer3],axis=3,name='Tensor_8_1')

# layer 8 - two convolution
conv_layer82, layer_weight82 = new_conv_layer(input=conv_layer81,
                                           num_input_channels=256,
                                           filter_size=3,
                                           num_filters=128,
                                           use_pooling=False,
                                           name='conv_layer8_2')

conv_layer83, layer_weight83 = new_conv_layer(input=conv_layer82,
                                           num_input_channels=128,
                                           filter_size=3,
                                           num_filters=128,
                                           use_pooling=False,
                                           name='conv_layer8_3')

# up-conv
deconv_layer4, deconv_weight4=new_deconv_layer(input=conv_layer83,
                                               img_size=256,
                                               num_output_channels=64,
                                               filter_size=3,
                                               num_filters=128,
                                               stride=[1,2,2,1],
                                               name='deconv_layer_4')

# combine tensor from the end of layer 1.
conv_layer91=tf.concat([conv_layer12,deconv_layer4],axis=3,name='Tensor_9_1')

# layer 9
conv_layer92, layer_weight92 = new_conv_layer(input=conv_layer91,
                                           num_input_channels=128,
                                           filter_size=3,
                                           num_filters=64,
                                           use_pooling=False,
                                           name='conv_layer9_2')

conv_layer93, layer_weight93 = new_conv_layer(input=conv_layer92,
                                           num_input_channels=64,
                                           filter_size=3,
                                           num_filters=64,
                                           use_pooling=False,
                                           name='conv_layer9_3')

# final layer
# final layer should be deconv or conv 1x1?

final_layer,final_weight= new_deconv_layer(input=conv_layer93,
                                           img_size=256,
                                           num_output_channels=2,
                                           filter_size=3,
                                           num_filters=64,
                                           stride=[1,1,1,1],
                                           name='output_layer')


print('calculate the cost')

ypred=tf.reshape(final_layer,[-1,num_classes*num_channels])
print (ypred.shape)
print (y_true.shape)

cost = tf.reduce_mean(tf.square(ypred-y_true))

optimizer=tf.train.AdagradOptimizer(learning_rate=0.0001).minimize(cost)

print('set up the session')

sess=tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))

sess.run(tf.global_variables_initializer())


saver = tf.train.Saver()

writer = tf.summary.FileWriter("//Users/pdhe/Pythoncode/TENSORFLOWSTUDY/demo")

merged_summary = tf.summary.merge_all()
writer.add_graph(sess.graph)

# in the command line type : >>> tensorboard --logdir //Users/pdhe/Pythoncode/TENSORFLOWSTUDY/demo



def random_mini_batches(X, Y, mini_batch_size):#, seed):
    m = X.shape[1]  # number of training examples
    mini_batches = []

    permutation = list(np.random.permutation(m)) # randomly permute a sequence and convert that to list
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m)) # shuffle the training set and reshape y to 4096 65 but it
    # is unnecessary if the shape is like that originally
    # after shuffling, compile them to minibatches
    num_complete_minibatches = int(
        math.floor(m / mini_batch_size))  # number of mini batches of size mini_batch_size in your partitioning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size] # = [k:k+1]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m] # last batch
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches





mini_batches=random_mini_batches(X_train, Y_train, 16)

for minibatch in mini_batches:  # in each minibatch,
    (x_batch, y_true_batch) = minibatch

    x_batch = x_batch.T
    y_true_batch = y_true_batch.T


total_iterations=0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    #global total_iterations# initialize a global variable
    total_iterations = 0
    # Start-time used for printing time-usage below.
    start_time = time.time()
    seed = 3
    costs = []
    for i in range(num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        # x_batch, x_batch = data.train.next_batch(train_batch_size)
        seed = seed + 1
        epoch_cost = 0.
        mini_batches = random_mini_batches(X_train, Y_train, 16)#, seed) # [4096 64(batch size) number of batches]
        # seed is not being utilized in the function random_mini_batches
        num_minibatches = len(mini_batches) # gives number of batches, can be integrated to random_mini_batches
        for minibatch in mini_batches: # in each minibatch,
            (x_batch, y_true_batch) = minibatch

            # x_batch = x_batch.T
            # y_true_batch = y_true_batch.T

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            # gives the run function a batch and optimizer, then it
            # optimize the model respect to this batch


            _, minibatch_cost = sess.run([optimizer, cost], feed_dict=feed_dict_train) # ??? why there is a _,?

            epoch_cost += minibatch_cost / num_minibatches
            # Print the cost every 100 epoch
        if i % 100 == 0:
            print("Cost after epoch %i: %f" % (i, epoch_cost))
        if i % 5 == 0:
            costs.append(epoch_cost) # append every 5 epoch cost
            #writer.add_summary(minibatch_cost)

    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(((time_dif))))
    np.save('trainingcost',costs)


saver.save(sess, "chkPoint/model.ckpt")

print('optimizing')
optimize(num_iterations=60) # very slow on cpu


def getActivations(layer,x,y,i):
    units = sess.run(layer, feed_dict={x: x, y_true: y})
#     plotNNFilter(units,i)
#
#
# def plotNNFilter(units,i):
    #filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    # n_columns = 6
    # n_rows = math.ceil(filters / n_columns) + 1
    # for i in range(filters):
        # plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")


xx=X_test[:,1]
yy=Y_test[:,1]
xx=np.reshape(xx,[1,xx.size])
yy=np.reshape(yy,[1,yy.size])
slice=32
def getActivations(layer,xx,yy,slice):

    units = sess.run(layer, feed_dict={x: xx, y_true: yy})
    plt.pyplot.figure(1, figsize=(20, 20))
    plt.pyplot.title('Filter ' + str(slice))
    plt.pyplot.imshow(units[0, :, :, slice], interpolation="nearest", cmap="gray")


getActivations(conv_layer12,xx,yy,32)
getActivations(conv_layer22,xx,yy,64)
getActivations(conv_layer32,xx,yy,128)
getActivations(conv_layer42,xx,yy,256)
getActivations(conv_layer52,xx,yy,512)
getActivations(conv_layer63,xx,yy,256)
getActivations(conv_layer73,xx,yy,128)
getActivations(conv_layer83,xx,yy,64)
getActivations(conv_layer93,xx,yy,32)

