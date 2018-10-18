import tensorflow as tf #For the ML part
import numpy as np #Scientific computing, arrays
import pandas as pd #Data analysis
from tensorflow.python.ops import control_flow_ops

import midi_convert

#Check if tensorflow works
#hello = tf.constant("Hello")
#sess = tf.Session()
#print(sess.run(hello))


###################### From midi to matrix ############################
"""
def get_songs(path):
    is_biggest = 0
    files = glob.glob('{}/*.mid*'.format(path)) #All midi-files in the folder
    songs = [] #An array to store the songs
    for f in tqdm(files):
        try:
            song = np.array(midi_convert.midi2matrix(f)) #Convert each song to a note state matrix
            if np.array(song).shape[0] > 50:
                songs.append(song)
                if(len(song) > is_biggest):
                    is_biggest = len(song)
                print( "\n" + "File: " + f)
        except Exception as e:
            raise e
        print("Longest song of size: {}".format(is_biggest))
    return songs

songs = get_songs('music/midi-songs')
print("{} songs processed".format(len(songs)))
"""

######################################################################


##################### Parameters #####################################





#epochs = 200 # Number of training ephochs (through the entire dataset)
#batch_size = 100 # Number of training examples to send through the model at a time


#####################################################################

####################### Variables ###################################


#w = tf.Variable(tf.random_normal([visible_size, hidden_size], 0.01), name = 'w') # Weight matrix
#hidden_bias = tf.Variable(tf.zeros([1, hidden_size], tf.float32, name = 'hidden_bias'))
#visible_bias = tf.Variable(tf.zeros([1, visible_size], tf.float32, name = 'visible_bias'))

#####################################################################

########################### Functions ###############################

def random_binary_vec(probs):
    # Returns vector of random 0:s and 1:s from a probability vector
    # This is for both h and x
    return tf.floor(probs + tf.random_uniform( tf.shape(probs), 0, 1) )


def gibbs_chain(x, W, bv, bh, k):
    def gibbs_step(count, k, xk):

        hk = random_binary_vec(tf.sigmoid( tf.matmul(xk, W) + bh) )
        xk = random_binary_vec(tf.sigmoid( tf.matmul(hk, tf.transpose(W)) + bv) )

        return count + 1, k, xk

    counter = tf.constant(0)
    [_,_, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter, gibbs_step, [counter, tf.constant(k), x])

    x_sample = tf.stop_gradient(x_sample)
    return x_sample

#########################################################################

def get_free_energy_cost(x, W, bv, bh, k):
    x_sample = gibbs_chain(x, W,bv, bh, k)

    def F(xx):
        return -tf.reduce_sum(tf.log(1 + tf.exp(tf.matmul(xx, W) + bh)), 1) - tf.matmul(xx, tf.transpose(bv))

    cost = tf.reduce_mean(tf.subtract(F(x), F(x_sample)))
    return cost

######################## Training update ################################

def get_cd_update(x, W, bv, bh, k, lr):
    x_sample = gibbs_chain(x, W, bv, bh, k)

    h = random_binary_vec(tf.sigmoid(tf.matmul(x, W) + bh))
    h_sample = random_binary_vec(tf.sigmoid(tf.matmul(x_sample, W) + bh))

    lr = tf.constant(lr, tf.float32) #Learning rate of CD

    size_bt = tf.cast(tf.shape(x)[0], tf.float32)

    w_adder = tf.multiply(lr/ size_bt, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
    visible_bias_adder = tf.multiply(lr / size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
    hidden_bias_adder = tf.multiply(lr / size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))

    update = [W.assign_add(w_adder), bv.assign_add(visible_bias_adder), bh.assign_add(hidden_bias_adder)]
    return update

##########################################################################


