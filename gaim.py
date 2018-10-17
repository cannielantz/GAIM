import tensorflow as tf #For the ML part
import numpy as np #Scientific computing, arrays
import pandas as pd #Data analysis
import msgpack #Exchange data among multiple languages
import glob #For pathname pattern expansion
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm #Progress meter for loops

import midi_convert

#Check if tensorflow works
#hello = tf.constant("Hello")
#sess = tf.Session()
#print(sess.run(hello))

is_biggest = 0
###################### From midi to matrix ############################

def get_songs(path):
    #is_biggest = 0
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

songs = get_songs('midi-songs')
print("{} songs processed".format(len(songs)))


######################################################################


##################### Parameters #####################################

lowestNote = midi_convert.min # Index of lowest note
highestNote = midi_convert.max
noteRange = highestNote - lowestNote

timesteps = 20
visible_size = 2 * noteRange * timesteps # Size of visible layer
hidden_size = 50 # Size of hidden layer

epochs = 200 # Number of training ephochs (through the entire dataset)
batch_size = 100 # Number of training examples to send through the model at a time
learning_rate = tf.constant(0.005, tf.float32) #0.005

#####################################################################

####################### Variables ###################################

x = tf.placeholder(tf.float32, [None, visible_size], name = 'x') # Our data
w = tf.Variable(tf.random_normal([visible_size, hidden_size], 0.01), name = 'w') # Weight matrix
hidden_bias = tf.Variable(tf.zeros([1, hidden_size], tf.float32, name = 'hidden_bias'))
visible_bias = tf.Variable(tf.zeros([1, visible_size], tf.float32, name = 'visible_bias'))

#####################################################################

########################### Functions ###############################

def random_binary_vec(probs):
    # Returns vector of random 0:s and 1:s from a probability vector
    # This is for both h and x
    return tf.floor(probs + tf.random_uniform( tf.shape(probs), 0, 1) )


def gibbs_chain(k):
    def gibbs_step(count, k, xk):

        hk = random_binary_vec(tf.sigmoid( tf.matmul(xk, w) + hidden_bias) )
        xk = random_binary_vec(tf.sigmoid( tf.matmul(hk, tf.transpose(w)) + visible_bias) )

        return count + 1, k, xk

    counter = tf.constant(0)
    [_,_, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter, gibbs_step, [counter, tf.constant(k), x])

    x_sample = tf.stop_gradient(x_sample)
    return x_sample

#########################################################################

######################## Training update ################################

x_sample = gibbs_chain(1)

h = random_binary_vec(tf.sigmoid(tf.matmul(x, w) + hidden_bias))
h_sample = random_binary_vec(tf.sigmoid(tf.matmul(x_sample, w) + hidden_bias))

size_bt = tf.cast(tf.shape(x)[0], tf.float32)

w_adder = tf.multiply(learning_rate/ size_bt, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
visible_bias_adder = tf.multiply(learning_rate / size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
hidden_bias_adder = tf.multiply(learning_rate / size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))

update = [w.assign_add(w_adder), visible_bias.assign_add(visible_bias_adder), hidden_bias.assign_add(hidden_bias_adder)]

##########################################################################

########################### Train model ####################################

with tf.Session() as sess:
    # Train model
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in tqdm(range(epochs)):
        for song in songs:
            song = np.array(song)
            song = song[:int(np.floor(song.shape[0] // timesteps) * timesteps)]
            song = np.reshape(song, [song.shape[0] // timesteps, song.shape[1] * timesteps])

            for i in range(1, len(song), batch_size):
                train_x = song[i:i + batch_size]
                sess.run(update, feed_dict={x: train_x})


    ###########################################################################

    ########################## Make music ####################################

    sample = gibbs_chain(1).eval( session=sess, feed_dict={ x: np.zeros((10, visible_size)) } )
    done = tf.constant("Sampling")
    print(sess.run(done))

    for i in range(sample.shape[0]):
        if not any(sample[i, :]):
            continue

        S = np.reshape(sample[i, :], (timesteps, 2*noteRange))
        midi_convert.matrix2midi(S, "generated/song_{}".format(i))



