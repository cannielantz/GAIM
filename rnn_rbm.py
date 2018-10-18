import tensorflow as tf
import numpy as np
import glob
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm

import RBM
import midi_convert
import generation

noteRange = midi_convert.span
timesteps = midi_convert.timesteps

visible_size = 2 * noteRange * timesteps # Size of visible layer of RBM
hidden_size = 50 # Size of hidden layer of RBM
hidden_rnn_size= 100 #size of each RNN hidden

def rnnrbm():
    x = tf.placeholder(tf.float32, [None, visible_size], name='x')  # Our data
    learning_rate = tf.placeholder(tf.float32)

    batch_size = tf.shape(x)[0] #100  # Number of training examples to send through the model at a time, batch size

    W = tf.Variable(tf.zeros([visible_size, hidden_size]), name="W")
    Wuh = tf.Variable(tf.zeros([hidden_rnn_size, hidden_size]), name="Wuh")
    Wuv = tf.Variable(tf.zeros([hidden_rnn_size, visible_size]), name="Wuv")
    Wvu = tf.Variable(tf.zeros([visible_size, hidden_rnn_size]), name="Wvu")
    Wuu = tf.Variable(tf.zeros([hidden_rnn_size, hidden_rnn_size]), name="Wuu")
    bh = tf.Variable(tf.zeros([1, hidden_size]), name="bh")
    bv = tf.Variable(tf.zeros([1, visible_size]), name="bv")
    bu = tf.Variable(tf.zeros([1, hidden_rnn_size]), name="bu")
    u0 = tf.Variable(tf.zeros([1, hidden_rnn_size]), name="u0")
    BH_t = tf.Variable(tf.zeros([1, hidden_size]), name="BH_t")
    BV_t = tf.Variable(tf.zeros([1, visible_size]), name="BV_t")

    def rnn_recurrence(u_tm1, sl):
        sl = tf.reshape(sl, [1, visible_size] )
        u_t = (tf.tanh( bu + tf.matmul(sl, Wvu ) + tf.matmul(u_tm1, Wuu) ))

        return u_t

    def visible_recurrence_bias(bv_t, u_tm1):
        bv_t = tf.add( bv, tf.matmul(u_tm1, Wuv))

        return bv_t

    def hidden_recurrence_bias(bh_t, u_tm1):
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh))

        return bh_t

    def generate_recurrence(count, k, u_tm1, primer_song, x, music):
        #Get bias vecrots from recurrent state of RNN
        bv_t = tf.add(bv, tf.matmul(u_tm1, Wuv))
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh) )

        #Run Gibbs step to get the music output. Prime the RBM with prev music output
        x_out = RBM.gibbs_chain(primer_song, W, bv_t, bh_t, k=25) #we used 1 before

        #Update RNN hidden state based on the musical output and the current hidden state
        u_t = tf.tanh( bu + tf.matmul( x_out , Wvu) + tf.matmul( u_tm1, Wuu) )

        #Add the new output to the musical piece
        music = tf.concat(0, [music, x_out])

        return count+1, k, u_t, x_out, x, music

    def generate(num, x=x, batch_size = batch_size, u0 = u0, visible_size = visible_size, prime_length = 100):
        """
                    manager for handling music generation. This function is one of the outputs of the build_rnnrbm function
        :param num:
        :param x:
        :param batch_size:
        :param u0:
        :param visible_size:
        :param prime_length:
        :return:
        """

        Uarr = tf.scan(rnn_recurrence, x, initializer=u0)
        long4 = np.floor(prime_length / timesteps)
        U = Uarr[int(long4), :, :]

        #[_, _, _, _, _, music] = control_flow_ops.While(lambda count, num_iter, *args: count < num_iter,
        #                                            generate_recurrence, [tf.constant(1, tf.int32), tf.constant(num), U,
        #                                            tf.zeros([1,visible_size], tf.float32), x,
        #                                            tf.zeros([1, visible_size], tf.float32)])

        time_steps = tf.constant(1, tf.int32)
        iterations = tf.constant(generation.nn_output_songs)
        u_t = tf.zeros([1, visible_size], tf.float32)
        music = tf.zeros([1, visible_size], tf.float32)
        loop_vars = [time_steps, iterations, U, u_t, x, music]

        [_, _, _, _, _, music] = tf.while_loop(lambda count, num_iter, *args: count < num_iter, generate_recurrence, loop_vars)


        return music

    #Reshape bias matrices to be the same size as the batch.
    tf.assign(BH_t, tf.tile(BH_t, [batch_size, 1]))
    tf.assign(BV_t, tf.tile(BV_t, [batch_size, 1]))

    #Scan through the rnn and generate the value for each hidden node in the batch
    u_t = tf.scan(rnn_recurrence, x, initializer=u0)

    #Scan through the rnn and generate the visible and hidden biases for each RB; in the batch
    BV_t = tf.reshape(tf.scan(visible_recurrence_bias, u_t, tf.zeros([1, visible_size], tf.float32)), [batch_size, visible_size])
    BH_t = tf.reshape(tf.scan(hidden_recurrence_bias, u_t, tf.zeros([1, hidden_size], tf.float32)), [batch_size, hidden_size])

    cost = RBM.get_free_energy_cost(x, W, BV_t, BH_t, k = 15)

    return x, cost, generate, W, bh, bv, x, learning_rate, Wuh, Wuv, Wvu, Wuu, bu, u0