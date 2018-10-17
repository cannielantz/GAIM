import tensorflow as tf
import numpy as np
import pandas as pd
import sys


from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
from matplotlib import pyplot as plt
from copy import deepcopy

import RBM
import midi_convert
import time
#import rnn_rbm


nn_output_songs = 3
primer_song = 'music/trainingSamples/22.mid'

def main( trained_weights ):
    x, cost, generate, W, bh, bv, x, lr, Wuh, Wuv, Wvu, Wuu, bu, o0 = rnn_rbm.rnnrbm() #build and get parameters odf the network

    tvars = [W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0]

    saver = tf.train.Saver(tvars) #Saver object to restore the weights of the model

    song_primer = RBM.get_songs(primer_song)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        saver.restore(sess, trained_weights ) #load that shit... that are the weights of the network

        #we generate num songs
        for i in tqdm(range(num)):
            generated_music = sess.run(generate(300), feed_dict = {x: song_primer}) #prime the newtork
            new_song_path = "generated/songs/{}_{}".format(i, primer_song.split("/")[-1])

            midi_convert.write_song(new_song_path, generated_music)

if __name__ == "__main__":
    main(sys.argv[1])