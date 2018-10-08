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

def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path)) #All midi-files in the folder
    songs = [] #An array to store the songs
    for f in tqdm(files):
        song = np.array(midi_convert.midi2matrix(f)) #Convert each song to a note state matrix
        if np.array(song).shape[0] > 50:
            songs.append(song)
    return songs

songs = get_songs('midi-songs')
print("{} songs processed".format(len(songs)))