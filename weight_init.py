import tensorflow as tf
import numpy as np
from tqdm import tqdm
import RBM
import rnn_rbm
import midi_convert

epochs = 100
lr = 0.0001

def main():
    songs = midi_convert.get_songs('music/midi-songs')

    x = tf.placeholder(tf.float32, [None, rnn_rbm.visible_size], name="x")  # The placeholder variable that holds our data
    W = tf.Variable(tf.random_normal([rnn_rbm.visible_size, rnn_rbm.hidden_size], 0.01), name="W")  # The weight matrix of the RBM
    Wuh = tf.Variable(tf.random_normal([rnn_rbm.hidden_rnn_size, rnn_rbm.hidden_size], 0.0001), name="Wuh")  # The RNN -> RBM hidden weight matrix
    bh = tf.Variable(tf.zeros([1, rnn_rbm.hidden_size], tf.float32), name="bh")  # The RNN -> RBM hidden bias vector
    Wuv = tf.Variable(tf.random_normal([rnn_rbm.hidden_rnn_size, rnn_rbm.visible_size], 0.0001), name="Wuv")  # The RNN -> RBM visible weight matrix
    bv = tf.Variable(tf.zeros([1, rnn_rbm.visible_size], tf.float32), name="bv")  # The RNN -> RBM visible bias vector
    Wvu = tf.Variable(tf.random_normal([rnn_rbm.visible_size, rnn_rbm.hidden_rnn_size], 0.0001), name="Wvu")  # The data -> RNN weight matrix
    Wuu = tf.Variable(tf.random_normal([rnn_rbm.hidden_rnn_size, rnn_rbm.hidden_rnn_size], 0.0001), name="Wuu")  # The RNN hidden unit weight matrix
    bu = tf.Variable(tf.zeros([1, rnn_rbm.hidden_rnn_size], tf.float32), name="bu")  # The RNN hidden unit bias vector
    u0 = tf.Variable(tf.zeros([1, rnn_rbm.hidden_rnn_size], tf.float32), name="u0")  # The initial state of the RNN


    BH_t = tf.Variable(tf.ones([1, rnn_rbm.hidden_size], tf.float32), name="BH_t")
    BV_t = tf.Variable(tf.ones([1, rnn_rbm.visible_size], tf.float32), name = "BV_t")


    saver = tf.train.Saver()
    update = RBM.get_cd_update(x, W, bv, bh, 1, lr)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        for epoch in tqdm(range(epochs)):
            for song in songs:
                sess.run(update, feed_dict={x: song})

        save_path = saver.save(sess, "parameter_checkpoints/initialized.ckpt")


if __name__ == "__main__":
    main()