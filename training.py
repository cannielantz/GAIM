import time
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import rnn_rbm
import midi_convert

batch_size = 100 # Nr of trainings ex. to feed into rnn_rbm at a time
epochs_to_save = 5 #Nr of epochs to run between saving each checkpoint
trained_weights = "parameter_checkpoints/initialized.ckpt"

def main(epochs):
    x, cost, generate, W, bh, bv, x, lr, Wuh, Wuv, Wvu, Wuu, bu, u0 = rnn_rbm.rnnrbm()

    tvars = [W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0]

    #opt_func = tf.train.GradientDescentOptimizer(learning_rate=lr)
    #gvs = opt_func.compute_gradients(cost, tvars)
    #gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
    #update = opt_func.apply_gradients(gvs)


    opt_func = tf.train.AdamOptimizer(learning_rate=lr)
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 1)
    update = opt_func.apply_gradients(zip(grads, tvars))


    songs = midi_convert.get_songs('music/midi-songs')

    saver = tf.train.Saver(tvars)
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        saver.restore(sess, trained_weights)

        print("Starting")
        for epoch in range(epochs):
            costs = []
            start = time.time()
            for s_ind, song in enumerate(songs):
                for i in range(1, len(song), batch_size):
                    tr_x = song[i:i + batch_size]
                    alpha = min(0.01, 0.1/float(i)) #Decrease learning rate
                    _, C = sess.run([update, cost], feed_dict={x: tr_x, lr: alpha})
                    costs.append(C)

            print("Epoch: {} cost: {} time: {}".format(epoch, np.mean(costs), time.time()-start))

            if(((epoch + 1) % epochs_to_save) == 0):
                saver.save(sess, "parameter_checkpoints/epoch_{}.ckpt".format(epoch))

if __name__ == "__main__":
    main(int(sys.argv[1]))


