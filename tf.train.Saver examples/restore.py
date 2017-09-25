# METHOD 1 
# import tensorflow as tf 

# v1 = tf.get_variable("v1", shape=[3])
# v2 = tf.get_variable("v2", shape=[5])

# saver = tf.train.Saver()

# with tf.Session() as sess:
#   # Restore variables from disk.
#   saver.restore(sess, "box/model.ckpt")

#   # Check the values of the variables
#   print("v1 : %s" % v1.eval())
#   print("v2 : %s" % v2.eval())


# METHOD 2 

import tensorflow as tf

W = tf.Variable(tf.zeros([2,3]), dtype=tf.float32, name="weights")
b = tf.Variable(tf.zeros([3]), dtype=tf.float32, name="biases")

# no need for init step 
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "box/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases: ", sess.run(b))
    