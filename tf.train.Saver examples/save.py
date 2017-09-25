# METHOD 1 

# import tensorflow as tf

# # Create some variables 
# v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
# v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

# # Create the op
# inc_v1 = v1.assign(v1+1)
# dec_v2 = v2.assign(v2-1)
# init_op = tf.global_variables_initializer()

# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()

# with tf.Session() as sess:
#   sess.run(init_op)
#   inc_v1.op.run()
#   dec_v2.op.run()

#   # Save the variables to disk.
#   save_path = saver.save(sess, "box/model.ckpt")


# METHOD 2 

import tensorflow as tf 

# save to file
# Remember to define the same dtype and shape when restoring
W = tf.Variable([[1.,2.,3.], [1.,2.,3.]], dtype=tf.float32, name="weights")
b = tf.Variable([1.,2.,3.], dtype=tf.float32, name="biases")

init = tf.global_variables_initializer()

saver = tf.train.Saver(sharded=False)
with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "box/save_net.ckpt")
    print("Save to path: ", save_path)
