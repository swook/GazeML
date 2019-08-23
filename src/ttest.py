import tensorflow as tf
hello = tf.constant('hello tensol')
sess = tf.Session()
print(sess.run(hello))