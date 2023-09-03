import tensorflow as tf

hello = tf.constant('hello, world!')
sess=tf.Session()
print(sess.run(hello))

print(tf.test.is_gpu_available())