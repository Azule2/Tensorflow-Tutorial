#Import os and TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
#Initialization of tensors

#shape and dtype
X = tf.constant(4, shape=(1,1), dtype=tf.float32)
#constant number 1 to 9
X = tf.constant([[1,2,3],[4,5,6]])
#tensorflow ones shape = 3,3
X = tf.ones((3,3))
#tensorflow zeros shape = 2,3
X = tf.zeros((2,3))

X = tf.eye(3) # I for the identity matrix (eye)
#random
X = tf.random.normal((3,3), mean=0, stddev=1)
X = tf.random.uniform((1,3), minval=1, maxval=5)
#range number start 1 , end 10 
X = tf.range(start=1, limit=10, delta=2)
#cast 
X = tf.cast(X, dtype= tf.float64)

print(X)
