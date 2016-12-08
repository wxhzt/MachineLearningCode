#coding:utf8
import numpy as np
from PIL import Image
import tensorflow as tf
def activation(source):
	# Tanh
	#return tf.nn.tanh(source)
	# Relu
	#x = tf.nn.relu(source)
	# Leaky ReLU
	#alpha = 0.001
	#return tf.maximum(alpha*source, source)
	# My evil slide of doom activation:
	alpha = 0.02
	beta = 1.1
	return tf.maximum(alpha*source, tf.sin(source)+(beta*source)) 

def xavier_init(shape, constant=1):
	val = 0.1 #constant * np.sqrt(2.0/float(np.sum(np.abs(shape[1:]))))
	return tf.random_uniform(shape, minval=-val, maxval=val)


# Convenience method for writing an output image from an encoded array
def save_image(struct, filename):
	#img_tensor = tf.image.encode_jpeg(decoded[0])
	print("Output mean: {}.  Low: {}  High: {}".format(struct[0].mean(), struct[0].min(), struct[0].max()))
	# Normalize to -1 - 1 and unfilter, then re-normalize for output.
	struct = struct[0]
	decoded_min = struct.min()
	decoded_max = struct.max()
	if decoded_min == decoded_max:
		decoded_max = 1.0
		decoded_min = 0
	decoded_norm = (struct-decoded_min)/(decoded_max-decoded_min)

	img_arr = np.asarray(decoded_norm*255, dtype=np.uint8)
	img = Image.fromarray(img_arr)
	img.save(filename)