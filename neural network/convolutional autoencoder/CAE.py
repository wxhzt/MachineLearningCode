#coding:utf8
import sys, os
from glob import glob
from random import choice, randint
from io import BytesIO
from itertools import cycle
import time

from PIL import Image
import numpy as np
import tensorflow as tf

import hashlib 
from utils import *

class CAE:
	'''
	convolutional auto encode
	'''
	def __init__(self, modelfile=None):
		'''
		initilize the variables and network structure
		'''
		self.LEARNING_RATE = 0.0001      
		self.TRAINING_ITERATIONS = 100000
		self.BATCH_SIZE = 1
		self.TRAINING_REPORT_INTERVAL = 100

		self.REPRESENTATION_SIZE = 200
		

		self.IMAGE_WIDTH = 192
		self.IMAGE_HEIGHT = 168
		self.IMAGE_DEPTH = 3

		self.input_batch = tf.placeholder(tf.float32, [self.BATCH_SIZE,  self.IMAGE_WIDTH, self.IMAGE_HEIGHT,self.IMAGE_DEPTH], name="image_input")
		self.encoded_batch = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.REPRESENTATION_SIZE], name="encoder_input") # Replace BATCH_SIZE with None
		self.keep_prob = tf.placeholder(tf.float32, name="keep_probability")
		self.output_objective = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.IMAGE_WIDTH,self.IMAGE_HEIGHT,  self.IMAGE_DEPTH], name="output_objective")

		self.decoder, self.encoder = self._build_model(self.input_batch,self.encoded_batch,self.keep_prob)

		# Get final ops
		#global_reconstruction_loss = output_objective*-tf.log(decoder+1.0e-6) + (1-output_objective)*-tf.log(1.0e-6+1-decoder)
		#global_reconstruction_loss = tf.reduce_sum(tf.abs(output_objective - decoder))
		self.global_reconstruction_loss = tf.reduce_sum(tf.square(self.output_objective - self.decoder))
		#global_reconstruction_loss = tf.nn.l2_loss(output_objective - decoder)
		self.global_representation_loss = tf.abs(1-tf.reduce_sum(self.encoder)) + tf.reduce_sum(tf.abs(self.encoder))
		self.global_loss = self.global_reconstruction_loss#+ global_representation_loss
		#global_optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(global_loss) #tf.clip_by_value(global_loss, -1e6, 1e6))
		self.global_optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).minimize(self.global_loss)
		self.sess = tf.Session()
		if modelfile!=None:
			self.saver = tf.train.Saver()
			self.sess.run(tf.initialize_all_variables())
		
			# If we already have a trained network, reload that. The saver doesn't save the graph structure, so we need to build one with identical node names and reload it here.
			# Right now the graph can be loaded with tf.import_graph_def OR the variables can be populated with Restore, but not both (yet).
			# The graph.pbtxt holds graph structure (in model folder).  model-checkpoint has values/weights.
			# TODO: Review when bug is fixed. (2015/11/29)
			if os.path.isfile(modelfile):
				saver.restore(self.sess, modelfile)
				print("load trained model")
			else:
				print("modelfile does not exist")
	
	def train(self, datafold):
		'''
		train the model with images in datafold
		'''

		self.saver = tf.train.Saver()
		self.sess.run(tf.initialize_all_variables())
		# Begin training
		for iteration in range(1, self.TRAINING_ITERATIONS):
			try:
				x_batch, y_batch = self._get_batch(self.BATCH_SIZE,datafold)
				loss1, _, encoder_output = self.sess.run(
					[self.global_loss, self.global_optimizer, self.encoder], 
					feed_dict={
						self.input_batch:x_batch, 
						self.encoded_batch:np.zeros((self.BATCH_SIZE, self.REPRESENTATION_SIZE)),
						self.output_objective:y_batch,
						self.keep_prob:0.5,
					}
				) 
				print("Iter {}: {} \n".format(iteration, loss1))
				if iteration % self.TRAINING_REPORT_INTERVAL == 0:
					# Checkpoint progress
					print("Finished batch {}".format(iteration))
					#time.sleep(1.0) # Sleep to avoid shared process killing us for resources.

			except KeyboardInterrupt:
				from IPython.core.debugger import Tracer
				Tracer()()
		
		# When complete, stop and let us play.
		print("Finished")
		self.saver.save(self.sess, "./model/final.model") #, global_step=iteration)
			
		
		

	def express(self, imagefile):
		'''
		get the low dimension expression of image
		'''
		# Render output sample
		filename = imagefile
		target = None
		img = Image.open(filename)
		print("Loaded image {}".format(filename))
		# Shrink image and embed in the middle of our target data.
		target_width, target_height = img.size
		max_dim = max(img.size)
		new_width = (self.IMAGE_WIDTH*img.size[0])//max_dim
		new_height = (self.IMAGE_HEIGHT*img.size[1])//max_dim
		# Center image in new image.
		newimg = Image.new(img.mode, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
		offset_x = int((self.IMAGE_WIDTH/2)-(new_width/2))
		offset_y = int((self.IMAGE_HEIGHT/2)-(new_height/2))
		box = (offset_x, offset_y, offset_x+new_width, offset_y+new_height)
		newimg.paste(img.resize((new_width, new_height)), box)
		# Copy to target
		target = np.asarray(newimg, dtype=np.float)/255
		y_batch = np.zeros([1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_DEPTH], dtype=np.float)
		print target.shape
		print y_batch.shape
		y_batch[0,:,:,:] = target
		encoded = self.sess.run(self.encoder, feed_dict={
						self.input_batch:y_batch, #todo
						self.keep_prob:1.0,
					})

		return encoded
	

	def reconstruct(self, imagefile, reimagefile):
		'''
		reconstruct image with the trained network
		'''
		encoded = self.express(imagefile)
		
		decoded = self.sess.run(self.decoder, feed_dict={
			self.input_batch:np.zeros(shape=[self.BATCH_SIZE, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_DEPTH]), 
			self.encoded_batch:encoded, 
			self.keep_prob:1.0,
		})

		save_image(decoded, reimagefile)
################################################################################################################################
	def _build_fc(self, input_source, hidden_size, weight=None, bias=None, activate=True):
		# Figure out size of input and create weight matrix of appropriate size.
		shape = input_source.get_shape().as_list()[-1]
		if weight is None:
			weight = tf.Variable(xavier_init([shape, hidden_size])) #tf.random_normal([shape, hidden_size]))
		if bias is None:
			bias = tf.Variable(tf.zeros([hidden_size,]))

		# Get preactivations
		result = tf.nn.bias_add(tf.matmul(input_source, weight), bias)

		# Sometimes activate
		if activate:
			result = activation(result)

		return result, weight, bias
	def _build_conv(self, source, filter_shape, strides, padding='SAME', activate=True, weight=None, bias=None):
		# If we don't have weights passed in, make some new ones.
		# TODO: Make sure shapes match up.	
		if not weight:
			weight = tf.Variable(xavier_init(filter_shape))
		if not bias:
			bias = tf.Variable(tf.zeros([filter_shape[-1],]))

		conv = tf.nn.bias_add(tf.nn.conv2d(source, filter=weight, strides=strides, padding=padding), bias)
		if activate:
			act = activation(conv) # Not relu6
		else:
			act = conv
		return act, weight, bias

	def _build_deconv(self, source, output_shape, filter_shape, strides, padding='SAME', activate=True, weight=None, bias=None):
		if not weight:
			weight = tf.Variable(xavier_init(filter_shape))
		deconv = tf.nn.conv2d_transpose(source, filter=weight, strides=strides, padding=padding, output_shape=output_shape)
		if not bias:
			#bias = tf.Variable(tf.zeros(output_shape[1:]))
			bias = tf.Variable(tf.zeros([deconv.get_shape()[-1],]))
		deconv = tf.nn.bias_add(deconv, -bias)
		#deconv = tf.nn.conv2d_transpose(source, filter=weight, strides=strides, padding=padding, output_shape=output_shape)
		if activate:
			act = activation(deconv)
		else:
			act = deconv
		return act, weight, bias

	def _build_max_pool(self, source, kernel_shape, strides):
		return tf.nn.max_pool(source, ksize=kernel_shape, strides=strides, padding='SAME')

	def _build_unpool(self, source, kernel_shape):
		input_shape =source.get_shape().as_list()
		return tf.image.resize_images(source, (input_shape[1]*kernel_shape[1], input_shape[2]*kernel_shape[2]))

	def _build_dropout(self, source, toggle):
		return tf.nn.dropout(source, toggle)

	def _build_lrn(self, source):
		return tf.nn.local_response_normalization(source)

	# Create model
	def _build_model(self, image_input_source, encoder_input_source, dropout_toggle):
		"""Image and Encoded are input placeholders.  input_encoded_interp is the toggle between input (when 0) and encoded (when 1).
		Returns a decoder and the encoder output."""
		# We have to match this output size.
		batch, input_height, input_width, input_depth = image_input_source.get_shape().as_list()
	
		filter_sizes = [64, 64, 64] # Like VGG net, except made by a stupid person.
	
		# Convolutional ops will go here.
		c0, wc0, bc0 = self._build_conv(image_input_source, [3, 3, input_depth, filter_sizes[0]], [1, 1, 1, 1], activate=False)
		c1 = self._build_max_pool(c0, [1, 2, 2, 1], [1, 2, 2, 1])
		c2, wc2, bc2 = self._build_conv(self._build_dropout(c1, dropout_toggle), [3, 3, filter_sizes[0], filter_sizes[1]], [1, 1, 1, 1])
		c3 = self._build_max_pool(c2, [1, 2, 2, 1], [1, 2, 2, 1])
		c4, wc4, bc4 = self._build_conv(self._build_dropout(c3, dropout_toggle), [3, 3, filter_sizes[1], filter_sizes[2]], [1, 1, 1, 1])
		c5 = self._build_max_pool(c4, [1, 2, 2, 1], [1, 2, 2, 1])
		conv_output = c5
	
		# Transition to FC layers.
		pre_flat_shape = conv_output.get_shape().as_list()
		flatten = tf.reshape(conv_output, [-1, pre_flat_shape[1]*pre_flat_shape[2]*pre_flat_shape[3]])
	
		# Dense connections
		fc0, wf0, bf0 = self._build_fc(flatten, 512)
		fc1, wf1, bf1 = self._build_fc(fc0, 512)
		fc2, wf2, bf2 = self._build_fc(self._build_dropout(fc1, dropout_toggle), self.REPRESENTATION_SIZE)
		fc_out = fc2
	
		# Output point and our encoder mix-in.
		mu_output, wmu, bmu = self._build_fc(fc_out, self.REPRESENTATION_SIZE)
		z_output, wz, bz = self._build_fc(fc_out, self.REPRESENTATION_SIZE)
		encoded_output = tf.random_normal(mean=mu_output, stddev=z_output, shape=z_output.get_shape()) #tf.nn.softmax(fc_out)
		encoded_input = self._build_dropout(encoder_input_source + encoded_output, dropout_toggle) # Mix input and enc.
		encoded_input.set_shape(encoded_output.get_shape()) # Otherwise we can't ascertain the size.
	
		# More dense connections on the offset.
		dfc2, dwf2, dbf2 = self._build_fc(encoded_input, 512, weight=tf.transpose(wf2), bias=tf.transpose(bf1))
		dfc1, dwf1, dbf1 = self._build_fc(dfc2, 512, weight=tf.transpose(wf1), bias=tf.transpose(bf0))
		dfc0, dwf0, dbf0 = self._build_fc(self._build_dropout(dfc1, dropout_toggle), flatten.get_shape().as_list()[-1], weight=tf.transpose(wf0))
	
		# Expand for more convolutional operations.
		unflatten = tf.reshape(dfc0, [-1, pre_flat_shape[1], pre_flat_shape[2], pre_flat_shape[3]]) #pre_flat_shape)
	
		# More convolutions here.
		dc5 = self._build_unpool(unflatten, [1, 2, 2, 1])
		dc4, wdc4, bdc4 = self._build_deconv(self._build_dropout(dc5, dropout_toggle), c3.get_shape().as_list(), [3, 3, filter_sizes[1], filter_sizes[2]], [1, 1, 1, 1])
		dc3 = self._build_unpool(dc4, [1, 2, 2, 1])
		dc2, wdc2, bdc2 = self._build_deconv(self._build_dropout(dc3, dropout_toggle), c1.get_shape().as_list(), [3, 3, filter_sizes[0], filter_sizes[1]], [1, 1, 1, 1])
		dc1 = self._build_unpool(dc2, [1, 2, 2, 1])
		dc0, wdc0, bdc0 = self._build_deconv(dc1, [batch, input_height, input_width, input_depth], [3, 3, input_depth, filter_sizes[0]], [1, 1, 1, 1], activate=False)
		deconv_output = dc0
	
		# Return result + encoder output
		return deconv_output, encoded_output
###########################################################################################################################################################################
	def _get_batch(self, batch_size, file_glob):
		batch = np.zeros([batch_size,  self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_DEPTH], dtype=np.float)
		labels = np.zeros([batch_size, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_DEPTH], dtype=np.float)
		index = 0
		gen = self._example_generator(file_glob)
		while index < batch_size:
			#for index, data in enumerate(gen):
			data = next(gen)
			x, y = data
			labels[index,:,:,:] = y[:,:,:]
			batch[index,:,:,:] = x[:,:,:]
			#if index >= batch_size:
			#	break
			index += 1
		return batch, labels

	# Define data-source iterator
	def _example_generator(self, file_glob, noise=0.0, cache=True):
		filenames = glob(file_glob)
		file_cache = dict()
		#for filename in cycle(filenames):
		while True:
			filename = choice(filenames)
			example = None
			target = None
			if cache and filename in file_cache:
				target = file_cache[filename]
			else:
				try:
					filename = choice(filenames)
					img = Image.open(filename)
					print("Loaded image {}".format(filename))
					# Shrink image and embed in the middle of our target data.
					target_width, target_height = img.size
					max_dim = max(img.size)
					new_width = (self.IMAGE_WIDTH*img.size[0])//max_dim
					new_height = (self.IMAGE_HEIGHT*img.size[1])//max_dim
					# Center image in new image.
					newimg = Image.new(img.mode, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
					offset_x = int((self.IMAGE_WIDTH/2)-(new_width/2))
					offset_y = int((self.IMAGE_HEIGHT/2)-(new_height/2))
					box = (offset_x, offset_y, offset_x+new_width, offset_y+new_height)
					newimg.paste(img.resize((new_width, new_height)), box)
					# Copy to target
					target = np.asarray(newimg, dtype=np.float)/255.0
					#example = np.swapaxes(example, 1, 2)
					file_cache[filename] = target
				except ValueError as e:
					print("Problem loading image {}: {}".format(filename, e))
					continue
			# Add noise
			if noise > 0:
				# Example is the noised copy.
				example = target + np.random.uniform(low=-noise, high=+noise, size=target.shape)
				# Re-normalize so we don't overflow/underflow.
				low = example.min()
				high = example.max()
				example = (example-low)/(high-low)
			else:
				example = target
			yield example, target	