#coding:utf8
import tensorflow as tf
import numpy as np


def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)


class Autoencoder:
    '''
        A simple Autorcoder written by tensorflow
    '''
    def __init__(self,n_input,n_hidden,activation_function=tf.nn.softplus,optimizer = tf.train.AdamOptimizer()):
        '''
            initialize the model.
            Args:
                n_input: the number of input node
                n_hidden: the number of hidden node
                activation_function: the activation_function at the output layer
                optimizer: the method to train the autocoder.
        '''
        # Network Parameters
        self.n_hidden = n_hidden
        self.n_input = n_input 
        self.transfer =activation_function

        #initialize weights
        self.weights = self._initialize_weights()
        #give the model, very simple
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        #give the cost function
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.sub(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        #train the model
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        '''
        train the autocoder 
        '''
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):
        '''
        calculate the total loss
        '''
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def transform(self, X):
        '''
        from the input layer to the hidden layer
        '''
        return self.sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, hidden = None):
        '''
        from the hidden layer to the reconstruction layer
        '''
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        '''
        from the inpute layer to the hidden layer
        '''
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])
    
   



if __name__ == "__main__":
    pass
   