import tensorflow as tf
import pdb

#TODO SHould I add gradient clipping?

class DDQN(tf.keras.Model):

    def __init__(self, num_bands, input_shape_dim, temporal_length):
        super().__init__()
        self.num_bands = num_bands
        self.input_shape_dim = input_shape_dim
        self.temporal_length = temporal_length
        self.init_model()
        
    def init_model(self):
        action_space = self.num_bands + 1
        total_input_shape = (self.temporal_length, self.input_shape_dim)
        input_shape = total_input_shape

        input_a = tf.keras.Input(shape=input_shape)
        lstm_lay = tf.keras.layers.LSTM(100)(input_a)
        # lstm_lay = tf.keras.layers.Dense(100, activation="relu")(input_a)

        advantage_dense = tf.keras.layers.Dense(10, activation='relu')(lstm_lay)
        advantage_lay = tf.keras.layers.Dense(action_space)(advantage_dense)
        value_dense = tf.keras.layers.Dense(10, activation='relu')(lstm_lay)
        value_lay = tf.keras.layers.Dense(1)(value_dense)

        Q_lay = value_lay +(advantage_lay -tf.math.reduce_mean(advantage_lay, axis=1, keepdims=True))
        self.m = tf.keras.Model(inputs=input_a, outputs=Q_lay)
        self.advantage_network = tf.keras.Model(inputs=input_a, outputs=advantage_lay)

    def call(self, input):
        return self.m(input)

    def advantage(self, input):
        return self.advantage_network(input)




