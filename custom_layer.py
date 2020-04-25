import keras
from keras import backend as K


class UserEmbedCell(keras.layers.Layer):

    def __init__(self, units, user_embedding, **kwargs):
        self.units = units
        self.state_size = units
        self.user_embedding = user_embedding
        super(UserEmbedCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel) + sum(self.user_embedding)
        return output, [output]



# layer = RNN(cell)
