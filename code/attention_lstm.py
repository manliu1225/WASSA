# Adapted from
# https://github.com/codekansas/keras-language-modeling/blob/master/attention_lstm.py
# Licensed under MIT

from keras import backend as K
from keras.engine import InputSpec
from keras.layers import LSTM, activations, Wrapper, Recurrent

class Attention(Wrapper):
    """
    This wrapper will provide an attention layer to a recurrent layer. 
    
    # Arguments:
        layer: `Recurrent` instance with consume_less='gpu' or 'mem'
    
    # Examples:
    
    ```python
    model = Sequential()
    model.add(LSTM(10, return_sequences=True), batch_input_shape=(4, 5, 10))
    model.add(TFAttentionRNNWrapper(LSTM(10, return_sequences=True, consume_less='gpu')))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop') 
    ```
    
    # References
    - [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449)
    
    
    """
    def __init__(self, layer, **kwargs):
        # assert isinstance(layer, Recurrent)
        # if layer.get_config()['consume_less']=='cpu':
        #     raise Exception("AttentionLSTMWrapper doesn't support RNN's with consume_less='cpu'")
        self.supports_masking = True
        super(Attention, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]
        nb_samples, nb_time, input_dim = input_shape

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        super(Attention, self).build()
        
        self.W1 = self.layer.init((input_dim, input_dim, 1, 1), name='{}_W1'.format(self.name))
        self.W2 = self.layer.init((self.layer.output_dim, input_dim), name='{}_W2'.format(self.name))
        self.b2 = K.zeros((input_dim,), name='{}_b2'.format(self.name))
        self.W3 = self.layer.init((input_dim*2, input_dim), name='{}_W3'.format(self.name))
        self.b3 = K.zeros((input_dim,), name='{}_b3'.format(self.name))
        self.V = self.layer.init((input_dim,), name='{}_V'.format(self.name))

        self.trainable_weights = [self.W1, self.W2, self.W3, self.V, self.b2, self.b3]

    def get_output_shape_for(self, input_shape):
        return self.layer.get_output_shape_for(input_shape)

    def step(self, x, states):
        # This is based on [tensorflows implementation](https://github.com/tensorflow/tensorflow/blob/c8a45a8e236776bed1d14fd71f3b6755bd63cc58/tensorflow/python/ops/seq2seq.py#L506).
        # First, we calculate new attention masks:
        #   attn = softmax(V^T * tanh(W2 * X +b2 + W1 * h))
        # and we make the input as a concatenation of the input and weighted inputs which is then
        # transformed back to the shape x of using W3
        #   x = W3*(x+X*attn)+b3
        # Then, we run the cell on a combination of the input and previous attention masks:
        #   h, state = cell(x, h).
        
        nb_samples, nb_time, input_dim = self.input_spec[0].shape
        h = states[0]
        X = states[-1]
        xW1 = states[-2]
        
        Xr = K.reshape(X,(-1,nb_time,1,input_dim))
        hW2 = K.dot(h,self.W2)+self.b2
        hW2 = K.reshape(hW2,(-1,1,1,input_dim)) 
        u = K.tanh(xW1+hW2)
        a = K.sum(self.V*u,[2,3])
        a = K.softmax(a)
        a = K.reshape(a,(-1, nb_time, 1, 1))
        
        # Weight attention vector by attention
        Xa = K.sum(a*Xr,[1,2])
        Xa = K.reshape(Xa,(-1,input_dim))
        
        # Merge input and attention weighted inputs into one vector of the right size.
        x = K.dot(K.concatenate([x,Xa],1),self.W3)+self.b3    
        
        h, new_states = self.layer.step(x, states)
        return h, new_states

    def get_constants(self, x):
        constants = self.layer.get_constants(x)
        
        # Calculate K.dot(x, W2) only once per sequence by making it a constant
        nb_samples, nb_time, input_dim = self.input_spec[0].shape
        Xr = K.reshape(x,(-1,nb_time,input_dim,1))
        Xrt = K.permute_dimensions(Xr, (0, 2, 1, 3))
        xW1t = K.conv2d(Xrt,self.W1,border_mode='same')     
        xW1 = K.permute_dimensions(xW1t, (0, 2, 3, 1))
        constants.append(xW1)
        
        # we need to supply the full sequence of inputs to step (as the attention_vector)
        constants.append(x)
        
        return constants

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))

        if self.layer.stateful:
            initial_states = self.layer.states
        else:
            initial_states = self.layer.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.layer.preprocess_input(x)
        

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.layer.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.layer.unroll,
                                             input_length=input_shape[1])
        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            return outputs
        else:
            return last_output
            

# from __future__ import absolute_import

# import keras
# from keras.layers import LSTM, activations

# class AttentionLSTM(LSTM):
#     def __init__(self, output_dim, attention_vec, attn_activation='tanh',
#                  attn_inner_activation='tanh', single_attn=False,
#                  n_attention_dim=None, **kwargs):
#         self.attention_vec = attention_vec
#         self.attn_activation = activations.get(attn_activation)
#         self.attn_inner_activation = activations.get(attn_inner_activation)
#         self.single_attention_param = single_attn
#         self.n_attention_dim = output_dim if n_attention_dim is None else n_attention_dim
#         super(AttentionLSTM, self).__init__(output_dim, **kwargs)

#     def build(self, input_shape):
#         super(AttentionLSTM, self).build(input_shape)

#         if hasattr(self.attention_vec, '_keras_shape'):
#             attention_dim = self.attention_vec._keras_shape[1]
#         else:
#             raise Exception('Layer could not be build: No information about expected input shape.')

#         self.U_a = self.layer.init((self.output_dim, self.output_dim), name='{}_U_a'.format(self.name))
#         self.b_a = keras.backend.zeros((self.output_dim,), name='{}_b_a'.format(self.name))

#         self.U_m = self.layer.init((attention_dim, self.output_dim), name='{}_U_m'.format(self.name))
#         self.b_m = keras.backend.zeros((self.output_dim,), name='{}_b_m'.format(self.name))

#         if self.single_attention_param:
#             self.U_s = self.layer.init((self.output_dim, 1), name='{}_U_s'.format(self.name))
#             self.b_s = keras.backend.zeros((1,), name='{}_b_s'.format(self.name))
#         else:
#             self.U_s = self.layer.init((self.output_dim, self.output_dim), name='{}_U_s'.format(self.name))
#             self.b_s = keras.backend.zeros((self.output_dim,), name='{}_b_s'.format(self.name))

#         self.trainable_weights += [self.U_a, self.U_m, self.U_s, self.b_a, self.b_m, self.b_s]

#         if self.initial_weights is not None:
#             self.set_weights(self.initial_weights)
#             del self.initial_weights

#     def step(self, x, states):
#         h, [h, c] = super(AttentionLSTM, self).step(x, states)
#         attention = states[4]

#         m = self.attn_inner_activation(keras.backend.dot(h, self.U_a) * attention + self.b_a)
#         # Intuitively it makes more sense to use a sigmoid (was getting some NaN problems
#         # which I think might have been caused by the exponential function -> gradients blow up)
#         s = self.attn_activation(keras.backend.dot(m, self.U_s) + self.b_s)

#         if self.single_attention_param:
#             h = h * keras.backend.repeat_elements(s, self.output_dim, axis=1)
#         else:
#             h = h * s

#         return h, [h, c]

#     def get_constants(self, x):
#         constants = super(AttentionLSTM, self).get_constants(x)
#         constants.append(keras.backend.dot(self.attention_vec, self.U_m) + self.b_m)
#         return constants
