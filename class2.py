import tensorflow as tf
#RNN Psuedocode Example
def mypsuedocode():
    my_rnn = (1, 2) #RNN()
    hidden_state = [0,0,0,0]
    sentence = ["I","love","recurrent","neural"]

    for word in sentence:
        prediction, hidden_state = my_rnn(word, hidden_state)

    next_word_prediction = prediction

#BYO RNN Cell
class MyRNNCell(tf.keras.layers.Layer):
    def __init__(self, rnn_units, input_dim, output_dim):
        super(MyRNNCell, self).__init__()

        #initialize weight matrices
        self.W_xh = self.add_weight([rnn_units, input_dim])
        self.W_hh = self.add_weight([rnn_units, rnn_units])
        self.W_hy = self.add_weight([output_dim, rnn_units])

        #initialize hidden state to zeros
        self.h = tf.zeros([rnn_units, 1])

    def call(self, x):
        #update hidden state
        self.h = tf.math.tanh(self.W_hh * self.h * self.W_xh * x)

        #compute output
        output = self.W_hy * self.h

        #return output and hidden state
        return output, self.h

#TF RNN Cell
rnn_units = 2
mymodel = tf.keras.layers.SimpleRNN(rnn_units)

#loss func
y = 1
predicted = 2
loss = tf.nn.softmax_cross_entropy_with_logits(y, predicted)


print('success')