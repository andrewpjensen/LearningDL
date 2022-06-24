import tensorflow as tf

#BYO Dense Layer
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()

        self.W = self.add_weight([input_dim, output_dim])
        self.b = self.add_weight([1, output_dim])

    def call(self, inputs):
        z = tf.matmul(inputs, self.W) + self.b
        output = tf.math.sigmoid(z)
        return output

#tensorflow implementation
layer = tf.keras.layers.Dense(units = 2)

#model implementation
n = 5
model = tf.keras.Sequential([
    tf.keras.layers.Dense(n),
    tf.keras.layers.Dense(2)
])

def class1summary():
    def compute_loss(x,y):
        pass
    x, y = 1,2
    model = tf.keras.Sequential([...])
    #choose optimizer
    optimizer = tf.keras.optimizer.SGD()

    while True:
        #forward pass through network
        prediction = model(x)
        
        with tf.GradientTape() as tape:
            #compute loss
            loss = compute_loss(y, prediction)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

print('success')