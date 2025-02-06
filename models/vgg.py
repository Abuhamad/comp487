import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from datasets.load_data import load_cifar10
import time

class VggBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters, n_convs, activation, initializer, regularizer, **kwargs):
        super(VggBlock, self).__init__()
        self.n_filters = n_filters
        self.n_convs = n_convs
        self.activation = activation
        self.initializer = initializer
        self.regularizer = regularizer
        self.conv_layers = [layers.Conv2D(n_filters, kernel_size=(3, 3), activation=activation, 
                                          kernel_initializer=initializer, kernel_regularizer=regularizer) for _ in range(n_convs)]
        self.max_pool = layers.MaxPooling2D(pool_size=(2, 2))
        
    def call(self, inputs):
        x = inputs
        for conv in self.conv_layers:
            x = conv(x)
        x = self.max_pool(x)
        return x
    

class Vgg(tf.keras.Model):
    def __init__(self, input_shape, output_shape, activation, output_activation, filters, poolings, fc_layers, loss, optimizer, 
                 epochs=50, batch_size=64, verbose=1, regularizer=tf.keras.regularizers.L2, regularizer_rate=1e-4, 
                 initializer=tf.keras.initializers.HeNormal(), dropout_rate=0.5, **kwargs):
        super(Vgg, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dropout_rate = dropout_rate
        self.filters = filters
        self.poolings = poolings
        self.fc_layers = fc_layers

        self.regularizer = regularizer
        self.regularizer_rate = regularizer_rate
        self.initializer = initializer
        self.activation = activation
        self.output_activation = output_activation
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = ['accuracy']
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = self.create_model()
    
    @property
    def metrics(self):
        return self._metrics
    @metrics.setter
    def metrics(self, value):
        self._metrics = value

    def create_model(self):
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=self.input_shape))
        
        # add convolutional and pooling layers
        for n_filters, n_convs in self.filters:
            model.add(VggBlock(n_filters, n_convs, self.activation, self.initializer, self.regularizer))
        
        # flatten the output of the convolutional layers
        model.add(layers.Flatten())

        # add fully connected layers
        for units in self.fc_layers:
            model.add(layers.Dense(units, activation=self.activation, kernel_initializer=self.initializer, 
                                   kernel_regularizer=self.regularizer(self.regularizer_rate)))
            model.add(layers.Dropout(self.dropout_rate))
        
        # add output layer
        model.add(layers.Dense(self.output_shape, activation=self.output_activation))
        
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        history = self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, 
                                 validation_data=(x_val, y_val), verbose=self.verbose)
        end_time = time.time()
        print(f"Time taken to train: {end_time - start_time:.2f} seconds")
        return history
    
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
    
    def predict(self, x):
        return self.model.predict(x)
    
if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10()
    model = Vgg(input_shape=(32, 32, 3), output_shape=10, activation='relu', output_activation='softmax', 
                filters=[(64, 2), (128, 2), (256, 3)], poolings=[2, 2, 0], fc_layers=[512, 256], 
                loss=losses.CategoricalCrossentropy(), optimizer=optimizers.Adam(), epochs=10)
    history = model.train(x_train, y_train, x_val, y_val)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")