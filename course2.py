import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import Tensorflow and Datasets and Keras API
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
#GPU 
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#load datasets and x reshape 
(x_train,y_train), (x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1,28*28).astype('float32') / 255.0

# Sequential API
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ]
)
model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu', name = 'my_layer'))
model.add(layers.Dense(10))



#Funtional API (A bit more Flexible)
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation='relu', name='first_layer')(inputs)
x = layers.Dense(256, activation='relu', name='second_layer')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs,outputs=outputs)

print(model.summary())
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)

model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=2)
model.evaluate(x_test,y_test,batch_size=32,verbose=2)
