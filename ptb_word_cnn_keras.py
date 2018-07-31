from __future__ import print_function
import tensorboard as tf
from tensorflow import keras
import reader
import numpy as np

kernel_size = 20
path = "data/simple-examples/data"

train_data, valid_data, test_data, vocab_size, word_to_id = reader.ptb_raw_data(path)
x_train = train_data[:-1]
x_train = [x_train[i:i+kernel_size] for i in range(len(x_train)-kernel_size)]
x_train = np.asarray(x_train)
y_train = train_data[1:]
y_train = np.asarray(y_train)
x_valid = valid_data[:-1]
x_valid = [x_valid[i:i+kernel_size] for i in range(len(x_valid)-kernel_size)]
x_valid = np.asarray(x_valid)
y_valid = valid_data[1:]
y_valid = np.asarray(y_valid)
x_test = test_data[:-1]
x_test = [x_test[i:i+kernel_size] for i in range(len(x_test)-kernel_size)]
x_test = np.asarray(x_valid)
y_test = test_data[1:]
y_test = np.asarray(y_test)
id_to_word = {value: key for (key, value) in word_to_id.items()}

def decode_text(text):
    return ' '.join([id_to_word.get(i, '?') for i in text])

print(x_train.shape)

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_length = kernel_size))
#model.add(keras.layers.Flatten())
model.add(keras.layers.Conv1D(filters = 1, kernel_size = 3, padding = "same", activation = keras.activations.tanh))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv1D(filters = 1, kernel_size = 3, padding = "same", activation = keras.activations.tanh))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(vocab_size, activation = keras.activations.softmax))

print (model.summary())

model.compile(
    loss = keras.losses.sparse_categorical_crossentropy,
    optimizer = keras.optimizers.Adadelta(),
    metrics = [keras.metrics.categorical_accuracy]
)
model.fit(x_train, y_train,
          epochs = 12,
          verbose = 1,
          validation_data = (x_valid, y_valid),
          shuffle = False
)

score = model.evaluate(x_test, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])