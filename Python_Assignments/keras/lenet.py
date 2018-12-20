import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# print(x_train[0])
x_train = np.array(x_train) / 255
x_test = np.array(x_test) / 255
# print(x_train[0])

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

lenet = keras.models.Sequential()

lenet.add(keras.layers.Conv1D(
    filters=6,
    kernel_size=5,
    activation='relu',
    use_bias=True,
    input_shape=(28, 28)))
lenet.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))

lenet.add(keras.layers.Conv1D(filters=16, kernel_size=5, activation='relu', use_bias=True))
lenet.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))

lenet.add(keras.layers.Flatten())
# lenet.add(keras.layers.Dropout(0.2))

lenet.add(keras.layers.Dense(120, activation='relu', use_bias=True))
# lenet.add(keras.layers.Dropout(0.2))
lenet.add(keras.layers.Dense(84, activation='relu', use_bias=True))
lenet.add(keras.layers.Dense(10, activation='softmax', use_bias=True))

lenet.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy']
)

lenet.fit(x_train, y_train, epochs=10, batch_size=128)

score = lenet.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score)
print('Test accuracy:', score)

y_pred = lenet.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

print()
print(confusion_matrix(y_test, y_pred))
