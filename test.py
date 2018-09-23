from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
dataset = datasets.load_iris()

x = dataset.data
y = dataset.target

y_labels = to_categorical(y, num_classes=3)

seed = 7
np.random.seed(seed)

def create_model(optimizer='rmsprop', init='glorot_uniform'):
	model = Sequential()
	model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
	model.add(Dense(units=6, activation='relu', kernel_initializer=init))
	model.add(Dense(units=3, activation='softmax', kernel_initializer=init))

	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	return model


def load_model(optimizer='rmsprop', init='glorot_uniform'):
	model = Sequential()
	model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
	model.add(Dense(units=6, activation='relu', kernel_initializer=init))
	model.add(Dense(units=3, activation='softmax', kernel_initializer=init))

	filepath = 'weights.best.h5'
	model.load_weights(filepath=filepath)

	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	return model


model = create_model()
history = model.fit(x, y_labels, validation_split=0.2, epochs=200, batch_size=5, verbose=0)
scores = model.evaluate(x, y_labels,verbose=0)

print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("model accuracy.png")


'''
model = create_model()
filepath = 'weights.best.h5'

checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]
model.fit(x, y_labels, validation_split=0.2, epochs=200, batch_size=5, verbose=0, callbacks=callback_list)

'''

'''
model = load_model()
scores = model.evaluate(x, y_labels, verbose=0)

print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))

'''


