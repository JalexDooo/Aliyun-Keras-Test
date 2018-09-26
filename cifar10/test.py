from keras.datasets import cifar10
from matplotlib import pyplot as plt
from scipy.misc import toimage
import numpy as np

(x_train, y_train), (x_validation, y_validation) = cifar10.load_data()

for i in range(0, 9):
	plt.subplot(331+i)
	plt.imshow(toimage(x_train[i]))

plt.savefig("cifar10-test.png")

