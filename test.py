from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


def create_model(units_list=[13], optimizer='adam', init='normal'):
	model = Sequential()
	units = units_list[0]
	model.add(Dense(units=units, activation='relu', input_dim=13, kernel_initializer=init))

	for units in units_list[1:]:
		model.add(Dense(units=units, activation='relu', kernel_initializer=init))

	model.add(Dense(units=1, kernel_initializer=init))
	model.compile(loss='mse', optimizer=optimizer)

	return model


datasets = datasets.load_boston()
x = datasets.data
y = datasets.target

seed = 7
np.random.seed(seed)

model = KerasRegressor(build_fn=create_model, epochs=200, batch_size=5, verbose=0)

param_grid = {}
param_grid['units_list'] = [[20], [13, 6]]
param_grid['optimizer'] = ['rmsprop', 'adam']
param_grid['init'] = ['glorot_uniform', 'normal']
param_grid['epochs'] = [100, 200]
param_grid['batch_size'] = [5, 20]

scaler = StandardScaler()
scaler_x = scaler.fit_transform(x)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
results = grid.fit(scaler_x, y)

print('Best: %f using %s' % (results.best_score_, results.best_params_))
means = results.cv_results_['mean_test_score']
stds = results.cv_results_['std_test_score']
params = results.cv_results['params']

for mean, std, param in zip(means, stds, params):
	print('%f (%f) with %r' % (mean, std, param))


