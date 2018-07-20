
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense


import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm

"""
	Script for the files:
		1. Primaria40decada.arff
"""


def create_model(neurons=1):
	# create model
	model = Sequential()
	model.add(Dense(64, input_dim=1024, activation='tanh', kernel_initializer="uniform") )
	model.add(Dense(32, activation='sigmoid', kernel_initializer="uniform"))
	model.add(Dense(4, activation='tanh', kernel_initializer="uniform"))
	#model.add(Dropout(dropout_rate))
	model.add(Dense(50, activation='softmax', kernel_initializer="uniform"))
	# Compile model

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	return model

data = arff.loadarff('Primaria40decada.arff')

df = pd.DataFrame(data[0])

ar = np.array(df);
X = ar[:, :-1] # transform the INPUT elements 
Y = ar[:, -1:] # transform the output elements (only last element)

## Manyually coinverting the classes to numbers:
X_conv = []

class_to_num_dict = { '-': 0, 'a': 1, 'c': 2, 'g' : 3, 't' : 4 }

for el in X:
	X_conv.append([ class_to_num_dict[zi] for zi in el ])

X = np.array(X_conv, dtype=np.int)
# end manually converting the classes

# one hot encode the output
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Y)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


# inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])]) convert back to class

# split test/validation
x_train, x_test, y_train, y_test = train_test_split(X, Y_onehot_encoded, test_size=0.33)


model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)

# Optimizing epocs and batch size
#batch_size = [10, 20]
#epochs = [5, 50]
#param_grid = dict(batch_size=batch_size, epochs=epochs)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(X, Y_onehot_encoded)

# Optimizing activation function - softplus
#activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#param_grid = dict(activation=activation)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(X, Y_onehot_encoded)

# dropout and weight constrain
#weight_constraint = [1, 2, 3, 4, 5]
#dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(X, Y_onehot_encoded)

# Number of Neurous

neurons = [4, 8, 16, 32, 64]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y_onehot_encoded) 


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))





"""
# Fit the model
history = model.fit(x_train, y_train, epochs=350, batch_size=20, validation_split=0.33, verbose=1)

score, acc = model.evaluate(x_test, y_test, verbose = 1, batch_size = 20)

print("Score", score)
print("Acc", acc)
"""
"""
('Score', 0.033529113298617165)
('Acc', 0.9945454561349117)

Epoch 350/350
897/897 [==============================] - 0s 131us/step - loss: 2.7945e-04 - acc: 1.0000 - val_loss: 0.0226 - val_acc: 0.9964
"""
#classes = model.predict(x_test, batch_size=20)














