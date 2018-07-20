from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax

from keras.models import Sequential
from keras.layers import Dense

"""
	Script for the files:
		1. Primaria40decada.arff
"""


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


model = Sequential()
model.add(Dense(256, input_dim=len(x_train[0]), activation='tanh', kernel_initializer="uniform"))
model.add(Dense(32, activation='sigmoid', kernel_initializer="uniform"))
model.add(Dense(4, activation='tanh', kernel_initializer="uniform"))
#model.add(Dense(10, activation='relu', kernel_initializer="uniform"))
model.add(Dense(len(y_train[0]), activation='softmax', kernel_initializer="uniform"))
# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = model.fit(x_train, y_train, epochs=200, batch_size=100, validation_split=0.33, verbose=1)

score, acc = model.evaluate(x_test, y_test, verbose = 1, batch_size = 10)

print("Score", score)
print("Acc", acc)

"""
('Score', 0.033529113298617165)
('Acc', 0.9945454561349117)

Epoch 350/350
897/897 [==============================] - 0s 131us/step - loss: 2.7945e-04 - acc: 1.0000 - val_loss: 0.0226 - val_acc: 0.9964
"""
#classes = model.predict(x_test, batch_size=20)














