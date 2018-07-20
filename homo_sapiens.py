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
		1. Escherichia_coli.arff
		2. Saccharomyces_cerevisiae.arff
		3. Homo_sapiens.arff
"""


#data = arff.loadarff('Escherichia_coli.arff')
#data = arff.loadarff('Saccharomyces_cerevisiae.arff')
data = arff.loadarff('Homo_sapiens.arff')

df = pd.DataFrame(data[0])

ar = np.array(df);
X = ar[:, :-1] # transform the INPUT elements 
Y = ar[:, -1:] # transform the output elements (only last element)

# one hot encode the output
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Y)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])]) convert back to class

# split test/validation
x_train, x_test, y_train, y_test = train_test_split(X, Y_onehot_encoded, test_size=0.2)


model = Sequential()
model.add(Dense(300, input_dim=64, activation='tanh', kernel_initializer="uniform"))
model.add(Dense(120, input_dim=64, activation='tanh', kernel_initializer="uniform"))
model.add(Dense(80, activation='relu', kernel_initializer="uniform"))
model.add(Dense(2, activation='softmax', kernel_initializer="uniform"))
# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size=200

# Fit the model
history = model.fit(x_train, y_train, epochs=15, batch_size=batch_size, validation_split=0.2, verbose=1)

classes = model.predict(x_test, batch_size=batch_size)

score, acc = model.evaluate(x_test, y_test, verbose = 1, batch_size=batch_size)

print("Score", score)
print("Acc", acc)

"""
('Score', 0.14707466438950606)
('Acc', 0.9754520925840735)
Epoch 150/150
44058/44058 [==============================] - 2s 50us/step - loss: 0.0039 - acc: 0.9987 - val_loss: 0.1637 - val_acc: 0.9736
"""




