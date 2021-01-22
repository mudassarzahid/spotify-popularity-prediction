from sklearn.model_selection import train_test_split
from sklearn import neural_network
from sklearn import metrics
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import pandas as pd
import numpy as np
import keras


data = pd.DataFrame(pd.read_csv('multiclass_data.csv'))
X = data[['acousticness', 'danceability', 'energy', 'valence', 'instrumentalness', 'loudness', 'tempo']]
Y = data['popularity']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=420, shuffle=True)

model = neural_network.MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5,), solver='lbfgs', random_state=18)
model.fit(X_train, Y_train)

predicted = model.predict(X_test)
print("Classification Report:\n %s:" % (metrics.classification_report(Y_test, predicted)))
