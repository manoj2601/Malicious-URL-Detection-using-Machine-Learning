#!/usr/bin/env python

import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from time import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from helper import *

# Reading data from csv file
data = pd.read_csv("data.csv")

# Labels
y = data["label"]

# Features
url_list = data["url"]



url_list = url_list.tolist()
X = []
for i in range(0, len(url_list)):
	url = url_list[i]
	features = []
	features.append(numDots(url))
	features.append(subDomainLevel(url))
	features.append(pathLevel(url))
	features.append(urlLength(url))
	features.append(numDash(url))
	features.append(numDashInHostName(url))
	features.append(atSymbol(url))
	features.append(tildeSymbol(url))
	features.append(numUnderscore(url))
	features.append(numPercent(url))
	features.append(numAmpersand(url))
	features.append(numHash(url))
	features.append(numNumericChars(url))
	features.append(ipAddr(url))
	features.append(hostnameLength(url))
	features.append(pathLength(url))
	features.append(numSensitiveWords(url))
	X.append(features)




y = y.tolist()
y = [(x == 'good') for x in y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=1000)
t1 = time()
mlp.fit(X_train, y_train)
t2 = time()
t3 = time()
predictions = mlp.predict(X_test)
t4 = time()
# 
cnt = 0
for i in range(0, len(predictions)):
	if(predictions[i] == y_test[i]):
		cnt+=1
print("Accuracy of our model is : "+str(cnt/len(predictions)))
conf_matrix = confusion_matrix(y_test,predictions)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.savefig('NN.png')

print("training time: "+str(t2-t1))
print("testing time: "+str(t4-t3))