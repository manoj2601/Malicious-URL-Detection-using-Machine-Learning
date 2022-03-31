#!/usr/bin/env python

import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from helper import *


from sklearn.tree import DecisionTreeRegressor 
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

logit = LogisticRegression()
logit.fit(X_train, y_train)

predictions = logit.predict(X_test)
# Accuracy of Our Model
print("Accuracy of our model is: ",logit.score(X_test, y_test))

print(predictions)

# mlp = MLPClassifier(hidden_layer_sizes=(30, 20, 10), max_iter=1000)
# mlp.fit(X_train, y_train)

# predictions = mlp.predict(X_test)
# # 
cnt = 0
for i in range(0, len(predictions)):
	if(predictions[i] == y_test[i]):
		cnt+=1
print("Accuracy of our model is : "+str(cnt/len(predictions)))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))