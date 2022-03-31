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
from content import *

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
	c = ContentFeatures(url)
	features = []
	features.append(c.url_host_is_ip())
	features.append(c.url_page_entropy())
	features.append(c.number_of_script_tags())
	features.append(c.script_to_body_ratio())
	features.append(c.length_of_html())
	features.append(c.number_of_page_tokens())
	features.append(c.number_of_sentences())
	features.append(c.number_of_punctuations())
	features.append(c.number_of_distinct_tokens())
	features.append(c.number_of_capitalizations())
	features.append(c.average_number_of_tokens_in_sentence())
	features.append(c.number_of_html_tags())
	features.append(c.number_of_hidden_tags())
	features.append(c.number_iframes())
	features.append(c.number_objects())
	features.append(c.number_embeds())
	features.append(c.number_of_hyperlinks())
	features.append(c.number_of_whitespace())
	features.append(c.number_of_included_elements())
	features.append(c.number_of_suspicious_elements())
	features.append(c.number_of_double_documents())
	features.append(c.number_of_eval_functions())
	features.append(c.average_script_length())
	features.append(c.average_script_entropy())
	features.append(c.number_of_suspicious_functions())
	X.append(features)




y = y.tolist()
y = [(x == 'good') for x in y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(30, 20, 10, 5), max_iter=2000)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)
# 
cnt = 0
for i in range(0, len(predictions)):
	if(predictions[i] == y_test[i]):
		cnt+=1
print("Accuracy of our model is : "+str(cnt/len(predictions)))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))