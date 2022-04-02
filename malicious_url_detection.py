
# coding: utf-8

# In[1]:


#!/usr/bin/env python

import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import time
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# In[2]:


# Reading data from csv file
data = pd.read_csv("data.csv")

# In[3]:


# Labels
y = data["label"]

# Features
url_list = data["url"]


# In[4]:


# Using Tokenizer
vectorizer = TfidfVectorizer()

# Store vectors into X variable as Our XFeatures
X = vectorizer.fit_transform(url_list)


# In[5]:


# Split into training and testing dataset 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


# Model Building using logistic regression
logit = LogisticRegression()
t1 = time.time()
logit.fit(X_train, y_train)
t2 = time.time()

# In[7]:


# Accuracy of Our Model
# print("Accuracy of LR model is: ",logit.score(X_test, y_test))


# Logistic regression : https://github.com/cozpii/Malicious-URL-detection
t3 = time.time()
predictions = logit.predict(X_test)
t4 = time.time()
# 
y_test = y_test.tolist()
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
plt.savefig('LR_tf.png')

print("training time: "+str(t2-t1))
print("testing time: "+str(t4-t3))