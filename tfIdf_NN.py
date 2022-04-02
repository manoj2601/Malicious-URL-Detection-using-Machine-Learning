import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Reading data from csv file
data = pd.read_csv("data.csv")


# Labels
y = data["label"]

# Features
url_list = data["url"]



# Using Tokenizer
vectorizer = TfidfVectorizer()

# Store vectors into X variable as Our XFeatures
X = vectorizer.fit_transform(url_list)



# Split into training and testing dataset 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


# Model Building using logistic regression
# logit = LogisticRegression()
# logit.fit(X_train, y_train)


# # In[7]:


# # Accuracy of Our Model
# print("Accuracy of LR model is: ",logit.score(X_test, y_test))


# Logistic regression : https://github.com/cozpii/Malicious-URL-detection


# NN
mlp = MLPClassifier(hidden_layer_sizes=(30, 20, 10), max_iter=1000)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)
cnt = 0
for i in range(0, len(predictions)):
	if(predictions[i] == y_test[i]):
		cnt+=1
print("Accuracy of NN model is : "+str(cnt/len(predictions)))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))