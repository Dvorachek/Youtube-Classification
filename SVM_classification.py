#category dictionary for value lookup
categories = { 1: 'Film & Animation', 10 : 'Music', 17: 'Sports', 20 : 'Gaming', 22: 'People & Blogs', 23 : 'Comedy', 24 : 'Entertainment', 25 : 'News & Politics', 26 : 'Howto & Style', 28 : 'Science & Technology' }

import csv
import numpy as np
import sklearn.datasets
import re
from sklearn.feature_extraction.text import CountVectorizer

#open the file
file = open ('balancedset.csv', 'r', encoding='utf8')

#read the lines into a list
data = []
[data.append(row) for row in csv.reader(file, delimiter=',')]

#split our data
trainX = data[1: int(len(data)*0.8)]
testX = data[int(len(data)*0.8)+1:]

trainY = np.zeros ((len(trainX),), dtype=np.int64)
testY = np.zeros ((len(testX),), dtype=np.int64)

#these are lists to be fed into sklearn
examples = []
text = []

#concatenating title, tags and description and add to list. Add category number to Y vector.
cnt = 0	
for row in trainX:
	string = str(row[2]) + ' ' + str(row[6]) + ' ' + str(row[15])
	examples.append(string)
	trainY[cnt] = int(row[4])
	cnt += 1

docs_new = []	
cnt = 0
for row in testX:
	string = str(row[2]) + ' ' + str(row[6]) + ' ' + str(row[15])
	docs_new.append(string)
	testY[cnt] = int(row[4])
	cnt += 1	

#put our data together in sklearn 'bunch'
dataset = sklearn.datasets.base.Bunch (data=examples, target=trainY) #, categories=categories)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2', min_df=2, ngram_range=(1, 1), stop_words='english')
X_train_tfidf = tfidf.fit_transform(dataset.data)

from sklearn.svm import SVC
clf = SVC(C=60, kernel='linear').fit (X_train_tfidf, trainY)

#Vectorize our testing documents
X_new_counts = tfidf.transform(docs_new)

#get predicted values
predicted = clf.predict(X_new_counts)

accuracy = (np.sum(predicted==testY))/len(testY)
print("Final accuracy: ", accuracy)