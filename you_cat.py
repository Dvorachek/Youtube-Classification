#category dictionary for value lookup
categories = { 1: 'Film & Animation', 10 : 'Music', 17: 'Sports', 20 : 'Gaming', 22: 'People & Blogs', 23 : 'Comedy', 24 : 'Entertainment', 25 : 'News & Politics', 26 : 'Howto & Style', 28 : 'Science & Technology' }

import csv
import numpy as np
import sklearn.datasets

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

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

#put our data together in sklearn 'bunch'
dataset = sklearn.datasets.base.Bunch (data=examples, target=trainY) #, categories=categories)
#vectorize
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(dataset.data)

#normalize over number of words in document
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#import the classifyer
from sklearn.naive_bayes import MultinomialNB

#do the fitting/training
clf = MultinomialNB().fit(X_train_tfidf, trainY)

#Vectorize our testing documents
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

#get predicted values
predicted = clf.predict(X_new_tfidf)

#loop through all testing docs and predictions -> get number right and guess vs. actual
cnt = 0
cnt_right = 0
for doc, category in zip(docs_new, predicted):
	if (trainY[category] == testY[cnt]):
		cnt_right += 1
	print ('{} {}'.format(trainY[category], testY[cnt]))

	cnt += 1
	
print (cnt_right)