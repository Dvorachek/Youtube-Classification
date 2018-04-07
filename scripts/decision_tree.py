from sklearn import tree
import numpy as np
import csv
import sklearn.datasets

#****************************
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


#Vectorize our testing documents
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

#**************************************************************

# get X
X = X_train_tfidf
# get Y
Y = trainY

# get X, Y test
X_test = X_new_tfidf
Y_test = testY

print("Training decision tree..")
# initialize the tree
clf = tree.DecisionTreeClassifier()

# train on train data
clf = clf.fit(X, Y)


print("Predicting values...")
# predict on testing data
predicted = clf.predict(X_test)

# print(predicted)

accuracy = (np.sum(predicted==Y_test))/len(Y_test)
print("Final accuracy: ", accuracy)