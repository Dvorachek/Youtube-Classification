import numpy as np
import csv
import sklearn.datasets


#open the file
file = open ('balancedset.csv', 'r', encoding='utf8')

#read the lines into a list
data = [row for row in csv.reader(file, delimiter=',')]

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
    examples.append("{} {} {}".format(row[2], row[6], row[15]))
    trainY[cnt] = int(row[4])
    cnt += 1

docs_new = []	
cnt = 0
for row in testX:
    docs_new.append("{} {} {}".format(row[2], row[6], row[15]))
    testY[cnt] = int(row[4])
    cnt += 1	

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

#put our data together in sklearn 'bunch'
dataset = sklearn.datasets.base.Bunch (data=examples, target=trainY)

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


from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor


# get X
X = X_train_tfidf
# get Y
Y = trainY

# get X, Y test
X_test = X_new_tfidf
Y_test = testY

#tree init..     name            classifier
classifiers = [("decision tree", tree.DecisionTreeClassifier()),
               ("random forests", RandomForestClassifier(max_features=1)),
               ("random forest regressor", RandomForestRegressor(max_features=1)),
               ("extra tress regressor", ExtraTreesRegressor(max_features=1))]

for alg in classifiers:
    print("="*80)
    print("Training {}..".format(alg[0]))
    
    # train on train data
    clf = alg[1].fit(X, Y)

    print("Predicting values...")
    # predict on testing data
    predicted = clf.predict(X_test)

    # print(predicted)

    accuracy = (np.sum(predicted==Y_test))/len(Y_test)
    print("Final accuracy: {0:.2f}\n".format(accuracy*100))
    
