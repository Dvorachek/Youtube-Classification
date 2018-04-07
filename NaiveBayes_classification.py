import csv
import numpy as np
import sklearn.datasets
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

def main():
	#category dictionary for value lookup
	categories = { 1: 'Film & Animation', 10 : 'Music', 17: 'Sports', 20 : 'Gaming', 22: 'People & Blogs', 23 : 'Comedy', 24 : 'Entertainment', 25 : 'News & Politics', 26 : 'Howto & Style', 28 : 'Science & Technology' }
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
		string = str(row[1]) + ' ' + str(row[2]) + ' ' + str(row[3])
		examples.append(string)
		trainY[cnt] = int(row[0])
		cnt += 1

	docs_new = []	
	cnt = 0
	for row in testX:
		string = str(row[1]) + ' ' + str(row[2]) + ' ' + str(row[3])
		docs_new.append(string)
		testY[cnt] = int(row[0])
		cnt += 1	

	#put our data together in sklearn 'bunch'
	dataset = sklearn.datasets.base.Bunch (data=examples, target=trainY) #, categories=categories)

	#normalize over number of words in document
	from sklearn.feature_extraction.text import TfidfVectorizer

	tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2', min_df=2, ngram_range=(1, 3), stop_words='english', analyzer='word')

	X_train_tfidf = tfidf.fit_transform(dataset.data)

	#import the classifyer
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.naive_bayes import GaussianNB
	from sklearn.naive_bayes import BernoulliNB

	#do the fitting/training
	clf = MultinomialNB().fit(X_train_tfidf, trainY)

	#Vectorize our testing documents
	X_new_counts = tfidf.transform(docs_new)

	#get predicted values
	predicted = clf.predict(X_new_counts)

	#loop through all testing docs and predictions -> get number right and guess vs. actual

	accuracy = (np.sum(predicted==testY))/len(testY)
	print("Final accuracy: ", accuracy)




	'''param_range = np.logspace(-6, -1, 5)
	train_scores, test_scores = validation_curve(
	    SVC(), X_train_tfidf, trainY, param_name="gamma", param_range=param_range,
	    cv=10, scoring="accuracy", n_jobs=1)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	plt.title("Validation Curve with SVM")
	plt.xlabel("$\gamma$")
	plt.ylabel("Score")
	plt.ylim(0.0, 1.1)
	lw = 2
	plt.semilogx(param_range, train_scores_mean, label="Training score",
	             color="darkorange", lw=lw)
	plt.fill_between(param_range, train_scores_mean - train_scores_std,
	                 train_scores_mean + train_scores_std, alpha=0.2,
	                 color="darkorange", lw=lw)
	plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
	             color="navy", lw=lw)
	plt.fill_between(param_range, test_scores_mean - test_scores_std,
	                 test_scores_mean + test_scores_std, alpha=0.2,
	                 color="navy", lw=lw)
	plt.legend(loc="best")
	plt.show()
	'''

	#below is for testing bernoullie naive bayes and tuning some parameters
	##################
	steps = [x * 0.05 for x in range(1,20)]
	stepss = [x * 0.1 for x in range(1,10)]

	# for step in steps:
		# print ("alpha = {}".format(step))
		# for bin in stepss:
	clf = BernoulliNB(alpha=0.05).fit(X_train_tfidf, trainY)

	#Vectorize our testing documents
	X_new_counts = tfidf.transform(docs_new)
	# X_new_tfidf = tfidf_transformer.transform(X_new_counts)

	#get predicted values
	predicted = clf.predict(X_new_counts)
	# predicted = clf.predict(X_new_tfidf.toarray())

	accuracy2 = (np.sum(predicted==testY))/len(testY)
	print("Final accuracy: ", accuracy)

	#loop through all testing docs and predictions -> get number right and guess vs. actual
	'''cnt = 0
	cnt_right = 0
	for doc, category in zip(docs_new, predicted):
		if (trainY[category] == testY[cnt]):
			cnt_right += 1
		#print ('{} {}'.format(trainY[category], testY[cnt]))

		cnt += 1
		
	print (cnt_right)
	print (float(cnt_right) / float(len(testY)))'''
	#########

	print ((accuracy, accuracy2))
	return ((accuracy, accuracy2))

if __name__ == "__main__":
	main()
