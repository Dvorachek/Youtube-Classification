import csv
import numpy as np
import sklearn.datasets
import re
from sklearn.feature_extraction.text import CountVectorizer

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

	from sklearn.feature_extraction.text import TfidfVectorizer

	tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2', min_df=2, ngram_range=(1, 2), stop_words='english')
	X_train_tfidf = tfidf.fit_transform(dataset.data)

	from sklearn.svm import SVC
	from sklearn.svm import LinearSVC
	from sklearn.svm import NuSVC

	# clf = SVC(C=10, kernel='linear').fit (X_train_tfidf, trainY)
	clf = LinearSVC(C=10).fit (X_train_tfidf, trainY)

	#Vectorize our testing documents
	X_new_counts = tfidf.transform(docs_new)

	#get predicted values
	predicted = clf.predict(X_new_counts)

	accuracy = (np.sum(predicted==testY))/len(testY)
	print("LinearSVC Final accuracy: ", accuracy)

	clf = SVC(C=60, kernel='linear').fit (X_train_tfidf, trainY)
	# clf = LinearSVC(C=10).fit (X_train_tfidf, trainY)

	#Vectorize our testing documents
	X_new_counts = tfidf.transform(docs_new)

	#get predicted values
	predicted = clf.predict(X_new_counts)

	accuracy2 = (np.sum(predicted==testY))/len(testY)
	print("SVC Final accuracy: ", accuracy)

	test = ["super cool gadget that fulfills all your tech needs. google. amazon. iphone. android. buy this 16gb harddrive right now and save your cloud computing computer compute", 
			"did you see that new movie? in theatres now - a trailer about a star war. blockbuster. movie. now playing. seen it. scene it. with amazing actors. action. drama. movies. cool film. 10/10. director. actor. star. huge budget. must see!", 
			"breaking news. russia invades canada. canadians vote to register as russian citizens. borders are being debated in parliament. news. coverage. current events. cbc reporting live on the groud. trump. terrorists. maga. make america great again.",
			"dog rescues baby kittens. cute. cat. kitten. dog. rescue. mother. animals. babies. adoption. meow woof. sasha a blind burmese mountain dog, without her own puppers, risks life and tail to save some cutie cutie kittens"]
	x = tfidf.transform(test)

	predicted = clf.predict(x)
	print ('prediction for {} is {}'.format(test, predicted))

	return ((accuracy, accuracy2))

if __name__ == "__main__":
	main()
