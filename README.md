# YouTube Video Classification
Using the Python3 library, scikit learn, we implemented various algorithms tasked with classifying a video into one of ten categories.
The video's title, tags and description are used in the classification.
## How To Use
Clone the repo and then extract **filteredset.7z** from the datasets folder and move it into the same location as *run_tests.py*.

Type python3 run_tests.py to execute.

This will take a long time as it runs seven different algorithms, ten times each, while shuffling the data each time.
## Results
LinearSVC: 97.19%
SVC: 96.89%
Random Forests: 94.89%
Decision Tree: 93.94%
Bernoulli Naive Bayes: 93.83%
Multinomial Naive Bayes: 90.29%
Extra Trees Regressor: 83.45%
