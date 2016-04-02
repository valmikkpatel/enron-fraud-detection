#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn import svm, grid_search
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")

# Function to compute fractions for the new features
def computeFraction( poi_messages, all_messages ):
    fraction = 0
    
    if all_messages != 'NaN' and all_messages != '0':
        fraction = float(poi_messages)/float(all_messages)

    return fraction



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','to_messages','from_messages','from_poi_proportion','to_poi_proportion','shared_receipt_poi_proportion','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)



### Task 2: Remove outliers
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)
data_dict.pop('LOCKHART EUGENE E',0)



### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

# Creating new features
for data in my_dataset:
	my_dataset[data]['from_poi_proportion'] = computeFraction(my_dataset[data]['from_poi_to_this_person'],my_dataset[data]['to_messages'])
	my_dataset[data]['to_poi_proportion'] = computeFraction(my_dataset[data]['from_this_person_to_poi'],my_dataset[data]['from_messages'])
	my_dataset[data]['shared_receipt_poi_proportion'] = computeFraction(my_dataset[data]['shared_receipt_with_poi'],my_dataset[data]['to_messages'])	

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Scaling the features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Splitting into training and testing features for validation
features_train, features_test, labels_train, labels_test = train_test_split(features,labels, test_size=0.3, random_state=42)

# Selecting the best features using SelectKBest
k_best = SelectKBest(k = 11)
features_train = k_best.fit_transform(features_train,labels_train)
scores = k_best.scores_
unsorted_pairs = zip(features_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
features_test = k_best.transform(features_test)




### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Decision Tree Classifier
parameters = {'min_samples_split':[2, 5, 10, 25]}
dt = DecisionTreeClassifier(splitter = 'best', criterion = 'entropy')
dt_clf = grid_search.GridSearchCV(dt, parameters)
dt_clf.fit(features_train,labels_train)
prediction = dt_clf.predict(features_test)

print "Decision Tree"
print "Precision Score: ",precision_score(labels_test, prediction)
print "Recall Score: ", recall_score(labels_test, prediction)
print dt_clf.best_params_, "\n"


# SVM Classifier
parameters = {'C':[1, 10, 50, 100, 500, 1000]}
svm = SVC(kernel = 'linear')
svm_clf = grid_search.GridSearchCV(svm, parameters, scoring = 'f1')
svm_clf.fit(features_train,labels_train)
prediction = svm_clf.predict(features_test)

print "SVC"
print "Precision Score: ",precision_score(labels_test, prediction)
print "Recall Score: ", recall_score(labels_test, prediction)
print svm_clf.best_params_, "\n"


# Naive Bayes Classifier
gnb_clf = GaussianNB()
gnb_clf.fit(features_train,labels_train)
prediction = gnb_clf.predict(features_test)

print "GaussianNB"
print "Precision Score: ",precision_score(labels_test, prediction)
print "Recall Score: ", recall_score(labels_test, prediction), "\n"




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Selecting Decision Tree Classifer as the final since it has the best precision and recall stats according to tester.py

clf = dt_clf



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
print "Data exported. Selected algorithm is Decision Trees"