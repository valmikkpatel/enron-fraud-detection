## Identifying Fraud from Enron Email Data
for Udacity's [Data Analyst Nanodegree](https://www.udacity.com/course/nd002), Project 5

### Introduction

Enron scandal has been one of the biggest financial scandal in history. It went bankrupt from a maximum valuation of $70 billion due to wisespread internal fraud and corruption. Due to resulting investigation and legal proceedings a significant amount of confidential information entered public domain.

Utilizing `scikit-learn` and machine learning methodologies, I built a "person of interest" (POI) identifier to predict guilty employees, using features from financial data and email data.

### Questions

> Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?

The goal of this project was to build a predictive model to identify POI using financial and email data of Enron employees. A POI is someone who was indicted for fraud, settled with the government, or testified in exchange for immunity.

The Enron dataset contains 146 Enron employees to investigate. Each sample in this dictionary containing 21 features. 18 people from this dataset labeled as POI. All of them have `poi` feature set as **True**. 

Outliers were found in the dataset by visualizing the financial data and combing through the list manually. Overall I was able to identify 3 outliers mentioned below.

- `TOTAL` : This record had extremely high values for financial data since it was a total of all records. There were other records with high values for financial data but those were executive employees who made large amounts of money.
- `THE TRAVEL AGENCY IN THE PARK` : This record did not represent an employee and was found by manual combing.
- `LOCKHART EUGENE E` : This entry contained no data and was founf by manual combing.



>What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.

 
I created three new features for the POI indentifier. These features were `from_poi_proportion`, `to_poi_proportion` and `shared_receipt_poi_proportion` which are respectively the proportion of messages received by POI, messages sent to POI and proportion of messages received with POI in CC. These variables were created since the absolute amount of communication with POI is not a good indicator of guilt. On the other hand the proportion of communication with POI can be a good feature. For example a IT manager might send 100s of mails to POIs but those will be a small fraction of total mails they send. On the other hand a guilty employee might send only a few emails to POIs but those will be a huge chunk of the total mail they send.

I used feature scaling on all features since I planned to use machine learning algoriths like Support Vector Machines which require feature scaling to be accurate. The scaling was achieved using the `MinMaxScaler` module.

In order to optimize and select the most important features I used the `SelectKBest` module from `sklearn`. Below are the features alongside their scores according to the module.

| Feature                 | Score↑ |
| :---------------------- | -----: |
| exercised_stock_options | 25.098 |
| total_stock_value | 24.468 |
| bonus | 21.060 |
| salary | 18.576 |
| to_poi_proportion |	 16.641 |
| deferred_income | 11.596 |
| long_term_incentive | 10.072 |
| restricted_stock | 9.347 |
| shared_receipt_poi_proportion | 9.296 |
| total_payments | 8.867 |
| loan_advances | 7.243 |
|expenses | 6.234|
|other | 4.205|
|from_poi_proportion | 3.211|
|director_fees | 2.108 |
|to_messages | 1.699 |
|deferral_payments | 0.217 |
|from_messages | 0.164 |
|restricted_stock_deferred | 0.065|

After looking at the scores I decided to keep the threshhold at 5 and selected the top 12 features to be used in the final algorithm since the other features had extremely less importance.



> What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms? 

The final algorithm that I ended up using was Decision Tree. Apart from this I also tried Support Vector Machines and Gaussian Naive Bayes algorithms. Below are the performances of the algorithms.

| Algorithm | Precision | Recall |
| :-------- | --------- | -----: |
| Decision Tree | 0.375 | 0.6 |
| Support Vector Machines | 0.333 | 0.2 |
| Gaussian Naive Bayes | 0.4 | 0.4 |

I took the decision to use Decision Tree as the final algorithm based on the above data.


> What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? 

Tuning the parameters of an algorithm is basically adjusting the values of different parameters to find the set of parameters that give the best performance for the given dataset. It is important to tune the parameters to get optimum performance.

For this project I manually tuned two parameters of Decision Tree namely `splitter` and `criterion`. I also used `GridSearchCV` to tune the `min_samples_split` parameter. The 5 settings investigated for `min_samples_split` were [2, 5, 10, 25].

> What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?

Validation is performed to ensure that a machine learning algorithm generalizes well.  A classic mistake is over-fitting, where the model is trained and performs very well on the training dataset, but markedly worse on the cross-validation and test datasets.

To validate my analysis I used stratified shuffle split cross validation developed by Udacity and defined in tester.py file. In this cross-validation method, the specified number of randomized folds are created in such a way that the percentage of samples for each class are preserved across folds. Whereas in stratified kfolds the sample is divided in to k folds of equal sizes and each fold is used as the test sample alternatively. Stratified shuffle split, used in this project, is a better cross validation since it gives finer control on the number of iterations and the proportion of samples in on each side of the train/test split.



> Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.

The evaluation metrics I used to judge performance were precision score and recall score. 

Precision score is the proportion of predicted POIs who are actually POIs. The average precision score I got was 0.35. Recall score is the proportion of actual POIs that are detected by the algorithm. The average recall score I got was 0.4.


### Conclusion

This was certainly and interesting and challenging dataset which works as a brilliant introduction to the world of machine learning. The most challenging aspect was working with sparse data.

###Related links
- [Documentation of scikit-learn 0.15][1]
- [sklearn tutorial][2]
- [Recursive Feature Elimination][3]
- [Selecting good features Part I: univariate selection][4]

[1]: http://scikit-learn.org/stable/documentation.html
[2]: http://amueller.github.io/sklearn_tutorial/
[3]: http://topepo.github.io/caret/rfe.html
[4]: http://blog.datadive.net/selecting-good-features-part-i-univariate-selection/
[5]: https://www.kaggle.com/c/the-analytics-edge-mit-15-071x/forums/t/7837/cross-validation-the-right-and-the-wrong-way
