#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Luca DeJesu, Mahir Morar, JeMarra Rivers, Colin Liu, and Chase Dannen
# Professor Anjum Chida 
# CS 4375 Section 001
# 9 May 2021


# Path for the stroke CSV file:
# /Users/luca/Downloads/healthcare-dataset-stroke-data.csv

# Get the pathname:
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
strokePath = input("Enter the path for the stroke CSV file: ")

# Read the CSV file into a pandas dataframe
strokeData = pd.read_csv(strokePath)



# Cleaning the data
# Attributes (x1,x2,...x10) (ignore the first column, its just an assigned ID number)
strokeData = strokeData.drop('id',axis=1)
# Class attribute is stroke = 1 = yes, stroke = 0 = no

# Get all counts of stroke occurences grouped by hypertension, smoking history, and heart disease.
hyper = strokeData.groupby(['hypertension', 'stroke']).agg({'stroke': 'count'}).rename(columns = {'stroke': 'count'}).reset_index()
smoking = strokeData.groupby(['smoking_status', 'stroke']).agg({'stroke': 'count'}).rename(columns = {'stroke': 'count'}).reset_index()
heart_disease = strokeData.groupby(['heart_disease', 'stroke']).agg({'stroke': 'count'}).rename(columns = {'stroke': 'count'}).reset_index()

# Lets take a look at the data groups prior to encoding:
print(smoking)
print(heart_disease)
print(hyper)



# Encode the gender as a number
strokeData['gender'] = strokeData['gender'].map({
'Male': int(0),
'Female':int(1),
'Other':int(2)})

# Encode marital history as a number
strokeData['ever_married'] = strokeData['ever_married'].map({
'Yes':int(1), 
'No':int(0)})

# Encode the work type as a number
strokeData['work_type'] = strokeData['work_type'].map({
'Private':int(3), 
'Self-employed':int(4),
'Govt_job':int(2), 
'children':int(1), 
'Never_worked':int(0)})

# Encode residence type as a number
strokeData['Residence_type'] = strokeData['Residence_type'].map({
'Urban':int(2), 
'Rural':int(1)})

# Encode smoking status
strokeData['smoking_status'] = strokeData['smoking_status'].map({
'formerly smoked':int(1),
'never smoked':int(2), 
'smokes':int(3),
'Unknown':int(0)})


########
# Fill the empty values with the average:
strokeData['bmi'] = strokeData['bmi'].fillna(strokeData['bmi'].mean())




# Splitting into training and testing sets

# Get the attributes as a matrix
X = strokeData.drop(['stroke'],axis = 1)
# Get the y values as a vector
y = strokeData.pop('stroke')

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=1)


# How many total strokes in the set?
actualStrokes = 0

for instance in y:
    if instance == 1:
        actualStrokes+=1


# Scale the data to the normal distribution for each feature
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Make a Gaussian Naive Bayes model and fit it to the training split
stroke_Model = GaussianNB()
stroke_Model.fit(X_train, y_train)


# Get predicted Y values
y_predict = stroke_Model.predict(X_test)

train_prediction = stroke_Model.predict(X_train)
test_prediction = stroke_Model.predict(X_test)

# Some explanation for what the GaussianNB did
# First, after trained on the training set, how many instances to predict in the test set?
testInstances = (stroke_Model.predict(X_test)).size


# How many strokes were predicted?
strokeCounter = 0
for instance in stroke_Model.predict(X_test):
    if instance == 1:
        strokeCounter = strokeCounter+1


print()
print("How many strokes were predicted?: ", strokeCounter, " of ", testInstances, " instances.")
print("Predicted incidence of stroke: %.2f" % ((strokeCounter/testInstances)*100), " percent." )
print("Actual incidence of stroke: %.2f" % ((actualStrokes/y.size)*100), " percent." )
print()

# We can calculate things like the true positives and false positives by using the indexes of the confusion matrix:
confusionMatrix = confusion_matrix(y_test, y_predict)
falsePositives = confusionMatrix[0][1]
truePositives = confusionMatrix[0][0]


falseNegatives = confusionMatrix[1][0]
trueNegatives = confusionMatrix[1][1]

truePosRate = truePositives/testInstances
falsePosRate = falsePositives/testInstances
falseNegRate = falseNegatives/testInstances

print("False positive rate: %.2f" % falsePosRate)
print("False negative rate: %.2f" % falseNegRate)
print()


# Get both training and testing accuracies.
train_score = stroke_Model.score(X_train,y_train)
test_score = stroke_Model.score(X_test,y_test)


print("Training accuracy: %.4f" % train_score)

print("Testing accuracy: %.4f" % test_score)

accuracy_score(y_test, y_predict)

# We get the probability for the test data and store it for use in the ROC curve
# Predict_proba gets the probability estimates for test vector 'X_test' - chance of stroke
y_prob = stroke_Model.predict_proba(X_test)[:,1]


# Get a ROC curve using false positive rate and true positive rate:
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_prob)
roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

print("TRUE POSITIVE RATE: ", truePosRate )

# Display ROC curve with seaborn:
sns.set_theme(style = 'white')
plt.figure(figsize = (5, 5))
plt.plot(false_positive_rate,true_positive_rate, color = 'red', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.title("Gaussian Naive Bayes R.O.C. Curve")
# Just a reference line
plt.plot([0, 1], [0, 1], linestyle = '--', color = 'blue')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Pie charts:
labels = 'Has hypertension, Stroke', 'Has Hypertension, No Stroke'
explode = (0, 0.1)

# These were found using the group by statements above: hyper, smoking, heart_disease.
hyperStrokeNumbers = [66, 432]
smokingStrokeNumbers = [112, 1562]
heartDiseaseStrokeNumbers = [47, 229]

fig1, ax1 = plt.subplots()
ax1.pie(hyperStrokeNumbers, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')

labels = 'Has smoked, Stroke', 'Has smoked, No Stroke'
explode = (0, 0.1)

fig1, ax1 = plt.subplots()
ax1.pie(smokingStrokeNumbers, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')

labels = 'Had heart disease, Stroke', 'Had heart disease, No Stroke'
explode = (0, 0.1)

fig1, ax1 = plt.subplots()
ax1.pie(heartDiseaseStrokeNumbers, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')

no_stroke = 5110 - actualStrokes
strokeNumbers = []
strokeNumbers.append(actualStrokes)
strokeNumbers.append(no_stroke)

colors = ["red", "blue"]
labels = 'Stroke', 'No Stroke'
explode = (0, .5)

fig1, ax1 = plt.subplots()
ax1.pie(strokeNumbers, explode=explode, labels=labels,colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')



# In[ ]:




