#!/usr/bin/env python
# coding: utf-8

# In[39]:


# Luca DeJesu, Mahir Morar, JeMarra Rivers, Colin Liu, and Chase Dannen
# Professor Anjum Chida 
# CS 4375 Section 001
# 9 May 2021



import pandas
import pydotplus
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split 



# Read in the file as a CSV file
dataset = pandas.read_csv("/Users/luca/Downloads/healthcare-dataset-stroke-data.csv")

print(dataset.head())



# Encode the data for strings into numbers using .map:
gender = {'Male': 0, 'Female': 1, 'Other': 2}
dataset['gender'] = dataset['gender'].map(gender)

marry = {'Yes': 1, 'No': 0}
dataset['ever_married'] = dataset['ever_married'].map(marry)

work = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
dataset['work_type'] = dataset['work_type'].map(work)

live = {'Urban': 0, 'Rural': 1}
dataset['Residence_type'] = dataset['Residence_type'].map(live)

smoke = {'never smoked': 0, 'smokes': 1, 'formerly smoked': 2, 'Unknown': 3}
dataset['smoking_status'] = dataset['smoking_status'].map(smoke)


# Show encoded data head
print(dataset.head())


# These features will make sure to ignore the useless ID number for attributes:
strokeFeatures = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'smoking_status']

X = dataset[strokeFeatures]

# The column for stroke is the class attribute.
y = dataset['stroke']

# Return a decision tree of depth 3.
strokeTree = DecisionTreeClassifier(max_depth=3)

# Get training and testing splits:
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=1)

# Fit to the training data:
strokeTree.fit(X_train, y_train)

# Get predicted Y values
y_predict = strokeTree.predict(X_test)

# Get predicted values
y_prob = strokeTree.predict_proba(X_test)[:,1]



# ROC Curve

# Get the false positive rate, the true positive rate, and make a ROC curve on it
# Thresholds: how many instances to be predicted based on a boundary?
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_prob)
roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

sns.set_theme(style = 'white')
plt.figure(figsize = (5, 5))
plt.plot(false_positive_rate,true_positive_rate, color = 'red', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.title("Decision Trees R.O.C. Curve")
# Just a reference line
plt.plot([0, 1], [0, 1], linestyle = '--', color = 'blue')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Show the text version of the tree before the graphical image
text_representation = tree.export_text(strokeTree, feature_names=strokeFeatures)
print(text_representation)

test_score = strokeTree.score(X_test,y_test)

print("Overall accuracy of the decision tree accuracy (on the test split): %.4f" % test_score)


classNames = ["0","1"]
imagePlot = plt.figure(figsize=(25,20))
_ = tree.plot_tree(strokeTree, 
                   feature_names=strokeFeatures,
                   class_names = classNames,
                   filled=True)

# Goes to downloads.
imagePlot.savefig("decision_tree_strokes.png")


# In[ ]:




