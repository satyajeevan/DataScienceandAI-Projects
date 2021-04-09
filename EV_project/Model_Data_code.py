# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:59:33 2021

@author: satya
"""

#Loading the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random

Ev=pd.read_csv(r'EvBatteryData.csv')
Ev.info()

#Renaming the columns
Ev.columns = ["NV","Current","Overweight","EVModel","CRating","WeeklyChargeCycles","AgeOfBattery","DrivingCond","ChargingType","OutsideTemp","AuxLoad","ReducedRange","Output"]


#Checking null values
Ev.isnull().sum()

# finding the categorical variables
categorical =[var for var in Ev.columns if Ev[var].dtype == 'O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)

Numerical = [var for var in Ev.columns if Ev[var].dtype!='O']
print('There are {} Numerical variables\n'.format(len(Numerical)))
print('The Numerical variables are :\n\n', Numerical)

#Convert all categorical values to upper string
for i in categorical:
    print(i)
    Ev[i]=Ev[i].str.upper()
    
# Dropping column EVmodel, because Based on the model we cannot say anything regarding its battery breakdown
Ev.drop(['EVModel'], axis=1,inplace=True)

categorical =[var for var in Ev.columns if Ev[var].dtype == 'O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)


#Checking Categorical Columns Unique values
print(Ev.DrivingCond.unique(),
Ev.ChargingType.unique(),
Ev.Output.unique())

#Replacing same value entries as one
Ev.DrivingCond = Ev.DrivingCond.replace("HIGHWAY DRIVING", "HIGHWAY")
Ev.DrivingCond = Ev.DrivingCond.replace(["HILL STATION", "HILL", "HILLWAY DRIVING","HILL ROAD"], "HILL")
Ev.DrivingCond = Ev.DrivingCond.replace("RAIN/LOW VISIBILITY", "RAINY")
Ev.DrivingCond = Ev.DrivingCond.replace(["NORMAL DRIVING IN CITY TRAFFIC", "CITY TRAFFIC","CITY","CITY DAYLIGHT"],"CITY_DAYLIGHT")
Ev.DrivingCond = Ev.DrivingCond.replace("CITY NIGHT", "CITY_NIGHT")

Ev.ChargingType = Ev.ChargingType.replace(["LEVEL 1", "LEVEL 1 "], "LEVEL_1")
Ev.ChargingType = Ev.ChargingType.replace("LEVEL 2", "LEVEL_2")
Ev.ChargingType = Ev.ChargingType.replace("DC FAST CHARGING", "DC_FAST_CHARGING")

#label encoding on categorical data

from sklearn import preprocessing
labelEncoder = preprocessing.LabelEncoder() 
  
mapping_dict ={} 
for col in categorical: 
    Ev[col] = labelEncoder.fit_transform(Ev[col]) 
  
    le_name_mapping = dict(zip(labelEncoder.classes_, 
                        labelEncoder.transform(labelEncoder.classes_))) 
    mapping_dict[col]= le_name_mapping 
print(mapping_dict) 


#head the data
Ev.head(5)

#Separating continuous columns
Ev_cont = Ev[['NV', 'Current', 'Overweight', 'CRating', 'WeeklyChargeCycles',
       'AgeOfBattery', 'OutsideTemp', 'AuxLoad', 'ReducedRange']]
Ev_cont.columns

#Discrete columns
Ev = Ev.drop(Ev_cont.columns, axis = 1)


#------------------------ Exploratory data analysis -----------------------------#

Ev_cont.head()


#Checking the distribution of the continuous data
sns.distplot(Ev_cont.NV, kde = True) #cont data have Bi-modal data with negatviely Skewed
sns.distplot(Ev_cont.Current, kde = True) #Current have normally distributed data
sns.distplot(Ev_cont.Overweight, kde = True) #Overweight have Bi-modal data with positive Skewness
sns.distplot(Ev_cont.CRating, kde = True) #CRating have Bi-Modal data with normal distribution
sns.distplot(Ev_cont.WeeklyChargeCycles, kde = True) #WeeklyChargeCycles have normally distributed data
sns.distplot(Ev_cont.AgeOfBattery, kde = True) #AgeOfBateery have positively Skewed data
sns.distplot(Ev_cont.OutsideTemp, kde = True)  #OutsideTemp have negatively Skewed data
sns.distplot(Ev_cont.AuxLoad, kde = True) #AuxLoad have normally distributed data
sns.distplot(Ev_cont.ReducedRange, kde = True) #ReducedRange have positively Skewed data

## Checking skewness value
Ev_cont.NV.skew() #-1.3455359316909081
Ev_cont.Current.skew() #0.9102858003558607
Ev_cont.Overweight.skew() #2.0048259917059172 
Ev_cont.CRating.skew() #0.35567044257189123
Ev_cont.WeeklyChargeCycles.skew() #0.24167952961061875
Ev_cont.AgeOfBattery.skew() #4.136964910147778
Ev_cont.OutsideTemp.skew() #-1.3623401980844692
Ev_cont.AuxLoad.skew() #-0.19485776964360774
Ev_cont.ReducedRange.skew() #0.996956533628171

## Checking kurtosis value
stats.kurtosis(Ev_cont.NV) #0.4712549893002951
stats.kurtosis(Ev_cont.Current) #1.9613925083392338
stats.kurtosis(Ev_cont.Overweight) #6.633391402628559
stats.kurtosis(Ev_cont.CRating) #0.16891783386370296
stats.kurtosis(Ev_cont.WeeklyChargeCycles) #-0.7123090569040027
stats.kurtosis(Ev_cont.AgeOfBattery) #42.548990581826615
stats.kurtosis(Ev_cont.OutsideTemp) #2.1627802224203645
stats.kurtosis(Ev_cont.AuxLoad) #-1.179344934464369
stats.kurtosis(Ev_cont.ReducedRange) #2.2476279353731226

#Checking for Outliers
sns.boxplot(data = Ev_cont, x = Ev_cont.NV) #Have Outliers
sns.boxplot(data = Ev_cont, x = Ev_cont.Current) #Have Outliers
sns.boxplot(data = Ev_cont, x = Ev_cont.Overweight) #Have Outliers
sns.boxplot(data = Ev_cont, x = Ev_cont.CRating) #Have Outliers
sns.boxplot(data = Ev_cont, x = Ev_cont.WeeklyChargeCycles) #No Outliers
sns.boxplot(data = Ev_cont, x = Ev_cont.AgeOfBattery) #Have Outliers
sns.boxplot(data = Ev_cont, x = Ev_cont.OutsideTemp) #Not considering under Outliers
sns.boxplot(data = Ev_cont, x = Ev_cont.AuxLoad) #No Outliers
sns.boxplot(data = Ev_cont, x = Ev_cont.ReducedRange) #Have Outliers

'''Conclusion for Outliers: Except columns WeeklyChargeCycles and AuxLoad 
rest all continous columns have Outliers present'''

## Data Distribution in one plot ##
sns.pairplot(Ev_cont, diag_kind="kde")

#histplot

Ev_cont.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

#correlation
corr = Ev_cont.corr()
corr

#checking outliers
fig, ax= plt.subplots(nrows=3, ncols=3,figsize=(16,8))
fig.tight_layout()

for i, column in enumerate(Ev_cont.columns):
  sns.boxplot(data = Ev_cont, x = Ev_cont[column],ax=ax[i//3,i%3])

#OUTLIER TREATMENT

#Separating the Outliers data from non-outlier data

Ev_outliers = Ev_cont[['NV', 'Current', 'Overweight', 'CRating', 'AgeOfBattery','ReducedRange']]
Ev_outliers.columns

#Nonoutliers

Ev_nonoutliers=Ev_cont.drop(Ev_outliers.columns,axis=1)
Ev_nonoutliers.columns

#Handling Outliers data by applying Winsorization

for data in Ev_outliers:
    
    #calculate the boundaries
    lower = Ev_outliers[data].quantile(0.17) 
    upper = Ev_outliers[data].quantile(0.83) 
    
    #Replacing the outliers
    Ev_outliers[data] = np.where(Ev_outliers[data] > upper, upper, 
                                   np.where(Ev_outliers[data] < lower, lower, Ev_outliers[data]))

#Checking the data for Outliers after applying Winsorization

fig, ax= plt.subplots(nrows=2, ncols=3,figsize=(16,8))
fig.tight_layout()

for i, column in enumerate(Ev_outliers.columns):
  sns.boxplot(data = Ev_cont, x = Ev_outliers[column],ax=ax[i//3,i%3])

sns.pairplot(Ev_cont, diag_kind="kde")

#Concatinating Non outliers and Outliers data

Ev_cont = pd.concat([Ev_nonoutliers,Ev_outliers], axis = 1)
Ev_cont.columns

#Checking the skewness values post outlier treatment

Ev_cont.NV.skew() #-1.3455359316909081
Ev_cont.Current.skew() #0.9102858003558607
Ev_cont.Overweight.skew() #2.0048259917059172 
Ev_cont.CRating.skew() #0.35567044257189123
Ev_cont.WeeklyChargeCycles.skew() #0.24167952961061875
Ev_cont.AgeOfBattery.skew() #4.136964910147778
Ev_cont.OutsideTemp.skew() #-1.3623401980844692
Ev_cont.AuxLoad.skew() #-0.19485776964360774
Ev_cont.ReducedRange.skew() #0.996956533628171

"""After applying th outliers the skewness values are near to zero
   which satisfies the normal distribution criteria"""

#Combining Continuous and Discrete data

Ev = pd.concat([Ev,Ev_cont], axis = 1)
Ev.columns

Ev.info()

Ev["Output"].value_counts()

## Displaying percentage wise
total = len(Ev['Output'])*1
ax = sns.countplot(x="Output", data=Ev,order=[0,1])
plt.title('Distribution of  Output')
plt.xlabel('Frequency')

for p in ax.patches:
        ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_y()+0.5, p.get_height()+5))


y = Ev["Output"] #Outputvariable
X = Ev.drop(["Output"], axis = 1) #Input variables
#Two ways for handling imbalanced dataset
X.columns

#Way1: Using RandomOverSampler method to balance the dataset
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
x_ros, y_ros = ros.fit_sample(X,y)
y_ros.value_counts()

corr_post=x_ros.corr()
corr_post


#Model Building
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_ros, y_ros,test_size=0.3,random_state=42)

######################## SVM model   #########################

from sklearn.svm import SVC

#grid search technique
from sklearn.model_selection import GridSearchCV 
 
# defining parameter range 
param_grid = {'C': [0.1,1,10,25],'gamma': ['auto'],'kernel': ['rbf','linear']} 

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,cv=4,n_jobs=-1)

# fitting the model for grid search 
sv_model_fit=grid.fit(x_train,y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 

#predict on test data
pred_test =sv_model_fit.predict(x_test)

#accuracy on test data
final_accuracy_test=np.mean(pred_test==y_test)
print(final_accuracy_test)


#predict on train data
pred_train= sv_model_fit.predict(x_train)

#accuracy on train data
final_accuracy_train=np.mean(pred_train==y_train)
print(final_accuracy_train)


#################Extreme gradient boosting ##########################

import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix

# n_jobs – Number of parallel threads used to run xgboost, learning_rate– Boosting learning rate ( “eta”)

xgb_clf = xgb.XGBClassifier(n_estimators=100,use_label_encoder=False)

param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.05,0.4,0.3],'learning_rate':[0.1,0.15,0.2]}

# Grid Search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(xgb_clf, param_test1,n_jobs=-1,scoring='accuracy')

xgboost_model=grid_search.fit(x_train, y_train)

cv_xg_clf = grid_search.best_estimator_

# Evaluation on Testing Data with model with hyperparameter
exg_db_test=accuracy_score(y_test,cv_xg_clf.predict(x_test))
exg_db_test
#confusion matrix of test data
confusion_matrix(y_test, cv_xg_clf.predict(x_test))

#best parameters
grid_search.best_params_

#training data accuracy
exg_db_train=accuracy_score(y_train,cv_xg_clf.predict(x_train))
exg_db_train
#plot the important features in the given data set
xgb.plot_importance(cv_xg_clf)

#####################  Random forest   ####################

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
"""
n_estimators = number of trees in the foreset
max_features = max number of features considered for splitting a node
max_depth = max number of levels in each decision tree
min_samples_split = min number of data points placed in a node before the node is split
"""

#Intialize the classifer
rf_clf_grid = RandomForestClassifier()

#tuning parameters
param_grid = {'max_depth': [80, 90],'n_estimators': [60, 100, 200],"max_features": [6,7,8,9,10], "min_samples_split": [2,3,10]}

grid_search =GridSearchCV(rf_clf_grid, param_grid,cv=10,verbose=3,n_jobs=-1)

#fit the model
RF_model= grid_search.fit(x_train,y_train)

#retrun the best parameters
grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_

cv_rf_clf_grid.fit(x_train,y_train)

from sklearn.metrics import accuracy_score, confusion_matrix
test_pred=RF_model.predict(x_test)

#test accuracy
RF_test_acc=accuracy_score(y_test, RF_model.predict(x_test))
RF_test_acc

#confusion matrix
cf_matrix=confusion_matrix(y_test, RF_model.predict(x_test))

#train accuracy
RF_train_acc=accuracy_score(y_train, RF_model.predict(x_train))
RF_train_acc

#confusion matrix
confusion_matrix(y_train, RF_model.predict(x_train))

#For avoiding the overfitting, past researches have included regularization parameters, cross validation[2] and increased the amount of training data

Best_model={'Model':["XGBoost","Random Forest","SVM"],'Accuracy':[exg_db_test,RF_test_acc,final_accuracy_test]}
EV_Analysis = pd.DataFrame(Best_model)  
EV_Analysis

########################################################################################
#Best Model Visualization - Random Forest(Train and Test Accuracy are very close)

import seaborn as sns

#Performance metrics for the classification model
#Recall = TP/(TP+FN)- Out of all the positive classes, how much we predicted correctly. It should be high as possible.
#Precision = TP/(TP+FP) 

group_names = ["True Pos","False Pos","False Neg","True Neg"]
group_counts = ["{0:.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

#Metrics for Binary Confusion Matrices
precision = cf_matrix[1,1] / sum(cf_matrix[:,1])
recall    = cf_matrix[1,1] / sum(cf_matrix[1,:])
f1_score  = 2*precision*recall / (precision + recall)
results=print("Precision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(precision,recall,f1_score))
results

y_test=pd.DataFrame(y_test)
#Classification report
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
# classification report
classification_test = classification_report(test_pred, y_test["Output"])

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(y_test["Output"],test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

#Area under curve -> 0.9 to 1 (Outstanding)
roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test

#We Are concluding upon Random forest model which gave us best fitted model

import pickle
# Open a file , where you want to store the data
file = open(r'Ev_RFmodel.pkl','wb')

# Dump information to that file
pickle.dump(RF_model,file)  
