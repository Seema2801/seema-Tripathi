#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Teema Tripathi
Batch : PGA21
Pune
"""


# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import neighbors
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt



# import dataset
path='/home/seema/.cache/.fr-cXZveU/Market Positioning of Mobile/Dataset/Mobile_data.csv'

# read the dataset
mob_price=pd.read_csv(path)


# EDA

mob_price.info()

mob_price.shape

mob_price.dtypes

# check  nulls
mob_price.isnull().count()

# check Zeroes 
mob_price[mob_price==0].count()


#  correlation : heatmap matrix
cor = mob_price.corr()
cor = np.tril(cor)
sns.heatmap(cor,vmin=-1,vmax=1,xticklabels=mob_price.columns,
yticklabels=mob_price.columns,square=False,annot=True,linewidths=1)

# from heatmap we conclude that there is high corelation between 'battery_power'and price_range,ram and price_range

sns.barplot(x='price_range',y='battery_power',data=mob_price,ci=None)
sns.barplot(x='price_range',y='ram',data=mob_price,ci=None)



    
#function to detect outliers:boxplot
for c in mob_price:
    fig=plt.figure()
    sns.boxplot(mob_price[c],color='yellow')
    
# outliers only in fc and px_height and these are domain outliers
    

# check 0
mob_price[mob_price==0].count()

# fc and pc can be 0 if their is no front camera that mean there fc is 0

#  impute 0 with other no. on px height
mob_price.px_height[mob_price.px_height==0]=mob_price.px_height.median()

# impute 0 with other no. on sc_w
mob_price.sc_w[mob_price.sc_w<=2]= np.random.randint(3,19)


# after imputing check 0
mob_price[mob_price==0].count()


# value counts
for c in mob_price:
    counts=mob_price[c].value_counts()
    print('features',c)
    print(counts)
    print('.....')
    

#distribution : Histrogram
fig=plt.figure()
for c in mob_price:
    fig=plt.figure()
    sns.distplot(mob_price[c],bins=10,color='green')



# check the distribution of the y-variable
mob_price.price_range.value_counts()
plt.count



sns.countplot(mob_price.price_range,color='brown')
plt.title('count of price_range')


# standardize the data (only features have to be standardized)
# StandardScaler
# MinMaxScaler

# make a copy of the dataset
mob_std = mob_price.copy()

ss = preprocessing.StandardScaler()
sv = ss.fit_transform(mob_std.iloc[:,:])
mob_std.iloc[:,:] = sv

# restore the original Y-value in the data_std
mob_std.price_range = mob_price.price_range

# compare the actual and transformed data
mob_price.head()
mob_std.head()

# shuffle the dataset
mob_std = mob_std.sample(frac=1)

mob_std.head()



# split the data into train/test
trainx,testx,trainy,testy=train_test_split(mob_std.drop('price_range',1),
                                           mob_std.price_range,
                                           test_size=0.30)


trainx.shape,trainy.shape
testx.shape,testy.shape


# cross-validation to determine the best K
cv_accuracy = []

n_list = np.arange(3,12,2); n_list

for n in n_list:
    model = neighbors.KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(model,trainx,trainy,cv=10,scoring='accuracy')
    cv_accuracy.append(scores.mean() )

print(cv_accuracy)  


bestK = n_list[cv_accuracy.index(max(cv_accuracy))]
print("best K = ", bestK)

# plot the Accuracy vs Neighbours to determine the best K
plt.plot(n_list,cv_accuracy)
plt.xlabel("Neighbours")
plt.ylabel("Accyuracy")
plt.title("Accuracy - Neighbours")



# build the model using the best K
m1 = neighbors.KNeighborsClassifier(n_neighbors=bestK,metric = "manhattan").fit(trainx,trainy)
# metric = "manhattan"


# predict on test data
p1 = m1.predict(testx)


# confusion matrix and classification report
df1=pd.DataFrame({'actual':testy,'predicted':p1})
df1
pd.crosstab(df1.actual,df1.predicted,margins=True)

print(classification_report(df1.actual,df1.predicted)) 

#--------------------------------------------------



# 2nd model after removing irrelevent feature and using  mahattan distance matrix


# function for feature selection
def bestFeatures(trainx,trainy):
    features = trainx.columns
    
    fscore,pval = f_classif(trainx,trainy)
    
    df = pd.DataFrame({'feature':features, 'fscore':fscore,'pval':pval})
    df = df.sort_values('fscore',ascending=False)
    return(df)


# call function
bestFeatures(trainx,trainy)




# drop  feature
mob_std.drop(columns=['pc','clock_speed','fc','talk_time','m_dep'],inplace=True)


# data_std.shape

# split the data into train/test
trainx2,testx2,trainy2,testy2=train_test_split(mob_std.drop('price_range',1),
                                           mob_std.price_range,
                                           test_size=0.3)
trainx2.shape,trainy2.shape
testx2.shape,testy2.shape

# cross-validation to determine the best K
cv_accuracy = []

n_list = np.arange(3,12,2); n_list

for n in n_list:
    model = neighbors.KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(model,trainx2,trainy2,cv=15,scoring='accuracy')
    cv_accuracy.append(scores.mean() )

print(cv_accuracy)  

bestK = n_list[cv_accuracy.index(max(cv_accuracy))]
print("best K = ", bestK)



# build the model using the best K
m2 = neighbors.KNeighborsClassifier(n_neighbors=bestK,metric='manhattan').fit(trainx2,trainy2)

# predict on test data
p2 = m2.predict(testx2)

# confusion matrix and classification report
df2=pd.DataFrame({'actual':testy2,'predicted':p2})
df2

pd.crosstab(df2.actual,df2.predicted,margins=True)
print(classification_report(df2.actual,df2.predicted))      

#-------------------------------------------


# model 3 based on maxmin standarized and manhattan

# copy dataset 
mob_maxmin=mob_price.copy()


# MinMaxScalar
mm=preprocessing.MinMaxScaler()

sv = mm.fit_transform(mob_maxmin.iloc[:,:])
mob_maxmin.iloc[:,:] = sv

# restore the original Y-value in the data_std
mob_maxmin.price_range = mob_price.price_range

# compare the actual and transformed data
mob_price.head()
mob_maxmin.head()

# shuffle the dataset
mob_maxmin = mob_maxmin.sample(frac=1)
mob_maxmin.head()



# split the data into train/test
trainx3,testx3,trainy3,testy3=train_test_split(mob_maxmin.drop('price_range',1),
                                           mob_maxmin.price_range,
                                           test_size=0.3)
trainx3.shape,trainy3.shape
testx3.shape,testy3.shape

# cross-validation to determine the best K
cv_accuracy = []

n_list = np.arange(3,12,2); n_list

for n in n_list:
    model = neighbors.KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(model,trainx3,trainy3,cv=10,scoring='accuracy')
    cv_accuracy.append(scores.mean() )

print(cv_accuracy)  

bestK = n_list[cv_accuracy.index(max(cv_accuracy))]
print("best K = ", bestK)


# build the model using the best K
m3 = neighbors.KNeighborsClassifier(n_neighbors=bestK,metric ="minkowski").fit(trainx3,trainy3)


# predict on test data
p3 = m3.predict(testx3)

# confusion matrix and classification report
df3=pd.DataFrame({'actual':testy3,'predicted':p3})
df3
pd.crosstab(df3.actual,df3.predicted,margins=True)
print(classification_report(df3.actual,df3.predicted))

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# model 4 without standarised


# shuffle the dataset
mob_price = mob_price.sample(frac=1)
mob_std.head()

# data_std.shape

# split the data into train/test
trainx4,testx4,trainy4,testy4=train_test_split(mob_price.drop('price_range',1),
                                           mob_price.price_range,
                                           test_size=0.25)
trainx.shape,trainy.shape
testx.shape,testy.shape

# cross-validation to determine the best K
cv_accuracy = []

n_list = np.arange(3,12,2); n_list

for n in n_list:
    model = neighbors.KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(model,trainx4,trainy4,cv=10,scoring='accuracy')
    cv_accuracy.append(scores.mean() )

print(cv_accuracy)  

bestK = n_list[cv_accuracy.index(max(cv_accuracy))]
print("best K = ", bestK)

# plot the Accuracy vs Neighbours to determine the best K
plt.plot(n_list,cv_accuracy)
plt.xlabel("Neighbours")
plt.ylabel("Accyuracy")
plt.title("Accuracy - Neighbours")

# build the model using the best K
m4 = neighbors.KNeighborsClassifier(n_neighbors=bestK,metric = "euclidean").fit(trainx4,trainy4)
# metric = "manhattan"
# predict on test data
p4 = m4.predict(testx4)

# confusion matrix and classification report
df4=pd.DataFrame({'actual':testy4,'predicted':p4})
df4
pd.crosstab(df4.actual,df4.predicted,margins=True)
print(classification_report(df4.actual,df4.predicted))


print(classification_report(df1.actual,df1.predicted))
print(classification_report(df2.actual,df2.predicted))
print(classification_report(df3.actual,df3.predicted))

#---------------------------------------------

'''
# conclusion

model1: using StandardScaler standardisation
we got
                  precision    recall  f1-score   support
           0       0.79      0.89      0.84       156
           1       0.62      0.59      0.60       155
           2       0.60      0.63      0.61       150
           3       0.88      0.77      0.82       139

    accuracy                           0.72       600
   macro avg       0.72      0.72      0.72       600
weighted avg       0.72      0.72      0.72       600


Model2:
    using feature_selection on Model1
               precision    recall  f1-score   support

           0       0.89      0.87      0.88       168
           1       0.68      0.71      0.70       154
           2       0.65      0.71      0.68       140
           3       0.93      0.80      0.86       138

    accuracy                           0.78       600
   macro avg       0.79      0.78      0.78       600
weighted avg       0.79      0.78      0.78       600


# Model3 using minmaxscalar

                 precision    recall  f1-score   support

           0       0.71      0.85      0.77       108
           1       0.58      0.50      0.54       129
           2       0.53      0.63      0.57       124
           3       0.85      0.67      0.75       139

    accuracy                           0.66       500
   macro avg       0.67      0.66      0.66       500


Model 4 : without standardisation 

                precision    recall  f1-score   support

           0       0.98      0.98      0.98       130
           1       0.91      0.95      0.93       110
           2       0.93      0.88      0.90       122
           3       0.95      0.96      0.95       138

    accuracy                           0.94       500
   macro avg       0.94      0.94      0.94       500
weighted avg       0.94      0.94      0.94       500



# model4 is best here because class of Y variable is balanced and data is small

'''




    