################################
#       1.prepare data         #
################################
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import random
import _pickle as cPickle
import os
from sklearn.model_selection import train_test_split

data_path = "/Users/pengchengliu/Documents/GitHub/Loan_Prediction/data/"

#craete a dicrectory if it doesn't exist
#if not makes this more robust
if not os.path.exists("featurescore"):
    os.makedirs("featurescore")
if not os.path.exists("model"):
    os.makedirs("model")    
if not os.path.exists("preds"):
    os.makedirs("preds")    

train =pd.read_csv(data_path+"train.csv")
test =pd.read_csv(data_path+"test.csv")

original = train

#print (sum(train.Loan_Status=='N')/(len(train.Loan_Status)))#bad loan rate:32%


################################
#   2.num variable rankings    #
################################

feature_type = pd.read_csv(data_path + "feature_type.csv")
numeric_feature = list(feature_type[feature_type.feature_type=='Numerical'].feature_name)

train_numeric = train[['Loan_ID']+numeric_feature]
train_rank = pd.DataFrame(train_numeric.Loan_ID,columns=['Loan_ID'])

test_numeric = test[['Loan_ID']+numeric_feature]
test_rank = pd.DataFrame(test_numeric.Loan_ID,columns=['Loan_ID'])


for feature in numeric_feature:
    train_rank['r'+feature] = train_numeric[feature].rank(method='max')
    test_rank['r'+feature] = test_numeric[feature].rank(method='max')
    
train_rank.to_csv(data_path+'train_x_rank.csv',index=None)
test_rank.to_csv(data_path+'test_x_rank.csv',index=None)

print (train_rank.shape)#(614, 6)
print (test_rank.shape)#(367, 6)

#####################################
#   3.discretization of rankings    #
#####################################
#create discretization features

train_x = train_rank.drop(['Loan_ID'],axis=1)
test_x = test_rank.drop(['Loan_ID'],axis=1)
train_unlabeled = train.drop(['Loan_Status'],axis=1)
train_unlabeled_x =  train_unlabeled.drop(['Loan_ID'],axis=1)

#discretization of train ranking features
#each 10% belongs to 1 level
train_x[train_x<int(len(train_rank)/10)] = 1
train_x[(train_x>=int(len(train_rank)/10))&(train_x<int(len(train_rank)/10*2))] = 2
train_x[(train_x>=int(len(train_rank)/10*2))&(train_x<int(len(train_rank)/10*3))] = 3
train_x[(train_x>=int(len(train_rank)/10*3))&(train_x<int(len(train_rank)/10*4))] = 4
train_x[(train_x>=int(len(train_rank)/10*4))&(train_x<int(len(train_rank)/10*5))] = 5
train_x[(train_x>=int(len(train_rank)/10*5))&(train_x<int(len(train_rank)/10*6))] = 6
train_x[(train_x>=int(len(train_rank)/10*6))&(train_x<int(len(train_rank)/10*7))] = 7
train_x[(train_x>=int(len(train_rank)/10*7))&(train_x<int(len(train_rank)/10*8))] = 8
train_x[(train_x>=int(len(train_rank)/10*8))&(train_x<int(len(train_rank)/10*9))] = 9
train_x[train_x>=int(len(train_rank)/10*9)] = 10
       
#nameing rule for discretization features, add "d" in front of orginal features
#for instance "x1" would have discretization feature of "dx1"
rename_dict = {s:'d'+s[1:] for s in train_x.columns.tolist()}
train_x = train_x.rename(columns=rename_dict)
train_x['Loan_ID'] = train.Loan_ID
train_x.to_csv(data_path+'train_x_discretization.csv',index=None)

#discretization of test ranking features

test_x[test_x<int(len(test_x)/10)] = 1
test_x[(test_x>=int(len(test_x)/10))&(test_x<int(len(test_x)/10*2))] = 2
test_x[(test_x>=int(len(test_x)/10*2))&(test_x<int(len(test_x)/10*3))] = 3
test_x[(test_x>=int(len(test_x)/10*3))&(test_x<int(len(test_x)/10*4))] = 4
test_x[(test_x>=int(len(test_x)/10*4))&(test_x<int(len(test_x)/10*5))] = 5
test_x[(test_x>=int(len(test_x)/10*5))&(test_x<int(len(test_x)/10*6))] = 6
test_x[(test_x>=int(len(test_x)/10*6))&(test_x<int(len(test_x)/10*7))] = 7
test_x[(test_x>=int(len(test_x)/10*7))&(test_x<int(len(test_x)/10*8))] = 8
test_x[(test_x>=int(len(test_x)/10*8))&(test_x<int(len(test_x)/10*9))] = 9
test_x[test_x>=int(len(test_x)/10*9)] = 10

test_x = test_x.rename(columns=rename_dict)
test_x['Loan_ID'] = test.Loan_ID
test_x.to_csv(data_path+'test_x_discretization.csv',index=None)

