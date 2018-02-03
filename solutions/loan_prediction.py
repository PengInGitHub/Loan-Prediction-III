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

data_path = "/Users/pengchengliu/Documents/GitHub/Loan_Prediction/data"

#craete a dicrectory if it doesn't exist
#if not makes this more robust
if not os.path.exists("featurescore"):
    os.makedirs("featurescore")
if not os.path.exists("model"):
    os.makedirs("model")    
if not os.path.exists("preds"):
    os.makedirs("preds")    

train =pd.read_csv("train.csv")

original = train

#print (sum(train.Loan_Status=='N')/(len(train.Loan_Status)))#bad loan rate:32%


################################
#   2.num variable rankings    #
################################

feature_type = pd.read_csv(data_path + "feature_type.csv")

numeric_feature = list(feature_type[feature_type.feature_type=='Numerical'].feature_name)

type(numeric_feature)

train_numeric = train[['Loan_ID']+numeric_feature]
train_rank = pd.DataFrame(train_numeric.Loan_ID,columns=['Loan_ID'])

for feature in numeric_feature:
    train_rank['r'+feature] = train_numeric[feature].rank(method='max')
train_rank.to_csv('train_x_rank.csv',index=None)
print (train_rank.shape)#(614, 7)



