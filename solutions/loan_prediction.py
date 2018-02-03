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

#craete a dicrectory if it doesn't exist
#https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
#if not makes this more robust
if not os.path.exists("featurescore"):
    os.makedirs("featurescore")
if not os.path.exists("model"):
    os.makedirs("model")    
if not os.path.exists("preds"):
    os.makedirs("preds")    

train =pd.read_csv("/Users/pengchengliu/Documents/GitHub/Loan_Prediction/data/train.csv")

original = train

print (sum(train.Loan_Status=='N')/(len(train.Loan_Status)))

