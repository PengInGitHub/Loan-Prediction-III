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

train =pd.read_csv("ppd_train_withid.csv")

train.rename(columns={'target':'y'}, inplace=True)
train.rename(columns={'Idx':'uid'}, inplace=True)

original = train

sum(train.target)/(len(train.target)-sum(train.target))

