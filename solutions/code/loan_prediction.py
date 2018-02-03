################################
#       1.prepare data         #
################################
import numpy as np
import pandas as pd
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

print (train_rank.shape)#(614, 5)
print (test_rank.shape)#(367, 5)

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

#############################################
#   4.frequency of ranking discretization   #
#############################################

#count of discretization of rankings
train_x['n1'] = (train_x==1).sum(axis=1)
train_x['n2'] = (train_x==2).sum(axis=1)
train_x['n3'] = (train_x==3).sum(axis=1)
train_x['n4'] = (train_x==4).sum(axis=1)
train_x['n5'] = (train_x==5).sum(axis=1)
train_x['n6'] = (train_x==6).sum(axis=1)
train_x['n7'] = (train_x==7).sum(axis=1)
train_x['n8'] = (train_x==8).sum(axis=1)
train_x['n9'] = (train_x==9).sum(axis=1)
train_x['n10'] = (train_x==10).sum(axis=1)
train_x[['Loan_ID','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']].to_csv(data_path+'train_x_nd.csv',index=None)

test_x['n1'] = (test_x==1).sum(axis=1)
test_x['n2'] = (test_x==2).sum(axis=1)
test_x['n3'] = (test_x==3).sum(axis=1)
test_x['n4'] = (test_x==4).sum(axis=1)
test_x['n5'] = (test_x==5).sum(axis=1)
test_x['n6'] = (test_x==6).sum(axis=1)
test_x['n7'] = (test_x==7).sum(axis=1)
test_x['n8'] = (test_x==8).sum(axis=1)
test_x['n9'] = (test_x==9).sum(axis=1)
test_x['n10'] = (test_x==10).sum(axis=1)
test_x[['Loan_ID','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']].to_csv(data_path+'test_x_nd.csv',index=None)

##############################################
#   5.feature importance of rank features    #
##############################################
#generate a variety of xgboost models to have rank feature importance

import xgboost as xgb
import sys
import _pickle as cPickle
import os

#load data
train_x = pd.read_csv(data_path+"train_x_rank.csv")
train_label=train[['Loan_Status','Loan_ID']]
#convert Y to 0 and N to 1
train_label['Loan_Status'] = train_label['Loan_Status'].map({'Y': 0, 'N': 1})

train_xy = pd.merge(train_x,train_label,on='Loan_ID')
y = train_xy.Loan_Status

#leave features only
train_x= train_xy.drop(["Loan_ID",'Loan_Status'],axis=1)
#convert to percentage 
X = train_x/len(train_x)
#to xgb.DMatrix format
dtrain = xgb.DMatrix(X, label=y)

#do the same to test table    
test = pd.read_csv(data_path+"test_x_rank.csv")
test_uid = test.Loan_ID
test = test.drop("Loan_ID",axis=1)
test_x = test/len(test)
dtest = xgb.DMatrix(test_x)

#define an xgb model to do calculate feature importance 
def pipeline(iteration,random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight):
    params={
    	'booster':'gbtree',
    	'objective': 'binary:logistic',
    	#'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
        'eval_metric': 'error',
    	'gamma':gamma,
    	'max_depth':max_depth,
    	'lambda':lambd,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight, 
        'eta': 0.04,
    	'seed':random_seed,
        }
    
    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=30000,evals=watchlist)
    model.save_model('./model/xgb{0}.model'.format(iteration))
    
    #predict test set
    test_y = model.predict(dtest)
    test_result = pd.DataFrame(columns=["Loan_ID","score"])
    test_result.uid = test_uid
    test_result.score = test_y
    test_result.to_csv("./preds/xgb{0}.csv".format(iteration),index=None,encoding='utf-8')
    
    #save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('./featurescore/feature_score_{0}.csv'.format(iteration),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)



if __name__ == "__main__":
    random_seed = list(range(1000,2000,10))
    gamma = [i/1000.0 for i in list(range(100,200,1))]
    max_depth = [6,7,8]
    lambd = list(range(100,200,1))
    subsample = [i/1000.0 for i in list(range(500,700,2))]
    colsample_bytree = [i/1000.0 for i in list(range(250,350,1))]
    min_child_weight = [i/1000.0 for i in list(range(200,300,1))]
    random.shuffle(random_seed)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    
#save params for reproducing
    with open('params.pkl','wb') as f:
        cPickle.dump((random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight),f)
    

#train 10 xgb
    for i in list(range(100)):
        pipeline(i,random_seed[i],gamma[i],max_depth[i%3],lambd[i],subsample[i],colsample_bytree[i],min_child_weight[i])


#calculate average feature score for ranking features

#get rank feature importance info from the xgboost models

#featurescore folder contains csv files called feature_score_* that tells feature importance

files = os.listdir('featurescore')
#save into a dict
fs = {}
for f in files:
    t = pd.read_csv('featurescore/'+f)
    t.index = t.feature
    t = t.drop(['feature'],axis=1)
    d = t.to_dict()['score']
    for key in d:
        if key in fs:
            fs[key] += d[key]
        else:
            fs[key] = d[key] 
       
#sort and organize the dict            
fs = sorted(fs.items(), key=lambda x:x[1],reverse=True)

t = []
type(t)
for (key,value) in fs:
    t.append("{0},{1}\n".format(key,value))

#save the overall importance scores of ranking features into csv
with open('rank_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(t)

##############################################
#     6.feature importance of raw features   #
##############################################
train_x = train_numeric
train_label=original[['Loan_Status','Loan_ID']]
#convert Y to 0 and N to 1
train_label['Loan_Status'] = train_label['Loan_Status'].map({'Y': 0, 'N': 1})

train_xy = pd.merge(train_x,train_label,on='Loan_ID')
train_xy, test_xy = train_test_split(train_xy, test_size = 0.2)

#train_y = train_xy[['Loan_ID']+['Loan_Status']]
#test_y = test_xy[['Loan_ID']+['Loan_Status']]
#label of xgb
y=train_xy.Loan_Status


###########
#  train  #
###########
#leave features only
X= train_xy.drop(["Loan_ID",'Loan_Status'],axis=1)
#to xgb.DMatrix format
dtrain = xgb.DMatrix(X, label=y)

###########
#   test  #
###########
#do the same to test table    
test = test_xy
test_uid = test.Loan_ID
test = test.drop(["Loan_ID",'Loan_Status'],axis=1)
dtest = xgb.DMatrix(test_x)

##########################
#   feature importance   #
##########################
#define an xgb model to do calculate feature importance 
def pipeline(iteration,random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight):
    params={
    	'booster':'gbtree',
    	'objective': 'binary:logistic',
    #	'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
        'eval_metric': 'error',
    	'gamma':gamma,
    	'max_depth':max_depth,
    	'lambda':lambd,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight, 
        'eta': 0.04,
    	'seed':random_seed,
        }
    
    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=30000,evals=watchlist)
    model.save_model('./model/xgb{0}.model'.format(iteration))
    
    #predict test set
    test_y = model.predict(dtest)
    test_result = pd.DataFrame(columns=["uid","score"])
    test_result.uid = test_uid
    test_result.score = test_y
    test_result.to_csv("./preds/xgb{0}.csv".format(iteration),index=None,encoding='utf-8')
    
    #save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('./featurescore/feature_score_{0}.csv'.format(iteration),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)
        

if __name__ == "__main__":
    random_seed = list(range(1000,2000,10))
    gamma = [i/1000.0 for i in list(range(100,200,1))]
    max_depth = [6,7,8]
    lambd = list(range(100,200,1))
    subsample = [i/1000.0 for i in list(range(500,700,2))]
    colsample_bytree = [i/1000.0 for i in list(range(250,350,1))]
    min_child_weight = [i/1000.0 for i in list(range(200,300,1))]
    random.shuffle(random_seed)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    
#save params for reproducing
    with open('params.pkl','wb') as f:
        cPickle.dump((random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight),f)
    


#train 10 xgb
    for i in list(range(100)):
        pipeline(i,random_seed[i],gamma[i],max_depth[i%3],lambd[i],subsample[i],colsample_bytree[i],min_child_weight[i])

#run from here
##################################
#   average feature importance   #
##################################

#get rank feature importance info from the xgboost models
import pandas as pd 
import os

#featurescore folder contains csv files called feature_score_* that tells feature importance

files = os.listdir('featurescore')
#save into a dict
fs = {}
for f in files:
    t = pd.read_csv('featurescore/'+f)
    t.index = t.feature
    t = t.drop(['feature'],axis=1)
    d = t.to_dict()['score']
    for key in d:
        if key in fs:
            fs[key] += d[key]
        else:
            fs[key] = d[key] 
       
#sort and organize the dict            
fs = sorted(fs.items(), key=lambda x:x[1],reverse=True)

t = []
type(t)
for (key,value) in fs:
    t.append("{0},{1}\n".format(key,value))

#save the overall importance scores of ranking features into csv
with open('raw_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(t)
