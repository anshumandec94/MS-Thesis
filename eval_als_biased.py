import lenskit
import pandas as pd
import numpy as np
from lenskit import batch,topn,util
from lenskit.algorithms import als
from lenskit.algorithms import basic
from lenskit.algorithms import item_knn
from lenskit import crossfold as xf
from lenskit.metrics import predict
'''
This code tests different configurations for the number of features of the reduced dimension space that performs best on
the RMSE metric for the ALS matrix factorization technique, including sanity check scores using the a baseline biasedscorer
and an item-item based scorer. 
'''
train=pd.read_csv("/project/naray190/ml-20m/ratings.csv")
test=pd.read_csv("/project/naray190/ml-20m/truncated_user_ratings.csv")
train= train[['userId','movieId','rating']]
test=test[['userId','movieId','rating']]
train.columns = ['user','item','rating']
test.columns=['user','item','rating']
algo_30als=als.BiasedMF(features=30,iterations=50,reg=0.1)
algo_40als=als.BiasedMF(features=40,iterations=50,reg=0.1)
algo_20als=als.BiasedMF(features=20,iterations=50,reg=0.1)
algo_25als=als.BiasedMF(features=25,iterations=50,reg=0.1)
algo_15als=als.BiasedMF(features=15,iterations=50,reg=0.1)
algo_50als=als.BiasedMF(features=50,iterations=50,reg=0.1)
algo_60als=als.BiasedMF(features=60,iterations=50,reg=0.1)
algo_10als=als.BiasedMF(features=10,iterations=50,reg=0.1)
algo_70als=als.BiasedMF(features=70,iterations=50,reg=0.1)
algo_80als=als.BiasedMF(features=80,iterations=50,reg=0.1)
algo_base=basic.Bias()
algo_ii=item_knn.ItemItem(nnbrs=20)
def eval(algo,train,test):
    fittable = util.clone(algo)
    algo.fit(train)
    users=test.user.unique()
    preds=algo.predict(test)

    rmse=predict.rmse(preds,test['rating'])
    return rmse


rmse_scores=pd.DataFrame(columns=['Algorithm','Dataset','RMSE'])
count=0
for train,test in xf.partition_users(train,5,xf.SampleFrac(0.2)):
    count=count+1
    newrow=pd.DataFrame({"Algorithm":['Item-Item'], "Dataset":[count], "RMSE":[eval(algo_ii,train,test)]})
    rmse_scores=rmse_scores.append(newrow)
    newrow=pd.DataFrame({"Algorithm" : ["ALS-20"], "Dataset" : [count], "RMSE": [eval(algo_20als,train,test)]})
    rmse_scores=rmse_scores.append(newrow)
    newrow=pd.DataFrame({"Algorithm" : ["ALS-30"], "Dataset" : [count], "RMSE": [eval(algo_30als,train,test)]})
    rmse_scores= rmse_scores.append(newrow)
    newrow=pd.DataFrame({"Algorithm" : ["ALS-40"], "Dataset" : [count], "RMSE" : [eval(algo_40als,train,test)]})
    rmse_scores= rmse_scores.append(newrow)
    newrow=pd.DataFrame({"Algorithm" : ["ALS-50"], "Dataset" : [count], "RMSE" : [eval(algo_50als,train,test)]})
    rmse_scores=rmse_scores.append(newrow)
    newrow=pd.DataFrame({"Algorithm" : ["ALS-60"], "Dataset" : [count], "RMSE" : [eval(algo_60als,train,test)]})
    rmse_scores=rmse_scores.append(newrow)
    newrow=pd.DataFrame({"Algorithm" : ['ALS-10'], "Dataset" : [count], "RMSE" : [eval(algo_10als,train,test)]})
    rmse_scores=rmse_scores.append(newrow)
    newrow=pd.DataFrame({"Algorithm" : ['ALS-70'], "Dataset":[count], "RMSE": [eval(algo_70als,train,test)]})
    rmse_scores=rmse_scores.append(newrow)
    newrow=pd.DataFrame({"Algorithm" : ['ALS-80'], "Dataset":[count], "RMSE": [eval(algo_80als,train,test)]})
    rmse_scores=rmse_scores.append(newrow)	
    newrow=pd.DataFrame({"Algorithm" : ['Baseline Biased Scorer'], "Dataset": [count], "RMSE" : [eval(algo_base,train,test)]})
    rmse_scores=rmse_scores.append(newrow)

 
    
	
    

rmse_scores.to_csv("RMSE_ALS_II.csv",sep=",",header=True,index=None)



