import pandas as pd
import numpy as np
from lenskit import batch,topn
from lenskit.metrics import topn as tn
from lenskit.algorithms import als
from lenskit.matrix import  CSR,RatingMatrix
from scipy import spatial
import matplotlib.pyplot as plt
from lenskit import topn
from IPython.core.debugger import set_trace
from lenskit.metrics import predict
import random
import lenskit.metrics.topn as topn

##
#In this script, we take our training and folding-in data(referred to as test) and build our recommender model and implement a
#a function to generate vectors for our folding-in ratings. 
#After this, we run tests on popular and unpopular movies by holding them our(using an 80% train, 20% split) by folding 
#in the 80% ratings data as new feature vectors, and testing on the 20% held out ratings(20% being held our per-user).
#For each user, we calculate RMSE, NDCG, Precision and Recall. 
#This is run 10 times, and the results are exported to a csv file called pop_held_out_metrics and unpop_held_out_metrics and 
#have the scores generated at each run. 
##
#Importing the documents
movie_data=pd.read_csv("/project/naray190/movie_data_20M.csv",sep=",")
movie_data.columns=['item','popularity','avgRating']
train=pd.read_csv('/project/naray190/ml-20m/ratings.csv')
test=pd.read_csv('/project/naray190/ml-20m/test_casual_user_ratings.csv')
#cleaning the columns
train=train[['userId','movieId','rating']]
test=test[['userId','movieId','rating']]
train.columns = ['user','item','rating']
test.columns=['user','item','rating']

#De-normalizing ratings
gbias=train['rating'].mean()
train['rating']-=gbias
test['rating']-=gbias
group=train.groupby('item')['rating']
item_biases=group.sum()/(group.count()+5)
train=train.join(pd.DataFrame(item_biases),on="item",how="inner",rsuffix="_im")
train=train.assign(rating=lambda df:df.rating-df.rating_im)
test=test.join(pd.DataFrame(item_biases),on="item",how="inner",rsuffix="_im")
test=test.assign(rating=lambda df:df.rating-df.rating_im)
group=train.groupby('user')['rating']
user_biases_train=group.sum()/(group.count()+5)
train=train.join(pd.DataFrame(user_biases_train),on="user",how="inner",rsuffix="_um")
train=train.assign(rating=lambda df:df.rating-df.rating_um)
group=test.groupby('user')['rating']
user_biases_test=group.sum()/(group.count()+5)
test=test.join(pd.DataFrame(user_biases_test),on="user",how="inner",rsuffix="_um")
test=test.assign(rating=lambda df:df.rating-df.rating_um)
train=train[['user','item','rating']]
test=test[['user','item','rating']]
#Fold-In function
class FoldIn(als.BiasedMF):
    def __init__(self,*args,**kwargs):
        super (FoldIn,self).__init__(*args,**kwargs)
        self.bias=None
    def fold_in(self,new_ratings):
        #set_trace()
        rmat, users, items = sparse_ratings(new_ratings,iidx=self.item_index_)
        n_users = len(users)trainhalfphopredicts=pd.DataFrame(columns=['user','item','score','rank'])
trainhalfphopredicts=recommend_for_users(algo_half,list(test_pho.user.unique()),algo_half.user_index_,algo_half.user_features_,trainhalfphopredicts,train_half_pho,20,item_biasesh,user_biases_trainh,gbiash)
        n_items = len(items)


        umat = np.full((n_users, self.features), np.nan)
        #set_trace()
        umat = als._train_matrix(rmat.N, self.item_features_, self.regularization)
        #set_trace()

        return umat,users

#custom-sparse ratings file
def sparse_ratings(ratings, scipy=False,uidx=None,iidx=None):
    """
    Convert a rating table to a sparse matrix of ratings.
    Args:
        ratings(pandas.DataFrame): a data table of (user, item, rating) triples.
        scipy: if ``True``, return a SciPy matrix instead of :py:class:`CSR`.
    Returns:
        RatingMatrix:
            a named tuple containing the sparse matrix, user index, and item index.
    """
    #set_trace()
    if(uidx is None):
        uidx = pd.Index(ratings.user.unique(), name='user')
    if(iidx is None):
        iidx = pd.Index(ratings.item.unique(), name='item')


    row_ind = uidx.get_indexer(ratings.user).astype(np.int32)
    col_ind = iidx.get_indexer(ratings.item).astype(np.int32)

    if 'rating' in ratings.columns:
        vals = np.require(ratings.rating.values, np.float64)
    else:
        vals = None

    matrix = CSR.from_coo(row_ind, col_ind, vals, (len(uidx), len(iidx)))
    #set_trace()
    if scipy:
        matrix = CSR.to_scipy(matrix)

    return RatingMatrix(matrix, uidx, iidx)

algo=FoldIn(features=25,iterations=50,reg=0.1)
algo.fit(train)
users=test.user.unique().tolist()
##Recommendation code
def recommend_for_users(algo,users,user_index,user_matrix,predicts,train,k):
    for user in users:
        uix=user_index.get_loc(user)
        uvfull=user_matrix[uix]
        user_movies=train.loc[train['user']==user]
        movie_list=set(user_movies['item'].tolist())
        candidates=set(train['item'].unique())-movie_list
        remove_movie=set(movie_data.loc[movie_data["popularity"]<10].item.values)
        candidates=candidates-remove_movie
        candidates=list(candidates)
        iix=algo.lookup_items(candidates)
        score=np.matmul(algo.item_features_[iix],uvfull)
        score=score+item_biases.loc[candidates]+user_biases_train[user]+gbias
        scores=pd.DataFrame({"item":candidates,"score":score})
        scores['user']=user
        scores=scores.sort_values('score',ascending=False)
        scores=scores.head(k)
        scores['rank']=scores['score'].rank(ascending=0)
        predicts=predicts.append(scores,sort=True)
    return predicts

pop_held_out_metrics=pd.DataFrame(columns=['run','ndcgf','ndcgp','precisionf','precisionp','recallf','recallp','rmsef','rmsep'])
unpop_held_out_metrics=pd.DataFrame(columns=['run','ndcgf','ndcgp','precisionf','precisionp','recallf','recallp','rmsef','rmsep'])
for i in range(10):
    #Holding out 20% of popular and unpopular movies
    train_full_pho=pd.DataFrame(columns=['user','item','rating'])
    train_pop_pho=pd.DataFrame(columns=['user','item','rating'])
    test_pho=pd.DataFrame(columns=['user','item','rating'])
    popIds=set(movie_data.loc[movie_data.popularity > 1450].item)
    for user in users:
        ufrating=train.loc[train.user == user]
        uprating=test.loc[test.user == user]
        homovies=set(uprating.sample(frac=0.2).item)
        testpop=uprating.loc[uprating.item.isin(homovies)]#test will hold original data
        trainfull=ufrating.loc[~ufrating.item.isin(homovies)]
        trainpop=uprating.loc[~uprating.item.isin(homovies)]
        train_full_pho=train_full_pho.append(trainfull,sort=True)
        train_pop_pho=train_pop_pho.append(trainpop,sort=True)
        test_pho=test_pho.append(testpop,sort=True)
        train_full_upo=pd.DataFrame(columns=['user','item','rating'])
        test_upo=pd.DataFrame(columns=['user','item','rating'])
        train_pop=pd.DataFrame(columns=['user','item','rating'])
        unpopIds=set(movie_data.item.unique())-popIds
    for user in users:
        ufrating=train.loc[train.user==user]
     
        unpopratings=ufrating.loc[ufrating.item.isin(unpopIds)]
    
        popratings=ufrating.loc[~ufrating.item.isin(unpopIds)]
        if(len(unpopratings)>5):
            homovies=set(unpopratings.sample(frac=0.2).item)
            trainfull=popratings.append(unpopratings.loc[~unpopratings.item.isin(homovies)],sort=True)
            train_full_upo=train_full_upo.append(trainfull,sort=True)
            test_upo=test_upo.append(unpopratings.loc[unpopratings.item.isin(homovies)],sort=True)
            train_pop=train_pop.append(test.loc[test.user == user],sort=True)

    trainfullphoumat,trainfullphoix=algo.fold_in(train_full_pho)
    trainpopphoumat,trainpopphoix=algo.fold_in(train_pop_pho)
    trainfullupoumat,trainfullupoix=algo.fold_in(train_full_upo)
    trainpopupoumat,trainpopupoix=algo.fold_in(train_pop)
    trainfullphopredicts=pd.DataFrame(columns=['user','item','score','rank'])
    trainpopphopredicts=pd.DataFrame(columns=['user','item','score','rank'])
    trainfullphopredicts=recommend_for_users(algo,list(train_full_pho.user.unique()),trainfullphoix,trainfullphoumat,trainfullphopredicts,train_full_pho,20)
    trainpopphopredicts=recommend_for_users(algo,list(train_pop_pho.user.unique()),trainpopphoix,trainpopphoumat,trainpopphopredicts,train_pop_pho,20)
    trainfullphopredicts=trainfullphopredicts.reset_index(drop=True)
    trainpopphopredicts=trainpopphopredicts.reset_index(drop=True)
    trainfullphopredicts=trainfullphopredicts.reset_index(drop=True)
    trainpopphopredicts=trainpopphopredicts.reset_index(drop=True)

    test_pho_rel=test_pho.copy()
    pos=test_pho.rating>0
    test_pho_rel.loc[pos,"rating"]=1
    neg=test_pho.rating<0
    test_pho_rel.loc[neg,"rating"]=0
    test_pho_rel=test_pho_rel.reset_index(drop=True)
    user_pho_ndcg=pd.DataFrame(columns=['user','ndcg-full','ndcg-pop'])
    for user in users:
        predictsf=trainfullphopredicts.loc[trainfullphopredicts.user == user]
        predictsp=trainpopphopredicts.loc[trainpopphopredicts.user == user]
        truth=test_pho_rel.loc[test_pho_rel.user == user]
        ndcgf=topn.ndcg(predictsf,truth)
        ndcgp=topn.ndcg(predictsp,truth)
        newrow=pd.DataFrame([[user,ndcgf,ndcgp]],columns=['user','ndcg-full','ndcg-pop'])
        user_pho_ndcg=user_pho_ndcg.append(newrow,sort=True)
    user_pho_precision=pd.DataFrame(columns=['user','prec_full','prec_pop'])
    for user in users:
        predictsf=trainfullphopredicts.loc[trainfullphopredicts.user == user]
        predictsp=trainpopphopredicts.loc[trainpopphopredicts.user == user]
        truth=test_pho_rel.loc[test_pho_rel.user == user]
        prec_full=topn.precision(predictsf,truth)
        prec_pop=topn.precision(predictsp,truth)
        newrow=pd.DataFrame([[user,prec_full,prec_pop]],columns=['user','prec-full','prec-pop'])
        user_pho_precision=user_pho_precision.append(newrow,sort=True)
    user_pho_recall=pd.DataFrame(columns=['user','recall-full','recall-pop'])
    for user in users:
        predictsf=trainfullphopredicts.loc[trainfullphopredicts.user == user]
        predictsp=trainpopphopredicts.loc[trainpopphopredicts.user == user]
        truth=test_pho_rel.loc[test_pho_rel.user == user]
        recall_full=topn.recall(predictsf,truth)
        recall_pop=topn.recall(predictsp,truth)
        newrow=pd.DataFrame([[user,recall_full,recall_pop]],columns=['user','recall-full','recall-pop'])
        user_pho_recall=user_pho_recall.append(newrow,sort=True)
    def predictui(train,uindex,iindex,umat,imat):
        res=pd.DataFrame(columns=['user','item','rating'])
        users=list(train.user.unique())
        for index,row in train.iterrows():
            user=row['user']
            item=row['item']
            uix=uindex.get_loc(user)
            iix=iindex.get_loc(item)
            uv=umat[uix]
            iv=imat[iix]
            pdt=np.dot(uv,iv)
            newrow=pd.DataFrame([[user,item,pdt]],columns=['user','item','rating'])
            res=res.append(newrow)

        return res
    preds=predictui(test_pho,trainfullphoix,algo.item_index_,trainfullphoumat,algo.item_features_)
    preds=preds.reset_index(drop=True)
    diff=preds['rating']-test_pho['rating']
    sqdif=diff.apply(np.square)
    rmsef=np.sqrt(sqdif.mean())
    predpop=predictui(test_pho,trainpopphoix,algo.item_index_,trainpopphoumat,algo.item_features_)
    predpop=preds.reset_index(drop=True)
    diffpop=predpop['rating']-test_pho['rating']
    sqdiffpop=diffpop.apply(np.square)
    rmsep=np.sqrt(sqdiffpop.mean())
    ndcgf=user_pho_ndcg['ndcg-full'].mean()
    ndcgp=user_pho_ndcg['ndcg-pop'].mean()
    precisionf=user_pho_precision['prec-full'].mean()
    precisionp=user_pho_precision['prec-pop'].mean()
    recallf=user_pho_recall['recall-full'].mean()
    recallp=user_pho_recall['recall-pop'].mean()
    newrow=pd.DataFrame([[i,ndcgf,ndcgp,precisionf,precisionp,recallf,recallp,rmsef,rmsep]],columns=['run','ndcgf','ndcgp','precisionf','precisionp','recallf','recallp','rmsef','rmsep'])
    pop_held_out_metrics=pop_held_out_metrics.append(newrow,sort=True)
    trainfullupopredicts=pd.DataFrame(columns=['user','item','score','rank'])
    trainpopupopredicts=pd.DataFrame(columns=['user','item','score','rank'])
    trainfullupopredicts=recommend_for_users(algo,list(train_full_upo.user.unique()),trainfullupoix,trainfullupoumat,trainfullupopredicts,train_full_upo,20)
    trainpopupopredicts=recommend_for_users(algo,list(train_pop.user.unique()),trainpopupoix,trainpopupoumat,trainpopupopredicts,train_pop,20)
    trainfullupopredicts=trainfullupopredicts.reset_index(drop=True)
    trainpopupopredicts=trainpopupopredicts.reset_index(drop=True)
    test_upo_rel=test_upo.copy()
    pos=test_upo.rating>0
    neg=test_upo.rating<0
    test_upo_rel.loc[pos,"rating"]=1
    test_upo_rel.loc[neg,"rating"]=0
    test_upo_rel=test_upo_rel.reset_index(drop=True)
    test_upo=test_upo.reset_index(drop=True)
    users_upo=set(test_upo.user.unique())
    user_upo_ndcg=pd.DataFrame(columns=['user','ndcg-full','ndcg-pop'])
    for user in users_upo:
        predictsf=trainfullupopredicts.loc[trainfullupopredicts.user == user]
        predictsp=trainpopupopredicts.loc[trainpopupopredicts.user == user]
        truth=test_upo_rel.loc[test_upo_rel.user == user]
        ndcg_full=topn.ndcg(predictsf,truth)
        ndcg_pop=topn.ndcg(predictsp,truth)
        newrow=pd.DataFrame([[user,ndcg_full,ndcg_pop]],columns=['user','ndcg-full','ndcg-pop'])
        user_upo_ndcg=user_upo_ndcg.append(newrow,sort=True)
    user_upo_precision=pd.DataFrame(columns=['user','prec-full','prec-pop'])
    for user in users_upo:
        predictsf=trainfullupopredicts.loc[trainfullupopredicts.user == user]
        predictsp=trainpopupopredicts.loc[trainpopupopredicts.user == user]
        truth=test_upo_rel.loc[test_upo_rel.user == user]
        prec_full=topn.precision(predictsf,truth)
        prec_pop=topn.precision(predictsp,truth)
        newrow=pd.DataFrame([[user,prec_full,prec_pop]],columns=['user','prec-full','prec-pop'])
        user_upo_precision=user_upo_precision.append(newrow,sort=True)
    user_upo_recall=pd.DataFrame(columns=['user','recall-full','recall-pop'])
    for user in users_upo:
        predictsf=trainfullupopredicts.loc[trainfullupopredicts.user == user]
        predictsp=trainpopupopredicts.loc[trainpopupopredicts.user == user]
        truth=test_upo_rel.loc[test_upo_rel.user == user]
        rec_full=topn.recall(predictsf,truth)
        rec_pop=topn.recall(predictsp,truth)
        newrow=pd.DataFrame([[user,rec_full,rec_pop]],columns=['user','recall-full','recall-pop'])
        user_upo_recall=user_upo_recall.append(newrow,sort=True)
    predupofull=predictui(test_upo,trainfullupoix,algo.item_index_,trainfullupoumat,algo.item_features_)
    predupofull=predupofull.reset_index(drop=True)
    dfull=predupofull['rating']-test_upo['rating']
    sqdfull=dfull.apply(np.square)
    rmsef=np.sqrt(sqdfull.mean())
    predupopop=predictui(test_upo,trainpopupoix,algo.item_index_,trainpopupoumat,algo.item_features_)
    predupopop=predupopop.reset_index(drop=True)
    dfpop=predupopop['rating']-test_upo['rating']
    sqdfpop=dfpop.apply(np.square)
    rmsep=np.sqrt(sqdfpop.mean())
    ndcgf=user_upo_ndcg['ndcg-full'].mean()
    ndcgp=user_upo_ndcg['ndcg-pop'].mean()
    precisionf=user_upo_precision['prec-full'].mean()
    precisionp=user_upo_precision['prec-pop'].mean()
    recallf=user_upo_recall['recall-full'].mean()
    recallp=user_upo_recall['recall-pop'].mean()
    newrow=pd.DataFrame([[i,ndcgf,ndcgp,precisionf,precisionp,recallf,recallp,rmsef,rmsep]],columns=['run','ndcgf','ndcgp','precisionf','precisionp','recallf','recallp','rmsef','rmsep'])
    unpop_held_out_metrics=unpop_held_out_metrics.append(newrow,sort=True)

pop_held_out_metrics.to_csv("/project/naray190/dr-project/data/TNM/pop_held_out_metrics.csv",sep=",",header=True,index=True)
unpop_held_out_metrics.to_csv("/project/naray190/dr-project/data/TNM/unpop_held_out_metrics.csv",sep=",",header=True,index=True)
