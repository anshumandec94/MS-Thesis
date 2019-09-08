import pandas as pd
from lenskit import batch,topn
from lenskit.metrics import topn as tn
from lenskit.algorithms import als
from scipy import spatial
import matplotlib.pyplot as plt
from lenskit import topn
from IPython.core.debugger import set_trace
from lenskit.metrics import predict
import random
import numpy as np
##
#In this script, we take our three training model data(namely, full half and none ratings data set) 
#After this, we run tests on popular and unpopular movies by holding them our(using an 80% train, 20% split) by folding 
#in the 80% ratings data as new feature vectors, and testing on the 20% held out ratings(20% being held our per-user).
#For each user, we calculate RMSE, NDCG, Precision and Recall. 
#This is run 10 times, and the results are exported to a csv file called pop_held_out_metrics_serious and unpop_held_out_metrics_serious and 
#have the scores generated at each run. 
##
train_full=pd.read_csv('/project/naray190/ml-20m/casual_pop_ratings.csv')
test=pd.read_csv('/project/naray190/ml-20m/test_ratings.csv')
train_full=train_full[['userId','movieId','rating']]
test=test[['userId','movieId','rating']]
train_full.columns = ['user','item','rating']
test.columns=['user','item','rating']
train_none=pd.read_csv('/project/naray190/ml-20m/ratings.csv')
train_none=train_none[['userId','movieId','rating']]
train_none.columns=['user','item','rating']
train_half=pd.read_csv("/project/naray190/ml-20m/half_casual_pop_ratings.csv")
train_half=train_half[["userId","movieId","rating"]]
train_half.columns=['user','item','rating']

gbias=train_none['rating'].mean()
train_none['rating']-=gbias
test['rating']-=gbias
group=train_none.groupby('item')['rating']
item_biases=group.sum()/(group.count()+5)
train_none=train_none.join(pd.DataFrame(item_biases),on="item",how="inner",rsuffix="_im")
train_none=train_none.assign(rating=lambda df:df.rating-df.rating_im)
test=test.join(pd.DataFrame(item_biases),on="item",how="inner",rsuffix="_im")
test=test.assign(rating=lambda df:df.rating-df.rating_im)
group=train_none.groupby('user')['rating']
user_biases_train=group.sum()/(group.count()+5)
train_none=train_none.join(pd.DataFrame(user_biases_train),on="user",how="inner",rsuffix="_um")
train_none=train_none.assign(rating=lambda df:df.rating-df.rating_um)
group=test.groupby('user')['rating']
user_biases_test=group.sum()/(group.count()+5)
test=test.join(pd.DataFrame(user_biases_test),on="user",how="inner",rsuffix="_um")
test=test.assign(rating=lambda df:df.rating-df.rating_um)

gbiasf=train_full['rating'].mean()
train_full['rating']-=gbiasf


group=train_full.groupby('item')['rating']
item_biasesf=group.sum()/(group.count()+5)
train_full=train_full.join(pd.DataFrame(item_biasesf),on="item",how="inner",rsuffix="_im")
train_full=train_full.assign(rating=lambda df:df.rating-df.rating_im)

group=train_full.groupby('user')['rating']
user_biases_trainf=group.sum()/(group.count()+5)
train_full=train_full.join(pd.DataFrame(user_biases_trainf),on="user",how="inner",rsuffix="_um")
train_full=train_full.assign(rating=lambda df:df.rating-df.rating_um)

gbiash=train_half['rating'].mean()
train_half['rating']-=gbiash


group=train_half.groupby('item')['rating']
item_biasesh=group.sum()/(group.count()+5)
train_half=train_half.join(pd.DataFrame(item_biasesh),on="item",how="inner",rsuffix="_im")
train_half=train_half.assign(rating=lambda df:df.rating-df.rating_im)

group=train_half.groupby('user')['rating']
user_biases_trainh=group.sum()/(group.count()+5)
train_half=train_half.join(pd.DataFrame(user_biases_trainh),on="user",how="inner",rsuffix="_um")
train_half=train_half.assign(rating=lambda df:df.rating-df.rating_um)

train_full=train_full[['user','item','rating']]
test=test[['user','item','rating']]
train_half=train_half[['user','item','rating']]
train_none=train_none[['user','item','rating']]
users=set(test.user.unique())
class FoldIn(als.BiasedMF):
    def __init__(self,*args,**kwargs):
        super (FoldIn,self).__init__(*args,**kwargs)
        self.bias=None
    def fold_in(self,new_ratings):
        #set_trace()
        rmat, users, items = sparse_ratings(new_ratings,iidx=self.item_index_)
        n_users = len(users)
        n_items = len(items)


        umat = np.full((n_users, self.features), np.nan)
        #set_trace()
        umat = als._train_matrix(rmat.N, self.item_features_, self.regularization)
        #set_trace()

        return umat,users


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
algo_full = FoldIn(features=25,iterations=50,reg=0.1)
algo_half =FoldIn(features=25,iterations=50,reg=0.1)
algo_none=FoldIn(features=25,iterations=50,reg=0.1)

###Holding Out 20% of test user's movies.

movie_data=pd.read_csv("/project/naray190/movie_data_20M.csv",sep=",")
movie_data.columns=['item','popularity','avgRating']
users=test.user.unique().tolist()
pop_held_out_metrics=pd.DataFrame(columns=['run','ndcgf','ndcgh','ndcgn','precisionf','precisionh','precisionn','recallf','recallh','recalln','rmsef','rmseh','rmsen'])
unpop_held_out_metrics=pd.DataFrame(columns=['run','ndcgf','ndcgh','ndcgn','precisionf','precisionh','precisionn','recallf','recallh','recalln','rmsef','rmseh','rmsen'])

for i in range(10):
    train_full_pho=pd.DataFrame(columns=['user','item','rating'])


    test_pho=pd.DataFrame(columns=['user','item','rating'])
    popIds=set(movie_data.loc[movie_data.popularity > 1450].item)
    train_half_pho=pd.DataFrame(columns=['user','item','rating'])
    all_users=set(train_full.user.unique())
    other_users=all_users-set(users)
    other_ratings=train_full.loc[train_full.user.isin(other_users)]

    for user in users:
        ufrating=train_full.loc[train_full.user == user]
        uprating=ufrating.loc[ufrating.item.isin(popIds)]
        homovies=set(uprating.sample(frac=0.2).item)
        testpop=ufrating.loc[ufrating.item.isin(homovies)]#test will hold original data
        trainfull=ufrating.loc[~ufrating.item.isin(homovies)]

        train_full_pho=train_full_pho.append(trainfull,sort=True)

        test_pho=test_pho.append(testpop,sort=True)

    train_full_pho=train_full_pho.append(other_ratings,sort=True)
    for user in users:
        ufrating=train_half.loc[train_half.user ==user]
        homovies=set(test_pho.loc[test_pho.user == user].item)
        trainhalf=ufrating.loc[~ufrating.item.isin(homovies)]

        train_half_pho=train_half_pho.append(trainhalf,sort=True)

    train_none_pho=pd.DataFrame(columns=['user','item','rating'])
    for user in users:
        ufrating=train_none.loc[train_none.user==user]
        homovies=set(test_pho.loc[test_pho.user == user].item)
        trainnone=ufrating.loc[~ufrating.item.isin(homovies)]
        train_none_pho=train_none_pho.append(trainnone,sort=True)


    all_users=set(train_none.user.unique())
    other_users=all_users-set(users)
    other_ratings=train_none.loc[train_none.user.isin(other_users)]
    train_none_pho=train_none_pho.append(other_ratings,sort=True)



    all_users=set(train_half.user.unique())
    other_users=all_users-set(users)
    other_ratings=train_half.loc[train_half.user.isin(other_users)]
    train_half_pho=train_half_pho.append(other_ratings,sort=True)

    ##UNPOPULAR REMOVAL
    train_full_upo=pd.DataFrame(columns=['user','item','rating'])
    test_upo=pd.DataFrame(columns=['user','item','rating'])

    unpopIds=set(movie_data.item.unique())-popIds

    for user in users:
        ufrating=train_full.loc[train_full.user==user]
        unpopratings=ufrating.loc[ufrating.item.isin(unpopIds)]
        valid_movies=set(movie_data.loc[movie_data.popularity>50].item)
        unpopratings=unpopratings.loc[unpopratings.item.isin(valid_movies)]
        popratings=ufrating.loc[~ufrating.item.isin(unpopIds)]
        if(len(unpopratings)>5):
            homovies=set(unpopratings.sample(frac=0.2).item)
            trainfull=popratings.append(unpopratings.loc[~unpopratings.item.isin(homovies)],sort=True)
            train_full_upo=train_full_upo.append(trainfull,sort=True)
            test_upo=test_upo.append(unpopratings.loc[unpopratings.item.isin(homovies)],sort=True)

    all_users=set(train_full.user.unique())
    other_users=all_users-set(users)
    other_ratings=train_full.loc[train_full.user.isin(other_users)]
    train_full_upo=train_full_upo.append(other_ratings,sort=True)

    train_half_upo=pd.DataFrame(columns=['user','item','rating'])
    for user in set(test_upo.user.unique()):
        ufrating=train_half.loc[train_half.user==user]
        homovies=set(test_upo.loc[test_upo.user == user].item)


        trainhalf=ufrating.loc[~ufrating.item.isin(homovies)]
        train_half_upo=train_half_upo.append(trainhalf,sort=True)
    all_users=set(train_half.user.unique())
    other_users=all_users-set(users)
    other_ratings=train_half.loc[train_half.user.isin(other_users)]
    train_half_upo=train_half_upo.append(other_ratings,sort=True)
    train_none_upo=pd.DataFrame(columns=['user','item','rating'])

    for user in set(test_upo.user.unique()):
        ufrating=train_none.loc[train_none.user==user]
        homovies=set(test_upo.loc[test_upo.user == user].item)
        trainnone=ufrating.loc[~ufrating.item.isin(homovies)]
        train_none_upo=train_none_upo.append(trainnone,sort=True)
    all_users=set(train_none.user.unique())
    other_users=all_users-set(users)
    other_ratings=train_none.loc[train_none.user.isin(other_users)]
    train_none_upo=train_none_upo.append(other_ratings,sort=True)

    def recommend_for_users(algo,users,user_index,user_matrix,predicts,train,k,item_biases,user_biases_train,gbias):
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

    algo_full.fit(train_full_pho)
    algo_half.fit(train_half_pho)
    algo_none.fit(train_none_pho)

    trainfullphopredicts=pd.DataFrame(columns=['user','item','score','rank'])
    trainfullphopredicts=recommend_for_users(algo_full,list(test_pho.user.unique()),algo_full.user_index_,algo_full.user_features_,trainfullphopredicts,train_full_pho,20,item_biasesf,user_biases_trainf,gbiasf)

    trainnonephopredicts=pd.DataFrame(columns=['user','item','score','rank'])
    trainnonephopredicts=recommend_for_users(algo_none,list(test_pho.user.unique()),algo_none.user_index_,algo_none.user_features_,trainnonephopredicts,train_none_pho,20,item_biases,user_biases_train,gbias)

    trainhalfphopredicts=pd.DataFrame(columns=['user','item','score','rank'])
    trainhalfphopredicts=recommend_for_users(algo_half,list(test_pho.user.unique()),algo_half.user_index_,algo_half.user_features_,trainhalfphopredicts,train_half_pho,20,item_biasesh,user_biases_trainh,gbiash)

    test_pho_rel=test_pho.copy()
    pos=test_pho.rating>0
    test_pho_rel.loc[pos,"rating"]=1
    neg=test_pho.rating<0
    test_pho_rel.loc[neg,"rating"]=0
    trainfullphopredicts=trainfullphopredicts.reset_index(drop=True)
    trainhalfphopredicts=trainhalfphopredicts.reset_index(drop=True)
    trainnonephopredicts=trainnonephopredicts.reset_index(drop=True)
    test_pho_rel=test_pho_rel.reset_index(drop=True)
    user_pho_ndcg=pd.DataFrame(columns=['user','ndcgf','ndcgh','ndcgn'])
    for user in users:
        predictsf=trainfullphopredicts.loc[trainfullphopredicts.user == user]
        predictsh=trainhalfphopredicts.loc[trainhalfphopredicts.user == user]
        predictsn=trainnonephopredicts.loc[trainnonephopredicts.user == user]
        truth=test_pho_rel.loc[test_pho_rel.user == user]

        ndcgf=topn.ndcg(predictsf,truth)
        ndcgh=topn.ndcg(predictsh,truth)
        ndcgn=topn.ndcg(predictsn,truth)
        newrow=pd.DataFrame([[user,ndcgf,ndcgh,ndcgn]],columns=['user','ndcgf','ndcgh','ndcgn'])
        user_pho_ndcg=user_pho_ndcg.append(newrow,sort=True)
    user_pho_precision=pd.DataFrame(columns=['user','precf','prech','precn'])
    for user in users:
        predictsf=trainfullphopredicts.loc[trainfullphopredicts.user == user]
        predictsh=trainhalfphopredicts.loc[trainhalfphopredicts.user == user]
        predictsn=trainnonephopredicts.loc[trainnonephopredicts.user == user]
        truth=test_pho_rel.loc[test_pho_rel.user == user]
        precf = topn.precision(predictsf,truth)
        prech = topn.precision(predictsh,truth)
        precn = topn.precision(predictsn,truth)
        newrow=pd.DataFrame([[user,precf,prech,precn]],columns=['user','precf','prech','precn'])
        user_pho_precision=user_pho_precision.append(newrow,sort=True)
    user_pho_recall=pd.DataFrame(columns=['user','recf','rech','recn'])
    for user in users:
        predictsf=trainfullphopredicts.loc[trainfullphopredicts.user == user]
        predictsh=trainhalfphopredicts.loc[trainhalfphopredicts.user == user]
        predictsn=trainnonephopredicts.loc[trainnonephopredicts.user == user]
        truth=test_pho_rel.loc[test_pho_rel.user == user]

        recf=topn.recall(predictsf,truth)
        rech=topn.recall(predictsh,truth)
        recn=topn.recall(predictsn,truth)
        newrow=pd.DataFrame([[user,recf,rech,recn]],columns=['user','recf','rech','recn'])
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
    test_pho=test_pho.reset_index(drop=True)
    preds_full=predictui(test_pho,algo_full.user_index_,algo_full.item_index_,algo_full.user_features_,algo_full.item_features_)
    preds_full=preds_full.reset_index(drop=True)
    difffull=preds_full['rating']-test_pho['rating']
    sqdifffull=difffull.apply(np.square)
    rmsef=np.sqrt(sqdifffull.mean())
    preds_half=predictui(test_pho,algo_half.user_index_,algo_half.item_index_,algo_half.user_features_,algo_half.item_features_)
    preds_half=preds_half.reset_index(drop=True)
    diffhalf=preds_half['rating']-test_pho['rating']
    sqdiffhalf=diffhalf.apply(np.square)
    rmseh=np.sqrt(sqdiffhalf.mean())
    preds_none=predictui(test_pho,algo_none.user_index_,algo_none.item_index_,algo_none.user_features_,algo_full.item_features_,)
    preds_none=preds_none.reset_index(drop=True)
    diffn=preds_none['rating']-test_pho['rating']
    sqdiffn=diffn.apply(np.square)
    rmsen=np.sqrt(sqdiffn.mean())
    ndcgf=user_pho_ndcg['ndcgf'].mean()
    ndcgh=user_pho_ndcg['ndcgh'].mean()
    ndcgn=user_pho_ndcg['ndcgn'].mean()
    precisionf=user_pho_precision['precf'].mean()
    precisionh=user_pho_precision['prech'].mean()
    precisionn=user_pho_precision['precn'].mean()
    recallf=user_pho_recall['recf'].mean()
    recallh=user_pho_recall['rech'].mean()
    recalln=user_pho_recall['recn'].mean()
    newrow=pd.DataFrame([[i,ndcgf,ndcgh,ndcgn,precisionf,precisionh,precisionn,recallf,recallh,recalln,rmsef,rmseh,rmsen]],columns=['run','ndcgf','ndcgh','ndcgn','precisionf','precisionh','precisionn','recallf','recallh','recalln','rmsef','rmseh','rmsen'])
    pop_held_out_metrics=pop_held_out_metrics.append(newrow,sort=True)

    algo_full.fit(train_full_upo)
    algo_half.fit(train_half_upo)
    algo_none.fit(train_none_upo)

    trainfullupopredicts=pd.DataFrame(columns=['user','item','score','rank'])
    trainfullupopredicts=recommend_for_users(algo_full,list(test_upo.user.unique()),algo_full.user_index_,algo_full.user_features_,trainfullupopredicts,train_full_upo,20,item_biasesf,user_biases_trainf,gbiasf)

    trainhalfupopredicts=pd.DataFrame(columns=['user','item','score','rank'])
    trainhalfupopredicts=recommend_for_users(algo_half,list(test_upo.user.unique()),algo_half.user_index_,algo_half.user_features_,trainhalfupopredicts,train_half_upo,20,item_biasesh,user_biases_trainh,gbiash)

    trainnoneupopredicts=pd.DataFrame(columns=['user','item','score','rank'])
    trainnoneupopredicts=recommend_for_users(algo_none,list(test_upo.user.unique()),algo_none.user_index_,algo_none.user_features_,trainnoneupopredicts,train_none_upo,20,item_biases,user_biases_train,gbias)

    trainhalfupopredicts=trainhalfupopredicts.reset_index(drop=True)
    trainfullupopredicts=trainfullupopredicts.reset_index(drop=True)
    trainnoneupopredicts=trainnoneupopredicts.reset_index(drop=True)
    test_upo_rel=test_upo.copy()
    pos=test_upo.rating>0
    neg=test_upo.rating<0
    test_upo_rel.loc[pos,"rating"]=1
    test_upo_rel.loc[neg,"rating"]=0
    test_upo_rel=test_upo_rel.reset_index(drop=True)
    tusers=list(test_upo.user.unique())

    user_upo_ndcg=pd.DataFrame(columns=['user','ndcg-full','ndcg-half','ndcg-none'])
    for user in tusers:
        predictsf=trainfullupopredicts.loc[trainfullupopredicts.user == user]
        predictsh=trainhalfupopredicts.loc[trainhalfupopredicts.user == user]
        predictsn=trainnoneupopredicts.loc[trainnoneupopredicts.user == user]
        truth=test_upo_rel.loc[test_upo_rel.user == user]
        ndcgf=topn.ndcg(predictsf,truth)
        ndcgh=topn.ndcg(predictsh,truth)
        ndcgn=topn.ndcg(predictsn,truth)
        newrow=pd.DataFrame([[user,ndcgf,ndcgh,ndcgn]],columns=['user','ndcg-full','ndcg-half','ndcg-none'])
        user_upo_ndcg=user_upo_ndcg.append(newrow,sort=True)
    ndcgf=user_upo_ndcg['ndcg-full'].mean()
    ndcgh=user_upo_ndcg['ndcg-half'].mean()
    ndcgn=user_upo_ndcg['ndcg-none'].mean()

    user_upo_precision=pd.DataFrame(columns=['user','precf','prech','precn'])
    for user in tusers:
        predictsf=trainfullupopredicts.loc[trainfullupopredicts.user == user]
        predictsh=trainhalfupopredicts.loc[trainhalfupopredicts.user == user]
        predictsn=trainnoneupopredicts.loc[trainnoneupopredicts.user == user]
        truth=test_upo_rel.loc[test_upo_rel.user == user]
        precf=topn.precision(predictsf,truth)
        prech=topn.precision(predictsh,truth)
        precn=topn.precision(predictsn,truth)
        newrow=pd.DataFrame([[user,precf,prech,precn]],columns=['user','precf','prech','precn'])
        user_upo_precision=user_upo_precision.append(newrow,sort=True)

    precisionf=user_upo_precision.precf.mean()
    precisionh=user_upo_precision.prech.mean()
    precisionn=user_upo_precision.precn.mean()

    user_upo_recall=pd.DataFrame(columns=['user','recf','rech','recn'])
    for user in tusers:
        predictsf=trainfullupopredicts.loc[trainfullupopredicts.user == user]
        predictsh=trainhalfupopredicts.loc[trainhalfupopredicts.user == user]
        predictsn=trainnoneupopredicts.loc[trainnoneupopredicts.user == user]
        truth=test_upo_rel.loc[test_upo_rel.user == user]
        recf=topn.recall(predictsf,truth)
        rech=topn.recall(predictsh,truth)
        recn=topn.recall(predictsn,truth)
        newrow=pd.DataFrame([[user,recf,rech,recn]],columns=['user','recf','rech','recn'])
        user_upo_recall=user_upo_recall.append(newrow,sort=True)
    recallf=user_upo_recall.recf.mean()
    recallh=user_upo_recall.rech.mean()
    recalln=user_upo_recall.recn.mean()
    test_upo=test_upo.reset_index(drop=True)
    preds_full=predictui(test_upo,algo_full.user_index_,algo_full.item_index_,algo_full.user_features_,algo_full.item_features_)
    preds_half=predictui(test_upo,algo_half.user_index_,algo_half.item_index_,algo_half.user_features_,algo_half.item_features_)
    preds_none=predictui(test_upo,algo_none.user_index_,algo_none.item_index_,algo_none.user_features_,algo_none.item_features_)
    preds_full=preds_full.reset_index(drop=True)
    preds_half=preds_half.reset_index(drop=True)
    preds_none=preds_none.reset_index(drop=True)
    diff=preds_full['rating']-test_upo['rating']
    difh=preds_half['rating']-test_upo['rating']
    difn=preds_none['rating']-test_upo['rating']
    sqdf=diff.apply(np.square)
    sqdh=difh.apply(np.square)
    sqdn=difn.apply(np.square)
    rmsef=np.sqrt(sqdf.mean())
    rmseh=np.sqrt(sqdh.mean())
    rmsen=np.sqrt(sqdn.mean())

    newrow=pd.DataFrame([[i,ndcgf,ndcgh,ndcgn,precisionf,precisionh,precisionn,recallf,recallh,recalln,rmsef,rmseh,rmsen]],columns=['run','ndcgf','ndcgh','ndcgn','precisionf','precisionh','precisionn','recallf','recallh','recalln','rmsef','rmseh','rmsen'])
    unpop_held_out_metrics=unpop_held_out_metrics.append(newrow,sort=True)

pop_held_out_metrics.to_csv("/project/naray190/dr-project/data/TNM/pop_held_out_metrics_serious.csv",sep=",",header=True,index=True)
unpop_held_out_metrics.to_csv("/project/naray190/dr-project/data/TNM/unpop_held_out_metrics_serious.csv",sep=",",header=True,index=True)


