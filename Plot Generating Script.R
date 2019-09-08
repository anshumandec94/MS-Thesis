library('ggplot2')
df<-rbind(data.frame(dataset=1,obs=all_top_100_overlap_count$overlapCount),
          data.frame(dataset=2,obs=poponly_top_100_overlap_count$overlapCount))
df$dataset<-as.factor(df$dataset)

ggplot(df,aes(x=obs,fill=dataset))+
  geom_histogram(binwidth=1,colour="black",position="dodge")+
  scale_fill_manual(breaks=1:2,values=c("blue","green"),labels=c("Full Vectors","Popular Only Vectors"))+
  xlab("Number of overlap in recommendations with the top 100 occuring movies")+
  ylab("Occurences in users")+
  scale_y_continuous(breaks = seq(0,1000,50))+
  scale_x_continuous(breaks = seq(0,20,1))+
  ggtitle("Histogram of the number of overlap movies with the top 100 movies recommended\nin the recommendations\n of a user vs their pop-only user")

df2<-rbind(data.frame(dataset=1,obs=full_pop_unpop_count$unpopCount),
          data.frame(dataset=2,obs=poponly_pop_unpop_count$unpopCount))
df2$dataset<-as.factor(df2$dataset)

ggplot(df2,aes(x=obs,fill=dataset))+
  geom_histogram(binwidth=1,colour="black",position="dodge")+
  scale_fill_manual(breaks=1:2,values=c("blue","green"),labels=c("Full Vectors","Popular Only Vectors"))+
  xlab("Number of unpopular movies in the recommendations")+
  ylab("Occurences in users")+
  scale_y_continuous(breaks = seq(0,1000,50))  
 
library('dplyr')
rmse<-group_by(RMSE_ALS,Algorithm)
rmse_sum<-summarise(rmse,avg_score=sprintf("%0.7f",mean(RMSE)))
rmse_sum

RMSE<-rbind(RMSE_ALS_10.60,RMSE_ALS_70.80)  
RMSE_ALS_70.80<-RMSE_ALS_70.80[!RMSE_ALS_70.80$Algorithm == "Baseline Biased Scorer", ]


df<-rbind(data.frame(dataset=1,obs=user_diversity$diversity),data.frame(dataset=2,obs=user_diversity_pop$diversity))
df$dataset<-as.factor(df$dataset)
ggplot(df,aes(x=obs,fill=dataset))+
  geom_histogram(binwidth = 0.1,colour="black",position=position_dodge(width=0.05))+
  scale_fill_manual(breaks=1:2,values=c("red","blue"),labels=c("Full Vector","Popular Only Vector"))+
  xlab("Diversity of User Recommendations")+
  ylab("Occurences in users")+
  scale_x_continuous(breaks=seq(0,1.5,0.1))+
  ggtitle("Comparative Histogram of Diversity of Recommendations for User Profiles\nDiversity is Pairwise cosine of movie feature vectors\nBaseline Diveristy of All Movies Recommended: 0.6110(Full) 0.6040(Popular Only)")

library('ggplot2')
df<-rbind(data.frame(dataset=1,obs=recsummaryfull$popcount),data.frame(dataset=2,obs=recsummarypop$popcount))
df$dataset<-as.factor(df$dataset)
ggplot(df,aes(x=obs,fill=dataset))+
  geom_histogram(binwidth = 1,colour="black",position=position_dodge(width=0.5))+
  scale_fill_manual(breaks=1:2,values=c("red","blue"),labels=c("Full Vector","Popular Only Vector"))+
  xlab("Number of popular movies recommended to user")+
  ylab("Occurences in users")+
  scale_x_continuous(breaks=seq(0,20,1))+
  ggtitle("Histogram of Popular Movies recommended to users")
  
ggplot()+
  geom_point(aes(y=recsummaryfull$avgrating,x=recsummaryfull$avgscore-recsummaryfull$avgrating,color="Full Profile"))+
  geom_point(aes(y=recsummarypop$avgrating,x=recsummarypop$avgscore-recsummarypop$avgrating,color="Popular Only Profile"))+
  labs(color="User Profile Used")+
  xlab("Average Rating of Movies Recommended-Average Score of Movies Recommended ")+
  ylab("Average Rating of Movies Recommended")+
  scale_x_continuous(breaks=seq(-2,2,0.5))+
  ggtitle("Plot of the Average Rating of a user's top-20\nvs The Difference of Average Rating of top-20 and Average Score of top-20")

  
  
ggplot(df, aes(x=obs,fill=dataset))+
  geom_histogram(binwidth=1450,colour="black",position="dodge")

ggplot()+
  geom_histogram(aes(x=(movie_recs$popularity),color="Full Profile"),position="dodge")+
  geom_histogram(aes(x=(movie_recs$popularity),color="Popular Profile"),position="dodge")+
  labs(color ="User Profile Used")
ggplot()+
  geom_count(aes(x=(log(movie_recs_pop$popularity)),y=(movie_recs_pop$impression),color="Popular movies only profile"))+
  geom_count(aes(x=(log(movie_recs$popularity)),y=(movie_recs$impression),color='All movies profile'))+
  xlab("Popularity(Number of ratings) on Logarithmic scale")+
  ylab("Number of Impressions(No of users the movie was recommended to")+
  ggtitle("Popularity vs Impressions where each dot is a movie recommended for a user profile")
  labs(color="User Rating Profile")
  
ggplot()+
  geom_histogram(bins=21,aes(x=user_overlap$overlap),colour="black")+
  xlab("Overlap in Top-20 Recommendations")+
  ylab("Occurences in Users")+
  scale_x_continuous(breaks=seq(0,20,1),minor_breaks = seq(0,20,1)) +
  scale_y_continuous(breaks=seq(0,300,10))+
  geom_vline(xintercept=4.33,colour="blue")+
  geom_text(aes(x=4,label="average overlap within group",y=100),angle=90)+
  geom_vline(xintercept=0,colour="red")+
  geom_text(aes(x=-0.2,label="average overlap outside group",y=100),angle=90)
  #ggtitle("Histogram of Overlap in Recommendations generated for a user's full and popular only profile")


  #(Pairwise Similarity)=0.00035(full),0.0010(popular only)\nData Grouped by Percentage of Popular Movies Rated by Users
ggplot()+
  geom_count(aes(x=ALS25Fr01filteredfvsp$full,y=ALS25Fr01filteredfvsp$popularonly))+
  scale_x_continuous(breaks=seq(0,20,1))+
  scale_y_continuous(breaks=seq(0,20,1))+
  xlab("Overlap with Top-20 Recommended for Full Profile")+
  ylab("Overlap with Top-20 Recommended for Popular Only Profile")
  #ggtitle("Plot of Overlap of Recommendations for users with Top-20 Popular Movies between Full and Popular Only Profile\nLegend indicates number of observations")
 

ggplot(data=half_remove_recommendations_summary)+
  geom_histogram(binwidth = 1,aes(x=half_remove_recommendations_summary$popcount),colour="black")


usrgrp1<-nrow(user_sim_group[user_sim_group$group == "Less than 80%" & user_sim_group$simscore>0.9, ])
usrgrpn<-nrow(user_sim_group[user_sim_group$group == "Less than 80%", ])

ggplot()+
  geom_histogram(binwidth=0.01,aes(x=user_simscore$simscore),colour="black")+
  scale_x_continuous(breaks=seq(0,1,0.1))+
  xlab("Cosine Similarity")+
  ylab("Number of users")+
  geom_vline(xintercept=0.003,colour="red")+
  geom_text(aes(x=-0.01,label="Average cosine similarity",y=100),angle=90)+
  geom_vline(xintercept=0.923,colour="blue")+
  geom_text(aes(x=0.9,label="most similar neighbor cosine similarity",y=100),angle=90)



ggplot(data=user_sim_group)+
  geom_histogram(binwidth=0.01,aes(x=user_sim_group$simscore),colour="black")+
  scale_x_continuous(breaks=seq(0,1,0.1))+
  xlab("Cosine Similarity")+
  ylab("Number of users")+
  facet_grid(~ group)+
  guides(fill=guide_legend(title="Popularity group"))
  #geom_vline(xintercept=0.003,colour="red")+
  #geom_text(aes(x=-0.01,label="Average cosine similarity",y=100),angle=90)+
  #geom_vline(xintercept=0.923,colour="blue")+
  #geom_text(aes(x=0.9,label="most similar neighbor cosine similarity",y=100),angle=90)






