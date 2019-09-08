library('dplyr')
movies<- group_by(ratings, movieId)
movie_data<-summarise(movies,popularity = n(), avgrating = mean(rating))
write.csv(movie_data,'/project/naray190/movie_data_20M.csv',row.names = FALSE)
popIds<-movie_data %>%
  filter(popularity > 1450) %>%
  select(movieId)
popIds<-as.vector(popIds)


 write.csv(user_data,'/project/naray190/user_data_20M.csv',row.names = FALSE)
user_data<-summarise(users,count=n(),popcount=sum(filter(users,movieId %in% popIds)))



summarize_if() 
userspnp<-mutate(ratings20Mfull,ispop=ifelse(ratings20Mfull$movieId%in%popIds$movieId,"Pop","Unpop")) %>% group_by(userId)
user_data<-summarise(userspnp,count=n(),popcount=sum(ispop=="Pop"),unpopcount=sum(ispop=="Unpop"),percentpop=popcount/count)

ratings20Mfull$ispop=ifelse(ratings20Mfull$movieId%in%popIds,"Pop","Unpop")

test_users<- user_data %>%
  filter(count>200 & unpopcount>1) %>%
  select(userId) %>%
  sample_n(size=1000)
test_users<- user_data %>%
  filter(count>50 & count<200 & unpopcount>1) %>%
  select(user) %>%
  sample_n(size=1000)
test_ratings<-ratings[which(ratings$userId %in% test_users$user & ratings$movieId %in% popIds$movieId), ]
truncated_ratings <- ratings[which(ratings$userId %in% test_users$user & ratings$movieId %in% popIds$movieId ), ]
other_users<-user_data%>%
  filter(typeofuser == "more than 201")%>%
  select(userId)
ousize=nrow(other_users)

other_users<-user_data%>%
  filter(count<200)%>%
  #sample_n(size=ousize/2)%>%
  select(userId)
all_users<-user_data%>%
  select(user)
rem_users<-setdiff(all_users,test_users)
truncated_data<-ratings[which(ratings$userId %in% other_users$userId & ratings$movieId %in% popIds$movieId), ]
casual_ratings<-ratings[which(ratings$userId %in% rem_users$user), ]
casual_users<-user_data%>%
  filter(typeofuser %in% c("0 to 50","51 to 200")) %>%
  select(userId)

truncated_ratings<-rbind(truncated_ratings,casual_ratings)
write.csv(test_ratings,"/project/naray190/ml-20m/test_ratings.csv",row.names=FALSE)
write.csv(truncated_data,"/project/naray190/ml-20m/half_casual_pop_ratings.csv",row.names=FALSE)
write.csv(truncated_data,"/project/naray190/ml-20m/casual_pop_ratings.csv",row.names = FALSE)
write.csv(truncated_ratings,"/project/naray190/ml-20m/popular_only_ratings.csv",row.names=FALSE)
write.csv(test_ratings, "/project/naray190/ml-20m/test_casual_user_ratings.csv",row.names=FALSE)
check<-user_data[user_data$userId %in% test_users$userId, "popcount"]
table(user_data$countgrp)
library('ggplot2')
ggplot(data=user_data,aes(x=(percentpop),fill=countgrp))+
  geom_histogram(binwidth=0.1,position="identity")+
  #geom_density(aes(y=0.1*..count..)) +
  facet_grid(~ countgrp) +
  labs(fill="Number of Items User has Rated")+
  xlab("Percentage of popular movies rated by the user") +
  ylab("Number of users") +
  ggtitle("Histogram of percentage of popular movies in user's rating profile ")
 
t1<-c
ggplot(data=user_data,aes(x=(popcount),fill=countgrp))+
  geom_histogram(binwidth=1,position="identity")+
  
  #geom_density(aes(y=0.1*..count..)) +
  facet_grid(~ countgrp)

ggplot(data=movie_data,aes(x=log(popularity),fill=popgroup))+
  geom_histogram(binwidth=0.1,position="identity")+
  scale_x_continuous(breaks=seq(0,15,1))+
  scale_y_continuous(breaks=seq(0,4000,500))+
  xlab("logarithmic scale of popularity(no of ratings)")+
  ylab("Number of movies")+
  guides(fill=guide_legend(title="Popularity group"))+
  ggtitle("Histogram of log popularity of movie")+
  geom_vline(xintercept =7.279,colour="red")+
  geom_text(aes(x=7.1,label="Popular movie cutoff",y=2000),angle=90)
  #geom_density(aes(y=..count..))+
  #facet_grid(~ popgroup)
names(user_data)[names(user_data) == "userId"] <- "user"  
user_sim_group<-merge(user_simscore,user_data,by=c("user")) 
user_sim_group$group<-cut(user_sim_group$percentpop,breaks=c(-Inf,0.8,0.9,Inf),labels=c("Less than 80%","80 to 89 %","90% or more"))  
  
user_data$typeofuser <- cut(user_data$count,breaks=c(-Inf,51,201,Inf),labels = c("0 to 50","51 to 200","more than 201"),right = FALSE)
movie_data$popgroup<-cut(movie_data$popularity,breaks=c(-Inf,11,1001,10001,Inf),labels=c("0 to 10","11 to 1000","1001 to 10000","more than 10000"))
movie_data$pop<-cut(movie_data$popularity,breaks=c(-Inf,1449,Inf),labels=c("Outside Top 2500","Top 2500"))