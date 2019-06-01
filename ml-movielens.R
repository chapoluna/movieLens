###################################
# MovieLens Recommendation System
# Fernando Scalice Luna
# HarvardX Data Science Program
###################################

###################################
# Create edx set and validation set
###################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1) # if using R 3.6.0: set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Required libs to run this project
library(stringr)
library(gridExtra)
library(gtable)
library(grid)

###################################
# Data Exploration
###################################

# Distribution of Ratings

edx %>% group_by(movieId) %>%
  summarise(n = n()) %>%
  ggplot(aes(n)) + 
  geom_histogram(fill = "darkred", color = "white", bins = 30) +
  scale_x_log10() +
  ggtitle("Movies Ratings") +
  xlab("Number of ratings (log10)") +
  ylab("Number of movies")

edx %>% group_by(userId) %>%
  summarise(n = n()) %>%
  ggplot(aes(n)) + 
  geom_histogram(fill = "darkred", color = "white", bins = 30) +
  scale_x_log10() +
  ggtitle("User Ratings") +
  xlab("Number of ratings (log10)") +
  ylab("Number of users")

edx %>% group_by(userId) %>%
  summarise(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(fill = "darkred", color = "white", bins = 30) +
  ggtitle("Ratings") +
  xlab("Rating") +
  ylab("Number of ratings")

mean(edx$rating)

# Exploring the AGE of the movies
# Extracting the Release Date
release_pattern <- "(\\()(\\d{4})(\\))"
edx_age <- edx %>% mutate(release = as.numeric(str_match(edx$title, release_pattern)[,3]))

# Checking if extraction is ok
identical(as.numeric(str_match(edx_age$title, release_pattern)[,3]), edx_age$release)

# Calculating the AGE
edx_age <- edx_age %>% mutate(age = 2019 - release)

# Distribution of rating by Age
edx_age %>% 
  ggplot(aes((age))) +
  geom_histogram(fill = "darkred", color = "white", bins = 30) +
  ggtitle("Ratings by Age") +
  xlab("Age") +
  ylab("Number of ratings")

  mean(edx_age$age)
  
# Number of movies by Age
edx_age %>% group_by(movieId) %>% summarise(n = n(), age = first(age)) %>%
    ggplot(aes((age))) +
    geom_histogram(fill = "darkred", color = "white", bins = 30) +
    ggtitle("Movies by Age") +
    xlab("Age") +
    ylab("Number of movies")

# Average Rating by Age
avg_rating_by_age <- edx_age %>% group_by(age) %>% summarise(avg_rating = mean(rating))

avg_rating_by_age %>%
  ggplot(aes(x = age, y = avg_rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Average rating by Movie Age") +
  xlab("Movie Age") +
  ylab("Average Age")

# Exploring single GENRES influence
edx_genres <- edx %>% separate_rows(genres, sep ="\\|")

avg_rating_by_genre <- edx_genres %>% group_by(genres) %>%
  filter(n() > 1000) %>%
  summarise(n = n(), avg_rating = mean(rating), se_rating = sd(rating)/sqrt(n))

avg_rating_by_genre.plot_a <- avg_rating_by_genre %>%
  ggplot(aes(x = genres, y = avg_rating, ymin = avg_rating - 2*se_rating, ymax = avg_rating + 2*se_rating)) +
  geom_point() +
  geom_errorbar() +
  ggtitle("Average ratings by Genre") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  xlab("Genre") +
  ylab("Average rating")

avg_rating_by_genre.plot_b <- avg_rating_by_genre %>%
  ggplot(aes(x = genres, y = n)) +
  geom_bar(stat = "identity") +
  ggtitle("Ratings by Genre") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  xlab("Genre") +
  ylab("Number of ratings")

g1 <- ggplotGrob(avg_rating_by_genre.plot_a)
g2 <- ggplotGrob(avg_rating_by_genre.plot_b)
g <- rbind(g1, g2, size = "first")
g$widths <- unit.pmax(g1$widths, g2$widths)
grid.newpage()
grid.draw(g)

###################################
# Building the Model
###################################

# The RMSE function to evaluate our models
RMSE <- function(true_ratings, predicted_ratings){ 
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# First Model - Simple Average
# Yu,i = mu + Eu,i

mu_hat <- mean(edx$rating) 
mu_hat

naive_rmse <- RMSE(validation$rating, mu_hat) 
naive_rmse

rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)

# Second Model - Movie Effects
# Yu,i = mu + bi + Eu,i

mu <- mean(edx$rating) 
movie_avgs <- edx %>%
  group_by(movieId) %>% summarize(b_i = mean(rating - mu))

movie_avgs %>% ggplot(aes(b_i)) +
  geom_histogram(fill = "darkred", color = "white", bins = 30)

predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>% 
  pull(b_i)

movie_rmse <- RMSE(validation$rating, predicted_ratings) 
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie Effect Model", RMSE = movie_rmse))
rmse_results

# Third Model - User Effects
# Yu,i = mu + bi + bu + Eu,i

user_avgs <- edx %>% left_join(movie_avgs, by='movieId') %>% 
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>% 
  left_join(user_avgs, by='userId') %>% 
  mutate(pred = mu + b_i + b_u) %>% 
  pull(pred)

user_rmse <- RMSE(validation$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results, tibble(method="Movie + User Effects Model", RMSE = user_rmse))

rmse_results

# Fourth Model - Age Effects
# Yu,i = mu + bi + bu + f(dui) + Eu,i

# From our data exploration, it seems movie age plays a role in estimating the rate of a movie. 
# Let's try the loess function to estimate and incorporate it in our model
total_span <- diff(range(avg_rating_by_age$age))
span <- 30/total_span

avg_rating_by_age.loess <- loess(avg_rating ~ age, degree = 1, span = span, data = avg_rating_by_age)

avg_rating_by_age %>% mutate(smooth = avg_rating_by_age.loess$fitted) %>% 
  ggplot(aes(age, avg_rating)) +
  geom_point(size = 3, alpha = .5, color = "black") + 
  geom_line(aes(age, smooth), size = 1, color="red")

age_avgs <- edx_age %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(age) %>%
  summarize(b_a = mean(predict(avg_rating_by_age.loess, newdata = age) - mu - b_i - b_u))

validation <- validation %>% 
  mutate(release = as.numeric(str_match(title, release_pattern)[,3])) %>% 
  mutate(age = 2019 - release)

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>% 
  left_join(user_avgs, by='userId') %>%
  left_join(age_avgs, by='age') %>%
  mutate(pred = mu + b_i + b_u + b_a) %>% 
  pull(pred)

age_rmse <- RMSE(validation$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results, tibble(method = "Movie, User and Age Effects Model", RMSE = age_rmse))
rmse_results
# Since it does not improve our model, we're not going to use it in the final model

# Time to tune our final model
lambdas <- seq(0, 5, .25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx_age$rating)
  
  b_i <- edx_age %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx_age %>%
    left_join(b_i, by="movieId") %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- edx_age %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>% .$pred
  
  return(RMSE(predicted_ratings, edx_age$rating))
})

qplot(lambdas, rmses)
lambdas[which.min(rmses)]

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie + User Effect Model",
                                 RMSE = min(rmses)))

rmse_results %>% knitr::kable()

######################################
# Predicting using the Validation set
#####################################

mu <- mean(validation$rating)
l <- lambdas[which.min(rmses)]

b_i <- validation %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))

b_u <- validation %>%
  left_join(b_i, by="movieId") %>% 
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

predicted_ratings <- validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>% .$pred

RMSE(predicted_ratings, validation$rating)

