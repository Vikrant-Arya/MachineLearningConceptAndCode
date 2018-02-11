#Item based collaborating filtering
import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
#Read file in file
ratings = pd.read_csv('/home/vikranth/Downloads/VikrantPersonal/Machine-Learning/DataScience-Python3/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('/home/vikranth/Downloads/VikrantPersonal/Machine-Learning/DataScience-Python3/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

#Merge both tables they will merge on basis of movie_id
ratings = pd.merge(movies, ratings)

'''
Now the amazing pivot_table function on a DataFrame will construct a user / movie rating matrix.
Note how NaN(Not a number) indicates missing data - movies that specific users didn't rate.

'''
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.pivot_table.html
movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
movieRatings.head()

#lets check for start war
starWarsRatings = movieRatings['Star Wars (1977)']

#See top entries
starWarsRatings.head()

#Co relation between star war and other movies
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corrwith.html
similarMovies = movieRatings.corrwith(starWarsRatings)

#Drop all the coulmn with NaN(Not a number) field
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html
similarMovies = similarMovies.dropna()
df = pd.DataFrame(similarMovies)
df.head(10)

# Sort by corelation 
similarMovies.sort_values(ascending=False)

'''
Result is not effective here. This result is also considering those who give rating of two movies "Star war"
and "Other movies", And then it will suggest other person with high corelation, so we have to consider other 
factor also like "Total Views of that movies"

Our results are probably getting messed up by movies that have only been viewed by a handful of people 
who also happened to like Star Wars. So we need to get rid of movies that were only watched by a 
few people that are producing spurious results.
Let's construct a new DataFrame that counts up how many ratings exist for each movie, 
and also the average rating while we're at it - that could also come in handy later.
'''

import numpy as np
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
movieStats.head()	

#Take all those movies that has rating more then 100
popularMovies = movieStats['rating']['size'] >= 100

#Sort on basis of rating and means desending order	
movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15]

# Join movie state on basis of title with similar movies(we already caculate co relation between them and 
# give that co-relation name similarity.
df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))

#check its entry it will ne like
'''
					(rating, size)	(rating, mean)	similarity
title			
101 Dalmatians (1996)	109				2.908257	0.211132
'''
df.head()

#sort on basis of similarity
print(df.sort_values(['similarity'], ascending=False)[:15])

#And you will get recommended movieslist