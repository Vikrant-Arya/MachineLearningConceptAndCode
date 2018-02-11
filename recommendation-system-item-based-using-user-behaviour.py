
#Item based collaborating filtering
# Method 2	
#In this method we will create recommended movies list for user behaviour
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
userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
userRatings.head()

'''
 pandas has a built-in corr() method that will compute a correlation score for every column pair in the matrix!
  This gives us a correlation score between every pair of movies 
  (where at least one user rated both movies - otherwise NaN's will show up.) That's amazing!
'''
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html
corrMatrix = userRatings.corr()
corrMatrix.head()

'''
min_periods : int, optional

Minimum number of observations required per pair of columns to have a valid result.
Currently only available for pearson and spearman correlation
'''
corrMatrix = userRatings.corr(method='pearson', min_periods=100)
corrMatrix.head()

'''
Now let's produce some movie recommendations for user ID 0, 
who I manually added to the data set as a test case. 
This guy really likes Star Wars and The Empire Strikes Back, but hated Gone with the Wind.
 I'll extract his ratings from the userRatings DataFrame, and use dropna() to get rid of missing data
  (leaving me only with a Series of the movies I actually rated:)
'''
myRatings = userRatings.loc[0].dropna()
myRatings

#pd.series() --> One-dimensional ndarray with axis labels (including time series).
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html

simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print ("Adding sims for " + myRatings.index[i] + "...")
    # Retrieve similar movies to this one that I rated
    sims = corrMatrix[myRatings.index[i]].dropna()
    # Now scale its similarity by how well I rated this movie
    sims = sims.map(lambda x: x * myRatings[i])
    
    # Add the score to the list of similarity candidates
    simCandidates = simCandidates.append(sims)
    
#Glance at our results so far:
print ("sorting...")
simCandidates.sort_values(inplace = True, ascending = False)
print (simCandidates.head(10))

'''
This is starting to look like something useful! Note that some of the same movies came up more than once,
because they were similar to more than one movie I rated. We'll use groupby()
to add together the scores from movies that show up more than once, so they'll count more:
'''
simCandidates = simCandidates.groupby(simCandidates.index).sum()

simCandidates.sort_values(inplace = True, ascending = False)
simCandidates.head(10)

'''
Remove my movie list from from similar movies list
'''
filteredSims = simCandidates.drop(myRatings.index)
print(filteredSims.head(10))