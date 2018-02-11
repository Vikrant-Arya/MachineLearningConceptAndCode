import numpy as np

#Create fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
    pointsPerCluster = float(N)/k
    #feature array
    X = []
    #predict array(Target array)
    y = []
    for i in range (k):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X, y


#%matplotlib inline
from pylab import *

(X, y) = createClusteredData(100, 5)

#to plot data on graph
#plt.figure(figsize=(8, 6))
#plt.scatter(X[:,0], X[:,1], c=y.astype(np.float)) #u.astype --> use for coloring data
#plt.show()

from sklearn import svm, datasets

C = 1.0
#support vector classfication
svc = svm.SVC(kernel='linear', C=C)
svc.fit(X, y)
#output
'''SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)'''


def plotPredictions(clf):
    #numpy.arange([start, ]stop, [step, ]dtype=None)
    xx, yy = np.meshgrid(np.arange(0, 250000, 10),
                     np.arange(10, 70, 0.5))
    '''
    Perform classification on samples in X.
    svm.svc().predict(X)
    Parameters: 
    X : {array-like, sparse matrix}, shape (n_samples, n_features)

    For kernel=”precomputed”, the expected shape of X is [n_samples_test, n_samples_train]

    Returns:    
    y_pred : array, shape (n_samples,)

    Class labels for samples in X.
    '''
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    #plot the pigure with size(W,H)
    plt.figure(figsize=(8, 6))

    #Reshape prediction in same frame
    Z = Z.reshape(xx.shape)
    # We are using automatic selection of contour levels;
    # this is usually not such a good idea, because they don't
    # occur on nice boundaries, but we do it here for purposes
    # of illustration.

    #Countourf --> https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.contourf.html
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    #ndarray.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)¶
    #Copy of the array, cast to a specified type.
    plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
    #Show Plot
    plt.show()
    
plotPredictions(svc)