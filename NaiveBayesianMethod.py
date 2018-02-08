import os
import io
import numpy
#pandas lib use for reading data from tabular form
from pandas import DataFrame
#Using sklearn library's function we can calculate naive baysian 
from sklearn.feature_extraction.text import CountVectorizer
#NB --> Naive baysian
from sklearn.naive_bayes import MultinomialNB

#Generator
def readFiles(path):
    """OS.walk() generate the file names in a directory tree by walking the tree either top-down or bottom-up.
    For each directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames)."""
    
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                #escape header
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            """https://github.com/coderc/MachineLearningConceptAndCode/blob/master/python/iterable-generators-yield.text"""
            yield path, message


#data from datafiles
def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)
#Create table with message and class --> spam/ham
data = DataFrame({'message': [], 'class': []},index=[])
data = data.append(dataFrameFromDirectory('/home/vikranth/Downloads/VikrantPersonal/Machine-Learning/DataScience-Python3/emails/spam/', 'spam'))
data = data.append(dataFrameFromDirectory('/home/vikranth/Downloads/VikrantPersonal/Machine-Learning/DataScience-Python3/emails/ham/', 'ham'))

#validate data 
data.head()

"""Now we will use a CountVectorizer to split up each message into its list of words,"""
vectorizer = CountVectorizer()
"""It will change word into number and store its frequency"""
counts = vectorizer.fit_transform(data['message'].values)
"""Naive Baysian classifier"""
classifier = MultinomialNB()
"""Set target for training set"""
targets = data['class'].values
"""use classifier to fit data set"""
classifier.fit(counts, targets)

"""example"""
examples = ['Free games now!!!', "Hi mike, how about a game of golf tomorrow?"]
"""Word count """
example_counts = vectorizer.transform(examples)
"""Check test data from trained classifier"""
predictions = classifier.predict(example_counts)
predictions
