import numpy as np
import pandas as pd
from sklearn import tree

#Read Hire database file
input_file = "/home/vikranth/Downloads/VikrantPersonal/Machine-Learning/DataScience-Python3/PastHires.csv"
df = pd.read_csv(input_file, header = 0)

#check some entry of file
df.head()

"""
	Years Experience	Employed?	Previousemployers	Level of Education	Top-tierschool	Interned	Hired
0		10					Y			4						BS				N				N		Y
1		0					N			0						PHD				Y				Y		Y
2		7					N			6						MS				N				N		N
"""

#for scikit-learn need everything in numerical format to solve decision tree
#So we have to convert Y--> 1 , N-->0 also level of educaiton bs--> 1, MS-->2, PHD-->3

d = {'Y': 1, 'N': 0}
df['Hired'] = df['Hired'].map(d) #changed
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
d = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(d) #changed
df.head()

'''
id YearsExperience	Employed?	Previous employers	Level of Education	Top-tier school	Interned	Hired
0		10				1				4					0				0				0			1
'''

#choose feature on which we have to take decision
features = list(df.columns[:6])
features

y = df["Hired"]
X = df[features]

#Create decision tree classifier

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

#using graphviz we can create decision tree graph

from IPython.display import Image  
from sklearn.externals.six import StringIO  
import pydotplus

dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=features)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  

#In out put image https://github.com/coderc/MachineLearningConceptAndCode/blob/master/output-images/decision-tree.png
#Attribute ares : features and gini(entropy), values(not-hired,hired), sample--> total input