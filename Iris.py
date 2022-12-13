import pandas as pd
from tools import*
from pprint import pprint
from sklearn import datasets


iris = datasets.load_iris()
df = pd.DataFrame(iris.data,columns= iris.feature_names)
data = tuple(df.itertuples(index=False,name=None))
for i in range(8):

    centroids=k_means(data,k=3)
    d= assign_data(centroids,data)


pprint(d,width=100)