import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import preprocessing
style.use('ggplot')

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
              [1,3],
              [8,9],
              [0,3],
              [5,4],
              [6,4],])

colors=['b','r','c','g','k']


class K_Means:
    def __init__(self,k=2,tol=0.0001,max_iter=300):
            self.k=k
            self.tol=tol
            self.max_iter=max_iter
        
    def fit (self, data):
        self.centroid={}
        #   Assigning first three points as initial centroids
        for i in range (self.k):
            self.centroid[i]=data[i]
        for i in range(self.max_iter):
            self.classification={}
            for i in range (self.k):
                self.classification[i]=[]
            
            #   distributing feature sets wrt centroid they are closest to
            for featureset in data:
                distance={}
                for i in self.centroid:
                    distance[i]=np.linalg.norm(featureset-self.centroid[i])
                min_i=99999
                min_val=99999
                for i in distance:
                    if distance[i]<min_val:
                        min_val=distance[i]
                        min_i=i
                self.classification[min_i].append(featureset)
            
            old_centroid=dict(self.centroid)
            #   changing centroid as the average of all featureset closest to the old centroid
            for i in self.centroid:
                self.centroid[i]=np.average(self.classification[i],axis=0)
            optimized=True
            
            #   Checking if the change in centroid isbelow tolerence
            for i in self.centroid:
                new=self.centroid[i]
                old=old_centroid[i]
                if(sum((new-old)/old*100.0)>self.tol):
                    optimized=False
                    print(sum((new-old)/old*100.0))
                    break
            if optimized:
                break
                
    def predict(self,data):
        dist=[]
        #   Checking which centroid is closest to the given featureset
        for i in self.centroid:
            dist.append(np.linalg.norm(data-self.centroid[i]))
        return dist.index(min(dist))
            
            

k=K_Means()
k.fit(X)

for i in k.classification:
    for j in k.classification[i]:
        plt.scatter(j[0],j[1],color=colors[i])

for i in k.centroid:
    plt.scatter(k.centroid[i][0],k.centroid[i][1],marker='*',s=100)

unknowns = np.array([[1,3],
                    [8,9],
                    [0,3],
                    [5,4],
                    [6,4],])

for i in unknowns:
    prediction=k.predict(i)
    print(prediction)
    plt.scatter(i[0],i[1],color=colors[prediction],marker='x',s=100)
plt.show()
