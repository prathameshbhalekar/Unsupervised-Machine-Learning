import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

style.use('ggplot')

X, y = make_blobs(n_samples=15, centers=5, n_features=2)

class Mean_Shift:
    def __init__(self,radius=None,radius_norm_step =100):
        self.radius_norm_step =radius_norm_step 
        self.radius=radius
    def fit(self,data):
        if self.radius==None:
            center=np.average(data,axis=0)
            self.radius=np.linalg.norm(center)/self.radius_norm_step
        weights=[i for i in range(self.radius_norm_step)][::-1]
        self.centroids={}
        for i in range(len(data)):
            self.centroids[i]=data[i]
            
        while True:
            new_centroids=[]
            for i in self.centroids:
                centroid=self.centroids[i]
                in_bandwidth=[]
                for featureset in data:
                    distance=np.linalg.norm(centroid-featureset)
                    if(distance==0):
                        distance=0.00000001
                    weight_index=int(distance/self.radius)
                    if(weight_index>self.radius_norm_step-1):
                        weight_index=self.radius_norm_step-1
                        
                    in_bandwidth+=([featureset]*(weights[weight_index]**2))
                new_centroids.append(tuple(np.average(in_bandwidth,axis=0)))
            unique_centroids=sorted(list(set(new_centroids)))        
            
            old_centroids=dict(self.centroids)
            
            self.centroids={}
            for i in range(len(unique_centroids)):
                self.centroids[i]=np.array(unique_centroids[i])
            
            optimized = True
            
            for i in self.centroids:
                old=old_centroids[i]
                new=self.centroids[i]
                if not np.array_equal(new,old):
                    optimized=False
            if optimized :
                self.centroid=old_centroids
                break
        
        visited=[]
        pop={}
        for i in self.centroids:
            pop[i]=[]
        print(self.centroids)
        for i in self.centroids:
            for j in self.centroids:
                if i!=j:
                    c1=self.centroids[i]
                    c2=self.centroids[j]
                    if(np.linalg.norm(c1-c2)<=self.radius):
                        pop[i].append(j)
        print(pop)
        n_centroid=dict(self.centroid)
        v=[]
        for i in pop:
            if i in v:
                print(i)
                continue
            for j in pop[i]:
                try:
                    del n_centroid[j]
                except:
                    pass
                v.append(j)
        print(n_centroid)
        self.centroid=n_centroid
        
j=Mean_Shift()
j.fit(X)
for i in X:
    plt.scatter(i[0],i[1],color='b')

for i in j.centroids:
    c=j.centroids[i]

    plt.scatter(c[0],c[1],color='r')
plt.show()
