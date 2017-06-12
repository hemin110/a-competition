# -*- coding: utf-8 -*-
'''
auther@ AIbert
hantaoer@foxmail.com
'''
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors as NNB
def getCluster(rawdata, flag, datatojudge, num_neighbors):

	neighbors = NNB(n_neighbors=num_neighbors,n_jobs=-1).fit(rawdata)
	locs = neighbors.kneighbors(datatojudge,return_distance=False)
	locs = locs.T
	rawflag = rawdata[flag]
	results = np.zeros(len(datatojudge))
	for i in range(num_neighbors):
		results = results + np.array(rawflag.iloc[locs[i]])
	index1 = ((results==0) | (results==num_neighbors)).T
	datatojudge = datatojudge[index1]
	results = results[index1]/num_neighbors
	targetflag = datatojudge[flag]
	index2 = np.array(targetflag==results)
	return datatojudge[index2]


if __name__ == '__main__':
	
	import matplotlib.pyplot as plt	
	a = pd.DataFrame(np.random.rand(16,2)*100)
	b = pd.DataFrame(np.random.rand(40,2)*100)
	a['flag'] = np.random.randint(0,2,16)
	b['flag'] = np.random.randint(0,2,40)
	mixture = getCluster(a,'flag', b, 2)
	plt.scatter(a[0],a[1],color=['grey','grey'],marker='o',s=40)
	mixture = mixture.append(a)
	plt.scatter(mixture[0],mixture[1],color=['r','g'],marker='o',s=20,alpha=0.6)
	plt.show()
