# -*- coding: utf-8 -*-
'''
auther@ AIbert
hantaoer@foxmail.com
'''
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class imbalanceData():			
	""" 
	    docstring for imbalanceData
	    K is percent of data to generate, N is the number of nodes in each neighbours group
		compared with the less category, such as 0.5,0.6,1,2 ..
		Null is not accepted, so you should fillna during preprocessing

	"""
	def __init__(self, data, flag):
		self.data = data
		self.flag = flag
		self.len = len(self.data)

	def new_data(self,data,groups,N,K):
		locs = groups.kneighbors(data.reshape(1,-1),return_distance=False)[0]
		loc = np.random.randint(0,K,size=N) # chose N neighbors
		alpha = (np.random.rand(N,1))*np.ones((1,self.data.shape[1]))
		res =  (self.data.iloc[locs[loc],:]-data)*alpha # N newdata generated
		return res

	def smote(self,N,K,random_state=100):
		if(N<=0):
			return pd.DataFrame()
		if(N<1):
			self.data = self.data.sample(int(N*self.len)+K,random_state=random_state)
			self.len = len(self.data)
			N = 1
		# select samples to do smote
		s1 = int(N)
		groups=NearestNeighbors(n_neighbors=K).fit(self.data)
		print '>>',
		# print('>>',endwith=" ")  # py3
		newdata = pd.DataFrame()
		np.random.seed(seed=random_state)
		for i in self.data.iterrows():
			locs = groups.kneighbors(i[1].reshape(1,-1),return_distance=False)[0]
			loc = np.random.randint(0,K,size=s1) 		# chose s1 neighbors
			alpha = (np.random.rand(s1,1)-0.2)*np.ones((1,self.data.shape[1]))
			newdata = newdata.append((self.data.iloc[locs[loc],:]-i[1])*alpha + i[1]) # N newdata generated
		s2 = N-s1
		if(s2 > 0):
			newdata = newdata.append(self.smote(s2,K,random_state=random_state))
		newdata[self.flag] = 1
		newdata.columns = self.data.columns
		return newdata

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	np.random.seed(seed=120)
	a = pd.DataFrame(np.random.rand(6,2)*20)
	b = pd.DataFrame(np.random.rand(20,2)*20)
	a['flag'] = 1
	b['flag'] = 0
	data = a.append(b)
	data.plot(kind='scatter',x=0,y=1,c='flag')
	plt.hold(True)
	b =	imbalanceData(a,flag='flag').smote(4,5,100)
	plt.scatter(x=b[0],y=b[1],c='r')
	plt.show()

