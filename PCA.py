# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:29:57 2017

@author: Q
"""

import numpy as np
import matplotlib.pyplot as plt
def loadData(filename,delim = '\t'):
    with open(filename) as fr:
        stringArr = [line.strip().split(delim) for line in fr.readlines()]
        dataArr = [list(map(float,line)) for line in stringArr]
    return np.mat(dataArr)
def pca(dataSet,topNfeat = 99999999):
    dataMean = np.mean(dataSet,axis = 0)
    meanRemoved = dataSet - dataMean
    covMat = np.cov(meanRemoved,rowvar=0)
    eigVals,eigVec = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVec[:,eigValInd]
    lowDDataMat = meanRemoved * np.mat(redEigVects)
    reconMat = lowDDataMat * redEigVects.T + dataMean
    return lowDDataMat,reconMat
    

    
    
    
    
#data = loadData('testSet.txt')
#lowdata,recondta = pca(data,1)
#fig = plt.figure(0)
#ax = fig.add_subplot(111)
#ax.scatter(data[:,0],data[:,1],s=90,marker='^',c='r')
#ax.scatter(recondta[:,0],recondta[:,1],s=30,marker='o',c='b')
#plt.show()

def svd(dataSet,N = 1):
    U,sigma,VT = np.linalg.svd(dataSet)
    sig = np.mat(np.eye(N)*sigma[:N])
    new = U[:,:N] * sig * VT[:N,:]
    return new

    
data = loadData('testSet.txt')
recondta = svd(data,1)
fig = plt.figure(0)
ax = fig.add_subplot(111)
ax.scatter(data[:,0],data[:,1],s=90,marker='^',c='r')
ax.scatter(recondta[:,0],recondta[:,1],s=30,marker='o',c='b')
plt.show()


