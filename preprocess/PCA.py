#coding:utf8

import numpy as np



def zeroMean(data):      
	'''
	去中心化
	'''
    mean=np.mean(data,axis=0) 
    newData=data-mean
    return newData,mean
	
def PCA(data,d):
	'''
	data: 待处理数据
	d: 将数据投射到d维空间
	'''
    newData,meanVal=zeroMean(data)                #1.去中心化
    covMat=np.cov(newData,rowvar=0)               #2.求协方差矩阵
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))#3.求特征值和特征向量
    eigValIndice=np.argsort(eigVals)              #  对特征值从小到大排序
    d_eigValIndice=eigValIndice[-1:-(d+1):-1]     #  最大的d个特征值的下标
    d_eigVect=eigVects[:,d_eigValIndice]          #  最大的d个特征值对应的特征向量
    lowDDataMat=newData*n_eigVect                 #4.映射到低维特征空间的数据
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal    #5.重构数据
    return lowDDataMat,reconMat
    
    
    
    
    
    
    
    




