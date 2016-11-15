#coding:utf8

import numpy as np

class KMeans:
    """
    - 参数
        n_clusters:
            聚类个数，即k
        max_iter:
            最大迭代次数
    """
    def __init__(self,n_clusters=5,max_iter=300):
        self.centroids = None  #中心点
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        '''
        进行聚类
        :param X: 数据集  array-like， shape = [n_samples,n_features]
        :return: self
        '''
        #对X进行类型检查
        if not isinstance(X,np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        m = X.shape[0]#m代表样本数量
        self.clusterAssment = np.empty((m,2))#m*2的矩阵，第一列存储样本点所属的族的索引值，
                                               #第二列存储该点与所属族的质心的平方误差

        self.centroids = self._randCent(X, self.n_clusters)

        clusterChanged = True
        for _ in range(self.max_iter):
            clusterChanged = False
            for i in range(m):#将每个样本点分配到离它最近的质心所属的族
                minDist = np.inf; minIndex = -1
                for j in range(self.n_clusters):
                    distJI = self._distEclud(self.centroids[j,:],X[i,:])
                    if distJI < minDist:
                        minDist = distJI; minIndex = j
                if self.clusterAssment[i,0] != minIndex:
                    clusterChanged = True
                    self.clusterAssment[i,:] = minIndex,minDist**2

            if not clusterChanged:#若所有样本点所属的族都不改变,则已收敛，结束迭代
                break
            for i in range(self.n_clusters):#更新质心，即将每个族中的点的均值作为质心
                ptsInClust = X[np.nonzero(self.clusterAssment[:,0]==i)[0]]#取出属于第i个族的所有点
                self.centroids[i,:] = np.mean(ptsInClust, axis=0)

        self.labels = self.clusterAssment[:,0]
        self.sse = sum(self.clusterAssment[:,1])
        return self

    def predict(self,X):#根据聚类结果，预测新输入数据所属的族
        #类型检查
        if not isinstance(X,np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        m = X.shape[0]#m代表样本数量
        preds = np.empty((m,))
        for i in range(m):#将每个样本点分配到离它最近的质心所属的族
            minDist = np.inf
            for j in range(self.n_clusters):
                distJI = self._distEclud(self.centroids[j,:],X[i,:])
                if distJI < minDist:
                    minDist = distJI
                    preds[i] = j
        return preds

    def _randCent(self, X, k):
        '''
        随机选取k个质心,必须在数据集的边界内
        :param X: 数据集  array-like， shape = [n_samples,n_features]
        :param k:质心的个数
        :return: k个质心， array-like, shape = [k,n_features]
        '''
        n = X.shape[1]        #特征维数
        centroids = np.empty((k,n))  #k*n的矩阵，用于存储质心
        for j in range(n):           #产生k个质心，一维一维地随机初始化
            minJ = min(X[:,j])
            rangeJ = float(max(X[:,j]) - minJ)
            centroids[:,j] = (minJ + rangeJ * np.random.rand(k,1)).flatten()
        return centroids

    def _distEclud(self, vecA, vecB):
        '''
        计算两点欧氏距离
        :return:
        '''
        return np.linalg.norm(vecA - vecB)
