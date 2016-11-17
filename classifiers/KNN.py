#coding:utf8
import numpy as np
import operator
class KNN:

    def __init__(self,k):
        '''
        :param k: int
        '''
        self.k = k

    def fit_predict(self,X,y,x):
        '''
        由于KNN没有训练过程，直接从k个最近的个体中选取类别数最多的一个。采用欧式距离进行计算
        :param X: array-like, shape=[n_sample, n_feature]
        :param y: array-like, shape=[n_sample,]
        :param x: 要预测的样本， array-like, [n_feature,]
        :return: label, x所属的类别
        '''
        dataSetSize=X.shape[0]
        diffMat = np.tile(x, (dataSetSize,1)) - X
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances**0.5
        sortedDistIndicies = np.argsort(distances)
        classCount={}
        for i in range(self.k):
            voteIlabel = y[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

if __name__ == "__main__":
    X =np.array([
        [1,2],
        [2,4],
        [1,1],
        [2,1]
    ])
    y = [1,2,1,1]
    knn = KNN(1)
    print knn.fit_predict(X,y,[5,5])  #应该返回2
