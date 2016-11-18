#coding:utf8
import numpy as np

class LogisticRegression():

    def __init__(self,alpha=0.1, maxCycles=1000):
        '''
        :param alpha:学习步长
        :param maxCycles: 最大循环次数
        :return:
        '''
        self.alpha = alpha
        self.maxCycles = maxCycles
        self.weight = None


    def _sigmoid(self, fx):
        '''
        sigmoid函数
        '''
        return 1.0/(1 + np.exp(-fx))

    
    def _gradDescent(self, feature, label, maxCycles):
        '''
        梯度下降法
        '''
        dataMat = np.mat(feature)                      #size: m*n
        labelMat = np.mat(label).transpose()        #size: m*1
        m, n = np.shape(dataMat)
        self.weight = np.ones((n, 1))
        for i in range(maxCycles):
            hx = self._sigmoid(dataMat * self.weight)
            error = labelMat - hx       #size:m*1
            self.weight = self.weight + self.alpha * dataMat.transpose() * error#根据误差修改回归系数
        return self


    def fit(self, train_x, train_y):
        '''
        梯度下降法训练
        :param train_x: 训练集
        :param train_y: 训练集标签
        '''
        return self._gradDescent(train_x, train_y, self.maxCycles)


    def predict(self, test_X):
        '''
        预测分类
        :param test_X:
        :param test_y:
        :return:
        '''
        dataMat = np.mat(test_X)
        hx = self._sigmoid(dataMat*self.weight)  #size:m*1
        return hx>0.5

if __name__ == "__main__":
    X =np.array([
        [1,2],
        [2,4],
        [1,1],
        [2,2]
    ])
    y = [True,True,False,False]
    lr = LogisticRegression()
    print lr.fit(X,y).predict([5,10])  #应该返回True