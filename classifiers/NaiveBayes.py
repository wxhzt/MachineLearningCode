#coding:utf8

import numpy as np
from sklearn.preprocessing import OneHotEncoder
"""
        贝叶斯模型的具体实现有三种：多项式模型，高斯模型，和伯努利模型。
"""
class MultinomiaNB:
    '''
    多项式模型
    '''
    def __init__(self,alpha=1.0):
        '''
        :param alpha: double
                      平滑参数
        :return: object
                 self
        '''
        self.alpha = alpha
        #self.fit_prior = fit_prior
        self.class_prior = None     #先验概率  P(y)
        self.classes = None
        self.conditional_prob = None #条件概率 P(X|y)

    def fit(self, X, y):
        """
        训练贝叶斯分类器
        :param X: array-like, shape = [n_samples, n_features]
                  特征向量
        :param y: array-like, shape = [n_samples,]
                  标签
        :return:  object
                  self
        """
        return self._compute_prior_proba(y)._compute_cond_proba(X,y)

    def predict(self, X):
        """
        对输入的特征向量预测分类
        :param X: array-like, shape = [n_features,]
                  特征向量
        :return:  int
                  返回X的分类
        """
        ypb = self.predict_single_feature(X)
        maxp = np.max(ypb)
        cindex = np.where(ypb == maxp)
        return self.classes[cindex]

    def predict_prob(self, X):
        '''
        对输入的特征向向量预测属于各个分类的可能性
        :param X: array-like, shape = [n_features]
                  特征向量
        :return:  array-like, shape = [n_classes,]
                  返回X的属于各个分类的可能性
        '''
        return self.predict_single_feature(X)

    def predict_single_feature(self, x):
        '''
        被predict_prob调用
        '''
        y = []
        for c in self.classes:
            pXy = 1
            for i in range(np.size(x)):
                pXy = pXy*self.conditional_prob[c][i][x[i]]
            y.append(self.class_prior[c]*pXy)
        return y/np.sum(y)

    def _compute_prior_proba(self,y):
        '''
        计算先验概率(Y)
        :param y: array-like, shape = [n_classes,]
                  label 数组
        :return:  object
                  self
        '''
        self.classes = np.unique(y)
        classnum = len(self.classes)
        self.class_prior = []
        sample_num = float(len(y))
        for c in self.classes:
            c_num = np.sum(np.equal(y,c))
            self.class_prior.append((c_num+self.alpha)/(sample_num+classnum*self.alpha))

        return self

    def _compute_cond_proba(self,X,y):
        '''
        计算P(X|Y)
        '''
        self.conditional_prob = {}
        for c in self.classes:
            self.conditional_prob[c] = {}
            for i in range(len(X[0])):  #for each feature
                feature = X[np.equal(y,c)][:,i]
                self.conditional_prob[c][i] = self._calculate_feature_prob(feature)

        return self

    def _calculate_feature_prob(self,feature):
        '''
        计算P(Xi|Yj)
        :param feature: 一个特定label中某个特征分布
        :return:
        '''
        values = np.unique(feature)
        total_num = float(len(feature))
        value_prob = {}
        for v in values:
            value_prob[v] = (( np.sum(np.equal(feature,v)) + self.alpha ) /( total_num + len(values)*self.alpha))
        return value_prob


class GaussianNB(MultinomiaNB):
    '''
    高斯模型
    '''
    def predict_single_feature(self, x):
        '''
        被predict_prob调用
        '''
        y = []
        for c in self.classes:
            pXy = 1
            for i in range(np.size(x)):
                pXy = pXy*self._get_xj_prob(self.conditional_prob[c][i][0],self.conditional_prob[c][i][1],x[i])
            y.append(self.class_prior[c]*pXy)

        return y/np.sum(y)

    def _compute_cond_proba(self,X,y):
        '''
        计算P(X|Y)
        '''
        self.conditional_prob = {}
        for c in self.classes:
            self.conditional_prob[c] = {}
            for i in range(len(X[0])):  #for each feature
                feature = X[np.equal(y,c)][:,i]
                self.conditional_prob[c][i] = self._calculate_feature_prob(feature)

        return self

    def _calculate_feature_prob(self,feature):
        '''
        计算高斯分布的均值和方差
        :param feature: 一个特定label中某个特征分布
        :return:
        '''
        mu = np.mean(feature)
        sigma = np.std(feature)
        return (mu,sigma)

    def _prob_gaussian(self,mu,sigma,x):
        '''
        计算概率密度
        :param mu:均值
        :param sigma:方差
        :param x: double
                  目标值
        :return:  double
                  概率密度
        '''
        return ( 1.0/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2)) )

    def _get_xj_prob(self,mu,sigma,target_value):
        return self._prob_gaussian(mu,sigma,target_value)

class BernoulliNB(MultinomiaNB):
    def fit(self, X, y):
        """
        训练贝叶斯分类器
        :param X: array-like, shape = [n_samples, n_features]
                  特征向量
        :param y: array-like, shape = [n_samples,]
                  标签
        :return:  object
                  self
        """
        self.enc = OneHotEncoder()
        self.enc.fit(X)

        return self._compute_prior_proba(y)._compute_cond_proba(self.enc.transform(X).toarray(),y)
    def predict(self, X):
        """
        对输入的特征向量预测分类
        :param X: array-like, shape = [n_samples, n_features]
                  特征向量
        :return:  array-like, shape = [n_samples,]
                  返回X的分类
        """
        ypb = self.predict_single_feature(self.enc.transform(X).toarray()[0])
        maxp = np.max(ypb)
        cindex = np.where(ypb == maxp)
        return self.classes[cindex]

if __name__ == '__main__':
    X = np.array([
                      [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
                      [4,5,5,4,4,4,5,5,6,6,6,5,5,6,6]
                ])
    X = X.T
    y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
    nb = BernoulliNB(alpha=1.0)
    nb.fit(X,y)
    print nb.predict(np.array([2,4]))#输出-1

