#coding:utf8
import numpy as np
from sklearn.model_selection import KFold
class StackEns(object):
    '''
    stack模型融合
    参考http://blog.csdn.net/wtq1993/article/details/51418958
    '''
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        #用来生成分割数据的索引
        folds = KFold(self.n_folds, shuffle=True, random_state=2016)
        #S_train  = 训练样本数 * 模型个数
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        #S_test = 测试样本数 * 模型个数
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            #S_test_i = 测试样本数 * 分块数
            S_test_i = np.zeros((T.shape[0], folds.get_n_splits(X)))

            for j, (train_idx, test_idx) in enumerate(folds.split(X)):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                clf.fit(X_train,y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

            S_test[:, i] = S_test_i.mean(1)

        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred

    def fit_predict_proba(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        #用来生成分割数据的索引
        folds = KFold(self.n_folds, shuffle=True, random_state=2016)
        #S_train  = 训练样本数 * 模型个数
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        #S_test = 测试样本数 * 模型个数
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            #S_test_i = 测试样本数 * 分块数
            S_test_i = np.zeros((T.shape[0], folds.get_n_splits(X)))

            for j, (train_idx, test_idx) in enumerate(folds.split(X)):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                clf.fit(X_train,y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

            S_test[:, i] = S_test_i.mean(1)

        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict_proba(S_test)[:]
        return y_pred