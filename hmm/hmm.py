#encoding:utf-8
import sys
import pickle
from copy import deepcopy

is_train = False

DEFAULT_PROB = 0.000000000001
MIN_PROB = -1 * float('inf')

train_path = "train.in"
test_path = "test.in"
output_path = "test.out"

#统计 各个次数 作为 各个概率
def train():
	print "start training ..."

	# 以下5个元素是HMM模型的参数
	V = set() # 观测集合
	Q = set() # 状态集合
	A = {} # 状态转移概率矩阵，P(状态|状态)，是一个二层dict 具体是 pre_state->(state->prob)
	B = {} # 观测概率矩阵，P(观测|状态)，是一个二层dict 具体是 state->(observ->prob)
	PI = {} # 初始状态概率向量

	# 统计模型参数
	with open(train_path, "rb") as infile:
		pre_s = -1 # t-1时刻的状态
		for line in infile:
			segs = line.rstrip().split('\t')
			if len(segs) != 2: # 遇到空行时
				pre_s = -1
			else:
				o = segs[0] # t时刻的观测o
				s = segs[1] # t时刻的状态s
				# 统计状态s到观测o的次数
				B[s][o] = B.setdefault(s, {}).setdefault(o, 0) + 1
				V.add(o)
				Q.add(s)
				if pre_s == -1: # 统计每个句子开头第一个状态的次数
					PI[s] = PI.setdefault(s, 0) + 1
				else: # 统计状态pre_s到状态s的次数
					A[pre_s][s] = A.setdefault(pre_s, {}).setdefault(s, 0) + 1
				pre_s = s #切换到下一个状态
	# 概率归一化
	for i in A.keys():
		prob_sum = 0
		for j in A[i].keys():
			prob_sum += A[i][j]
		for j in A[i].keys():
			A[i][j] = 1.0 * A[i][j] / prob_sum

	for i in B.keys():
		prob_sum = 0
		for j in B[i].keys():
			prob_sum += B[i][j]
		for j in B[i].keys():
			B[i][j] = 1.0 * B[i][j] / prob_sum

	prob_sum = sum(PI.values())
	for i in PI.keys():
		PI[i] = 1.0 * PI[i] / prob_sum
	print "finished training ..."

	return A, B, PI, V, Q

def saveModel(A, B, PI, V, Q):
	with open("A.param", "wb") as outfile:
		pickle.dump(A, outfile)
	with open("B.param", "wb") as outfile:
		pickle.dump(B, outfile)
	with open("PI.param", "wb") as outfile:
		pickle.dump(PI, outfile)
	with open("V.param", "wb") as outfile:
		pickle.dump(V, outfile)
	with open("Q.param", "wb") as outfile:
		pickle.dump(Q, outfile)

#维特比
def predict(X, A, B, PI, V, Q):
	W = [{} for t in range(len(X))] #相当于书上的δ
	path = {}
	for s in Q:
		W[0][s] = 1.0 * PI.get(s, DEFAULT_PROB) * B.get(s, {}).get(X[0], DEFAULT_PROB) #0时刻状态为s的概率
		path[s] = [s]
	for t in range(1, len(X)):
		new_path = {}
		for s in Q: #两轮循环暴力求解
			max_prob = MIN_PROB
			max_s = ''
			for pre_s in Q:
				prob = W[t-1][pre_s] * \
					   A.get(pre_s, {}).get(s, DEFAULT_PROB) * \
					   B.get(s, {}).get(X[t], DEFAULT_PROB)
				(max_prob, max_s) = max((max_prob, max_s), (prob, pre_s)) #全由第一个prob决定
			W[t][s] = max_prob #t时刻状态为s的最大概率
			tmp = deepcopy(path[max_s])
			tmp.append(s)
			new_path[s] = tmp
		path = new_path
	(max_prob, max_s) = max((W[len(X)-1][s], s) for s in Q)# 最后一个时刻各个状态的概率的最大的
	return path[max_s]

def getModel():
	with open("A.param", "rb") as infile:
		A = pickle.load(infile)
	with open("B.param", "rb") as infile:
		B = pickle.load(infile)
	with open("PI.param", "rb") as infile:
		PI = pickle.load(infile)
	with open("V.param", "rb") as infile:
		V = pickle.load(infile)
	with open("Q.param", "rb") as infile:
		Q = pickle.load(infile)		
	return A, B, PI, V, Q

def test(A, B, PI, V, Q):
	print "start testing"
	with open(test_path, "rb") as infile, \
		 open(output_path, "wb") as outfile:
		X_test = []
		y_test = []
		for line in infile:
			segs = line.strip().split('\t')
			if len(segs) != 2: # 遇到空行时
				if len(X_test) == 0:#一整句 比如NBAD
					continue
				preds = predict(X_test, A, B, PI, V, Q)
				for vals in zip(X_test, y_test, preds):
					outfile.write("\t".join(vals) + "\n")	
				outfile.write("\n")
				X_test = []
				y_test = []
			else:
				o = segs[0] # t时刻的观测o
				s = segs[1] # t时刻的状态s		
				X_test.append(o)
				y_test.append(s)

	print "finished testing"

def main():
	if is_train:
		A, B, PI, V, Q = train()
		saveModel(A, B, PI, V, Q)
	else:
		A, B, PI, V, Q = getModel()

	test(A, B, PI, V, Q)

if __name__ == '__main__':
	main()
