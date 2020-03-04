import numpy as np
from decision_tree import Decision_Tree 


class Random_Forets:
    
    def __init__(self,
                 n_trees,
                 max_depth,
                 classifier=False,
                 Loss='MSE'):
        # 输入参数 树的个数  树的最大深度
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.classifier = classifier
        self.Loss = Loss
        pass
    
    def fit(self,X , Y):
        # 循环训练每一颗决策树
        self.tree=[]
        for s_t in range(self.n_trees):
            X_Sample, Y_Sample = boost_trap(X, Y)
            single_tree = Decision_Tree(max_depth = self.max_depth,
                                        classifier=True,
                                        Loss="Gini"
                                        )
            single_tree.fit(X_Sample, Y_Sample)
            self.tree.append(single_tree)
            pass
        pass
    
    def predict(self, X):
        # 对输入数据进行预测
        tree_pre = np.array([[t._trave(t.root, x) for x in X] for t in self.tree])
        return self._vote(tree_pre)
    
    def _vote(self, prediction):
        #对每棵树的输出计算平均
        if self.classifier:
            # 如果是分类问题则返还最多的种类
            return np.array([np.bincount(x).argmax() for x in prediction.T])
            
        # 如果是回归问题则返还预测平均值
        return np.array([np.mean(x) for x in prediction.T])

    pass
def boost_trap(X, Y):
    #bagging 
    #有放回抽样
    N, M = X.shape
    idxs = np.random.choice(N, N, replace = True)
    return X[idxs], Y[idxs]