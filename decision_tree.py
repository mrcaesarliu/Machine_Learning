import numpy as np

class Node:
    # 定义树的节点 类
    def __init__(self,left,right,feature,threholds):
        self.left = left
        self.right = right
        self.feature = feature
        self.threholds = threholds
        
        pass
    pass

class Leaf:
    # 定义树的叶类
    def __init__(self,value):
        self.value = value
        pass
    pass


class Decision_Tree:
    
    def __init__(self,max_depth,
                 classifier=False,
                 Loss = "MSE"
                 ):
        #树的深度
        #分类还是回归
        #损失函数类型
        self.max_depth = max_depth
        self.classifier = classifier
        self.Loss = Loss

    
    def fit(self, X, Y):
        if self.classifier:
            self.n_class = max(Y)+1
            pass
        self.n_feats = X.shape[1]
        self.root = self._grow(X, Y)
    
    def _grow(self, X, Y, cur_depth = 0):
        '''
        if len(set(Y) == 1):
            return Leaf(set(Y))
        '''
        #树的递归分裂
        s_Y = set(Y)
        # 如果节点一侧只存在一个样本则返回样本值
        if len(s_Y) == 1:
            if self.classifier:
                P = np.zeros(self.n_class)
                P[Y[0]] = 1
                return Leaf(P)
            return Leaf(Y[0])
        
        # 如果深度到了设定深度停止分裂，返还叶节点
        if cur_depth>= self.max_depth:
            v = np.mean(Y)
            if self.classifier:
                v = np.bincount(Y, minlength = self.n_class)/len(Y)
            return Leaf(v)
        # 分裂过程
        cur_depth+= 1
        M, N = X.shape
        # 打乱特征值顺序
        feat_idxs = np.random.choice(N, self.n_feats, replace=False)
        # 计算当前深度节点的特征值 阈值       
        feat, thresh = self._segment(feat_idxs, X, Y, cur_depth)
        #区分节点左边和右边的数据
        left_id = np.argwhere(X[:, feat]<= thresh).flatten()
        right_id = np.argwhere(X[:, feat]>= thresh).flatten()
        print('depth %s'%(cur_depth))
        print('left %s'%(len(left_id)))
        print('right %s'%(len(right_id)))
        # 递归分裂树
        left = self._grow(X[left_id, :], Y[left_id], cur_depth)
        right = self._grow(X[right_id, :], Y[right_id], cur_depth)
        
        
        return Node(left,right,feat,thresh)
    
    def _segment(self, feat_idxs, X, Y, cur_depth):
        #初始化最优gain
        best_gain = -np.inf
        split_idx = None
        split_thresh = None
        
        # 遍历每一个feature
        for i in feat_idxs:
            vals = X[:, i]
            levels = np.unique(vals)
            thresh = (levels[:-1]+levels[1:])/2
            # 在一个feature 中计算每个样本feature作为 阈值的gain
            gain = np.array([self._impurity(Y, t, vals) for t in thresh])
            if  any(gain):
                #求出熵增最大的feature 和阈值
                if gain.max()>best_gain:
                    split_idx = i
                    best_gain = gain.max()
                    split_thresh = thresh[gain.argmax()]
                    # if 循环结束
                    pass
                pass
            # for 循环结束
            pass
        # 返回计算得到的特征值下标 和特征值阈值
        return split_idx, split_thresh
    
    def _impurity(self, Y, thresh, vals):
        # 计算当前 thresh 下 vals特征下面的损失函数
        # 计算样本 及输入阈值的熵增
        # 当前时刻的loss
        if self.Loss == "MSE" :
            parent_loss = mse(Y)
        if self.Loss == "Gini" :
            parent_loss = Gini(Y)
        
        left_vals = np.argwhere(vals<=thresh).flatten()
        right_vals = np.argwhere(vals>=thresh).flatten()
        
        n = len(Y)
        left_n, right_n =len(left_vals),len(right_vals)
        err_l = mse(Y[left_vals])
        err_r = mse(Y[right_vals])
        
        child_loss = (left_n/n)*err_l + (right_n/n)*err_r
        
        ig = parent_loss - child_loss
        
        return ig
    
    def predict(self, X):
        
        
        return np.array([self._trave(self.root, x) for x in X])
    
    def _trave(self, node, X):
        #预测函数
        #递归遍历每一个节点 
        
        if isinstance(node,Leaf):
            if self.classifier:
                return node.value.argmax()
            return node.value
            pass
        if X[node.feature] <= node.threholds:
            return self._trave(node.left, X)
        return self._trave(node.right, X)
        pass
    pass

# Decision Tree 常用的几种损失函数
def mse(vals):
    # 残差平方和损失函数
    return np.mean( (vals - np.mean(vals) )**2 )


def Gini(vals):
    # 1-sum( ( Ni/N )^2 )
    N_i = np.bincount(vals)
    N = np.sum(N_i)
    
    return 1 - sum( [(i / N)**2 for i in N_i] )