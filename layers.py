import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.LeakyRelu = nn.LeakyReLU(self.alpha)

    def _prepare_attentional_mechanism_input(self, Wh):
        """ 我们的目标是得到一个N * N* 2 * self.out_features矩阵，因为我们要得到每一组 Whi||Whj
            那么目标就是想办法得到一个3维张量(i,j,c)用前2维(i,j)来确定一组关系，用c来表示 Whi和Whj在dim=1 concat的特征
            这样第i行 的(0~N-1)列，就代表了hi和(h0~hN-1)的 Whi||Whj
        """
        N = Wh.size()[0]  # 节点数量
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        """  原本是140 * 1433 这样之后每个行向量都重复了140次，整个矩阵变成了(N*N，1433),N就是重复次数，dim就是在哪个维度上重复
             # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
             # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        """
        Wh_repeated_alternating = Wh.repeat(N, 1)
        """ 在这里wh.repeat(140,1)是先在列上复制一倍，再在行上复制140倍，等于说是把140个Wh在行维度做concat
            # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
            # '----------------------------------------------------' -> N times
        """

        """最后我们让2个矩阵在dim=1上做concat，这样就得到了(N*N，2*out_features)的矩阵，最后view(reshape)就达到了我们的目的
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN"""
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)
        return all_combinations_matrix.view(N, N, 2*self.out_features)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)  # (N，N, 2 * out_features)
        e = self.LeakyRelu(torch.matmul(a_input, self.a).squeeze(2))  # (N,N,1) -> (N,N)

        zero_vec = -9e15*torch.ones_like(e)  # 非常小的数字，softmax之后就是0，这里不能用0，因为e^0=1
        """np.where(condition,x,y) 当where内有三个参数时，第一个参数表示条件，当条件成立时where方法返回x，当条件不成立时where返回y"""
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # (N，N) * (N, out_features) -> (N，out_features)
        if self.concat:
            """说明不是最后一层，需要加一个激活函数ELU"""
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'





