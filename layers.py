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


class GATLayer(nn.Module):
    """GAT层 for ppi"""

    # 定义常量
    # 这里边信息是一个[2,N]shape的张量，row=0是源点编号，row=1是入点编号
    src_nodes_dim = 0  # 定义出边点的编号
    trg_nodes_dim = 1  # 定义入边点的编号

    # 后面我们计算features经常用的是[N,Head,features]的三维张量
    nodes_dim = 0  # 定义节点维度
    head_dim = 1  # 定义head维度

    def __init__(self, num_in_features, num_out_features, num_of_heads,
                 concat=True, activation=nn.ELU(), dropout_prob=0.6,
                 add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection

        # W，这里是直接得到了head个W
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # a, 原来每个head的 a 是(1, 2*Fout) 这里是把它拆成2半，同时直接定义了heads个，前面加个1是为了后面好算
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        # 定义bias
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        # 定义短接
        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights
        self.attention_weights = None  # 保存attention数据，用于可视化

        self.init_params()  # 初始化参数

    def init_params(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, data):
        """
        采用N=2000, Fin=50, Fout=64, heads=4 来模拟
        node_features的shape为(2000,50), (N,Fin)
        edge_index的shape为(2,E)
        形如 [[    0,     0,     0,  ..., 44905, 44905, 44905],
             [  372,  1101,   766,  ..., 44608, 44831, 44905]]
        :param data:(node_features, edge_index)
        """
        in_nodes_features, edge_index = data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        """Step 1: Linear Projection + regularization"""
        in_nodes_features = self.dropout(in_nodes_features)  # 输入的features首先进行dropout
        # (N,Fin) -> (N, NH, FOUT)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)

        """Step 2: 他这里是用边来计算attention的，由于你聚合每一个点时，不需要全体图的信息，只需要i的邻接顶点j的信息
        这相当于你用edge的两个端点来确定(i,j)，故使用edge集来计算，有助于减少开销"""

        # (N,NH,FOUT) * (1,NH,FOUT) -> (N,NH,FOUT)首先是点积,再把最后一维相加这样等于是计算节点i在每个head的 dot(a.T,Wi)
        # sum(dim=-1) -> (N,NH)
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # 出节点得分 (E,NH)，入节点得分 (E,NH) 出节点对应的特征，这里由于是无向图，所以包含了全部边集的顶点 (E,NH,FOUT)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target,
                                                                                           nodes_features_proj,
                                                                                           edge_index)
        # (E,NH) 实际上就对应了论文上 leakRelu(a.T * (Wi||Wj)) 这一步
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # 计算 aij，但得到的似乎是aji？ (E,NH,1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim],
                                                              num_of_nodes)
        attentions_per_edge = self.dropout(attentions_per_edge)

        """Step3 邻居聚合"""

        # 每条边乘上权重 (E,NH,FOUT)，这里又有点反直觉，但似乎是对的，此时 是aji * nodei
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # 进行聚合 shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index,
                                                      in_nodes_features, num_of_nodes)

        """Step4 Residual/skip connections,concat,bias"""
        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)

        return out_nodes_features, edge_index

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        """聚合
        :param nodes_features_proj_lifted_weighted: 每条边的得分，此时已经乘上权重了，是 aji * nodei (E,NH,FOUT)
        :param edge_index: 边集 (2,E)
        :param in_nodes_features:输入矩阵 (N,Fin)
        :param num_of_nodes:节点数量
        """
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)  # (N,NH,FOUT)

        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim],
                                                        nodes_features_proj_lifted_weighted)  # (E,NH,FOUT)

        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features  # (N,NH,FOUT)

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """把原本全部顶点的信息对应到边集上，实现减小开支的目的
        :param scores_source: 出节点得分 (N,NH)
        :param scores_target: 入节点得分 (N,NH)
        :param nodes_features_matrix_proj: 节点特征 (N,NH,FOUT)
        :param edge_index: 边集合 (2,E)
        """
        src_nodes_index = edge_index[self.src_nodes_dim]  # 出节点索引
        trg_nodes_index = edge_index[self.trg_nodes_dim]  # 入节点索引

        """在pytorch中 index_select(dim,index)这种方式比普通的[index]更快"""
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)  # 出节点得分 (E,NH)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)  # 入节点得分 (E,NH)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)
        # 出节点对应的特征，这里由于是无向图，所以包含了全部边集的顶点 (E,NH,FOUT)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        :param scores_per_edge: 每条边的得分，(E,NH) 实际上就对应了论文上 leakRelu(a.T * (Wi||Wj)) 这一步
        :param trg_index: 入节点的索引 (E)
        :param num_of_nodes: 顶点数量
        :return:
        """
        scores_per_edge = scores_per_edge - scores_per_edge.max()  # 改善数值稳定性
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # 计算每个节点的邻居(出边)分数的总和，shape(E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index,
                                                                                num_of_nodes)

        # 此时得到的是aji，这是有道理的，因为后面它直接乘的是Wi，这样就得到了aji*Wi，那么只是换了一下顺序，结果是对的
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)  # 防止除0

        return attentions_per_edge.unsqueeze(-1)   # 最后增加一维，(E, NH，1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        """计算每个节点的邻居(出边)分数的总和
        :param exp_scores_per_edge: 每条边的得分(exp过) (E,NH)
        :param trg_index: 入节点的索引 (E)
        :param num_of_nodes: 顶点数量
        """

        # 把trg_index广播成和exp_scores_per_edge一个shape (E)->(E,NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes

        # 建立一个张量(N,NH)
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # 得到了每个i节点邻居(边)的总和 （N,NH)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        """记录一下过程
           tensor.scatter_add_(dim, index, values) dim 用来指定tensor的第几个维度，index, values二者的shape要求一样
           假定dim=0,tensor为3维, index, values也为3维(其实不需要) 则tensor[index[i][j][k]][j][k] = values[i][j][k]
           这里dim=0,neighborhood_sums为2维(N,NH), trg_index_broadcasted shape为(E,NH)，exp_scores_per_edge为每条边的得分，
           exp_scores_per_edge的每条边的得分其实暗含了方向也就是score -> target, 例如第1个值应当为 0 -> s(某个入节点) 的得分，而
           trg_index_broadcasted的第一个值为s，那我如果neighborhood_sums[s] += exp_scores_per_edge的第一个值，直觉上感觉是错的，
           其实不然，由于这是一个无向图，所以我这样做也只是累加的顺序出了问题，因为在后面一定会一个 s -> 0 这条边对应的值，这样也就得到了
           每个i节点邻居(边)的总和
        """

        return neighborhood_sums.index_select(self.nodes_dim, trg_index)  # 返回入节点的邻居总和

    def explicit_broadcast(self, this, other):
        """把this广播成和other一个shape"""

        # 把this的shape增加到和other一样
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        return this.expand_as(other)  # 把一个tensor变成和函数括号内一样形状的tensor

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        """最后的heads聚合部分
        :param attention_coefficients: 每条边的attention aji (E,NH,1)
        :param in_nodes_features: 输入矩阵 (N,FIN)
        :param out_nodes_features: 输出矩阵 (N,NH,FOUT)
        """
        if self.log_attention_weights:
            # 是否存储attention值
            self.attention_weights = attention_coefficients

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                out_nodes_features += in_nodes_features
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads,
                                                                             self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)














