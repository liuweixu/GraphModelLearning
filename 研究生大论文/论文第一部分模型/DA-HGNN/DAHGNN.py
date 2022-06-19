
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sparse_to_adjlist
import os
from scipy.io import loadmat

'''
    增加EvolveGCN部分, 使用的是LSTM方法
'''
class EGCN(torch.nn.Module):
    def __init__(self, args, activation, device='cuda:0'):
        super(EGCN, self).__init__()

        self.device = device
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        for i in range(1,2): # 这部分算是垂直上下的, 因为DAHGNN只有一层图神经网络层，所以只有一个RNN层

            grcu_i = GRCU()
            #print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self,A_list, X_list):

        for unit in self.GRCU_layers: #说明只有一个RNN层 
            X_list = unit(A_list,X_list)

        out = X_list[-1] #输出最后的node embedding结果
        return out

'''
    LSTM, 这部分对应的是每一个RNN传递情况，算是从左到右的
'''
class GRCU(torch.nn.Module):
    def __init__(self):
        super(GRCU, self).__init__()
        self.evolve_weights = mat_GRU_cell(cell_args)
        self.DAHGNN_init_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.reset_param(self.GCN_init_weights)
        self.dahgnnModel = DA_HGNN(num_feature=784,sigma=0.4,multi_head=4)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,A_list,X_list):
        DAHGNN_weights = self.DAHGNN_init_weights
        out_seq = []
        for t,Ahat in enumerate(A_list): #t应该指的是时间段
            node_embs = X_list[t]
            #first evolve the weights from the initial and use the new weights with the node_embs
            DAHGNN_weights = self.evolve_weights(DAHGNN_weights)#,node_embs,mask_list[t])
            node_embs = self.dahgnnModel(node_embs, Ahat) # 这部分要喂入节点特征矩阵X和同构图矩阵A
            out_seq.append(node_embs)

        return out_seq
'''
    LSTM块
'''
class mat_GRU_cell(torch.nn.Module):
    def __init__(self,DAHGNN_weights):
        super(mat_GRU_cell, self).__init__()
        self.update = mat_GRU_gate(DAHGNN_weights.shape[0],
                                   DAHGNN_weights.shape[1],
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(DAHGNN_weights.shape[0],
                                   DAHGNN_weights.shape[1],
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(DAHGNN_weights.shape[0],
                                   DAHGNN_weights.shape[1],
                                   torch.nn.Tanh())

    def forward(self,prev_Q):
        z_topk = prev_Q

        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q #更新参数

'''
    LSTM门
'''
class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super(mat_GRU_gate, self).__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out
'''
    总部分
    不过里面的top_k似乎没啥用，就留着
'''
class DA_HGNN(nn.Module):
    def __init__(self, num_feature, sigma, multi_head):
        super(DA_HGNN, self).__init__()
        self.Linear = nn.Linear(8*4,256,device='cuda:0') # 32 256
        self.Elu = nn.ELU()
        self.attentions = [DA_HGAN(in_features=256,sigma=sigma,multi_head=multi_head) for _ in
                           range(multi_head)]   #多头注意力
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.net_SingleHGCN = SingleHGCN(num_feature=num_feature)
        self.net_DA_HGAN_2 = DA_HGAN(sigma=sigma,in_features=256)
    def forward(self, X, A): # 喂入节点特征矩阵和同构图矩阵
        X,E,H = self.net_SingleHGCN(X, A)  #单层超图卷积网络
        # X,H,w_nodes = zip
        # X, E = self.net_SingleHGCN((X,H))  # 单层超图卷积网络
        X = torch.cat([att(X,E,H) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
        X = self.Linear(X) # output [, 256]
        X = self.Elu(X)
        X_tilde = self.net_DA_HGAN_2(X,E,H)
        return X_tilde

class SingleHGCN(nn.Module):
    def __init__(self, num_feature):
        super(SingleHGCN,self).__init__()
        self.theta = Parameter(
            torch.normal(0, 0.01, size=(num_feature,256), requires_grad=True,device=torch.device('cuda:0')))
        self.top_k = top_k+1
    def forward(self, X, A):
        '''
        :param A:同构图矩阵, 主要用于创建超边
        :param X:节点特征矩阵, 要把文本特征纳入考虑预计的shape为(100,2+70), 类比EGCN论文的H矩阵
        :param H:关系矩阵, 类比GCN的邻接矩阵
        '''
        # X,H = zip
        # X = X.reshape(A.shape[0], -1)  不需要reshape
        H = self.hyper_create(A) # 构建超边
        X,E = self.singleHGCN(X,H)
        return X,E,H
    #超边构建，A是同构图
    def hyper_create(self, A):
        adj_lists = sparse_to_adjlist(A)
        H = torch.empty(len(adj_lists), len(adj_lists))
        for num in range(len(adj_lists)):
            H[num, list(adj_lists[num])] = 1
        return H

    #单层卷积操作
    def singleHGCN(self,X,H):
        '''
        单层超图卷积网络，用于学习节点特征和超边特征的低纬嵌入
        :param X:初始节点的特征矩阵
        :param H:超图关联矩阵
        :param De:超边度的对角矩阵
        :param Dv:顶点度的对角矩阵
        实验过程中默认节点数n等于超边数m
        :return:
        '''
        Dv = torch.diag(torch.pow(torch.sum(H, dim=1),-1 / 2))
        De = torch.diag(torch.pow(torch.sum(H, dim=0), -1 / 2))
        # X = torch.mm(torch.mm(torch.mm(torch.mm(De,H.T),Dv),X),self.theta)   #低维节点特征嵌入X
        # E = torch.mm(torch.mm(torch.mm(Dv,H),De),X)           #超边特征嵌入E
        E = torch.mm(torch.mm(torch.mm(torch.mm(De, H.T), Dv), X), self.theta)  # 低维节点特征嵌入X
        X = torch.mm(torch.mm(torch.mm(Dv, H), De), E)  # 超边特征嵌入E
        return X,E

class DA_HGAN(nn.Module):
    def __init__(self,sigma,in_features,alpha=0.2,multi_head=1):
        '''
        :param sigma: 相似度的阈值
        :param alpha: LeakyRelu的参数，默认0.2
        '''
        super(DA_HGAN, self).__init__()
        self.W = Parameter(
            torch.zeros(size=(in_features, int(8)),requires_grad=True, device=torch.device('cuda:0')))
        self.alpha_x = Parameter(
            torch.zeros(size=(2 * int(8), 1), requires_grad=True, device=torch.device('cuda:0')))
        self.alpha_e = Parameter(
            torch.zeros(size=(2 * int(8), 1), requires_grad=True, device=torch.device('cuda:0')))
        self.sigma = sigma
        self.net_ELU = nn.ELU()
        self.leakyrelu = nn.LeakyReLU(alpha)
        
    def forward(self,X,E,H):
        return self.DA_HGAN(X,E,H)

    def DA_HGAN(self,Xi,Ek,H):
        '''
        :param X:低维节点特征嵌入 shape:nxd
        :param E:超边特征嵌入     shape:mxd
        :sigma 是预定义的阈值 超参数
        :return:

        密度感知超图注意网络主要由两部分组成：密度感知注意顶点聚合和密度感知注意超边聚合。
        '''
        rho_xi = self.node_density(Xi,H,self.sigma)  #节点密度
        rho_hyper_edge = self.hyper_edge_density(rho_xi,H)  #超边密度
        E = self.attention(Xi,Ek,rho_xi,node=True,H=H,X=Xi)    #节点的注意力值，node为true表示计算的是节点注意力值
        X = self.attention(Ek,Xi,rho_hyper_edge,node=False,H=H,X=E)   #超边的注意力值

        return X

    #通过余弦相似度计算密度
    def node_density(self,X,H,sigma:float):
        neiji = torch.mm(X, X.T)    #内积
        mochang = torch.sqrt(torch.sum(X * X, dim=1).reshape(1, -1) * (torch.sum(X * X, dim=1).reshape(-1, 1))) #模长
        cosim = neiji/mochang   #余弦相似度矩阵

        #矩阵元素小于sigma的全部置为0，对角线元素也置0，因为不需要自己和自己的相似度
        cosim = torch.where(cosim>sigma,cosim,torch.zeros_like(cosim))\
                *(torch.ones_like(cosim,device='cuda:0')-torch.eye(cosim.shape[0],device='cuda:0'))
        #节点和超边的关系矩阵H的内积的每一行，可以表示每个节点的所有相邻节点
        xx = torch.where(torch.mm(H,H.T) > 0, float(1), float(0)) \
             * (torch.ones_like(cosim,device='cuda:0') - torch.eye(cosim.shape[0],device='cuda:0'))
        #将每个节点与相邻节点的相似度相加，就是该节点的密度
        rho = torch.sum(cosim*xx,dim=1).reshape(-1,1)
        return rho
    #计算节点与边的注意力值,然后计算标准化密度，最后得出注意力权重
    def attention(self,Xi:torch.Tensor,Ek:torch.Tensor,rho:torch.Tensor,node:bool,H,X):
        '''
        :param Xi:节点嵌入矩阵
        :param Ek: 超边嵌入矩阵
        :param rho:节点密度
        :return:注意力权重
        '''
        # 将WX和WE拼接

        a_input = self._prepare_attentional_mechanism_input(
            Xi, Ek,node)  # 实现论文中的特征拼接操作 Wh_i||Wh_j ，得到一个shape = (N ， N, 2 * out_features)的新特征矩阵
        if node:
            a_x = self.leakyrelu(torch.matmul(a_input, self.alpha_x).squeeze(2))
        else:
            a_x = self.leakyrelu(torch.matmul(a_input,self.alpha_e).squeeze(2))
        rho_tilde =torch.tensor([(enum-torch.min(rho))/(torch.max(rho)-torch.min(rho))*torch.max(a_x) for enum in rho],
                                device=torch.device('cuda:0')).reshape(-1,1)
        a_x_tilde = a_x+rho_tilde
        zero_vec = -1e12 * torch.ones_like(a_x_tilde)  # 将没有连接的边置为负无穷
        if node:
            attention = torch.where(H > 0, a_x_tilde, zero_vec)  # [N, N]   #node为true，计算coe_x_e，a_x_tilde代表的是a_x_e,是节点-超边的形状
        else:
            attention = torch.where(H.T > 0, a_x_tilde, zero_vec)           ##node为false，计算coe_e_x，a_x_tilde其实是a_e_x,是超边-节点的形状
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        # attention = F.softmax(attention, dim=0)
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, 0.2, training=self.training)  # dropout，防止过拟合
        if node:
            h_prime = self.net_ELU(torch.matmul(attention.T, torch.mm(X,self.W)))  # [N, N].[N, out_features] => [N, out_features]
        else:
            h_prime = self.net_ELU(torch.matmul(attention.T, X))  # [N, N].[N, out_features] => [N, out_features]
        return h_prime

    def hyper_edge_density(self,rho_x:torch.Tensor,H:torch.Tensor):
        '''
        计算超边密度
        :param rho_x:  节点密度
        :param H:       关系矩阵
        :return:
        '''
        rho_hyper_edge = torch.sum(H*rho_x,dim=0)
        return rho_hyper_edge.reshape(-1,1)

    def _prepare_attentional_mechanism_input(self, Xi,Ek,node:bool):

        WX = torch.mm(Xi, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        WE = torch.mm(Ek, self.W)
        WXN = WX.size()[0]  # number of nodes
        WEN = WE.size()[0]
        WX_repeated_in_chunks = WX.repeat_interleave(WEN, dim=0)
        # repeat_interleave(self: Tensor, repeats: _int, dim: Optional[_int]=None)
        # 参数说明：
        # self: 传入的数据为tensor
        # repeats: 复制的份数
        # dim: 要复制的维度，可设定为0/1/2.....
        WE_repeated_alternating = WE.repeat(WXN, 1)

        all_combinations_matrix = torch.cat([WX_repeated_in_chunks, WE_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(WXN, WEN, 2 * self.W.shape[1])

'''
    下面是读取函数
'''
# 用于读取训练数据和测试数据的所有路径
def load_data_fraud():
    train_datasets = os.listdir('train_folder')
    test_datasets = os.listdir('test_folder')
    return train_datasets, test_datasets

# 读取每一个路径对应的mat文件
def load_data_mat(matPath):
    dataMat = loadmat(matPath)
    A_RUR = torch.from_numpy(dataMat['rur'].todense())
    A_RPR = torch.from_numpy(dataMat['rpr'].todense())
    label = torch.from_numpy(np.array(dataMat['label']))
    feature1 = torch.from_numpy(dataMat['features'].todense())
    feature2 = torch.from_numpy(dataMat['text'])
    SA = Self_Attention(feature2.shape)
    output = SA(feature2)
    feature_result = torch.concat([feature1, output], 1)
    return A_RUR, A_RPR, feature_result, label

'''
    下面是对词向量降维，其权重需要学习
'''
class Self_Attention(nn.Module):
    def __init__(self, input_shape, dropout_rate=0.0):
        super(Self_Attention, self).__init__()
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.k = input_shape[-1]
        self.W_layer = nn.Linear(self.k, self.k)
        self.U_weight = Parameter(
            torch.normal(0, 0.01, size=(input_shape[1], 1), requires_grad=True,device=torch.device('cuda:0')) # [70,1]
        )

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError("The dim of inputs is required 3 but get {}".format(len(inputs.shape)))
    
        x = self.W_layer(inputs) #[100, 70, 768]
        score = torch.matmul(x.transpose(1, 2), self.U_weight)
        score = self.dropout_layer(score) #[100, 768, 1]

        score = F.softmax(score, dim=1) #[100, 768, 1]

        output = torch.matmul(inputs, score) #[100, 70, 1]
        output /= inputs[1]**0.5 #这部分存疑
        output = torch.squeeze(output, dim=-1) #[100, 70]
        
        return output