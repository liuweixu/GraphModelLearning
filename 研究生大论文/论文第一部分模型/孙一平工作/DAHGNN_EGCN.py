import torchvision.datasets
from torch.nn.parameter import Parameter
from torchvision import transforms
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as u
import math


class EGCN(torch.nn.Module):
    def __init__(self,sigma,num_feature,multi_head=16,device='cpu'):
        # 初始化权重矩阵
        super().__init__()
        self.device = device  # 在什么设备上跑
        self.GRCU_layers = []
        # self._parameters = nn.ParameterList()  # 可学习参数
        self.net_SingleHGCN = SingleHGCN(num_feature=num_feature)
        self.Linear = nn.Linear(8 * 4, 256, device='cuda:0')
        self.Elu = nn.ELU()

        for i in range(1, 3):  # 多少层，就多少次循环
            multi_head = multi_head/4
            grcu_i = GRCU(multi_head=multi_head,sigma=sigma)
            # print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i.to(self.device))  # 将每一层的RNN存起来备用
            # self._parameters.extend(list(self.GRCU_layers[-1].parameters()))  # 将最新一层的参数加进来
        self.GRCU_layers = self.GRCU_layers
        for i, GRCU_layer in enumerate(self.GRCU_layers):
            self.add_module('attention_{}'.format(i), GRCU_layer)

    def parameters(self):
        return self._parameters

    def forward(self,Nodes_list):
        Nodes_list = Nodes_list.reshape(3, -1, Nodes_list.shape[1], Nodes_list.shape[2], Nodes_list.shape[3])
        node_feats = Nodes_list[-1]
        X_list,E_list,H_list,dist_list,radius_list= [],[],[],[],[]
        for node_data in Nodes_list:
            X, E, H, dist, radius = self.net_SingleHGCN(node_data)  # 单层超图卷积网络
            X_list.append(X)
            E_list.append(E)
            H_list.append(H)
            dist_list.append(dist)
            radius_list.append(radius)
        for i,unit in enumerate(self.GRCU_layers):
            # unit就是GRCU
            X_list = unit(H_list, X_list,E_list,dist_list,radius_list)  # ,nodes_mask_list)
            if i == 0:
                X_list = torch.concat(X_list, dim=0)
                X_list = self.Linear(X_list)
                X_list = self.Elu(X_list)
                X_list = [X_list[:400],X_list[400:800],X_list[800:1200]]
        out = torch.concat(X_list,dim=0)
        return out

class GRCU(torch.nn.Module):
    #这里就是竖直方向传递
    def __init__(self,sigma,multi_head):
        super().__init__()
        self.multi_head = multi_head

        #多头注意力机制
        self.attentions = [DA_HGAN(in_features=256, sigma=sigma, multi_head=multi_head) for _ in
                           range(int(multi_head))]  # 多头注意力
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, H_list, X_list,E_list,dist_list,radius_list):  # ,mask_list):
        # GCN_weights = self.GCN_init_weights.to("cuda:0") #这个self.GCN_init_weights是a_x,和a_e

        out_seq = []
        for t, Ahat in enumerate(H_list):
            node_embs = X_list[t]#获得了Xi和Ek矩阵，准备扔进DAHGNN
            edge_embs = E_list[t]
            dist = dist_list[t]
            radius = radius_list[t]
            # first evolve the weights from the initial and use the new weights with the node_embs
            #将需要更新的alpha系列参数拼在一起，这样只需要一个RNN
            GCN_weights_x = torch.concat([att.alpha_x for att in self.attentions],dim=1)
            GCN_weights_e = torch.concat([att.alpha_e for att in self.attentions],dim=1)
            GCN_weights = torch.concat([GCN_weights_x,GCN_weights_e],dim=1)
            GCN_weights = GCN_weights.to('cuda:0')
            self.evolve_weights = mat_GRU_cell(GCN_weights)  # cell_args(162,100)
            GCN_weights = self.evolve_weights(GCN_weights)  # ,node_embs,mask_list[t]),这一步是获得RNN更新后的参数矩阵
            for i,att in enumerate(self.attentions):
                att.alpha_x = Parameter(GCN_weights[:,i].reshape(16,-1))
                if self.multi_head == 1:
                    att.alpha_e = Parameter(GCN_weights[:,i+1].reshape(16, -1))
                    continue
                att.alpha_e = Parameter(GCN_weights[:,4+i].reshape(16,-1))
            node_embs = torch.cat([att(node_embs, edge_embs, Ahat, dist, radius) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接

            out_seq.append(node_embs)

        return out_seq

class mat_GRU_cell(torch.nn.Module):
    #横向传递
    def __init__(self, GCN_weights):
        super().__init__()
        self.update = mat_GRU_gate(GCN_weights.shape[0],  # 16
                                   GCN_weights.shape[1],  # 8
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(GCN_weights.shape[0],  # 162
                                   GCN_weights.shape[1],
                                  torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(GCN_weights.shape[0],  # 162
                                   GCN_weights.shape[1],
                                   torch.nn.Tanh())

    def forward(self, prev_Q):  # ,prev_Z,mask):
        z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q  # new_Q就是更新好后的参数，也就是我们要的a_x

class mat_GRU_gate(torch.nn.Module):
    def __init__(self, rows, cols, activation):
        '''
        rows 162
        clos 100
        activation:sigmoid
        '''
        super().__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W1 = Parameter(torch.Tensor(rows, rows)).to('cuda:0')  # W.shape 16,16
        self.reset_param(self.W1)

        self.U = Parameter(torch.Tensor(rows, rows)).to('cuda:0')  # U.shape 16,16
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows, cols)).to('cuda:0')  # bias.shape,16,8

    def reset_param(self, t):
        # 统计矩阵的列数，取平方根倒数，然后归一化
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        out = self.activation(self.W1.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out


class DA_HGNN(nn.Module):
    def __init__(self,num_feature,sigma,multi_head):
        super(DA_HGNN, self).__init__()
        self.Linear = nn.Linear(8*4,256,device='cuda:0')
        self.Elu = nn.ELU()
        self.attentions = [DA_HGAN(in_features=256,sigma=sigma,multi_head=multi_head) for _ in
                           range(multi_head)]   #多头注意力
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.net_SingleHGCN = SingleHGCN(num_feature=num_feature)
        self.net_DA_HGAN_2 = DA_HGAN(sigma=sigma,in_features=256)
    def forward(self,X):
        X,E,H,dist,radius = self.net_SingleHGCN(X)  #单层超图卷积网络
        # X,H,w_nodes = zip
        # X, E = self.net_SingleHGCN((X,H))  # 单层超图卷积网络
        X = torch.cat([att(X,E,H,dist,radius) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
        X = self.Linear(X)
        X = self.Elu(X)
        X_tilde = self.net_DA_HGAN_2(X,E,H,dist,radius)
        return X_tilde

class SingleHGCN(nn.Module):
    def __init__(self,num_feature):
        super(SingleHGCN,self).__init__()
        self.theta = Parameter(
            torch.normal(0, 0.01, size=(num_feature,256), requires_grad=True,device=torch.device('cuda:0')))
    def forward(self,X):
        # X,H = zip
        X = X.reshape(X.shape[0],-1 )
        H,dist,radius = self.euclideanDistance(X)
        X,E = self.singleHGCN(X,H)
        return X,E,H,dist,radius

    # 欧式距离计算出超边关系矩阵
    def euclideanDistance(self,X):
        A = torch.sum(X**2,dim=1).reshape(-1,1)
        B = torch.sum(X**2,dim=1).reshape(1,-1)
        C = torch.mm(X,X.T)
        dist_matric = torch.abs(A+B-2*C)
        radius = torch.mean(dist_matric)
        edge = torch.where(dist_matric < radius/5, float(1), float(0))
        x = torch.repeat_interleave(edge, edge.shape[0], dim=1)
        edge = edge.repeat(1, edge.shape[0])
        H = torch.where(torch.sum(edge * x, dim=0).reshape(edge.shape[0], -1) > 0, 1., 0.)
        return H,dist_matric,radius
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
        # E = torch.mm(torch.mm(torch.mm(De, H.T), Dv), X)  # 低维节点特征嵌入X
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
            torch.ones(size=(in_features, int(8)),requires_grad=True, device='cuda:0'))
        self.alpha_x = Parameter(
            torch.zeros(size=(2 * int(8), 1), requires_grad=True, device=torch.device('cuda:0')))
        self.alpha_e = Parameter(
            torch.zeros(size=(2 * int(8), 1), requires_grad=True, device=torch.device('cuda:0')))
        self.sigma = sigma
        self.net_ELU = nn.ELU()
        self.leakyrelu = nn.LeakyReLU(alpha)
        
    def forward(self,X,E,H,dist,radius):
        return self.DA_HGAN(X,E,H,dist,radius)

    def DA_HGAN(self,Xi,Ek,H,dist,radius):
        '''
        :param X:低维节点特征嵌入 shape:nxd
        :param E:超边特征嵌入     shape:mxd
        :sigma 是预定义的阈值 超参数
        :return:

        密度感知超图注意网络主要由两部分组成：密度感知注意顶点聚合和密度感知注意超边聚合。
        '''
        rho_xi = self.node_density(Xi,H,self.sigma)  #节点密度
        # R_CB_Xi, edge = R_CB_Xi_create2(dist, radius)
        # relative_closure_Xi = relative_closure_cal2(R_CB_Xi, edge)
        # relative_closure_s = relative_closure_s_cal2(R_CB_Xi)
        # w_r_bc = w_r_bc_S2(relative_closure_s, relative_closure_Xi, range(edge.shape[0]))
        # rho_xi = w_single_node2(w_r_bc, relative_closure_s)

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
            h_prime = self.net_ELU(torch.matmul(attention.T, torch.mm(X,self.W.to('cuda:0'))))  # [N, N].[N, out_features] => [N, out_features]
        else:
            h_prime = self.net_ELU(torch.matmul(attention.T, X.to('cuda:0')))  # [N, N].[N, out_features] => [N, out_features]
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
        WX = torch.mm(Xi, self.W.to('cuda:0'))  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        WE = torch.mm(Ek, self.W.to('cuda:0'))
        WXN = WX.size()[0]  # number of nodes
        WEN = WE.size()[0]
        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #
        WX_repeated_in_chunks = WX.repeat_interleave(WEN, dim=0)
        # repeat_interleave(self: Tensor, repeats: _int, dim: Optional[_int]=None)
        # 参数说明：
        # self: 传入的数据为tensor
        # repeats: 复制的份数
        # dim: 要复制的维度，可设定为0/1/2.....
        WE_repeated_alternating = WE.repeat(WXN, 1)
        # repeat方法可以对 Wh 张量中的单维度和非单维度进行复制操作，并且会真正的复制数据保存到内存中
        # repeat(N, 1)表示dim=0维度的数据复制N份，dim=1维度的数据保持不变

        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
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
        # eN || eN

        all_combinations_matrix = torch.cat([WX_repeated_in_chunks, WE_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(WXN, WEN, 2 * self.W.shape[1])

def load_data_fashion_mnist(batch_size,resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans =transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data",train=True,transform=trans,download=True)
    mnist_test =torchvision.datasets.FashionMNIST(
        root='../data',train=False,transform=trans,download=True
    )
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=0),
            (data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=0)))
def load_data_mnist(batch_size,resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans =transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(
        root="../data",train=True,transform=trans,download=True)
    mnist_test =torchvision.datasets.MNIST(
        root='../data',train=False,transform=trans,download=True
    )
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=0),
            (data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=0)))
def euclideanDistance(X):
    A = torch.sum(X ** 2, dim=1).reshape(-1, 1)
    B = torch.sum(X ** 2, dim=1).reshape(1, -1)
    C = torch.mm(X, X.T)
    dist_matric = torch.abs(A + B - 2 * C)
    return dist_matric,torch.mean(dist_matric)
def euclideanDistance2(X,Y):
    i,j = X.shape[0],X.shape[1]
    X = X.repeat(1,Y.shape[1]).T
    X = X.reshape(-1,j,i)
    A = torch.sum(X** 2, dim=1).reshape(X.shape[0],-1, 1)
    B = torch.sum(Y ** 2, dim=1).reshape(Y.shape[0],1, -1)
    X = torch.transpose(X, 1, 2)
    C = torch.bmm(X, Y)
    dist_matric = torch.abs(A + B - 2 * C)
    return dist_matric
def R_CB_Xi_create2(dist_matric:torch.Tensor,radius):
    '''
    :param dist_matric:距离矩阵
    :param radius: 画圈的半径
    :return:
    '''
    edge = torch.where(dist_matric < radius, float(1), float(0))
    # edge = torch.tensor([[1.,1,0,0],[1,1,1,1],[0,1,1,1],[0,1,1,1]])
    dist,_ = euclideanDistance(edge)
    dist = dist+torch.tril(torch.ones_like(edge),diagonal=0)
    dist = torch.nonzero(torch.where(dist > 0, 0., 1.))
    edge[dist[:, 1], :] = 0
    R_CB_Xi = torch.ones_like(edge,device='cuda:0')*torch.tensor(range(1,edge.shape[0]+1),device='cuda:0')
    R_CB_Xi = edge*R_CB_Xi
    return R_CB_Xi,edge
def relative_closure_cal2(R_CB_Xi,edge):
    edge = torch.repeat_interleave(edge, edge.shape[0], dim=1)
    R_CB_Xi = R_CB_Xi.repeat(1, edge.shape[0])
    relative_closure_Xi = R_CB_Xi*edge
    return relative_closure_Xi
def relative_closure_s_cal2(relative_closure_Xi):
    relative_closure_S = relative_closure_Xi[torch.nonzero(torch.sum(relative_closure_Xi,dim=1))[:,0]]
    return relative_closure_S
def w_r_bc_S2(relative_closure_s,relative_closure_Xi,S):
    relative_closure_Xi = relative_closure_Xi.T
    relative_closure_Xi = relative_closure_Xi.reshape(-1,relative_closure_Xi.shape[1],relative_closure_Xi.shape[1])
    dist = euclideanDistance2(relative_closure_s,relative_closure_Xi)
    num_zero = (torch.ones_like(torch.count_nonzero(torch.transpose(dist,1,2),dim=1))*len(S)-torch.count_nonzero(torch.transpose(dist,1,2),dim=1)).T
    w_r_bc = 1/len(S)*torch.sum(num_zero/torch.sum(num_zero,dim=0),dim=1)
    return w_r_bc
def w_single_node2(w_r_bc,relative_closure_S):
    relative_closure_S = torch.where(relative_closure_S>0,1,0)
    w_node = torch.sum(w_r_bc.reshape(relative_closure_S.shape[0],-1)*relative_closure_S,dim=0)
    return w_node

def hiper_edge_create(dist_matric:torch.Tensor,radius):
    edge = torch.where(dist_matric < radius, float(1), float(0))
    x = torch.repeat_interleave(edge,4,dim=1)
    edge = edge.repeat(1,4)
    H = torch.where(torch.sum(edge*x,dim=0).reshape(4,-1)>0,1.,0.)
    return H

batch_size = 1200   #一个batch中包含3个小batch，一个小batch 400
train_iter,test_iter = load_data_mnist(batch_size=batch_size)
args = u.Namespace({'feats_per_node':256,
                    'layer_1_feats':256,
                    'layer_2_feats':256})
net =EGCN(num_feature=784,sigma=0.4)
for i,(X,y) in enumerate(train_iter):
    X = X.to("cuda:0")
    y_h = net(X)
    break


