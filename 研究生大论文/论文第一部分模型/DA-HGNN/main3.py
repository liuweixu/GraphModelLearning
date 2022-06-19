# import ESOINN
import time
from DAHGNN import load_data_fraud, load_data_mat
# from DAHGNN import load_data_mnist
from DAHGNN import EGCN
# from DAHGNN2 import DA_HGNN
# from DAHGNN3 import DA_HGNN
from torch import nn
import torch
from d2l import torch as d2l

train_datasets, test_datasets = load_data_fraud()

def evaluate_accuracy_gpu(net,data_iter,device=None):
    if isinstance(net,nn.Module):
        net.eval()  #使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval，eval（）时，
                    # 框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大！！！！！！
    if not device:
        device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    for X,y in data_iter:
        if isinstance(X,list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X),y),y.numel())    #numel()返回元素个数
        break
    return metric[0]/metric[1]

def train(net,train_datasets,test_iter,num_epochs,lr,device):
    def init_weights(m):
        if  type(m) == 'DA_HGNN':
            for parameter in m.parameters():
                if parameter.shape == (256,):
                    continue
                nn.init.xavier_uniform_(parameter)#xavier参数初始化，使每一层的参数均值和方差都保持不变
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights) #对net里的每一个Parameter都run一下init_weights函数
    print('training on ',device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        time1 = time.time()
        metric = d2l.Accumulator(3) #是[0.0, 0.0, 0.0]
        net.train()
        if epoch % 50 == 0:
            lr/=2
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        # for i,(X,y) in enumerate(train_iter):
        # for train_data in train_datasets:
        for base_num in range(0, len(train_datasets), 10):
            A_RUR_lists = []
            A_RPR_lists = []
            X_lists = []
            label_lists = []
            optimizer.zero_grad()
            for number in range(base_num*10, (base_num+1)*10):
                train_data = '../datasets/matFolder' + train_datasets[number]
                A_RUR, A_RPR, X, label = load_data_mat(train_data) # X是[100, 72]
                A_RUR = A_RUR.to(device)
                A_RPR = A_RPR.to(device)
                X = X.to(device)
                label = label.to(device)
                A_RUR_lists.append(A_RUR)
                A_RPR_lists.append(A_RPR)
                X_lists.append(X)
                label_lists.append(label)
            # y_h = net((X,H))
            y1 = net(A_RUR_lists, X_lists)
            y2 = net(A_RPR_lists, X_lists)
            y_result = (y1 + y2) / 2
            # TODO 应该是直接对y1和y2取平均，然后和label直接计算损失
            l = loss(y_result,label_lists[-1])
            l.backward()
            optimizer.step()    #进行参数优化
            with torch.no_grad():
                metric.add(l*X.shape[0],d2l.accuracy(y_h,y),X.shape[0])
        train_l = metric[0]/metric[2]
        train_acc = metric[1]/metric[2]
        test_acc = evaluate_accuracy_gpu(net,test_iter)
        time2 = time.time()
        time3 = (time2 - time1) / 60
        print(f'loss {train_l:.3f},train acc {train_acc:.3f},test acc {test_acc:.3f}'
          f'on {str(device)},have finished{epoch},spend time {time3:.3f} min')

train_iter,test_iter = load_data_fraud()
'''
    模型部分
    在nn.Linear(8,10)中改为(8,100)，应该是输出100维
'''
net = nn.Sequential(EGCN(), nn.Linear(8,100), nn.Softmax(dim=1))
net.to(device=torch.device('cuda:0'))

train(net,train_datasets,test_datasets,num_epochs=2000,lr=0.002,device='cuda:0')





