import ESOINN
from DAHGNN import load_data_fashion_mnist
from DAHGNN_primary import load_data_mnist
from DAHGNN_EGCN import EGCN
from torch import nn
import torch
from d2l import torch as d2l
import time

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
        if X.shape==torch.Size([400, 1, 28, 28]):
            continue
        y = y.to(device)

        metric.add(d2l.accuracy(net(X),y),y.numel())    #numel()返回元素个数
    return metric[0]/metric[1]

def train(net,train_iter,test_iter,num_epochs,lr,device):
    def init_weights(m):
        if  type(m) == EGCN:
            for parameter,(name,params) in zip(m.parameters(),net.named_parameters()):
                # if parameter.shape == (256,) or name == "0.attention_0.W" or name == "0.attention_1.W" or name=="0.attention_2.W"\
                #         or name=="0.attention_3.W" or name=="0.net_DA_HGAN_2.W":
                if parameter.shape == (256,):
                    continue
                nn.init.xavier_uniform_(parameter)#xavier参数初始化，使每一层的参数均值和方差都保持不变
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights) #对net里的每一个Parameter都run一下init_weights函数
    print('training on ',device)
    # net.to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        print("=============更新之前===========")
        temp = 0  # 控制打印的参数个数
        for name, parms in net.named_parameters():
            print('-->name:', name)
            # print('-->para:', parms)
            print('-->grad_requirs:', parms.requires_grad)
            print('-->grad_value:', parms.grad)
            print("===")
        time1 = time.time()
        metric = d2l.Accumulator(3)
        net.train()
        if epoch % 100 == 0:
            lr/=2
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        for i,(X,y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to('cuda:0'), y.to('cuda:0')
            X, y = X.to('cuda:0'), y.to('cuda:0')

            y_h = net(X)
            l = loss(y_h,y)
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
        # print("=============更新之后===========")
        # for name, parms in net.named_parameters():
        #     print('-->name:', name)。
        #     # print('-->para:', parms)
        #     print('-->grad_requirs:', parms.requires_grad)
        #     print('-->grad_value:', parms.grad)
        #     print("===")
        # print(optimizer)

batch_size = 1200   #一个batch中包含3个小batch，一个小batch 400

train_iter,test_iter = load_data_mnist(batch_size=batch_size)
net =nn.Sequential(EGCN(num_feature=784,sigma=0.4),
                   nn.Linear(8,10),
                   nn.Softmax(dim=1))
net.to(device=torch.device('cuda:0'))

train(net,train_iter,test_iter,num_epochs=2000,lr=0.002,device='cuda:0')





