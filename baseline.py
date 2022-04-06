import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model.base_model import Net
from utils.dataloder import MNIST_CLR
from tqdm import tqdm
import random
import argparse
from utils.metric import *
import warnings
from utils.log_file import Logger
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now()) 
#warnings.filterwarnings('ignore')
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter('runs/baseline/'+TIMESTAMP)


if not os.path.exists('log/baseline'):
    os.makedirs('log/baseline')
log_file_name='log/baseline/'+TIMESTAMP[:-1]+'.log'
    
log=Logger(log_file_name, level='info')
log.logger.info('TIME: %s' , TIMESTAMP[:-1])

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [bs, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [bs, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def model_train(model, train_dataloader, optimizer,criterion, epoch, left_noise_level, right_noise_level, device):
    model.train()
    with tqdm(total=len(train_dataloader), desc="Batch") as pbar:
        losses = 0
        count = 0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data

            input_left = inputs[:,:,:14,:]
            input_right = inputs[:,:,14:,:]
            # 将数据迁移到device中，如device为GPU，则将数据从CPU迁移到GPU；如device为CPU，则将数据从CPU迁移到CPU（即不作移动）
            input_left, input_right, labels = input_left.to(device), input_right.to(device), labels.to(device)
            # 清空参数的梯度值
            optimizer.zero_grad()

            # 前馈+反馈+优化
            _,_,outputs = model(input_left, input_right)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses += loss.detach().cpu().numpy()
            count += 1

            pbar.set_postfix({'loss' : '{0:1.5f}'.format(loss.detach().cpu().numpy())})
            pbar.update(1)

    return

def model_val(model, test_dataloader, epochs, left_noise_level, right_noise_level, device):
    model.eval()

    num_cls=10
    
    list_label=[]
    list_pred=[]
    num_label=[]
    softmax = nn.Softmax(dim=1)
    correct = 0
    total = 0

    with torch.no_grad():
        for i,data in enumerate(test_dataloader, 0):
            inputs, labels = data
            input_left = inputs[:,:,:14,:]
            input_right = inputs[:,:,14:,:]
            input_left, input_right, labels = input_left.to(device), input_right.to(device), labels.to(device)
            left_out,right_out,outputs = model(input_left, input_right)
            pre_left = torch.mm(left_out,torch.transpose(model.fc.weight[:,:50],0,1)) + model.fc.bias/2
            pre_right = torch.mm(right_out,torch.transpose(model.fc.weight[:,50:],0,1)) + model.fc.bias/2
            prediction=softmax(outputs)

            ### 存储 one-hot label/pred
            alabel=labels.unsqueeze(1)                                    
            onehot=make_one_hot(alabel,num_cls) 

            for item in onehot:
                list_label.append(item.cpu().data.numpy())

            for item in prediction.cpu():
                list_pred.append(item.cpu().data.numpy())

            _, predicted = torch.max(outputs.data, 1)

            # 统计样本个数
            total += labels.size(0)
            # 统计正确预测样本个数
            correct += (predicted == labels).sum().item()

    pred_array=np.array(list_pred)
    gt_array=np.array(list_label)

    stats = calculate_stats(pred_array,gt_array)
    mAP = np.mean([stat['AP'] for stat in stats])
    acc= correct / total


    return mAP,acc


def model_test(model, test_dataloader, left_noise_level, right_noise_level,device):
    model.eval()
    # CIFAR10 类别内容
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    class_correct = list(0. for i in range(10))
    class_correct_left=list(0. for i in range(10))
    class_correct_right=list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for i,data in enumerate(test_dataloader, 0):
            inputs, labels = data
            input_left = inputs[:,:,:14,:]
            input_right = inputs[:,:,14:,:]
            input_left, input_right, labels = input_left.to(device), input_right.to(device), labels.to(device)
            # 前馈获得网络预测结果
            left_out,right_out,outputs = model(input_left, input_right)
            pre_left = torch.mm(left_out,torch.transpose(model.fc.weight[:,:50],0,1)) + model.fc.bias/2
            pre_right = torch.mm(right_out,torch.transpose(model.fc.weight[:,50:],0,1)) + model.fc.bias/2
            # 在预测结果中，获得预测值最大化的类别id
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_left = torch.max(pre_left.data, 1)
            _, predicted_right = torch.max(pre_right.data, 1)

            # 对于batch内每一个样本，获取其预测是否正确
            c = (predicted == labels).squeeze()
            c_left = (predicted_left == labels).squeeze()
            c_right = (predicted_right == labels).squeeze()
            # 对于batch内样本，统计每个类别的预测正确率
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_correct_left[label] += c_left[i].item()
                class_correct_right[label] += c_right[i].item()
                class_total[label] += 1

    for i in range(10):
        log.logger.info('Accuracy of %2s : %.3f %%, left Acc: %.3f, right Acc: %.3f' % (classes[i], 100 * class_correct[i] / class_total[i],100 * class_correct_left[i] / class_total[i],100 * class_correct_right[i] / class_total[i]) )

    return

def main():
    parser = argparse.ArgumentParser(description='MNIST_Classifier')
    parser.add_argument('--use_pretrain', type=int, default=0, help='whether to init from ckpt')
    parser.add_argument('--ckpt_file', type=str, default='location_cluster_net_norm_006_0.680.pth', help='pretrained model name')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='training batch size')
    parser.add_argument('--epoch', type=int, default=1, help='training epoch')
    parser.add_argument('--gpu_ids', type=str, default=None, help='USING GPU IDS e.g.\'[0,4]\'')
    parser.add_argument('--num_threads', type=int, default=12, help='number of threads')
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()


    for arg in vars(args):
        log.logger.info('{:<16} : {}'.format(arg, getattr(args, arg)))

    setup_seed(args.seed)

    # 根据环境参数选取对应运行平台：cpu or gpu
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    log.logger.info('Running Device: %s', device)

    if(device!='cpu'):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # 对获取的图像数据做ToTensor()变换和归一化
    transform = transforms.Compose([transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    # 利用torchvision提供的CIFAR10数据集类，实例化训练集和测试集提取类
    trainset = MNIST_CLR(data_root='/Users/ykw/lab/project/game/MNIST/data/MNIST/raw', data_list_file=None, args=args, train=True,transforms=transform,target_transform=None)
    testset = MNIST_CLR(data_root='/Users/ykw/lab/project/game/MNIST/data/MNIST/raw', data_list_file=None, args=args, train=False,transforms=transform,target_transform=None)

    # 利用torch提供的DataLoader, 实例化训练集DataLoader 和 测试集DataLoader
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    

    # 卷积神经网络实例化
    net = Net().to(device)
    log.logger.info(net)

    # 实例化损失函数和SGD优化子
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    best_acc=-1
    best_map=-1
    path='ckpt/baseline'
    for e in range(args.epoch):
        log.logger.info('Epoch is %03d' % e)
        model_train(net, trainLoader, optimizer, criterion, e,None,None,device)
        acc,mAP=model_val(net, testLoader,e,None,None,device)
        if(acc>best_acc):
            best_acc=acc
            best_map=mAP
            best_model=net
            log.logger.info('best acc: %.3f, mAP: %.3f' % (best_acc,best_map))
        else:
            log.logger.info('No change! best acc: %.3f, mAP: %.3f' % (best_acc,best_map))
            if not os.path.exists(path):
                os.makedirs(path)
            PATH = 'ckpt/baseline/baseline.pth'
            torch.save(best_model.state_dict(), PATH)
    
    model_test(best_model, testLoader, None,None,device)



if __name__ == '__main__':
    main()