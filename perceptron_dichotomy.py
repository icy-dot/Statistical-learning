'''
MNIST 数据集来自美国国家标准与技术研究所, National Institute of Standards and Technology (NIST). 训练集 (training set) 由来自 250 个不同人手写的数字构成, 其中 50% 是高中学生, 50% 来自人口普查局 (the Census Bureau) 的工作人员. 测试集(test set) 也是同样比例的手写数字数据.
————————————————
数据下载链接：https://www.kaggle.com/oddrationale/mnist-in-csv?select=mnist_test.csv

该mnist_train.csv文件包含 60,000 个训练示例和标签。将mnist_test.csv含有10,000个测试实例和标签。每行包含 785 个值：第一个值是标签（从 0 到 9 的数字），其余 784 个值是像素值（从 0 到 255 的数字）。
一行为一个样本
'''

def loadData(file):
    print('loading file')
    dataArr = []
    labelArr = []
    fr = open(file,'r')
    cur = fr.readlines()
    for f in cur:
        curline = f.strip().split(',')
        if int(curline[0]) >= 5:
            labelArr.append(1)
        else:
            labelArr.append(-1)
        dataArr.append([int(nums)/255 for nums in curline[1:]] )
    return dataArr,labelArr

def proceptron(dataArr,labelArr,iter=50):
    print('start to train')
    datamat = np.mat(dataArr)
    labelmat = np.mat(labelArr).T
    m,n = np.shape(datamat)
    w = np.zeros((1, np.shape(datamat)[1]))
    b = 0
    h = 0.0001
    
    for k in range(iter):
        for i in range(m):
            xi = datamat[i]
            yi = labelmat[i]
            if -1 * yi*(w * xi.T + b)>=0:
                w = w + h*yi*xi
                b = b + h*yi
        print('round %d:%d training'%(k,iter))
    return w,b

def model_test(dataArr,labelArr,w,b):
    datamat = np.mat(dataArr)
    labelmat = np.mat(labelArr).T
    m,n = np.shape(datamat)
    errocnt = 0
    for i in range(m):
        xi = datamat[i]
        yi = labelmat[i]
        result = -1*yi*(w*xi.T+b)
        if  result >= 0:
            errocnt += 1
    accrurate =1- (errocnt/m)
    return accrurate

import numpy as np
import time
if __name__ == '__main__':
    start = time.time()
    traindata,trainlabel = loadData(r'C:\Users\Administrator\Desktop\机器学习\mnist_train.csv')
    testdata,testlabel = loadData(r'C:\Users\Administrator\Desktop\机器学习\mnist_test.csv')
    iter = 30
    w,b = proceptron(traindata,trainlabel,iter)
    accrurate = model_test(testdata,testlabel,w,b)
    end = time.time()
    print("the accuracy rate is:",accrurate)
    print('time span:',end-start)
