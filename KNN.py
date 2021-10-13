'''
KNN算法，根据最邻近的K个值，来对新输入的样本进行分类
关键点在于K值怎么确定，怎么才算最近邻
'''

'''
1、导入数据
2、计算点与点之间的距离
3、训练模型以及测试：数据处理、计算距离、找出距离最大的前K个值、输出判断的类别、预测值与真实值的差异、计算正确率
'''
import time
import numpy as np

def loadData(filename):
    print('start to loadfile')
    dataArr = []
    labelArr = []
    fr = open(filename,'r')
    for line in fr.readlines():
        curline = line.strip().split(',')
        dataArr.append([int(nums) for nums in curline[1:]])
        labelArr.append(int(curline[0]))
    return dataArr,labelArr

def caldist(x1,x2):
    return (np.sqrt(np.sum(np.square(x1-x2))))

def getclosest(traindatamat,trainlabelmat,x,topk):
    distlist = [0] * len(traindatamat)
    for i in range(len(traindatamat)):
        x1 = traindatamat[i]
        dist = caldist(x1,x)
        distlist[i] = dist
    topklist = np.argsort(np.array(distlist))[:topk]#这个地方返回的是前topk大的值的下标
    labellist = [0] * 10
    for index in topklist:
        labellist[int(trainlabelmat[index])] += 1
    return labellist.index(max(labellist))


def model_test(traindataArr,trainlabelArr,testdataArr,testlabelArr,topk):
    print('start to test')
    traindatamat = np.mat(traindataArr)
    trainlabelmat = np.mat(trainlabelArr).T
    testdatamat = np.mat(testdataArr)
    testlabelmat = np.mat(testlabelArr).T
    
    erro = 0
    for i in range(10):
        x = testdatamat[i]
        y = getclosest(traindatamat,trainlabelmat,x,topk)
        if y != testlabelmat[i]:
            erro += 1
    accrurate = 1 - (erro / 10 )
    return accrurate


if __name__ == '__main__':
    start = time.time()
    traindataArr,trainlabelArr = loadData(r'C:\Users\Administrator\Desktop\机器学习\mnist_train.csv')
    testdataArr,testlabelArr = loadData(r'C:\Users\Administrator\Desktop\机器学习\mnist_test.csv')
    topk = 25
    accrurate = model_test(traindataArr,trainlabelArr,testdataArr,testlabelArr,topk)
    end = time.time()
    print('accrurate is :',accrurate*100,'%')
    print('span time :',end-start)
