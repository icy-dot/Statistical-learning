#首先获取数据
#定义朴素贝叶斯分类，里面要用到计算先验概率以及条件概率的功能
#测试准确率
import time
import numpy as np

def loaddata(file):
    dataArr = []
    labelArr = []
    fr = open(file,'r')
    for line in fr.readlines():
        curline = line.strip().split(',')
        dataArr.append([int(int(nums)>128) for nums in curline[1:]])
        labelArr.append(int(curline[0]))
    return dataArr,labelArr

def getallprosibility(traindataArr,trainlabelArr):
    feature = 784
    classnums = 10
    
    #先验概率
    py = np.zeros((classnums,1))
    for i in range(classnums):
        py[i] = ((np.sum(np.mat(trainlabelArr)==i))+1) /  (len(trainlabelArr)+10)
    py = np.log(py)
    #条件概率
    px_y = np.zeros((classnums,feature,2))
    for n in range(len(trainlabelArr)):
        label =  trainlabelArr[n]
        x = traindataArr[n]
        for j in range(feature):
            px_y[label][j][x[j]] += 1
            
    for label in range(classnums):
        for j in range(feature):
            px_y0 = px_y[label][j][0]
            px_y1 = px_y[label][j][1]
            px_y[label][j][0] = np.log((px_y0+1)/(px_y0+px_y1+2))
            px_y[label][j][1] = np.log((px_y1+1)/(px_y0+px_y1+2))
    return py,px_y

def navieBayes(py,px_y,x):
    feature = 784
    classnums = 10
    p = [0]*classnums
    for i in range(classnums):
        sum = 0
        for j in range(feature):
            sum += px_y[i][j][x[j]]
        p[i] = sum + py[i]
    return p.index(max(p))


def model_test(py,px_y,testdataArr,testlabelArr):
    erro = 0
    for i in range(len(testdataArr)):
        predict = navieBayes(py,px_y,testdataArr[i])
        if predict != testlabelArr[i]:
            erro += 1
    return 1-erro/len(testlabelArr)


if __name__ == '__main__':
    start = time.time()
    traindataArr,trainlabelArr = loaddata(r'C:\Users\Administrator\Desktop\机器学习\mnist_train.csv')
    testdataArr,testlabelArr = loaddata(r'C:\Users\Administrator\Desktop\机器学习\mnist_test.csv')
    py,px_y = getallprosibility(traindataArr,trainlabelArr)
    accru = model_test(py,px_y,testdataArr,testlabelArr)
    end = time.time()
    print('accu is :',accru)
    print('span time:',end-start)

    
