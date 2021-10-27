#加载数据
#寻找最大类别
#计算经验熵
#计算条件经验熵
#计算信息增益，输出信息增益最大的特征以及信息增益值
#更新数据
#预测
#计算正确率
import numpy as np
def loadData(file):
    dataArr = []
    labelArr = []
    fr = open(file,'r')
    for line in fr.readlines():
        curline = line.strip().split(',')
        dataArr.append([int(int(nums)>128) for nums in curline[1:]])
        labelArr.append(int(curline[0]))
    return dataArr,labelArr

def majorClass(labelArr):
    classDict = {}
    for i in range(len(labelArr)):
        if labelArr[i] in classDict.keys():
            classDict[labelArr[i]] += 1
        else:
            classDict[labelArr[i]] = 1
    classSort = sorted(classDict.items(), key=lambda x: x[1], reverse=True)
    return classSort[0][0]

def calc_H_D(trainLabelArr):#---这个地方是要用数组计算的，输入的参数要是数组才行，不能是列表
    H_D = 0
    trainLabelSet = set([label for label in trainLabelArr])
    for i in trainLabelSet:
        p = trainLabelArr[trainLabelArr == i].size / trainLabelArr.size
        H_D += -1 * p * np.log2(p)
    return H_D

def calcH_D_A(trainDataArr_DevFeature,trainLabelArr):
    H_D_A = 0
    traindataset = set([label for label in trainDataArr_DevFeature])
    for i in traindataset:
        H_D_A += trainDataArr_DevFeature[trainDataArr_DevFeature==i].size/trainDataArr_DevFeature.size*calc_H_D(trainLabelArr[trainDataArr_DevFeature==i])
    return H_D_A


def calBestFeature(trainDataList,trainLabelList):
    trainDataArr = np.array(trainDataList)
    trainLabelArr = np.array(trainLabelList)
    featureNum = trainDataArr.shape[1]
    maxG_D_A = -1
    maxFeature = -1
    H_D = calc_H_D(trainLabelArr)
    for feature in range(featureNum):
        trainDataArr_DevideByFeature = np.array(trainDataArr[:, feature].flat)
        G_D_A = H_D - calcH_D_A(trainDataArr_DevideByFeature, trainLabelArr)
        if G_D_A > maxG_D_A:
            maxG_D_A = G_D_A
            maxFeature = feature
    return maxFeature, maxG_D_A

def getSubDataArr(trainDataArr, trainLabelArr, A, a):
    retDataArr = []
    retLabelArr = []
    for i in range(len(trainDataArr)):
        if trainDataArr[i][A] == a:
            retDataArr.append(trainDataArr[i][0:A] + trainDataArr[i][A+1:])
            retLabelArr.append(trainLabelArr[i])
    return retDataArr, retLabelArr

def createTree(*dataSet):
    Epsilon = 0.1
   
    trainDataList = dataSet[0][0]
    trainLabelList = dataSet[0][1]
    print('start a node', len(trainDataList[0]), len(trainLabelList))

    classDict = {i for i in trainLabelList}
    if len(classDict) == 1:
        return trainLabelList[0]
    if len(trainDataList[0]) == 0:
        return majorClass(trainLabelList)
    Ag, EpsilonGet = calBestFeature(trainDataList, trainLabelList)
    if EpsilonGet < Epsilon:
        return majorClass(trainLabelList)
    treeDict = {Ag:{}}
    treeDict[Ag][0] = createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 0))
    treeDict[Ag][1] = createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 1))

    return treeDict

def predict(testDataList,tree):
     while True:
        (key, value), = tree.items()
        if type(tree[key]).__name__ == 'dict':
            dataVal = testDataList[key]
            del testDataList[key]
            tree = value[dataVal]
            if type(tree).__name__ == 'int':
                return tree
        else:
            return value


def model_test(testdatalist,testlabellist,tree):
    erro = 0
    for i in range(len(testdatalist)):
        if testlabellist[i] != predict(testdatalist[i],tree):
            erro += 1
    return 1-erro/len(testdatalist)

if  __name__ == '__main__':
    trainDataList, trainLabelList = loadData(r'C:\Users\Administrator\Desktop\机器学习\mnist_train1.csv')
    testDataList, testLabelList = loadData(r'C:\Users\Administrator\Desktop\机器学习\mnist_test1.csv')
    print('start create tree')
    tree = createTree((trainDataList, trainLabelList))
    print('tree is:', tree)
    print('start test')
    accur = model_test(testDataList, testLabelList, tree)
    print('the accur is:', accur)

    
