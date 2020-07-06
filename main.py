import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.pyplot import MultipleLocator
trainnum = 3000
testnum = 1000

def plotweights(weightsarray):
    imagenum,cycles = np.shape(weightsarray) #imagenum = 58
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.plot(range(cycles),list(weightsarray[i]),'r-',label=u"线条")
        plt.ylabel('w'+str(i+1))
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()

    for i in range(16,32):
        plt.subplot(4,4,i-15)
        plt.plot(range(cycles),list(weightsarray[i]),'r-',label=u"线条")
        plt.ylabel('w'+str(i+1))
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()

    for i in range(32,48):
        plt.subplot(4,4,i-31)
        plt.plot(range(cycles),list(weightsarray[i]),'r-',label=u"线条")
        plt.ylabel('w'+str(i+1))
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()

    for i in range(48,imagenum):
        plt.subplot(4,4,i-47)
        plt.plot(range(cycles),list(weightsarray[i]),'r-',label=u"线条")
        plt.ylabel('w'+str(i+1))
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()

def plotloss(losslist):
    x = len(losslist)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("损失函数变化曲线")
    #plt.xlim(xmax=x, xmin=0)
    #plt.ylim(ymax=200, ymin=0)
    #plt.annotate("(3,6)", xy=(3, 6), xytext=(4, 5), arrowprops=dict(facecolor='black', shrink=0.1))
    plt.xlabel("次数")
    plt.ylabel("损失")
    plt.plot(range(x),losslist,'r-',label=u"线条")
    plt.show()

def loss(weights,datamat,labelmat):
    """
    计算损失
    :param weights:n*1的矩阵
    :param datamat:m*n的矩阵，n是特征维度，m是样本数
    :param labelmat:m*1的矩阵
    :return:浮点数
    """
    m,n = np.shape(datamat)
    result = 0.0
    for i in range(m):
        x = datamat[i]*weights
        x = x[0,0]
        label = labelmat[i,0]
        result = result + max(x, 0) - x * label + math.log(1 + math.exp(-abs(x)))
    return result*(1.0/m)

def sigmoidmat(x):
    """
    :param x:m*1的列向量
    :return: m*1的列向量
    """
    m,n = np.shape(x)
    results = np.ones((m,1))
    for i in range(m):
        results[i,0] = sigmoid(x[i,0])
    return results

def sigmoid(x):
    return math.exp(min(0,x))/(1.0 + math.exp(-abs(x)))

def getsum(a,b):
    """
    两个list的对应位置元素相加
    :param a:list
    :param b: list
    :return: list
    """
    n = len(a)
    sum = []
    for i in range(n):
        sum.append(a[i]+b[i])
    return sum

def loadtraindata(filepath): #load train data
    datain = []
    lablemat = []
    with open(filepath) as f:
        reader = csv.reader(f)
        reader = list(reader)
        size = len(reader[0])
        for i in range(trainnum):
            lablemat.append(reader[i][size-1])
            reader[i] = reader[i][1:size-1]
            reader[i].insert(0,1.0)
            datain.append(reader[i])
    lablemat = [int(x) for x in lablemat]
    datain = [[float(x) for x in row] for row in datain]
    return guiyi(datain),lablemat

def loadtraindata1(filepath): #load train data
    datain = []
    lablemat = []
    with open(filepath) as f:
        reader = csv.reader(f)
        reader = list(reader)
        size = len(reader[0])
        for i in range(trainnum):
            lablemat.append(reader[i][size-1])
            reader[i] = reader[i][1:size-1]
            floatlist = [float(x) for x in reader[i]]
            sqrtlist = [x**2 for x in floatlist]
            floatlist = floatlist + sqrtlist
            floatlist.insert(0,1.0)
            datain.append(floatlist)
    lablemat = [int(x) for x in lablemat]
    #datain = [[float(x) for x in row] for row in datain]
    return guiyi(datain),lablemat

def guiyi(datain):
    """
    :param datain:list
    :return: list
    """
    datain1 = datain
    datamat = np.mat(datain).transpose()
    n,m = np.shape(datamat)
    minlist = []
    maxlist = []
    for i in range(1,n):
        row = datamat[i].tolist()
        row = row[0]
        maxlist.append(max(row))
        minlist.append(minexcept0(row))
    for i in range(m):
        for j in range(1,n):
            if minlist[j-1]>=maxlist[j-1]:
                continue
            if datain1[i][j] != 0:
                datain1[i][j] = (maxlist[j-1] - datain1[i][j])/(maxlist[j-1] - minlist[j-1])
    return datain1

def minexcept0(a):
    """
    :param a:list
    :return: min num(not 0)
    """
    rmin = 1000
    for i in range(len(a)):
        if a[i] != 0:
            if a[i]<rmin:
                rmin = a[i]
    return rmin

def loadtestdata1(filepath):
    datamat = []
    lablemat = []
    with open(filepath) as f:
        reader = csv.reader(f)
        reader = list(reader)
        size = len(reader[0])
        for i in range(trainnum,testnum+trainnum):
            lablemat.append(reader[i][size-1])
            reader[i] = reader[i][1:size-1]
            floatlist = [float(x) for x in reader[i]]
            sqrtlist = [x ** 2 for x in floatlist]
            floatlist = floatlist + sqrtlist
            floatlist.insert(0, 1.0)
            datamat.append(floatlist)
    lablemat = [int(x) for x in lablemat]
    #datamat = [[float(x) for x in row] for row in datamat]
    return guiyi(datamat),lablemat

def loadtestdata(filepath): #添加特征数
    datain = []
    lablemat = []
    with open(filepath) as f:
        reader = csv.reader(f)
        reader = list(reader)
        size = len(reader[0])
        for i in range(trainnum,trainnum+testnum):
            lablemat.append(reader[i][size-1])
            reader[i] = reader[i][1:size-1]
            reader[i].insert(0,1.0)
            datain.append(reader[i])
    lablemat = [int(x) for x in lablemat]
    datain = [[float(x) for x in row] for row in datain]
    return guiyi(datain),lablemat

def gradAscent1(datamat,labelin,testdatamat,testlabelin,Lamda):
    labelmat = np.mat(labelin).transpose()
    testlabel = np.mat(testlabelin).transpose()
    m,n = np.shape(datamat)
    maxcycles = 10
    weights = np.zeros((n,1))
    #weightsarray = weights
    losslist = [loss(weights,datamat,labelmat)]
    testlosslist = [loss(weights,testdatamat,testlabel)]
    alpha = 0.001
    for k in range(maxcycles):
        h = sigmoidmat(datamat*weights)
        error = (h - labelmat)
        weights = weights*(1-alpha*Lamda) - alpha*datamat.transpose()*error
        #weightsarray = np.column_stack((weightsarray,weights.tolist()))
        losslist.append(loss(weights,datamat,labelmat))
        testlosslist.append(loss(weights,testdatamat,testlabel))
    #plot2loss(losslist,testlosslist)
    #plotweights(weightsarray)
    #print(min(losslist))
    return weights,losslist,testlosslist
def stocgradascent1(datain,labelin,cycles=100):
    datamat = np.mat(datain)
    labelmat = np.mat(labelin).transpose()
    m,n = np.shape(datamat)
    weights = np.zeros((n,1))
    weightsarray = weights
    losslist = [loss(weights,datamat,labelmat)]
    for j in range(cycles):
        dataindex = list(range(m))
        for i in range(m):
            alpha = 1.0/(10+j+i) + 0.01
            randindex = int(random.uniform(0,len(dataindex)))
            x = datamat[dataindex[randindex]]*weights
            h = sigmoid(x[0,0])
            error = labelin[dataindex[randindex]] - h
            weights = weights + alpha*error*datamat[dataindex[randindex]].transpose()
            del(dataindex[randindex])
        weightsarray = np.column_stack((weightsarray,weights.tolist()))
        losslist.append(loss(weights,datamat,labelmat))
    plotloss(losslist)
    #plotweights(weightsarray)
    print(min(losslist))
    return weights
def stocgradascent(datain,labelin,alpha,cycles): #小批量梯度下降法
    labelmat = np.mat(labelin).transpose()
    datamat = np.mat(datain)
    m,n = np.shape(datamat)
    weights = np.zeros((n,1)) #初始化为0的数组
    #weightsarray = weights
    losslist = [loss(weights,datamat,labelmat)]
    mini_batch = 30
    for k in range(cycles):
        for j in range(int(m/mini_batch)):
            for i in range(mini_batch*j,mini_batch*j+mini_batch):
                x = datamat[i]*weights
                h = sigmoid(x[0,0])
                error = labelin[i] - h
                weights = weights + 1.0/mini_batch * alpha*error*datamat[i].transpose()
            #weightsarray = np.column_stack((weightsarray,weights.tolist()))
            losslist.append(loss(weights,datamat,labelmat))
    plotloss(losslist)
    #plotweights(weightsarray)
    print(min(losslist))
    return weights
def classifyvector(testdata,weights):
    """
    计算分类结果
    :param testdata: m*n的矩阵
    :param weights: n*1的列向量
    :return:长度为testnum的列表
    """
    m,n = np.shape(testdata)
    classifylist = []
    for i in range(m):
        x = testdata[i]*weights
        prob = sigmoid(x[0,0])
        if prob > 0.5:
            classifylist.append(1)
        else:
            classifylist.append(0)
    return classifylist
def gradAscent(datamat,labelin,alpha):
    labelmat = np.mat(labelin).transpose()
    m,n = np.shape(datamat)
    maxcycles = 1000
    weights = np.zeros((n,1))
    #weightsarray = weights
    losslist = [loss(weights,datamat,labelmat)]
    Lamda = 0.01
    for k in range(maxcycles):
        h = sigmoidmat(datamat*weights)
        error = (h - labelmat)
        weights = weights*(1-Lamda*alpha) - alpha*datamat.transpose()*error
        #weightsarray = np.column_stack((weightsarray,weights.tolist()))
        losslist.append(loss(weights,datamat,labelmat))
    plotloss(losslist)
    #plotweights(weightsarray)
    #print(min(losslist))
    return weights
def plot7trainloss(trainlosslist,alphalist):
    num = len(trainlosslist)
    cycles = len(trainlosslist[0])
    x = list(range(cycles))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("训练集的损失函数变化曲线")
    plt.xlabel("次数")
    plt.ylabel("损失")
    plt.plot(x, trainlosslist[0], 'r-', color='green', label="Lamda = %f"%alphalist[0])
    plt.plot(x, trainlosslist[1], 'r-', color='red', label="Lamda = %f" % alphalist[1])
    plt.plot(x, trainlosslist[2], 'r-', color='orange', label="Lamda = %f" % alphalist[2])
    plt.plot(x, trainlosslist[3], 'r-', color='cyan', label="Lamda = %f" % alphalist[3])
    plt.plot(x, trainlosslist[4], 'r-', color='blue', label="Lamda = %f" % alphalist[4])
    #plt.plot(x, trainlosslist[5], 'r-', color='blue', label="alpha = %f" % alphalist[5])
    #plt.plot(x, trainlosslist[6], 'r-', color='purple', label="alpha = %f" % alphalist[6])
    plt.legend()
    plt.show()
def plot7testloss(trainlosslist,alphalist):
    num = len(trainlosslist)
    cycles = len(trainlosslist[0])
    x = list(range(cycles))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("测试集的损失函数变化曲线")
    plt.xlabel("次数")
    plt.ylabel("损失")
    plt.plot(x, trainlosslist[0], 'r-', color='green', label="Lamda = %f"%alphalist[0])
    plt.plot(x, trainlosslist[1], 'r-', color='red', label="Lamda = %f" % alphalist[1])
    plt.plot(x, trainlosslist[2], 'r-', color='orange', label="Lamda = %f" % alphalist[2])
    plt.plot(x, trainlosslist[3], 'r-', color='cyan', label="Lamda = %f" % alphalist[3])
    plt.plot(x, trainlosslist[4], 'r-', color='blue', label="Lamda = %f" % alphalist[4])
    #plt.plot(x, trainlosslist[5], 'r-', color='blue', label="alpha = %f" % alphalist[5])
    #plt.plot(x, trainlosslist[6], 'r-', color='purple', label="alpha = %f" % alphalist[6])
    plt.legend()
    plt.show()
def test(filepath):
    traindata,trainlable = loadtraindata(filepath)
    traindatamat = np.mat(traindata)
    testdata,testlable = loadtestdata(filepath)
    testdatamat = np.mat(testdata)
    trainerrorlist = []
    testerrorlist = []
    trainlosslist= []
    testlosslist = []
    Lamdalist = [0,0.05,1.0,2.0,2.5]
    for i in range(len(Lamdalist)):
        weights,trainloss,testloss = gradAscent1(traindatamat,trainlable,testdatamat,testlable,Lamdalist[i])
        trainlosslist.append(trainloss)
        testlosslist.append(testloss)
        trainerrorlist.append(cacuerror(traindatamat,trainlable,weights))
        testerrorlist.append(cacuerror(testdatamat,testlable,weights))
    plot7trainloss(trainlosslist,Lamdalist)
    plot7testloss(testlosslist,Lamdalist)
    print(trainerrorlist)
    print(testerrorlist)
    #print("训练集大小:%d,测试集大小:%d,测试集分类错误率:%f"%(trainnum,testnum,testerror))
    #print("训练集分类错误率:%f"%trainerror)
def testalone(filepath):
    traindata, trainlable = loadtraindata(filepath)
    traindatamat = np.mat(traindata)
    testdata, testlable = loadtestdata(filepath)
    testdatamat = np.mat(testdata)
    weights = gradAscent(traindatamat,trainlable,0.001)
    trainerror = cacuerror(traindatamat,trainlable,weights)
    testerror = cacuerror(testdatamat,testlable,weights)
    print(trainerror)
    print(testerror)
def cacuerror(datamat,labellist,weights):
    """
    :param datamat: m*n矩阵
    :param labellist: 长度为m的list
    :param weights: n*1的矩阵
    :return: 错误率
    """
    errorcount = 0
    classifylist = classifyvector(datamat, weights)
    for i in range(len(labellist)):
        if classifylist[i] != labellist[i]:
            errorcount += 1
    errorRate = (float(errorcount) / len(labellist))
    return errorRate
def plot2loss(trainlosslist,testlosslist):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("训练集和测试集的损失函数变化曲线")
    # plt.xlim(xmax=x, xmin=0)
    # plt.ylim(ymax=200, ymin=0)
    # plt.annotate("(3,6)", xy=(3, 6), xytext=(4, 5), arrowprops=dict(facecolor='black', shrink=0.1))
    plt.xlabel("次数")
    plt.ylabel("损失")
    #ax = plt.gca()
    #ax.xaxis.set_major_locator(x_major_locator)
    #plt.xticks([0.00001,0.0001,0.001,0.01,0.1,1.0],[0.00001,0.0001,0.001,0.01,0.1,1.0])
    x = list(range(len(trainlosslist)))
    plt.plot(x, trainlosslist, 'r-',color = 'g', label="train set")
    plt.plot(x, testlosslist,'r-',color = 'b', label="test set")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #alphalist = [0.00001, 0.00005, 0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.003, 0.009,0.01, 0.03, 0.09, 0.1, 0.3,0.6, 0.9, 1.0]
    #trainerror = [0.11766666666666667, 0.08733333333333333, 0.08, 0.07466666666666667, 0.07166666666666667, 0.07066666666666667,0.07066666666666667, 0.06833333333333333, 0.081, 0.216, 0.073, 0.24666666666666667, 0.22066666666666668, 0.088, 0.07, 0.088666666666666667,0.07766666666666666, 0.06966666666666667]
    #testerror = [0.11, 0.079, 0.072, 0.068, 0.067, 0.07, 0.073, 0.073, 0.075, 0.224, 0.076, 0.241, 0.234, 0.079, 0.075, 0.078,0.081, 0.076]
    #ploterror(trainerror,testerror,alphalist)
    #alphalist1 = [1.1, 1.2, 0.9, 0.0003, 0.0005, 0.0007, 0.0009, 0.002, 0.004, 0.005, 0.006, 0.007, 0.008,0.2, 0.4, 0.5, 0.7, 0.8, 1.3, 1.4, 1.5]
    #trainerror1 = [0.094, 0.066, 0.07766666666666666, 0.07266666666666667, 0.072, 0.071, 0.06933333333333333, 0.06566666666666666, 0.069, 0.066, 0.07066666666666667, 0.21666666666666667, 0.09433333333333334, 0.078, 0.094, 0.06866666666666667, 0.09166666666666666, 0.15066666666666667, 0.072, 0.091, 0.06466666666666666]
    #testerror1 = [0.098, 0.074, 0.081, 0.068, 0.069, 0.073, 0.073, 0.071, 0.071, 0.073, 0.074, 0.206, 0.089, 0.076, 0.098, 0.077, 0.083, 0.141, 0.074, 0.084, 0.073]
    test("e:\\机器学习\\结课\\题目4 个人收入预测.csv")