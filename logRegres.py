import numpy as np

def loadDataSet(filename="testSet.txt"):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(intX):
    return 1.0/(1 + np.exp(-intX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        weights = weights + alpha * dataMatrix.transpose() * (labelMat - h)
    return weights

def stocGradAscent(dataMatrix, classLabels, numIter = 150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.001
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del dataIndex[randIndex]
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = dataArr.shape[0]
    xcord1,ycord1 = [],[]
    xcord2,ycord2 = [],[]
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    x = np.arange(-3.0,3.0,0.1)
    y = (-weights[0] - weights[1] * x)/weights[2]
    y = y.reshape((60,1))
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob >= 0.5:
        return 1
    else:
        return 0
def colicTest():
    frTrain = open("horseColicTraining.txt")
    frTest = open("horseColicTest.txt")
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split()
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = float(errorCount) / numTestVec
    print ("Error rate is: {0:0.2f}%".format(errorRate * 100))
    return errorRate
def myColicTest(trainFile, testFile):
    trainingData = np.loadtxt(trainFile)
    testData = np.loadtxt(testFile)
    trainingSet = trainingData[:,:-1]
    trainingLabels = trainingData[:,-1]
    testSet = testData[:,:-1]
    testLabels = testData[:,-1]
    trainWeights = stocGradAscent(trainingSet, trainingLabels, 500)
    errorCount = 0
    numTestVec = len(testLabels)
    for i in range(numTestVec):
        if int(classifyVector(testSet[i,:], trainWeights)) != int(testLabels[i]):
            errorCount += 1
    errorRate = float(errorCount) / numTestVec
    print ("Error rate is :{0:0.2f}%".format(float(errorCount*100)/numTestVec))
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += myColicTest("horseColicTraining.txt","horseColicTest.txt")
    print ("My: After {} iterations the everage error rate is: {:0.2f}%".format(numTests, float(errorSum)*100/(numTests)))
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("Book: After {} iterations the everage error rate is: {:0.2f}%".format(numTests, float(errorSum)*100/(numTests)))
    
    