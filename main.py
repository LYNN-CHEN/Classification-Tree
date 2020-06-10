import csv

#建立决策树模型
class Tree:
    def __init__(self, value=None, left=None, right=None, results=None, col=-1, data=None):
        self.value = value
        self.left = left
        self.right = right
        self.results = results
        self.col = col
        self.data = data

#分别计算dataset/分割后的list中quality大于6和小于等于6的元素个数
#返回一个字典储存结果
def countResult(dataset,value=6):       
    results = {'above':0,'else':0}
    for data in dataset:
        if data[-1] > value:
            results['above'] += 1
        else:
            results['else'] +=1
    return results

#根据 gini = 1 - Σp² 公式计算数据集中的gini值
def calGini(lines):     
    length = len(lines)     #得到lines总数据量
    results = countResult(lines)   #计算lines中above和else的个数   
    pSquare = 0
    if length != 0:
        for i in results:
            pSquare += (results[i]/length)**2
            Gini = 1 - pSquare
        return Gini

#根据所给的value来分叉
#大于value → 归为listBig；反之归为listSmall
def splitData(rows_split,col,value):
    listBig = []
    listSmall = []
    for row in rows_split:
        if row[col] > value:
            listBig.append(row)
        else:
            listSmall.append(row)
    return listBig,listSmall

#创建分支
def createBranch(rows):       
    global testLabel
    currentGain = calGini(rows)     #得到当前的gini指数
    length = len(testLabel)     #可以进行考虑的特征数
    bestFeatureIndex = None     #初始化索引值、最佳分割点和最佳giniGain
    bestGiniGain = 0
    bestSplitValue = None

    for col in range (0,length-1):      #遍历所有特征
        #去除重复取值
        setColValue = []        
        for i in rows:
            if i[col] not in setColValue:
                setColValue.append(i[col])
        #取去重得到的list中相邻两元素的中间值构成一个新的list
        middleValue = []        
        i = 0
        while i < (len(setColValue)-1):
            middleValue.append((setColValue[i]+setColValue[i+1])/2)
            i += 2
        #在新的list中测试 尝试得到每个特征对应的最佳分割点的值value
        for value in middleValue:
            listB,listS = splitData(rows,col,value)        #根据该分割点分出两部分list
            p = len(listB)/len(rows)
            giniBig = calGini(listB)
            giniSmall = calGini(listS)
            giniGain = currentGain - p*giniBig - (1-p)*giniSmall      #算该分割点的giniGain
            if giniGain > bestGiniGain:
                bestGiniGain = giniGain     #获取最小时候的giniGain以及此时的特征索引值
                bestFeatureIndex = col
                bestSplitValue = value

    #当最佳gini增益大于0时，证明还未分叉结束，继续递归分叉
    if bestGiniGain > 0:
        left = createBranch(listB)      #左支为listBig
        right = createBranch(listS)     #右支为listSmall
        return Tree(col=bestFeatureIndex, value = bestSplitValue, left = left, right=right)
    #否则 结束递归，将最终该分叉下的above和below的情况储存到results字典中，方便之后的分类
    else:
        return Tree(results=countResult(rows), data=rows)

#def prune(tree):



#定义分类函数，对给定的testData预测其quality是否大于6并判断预测值是否准确
def classify(testData, tree):
    #当tree.results != None，意味着该节点不具有subtree，停止继续往下查找
    if tree.results != None:        
        above = tree.results['above']       #获取当前results中above和else的个数
        below = tree.results['else']
        if above >= below:      #当分类的results中above占多数时，预测该testData的quality也是above（>6），标记为1
            prediction = 1
        else:       #反之，预测其<=6，标记为0
            prediction = 0
        if testData[-1] > 6 :     #检查实际quality的大小
            fact = 1
        else:
            fact = 0  
        if fact == prediction:      #预测准确则返回True；反之返回False
            return True
        else:
            return False              
    #若还没到达最低端
    else:
        value = testData[tree.col]       #获取该特征（col）下给定数据的value是多少
        if value >= tree.value:     #与该节点的分割value对比，若大于，则往左查找；小于则往右
            branch = tree.left
        else:
            branch = tree.right
        return classify(testData,branch)        #递归查找

#创建数据库和标签列
def main(file):
    label = file[0]      #标签名
    testLabel = label
    del testLabel[-1]
    file.pop(0)
    dataset = []        #只存储数据
    for i in file:
        i = list(map(float,i)) 
        if i not in dataset:
            dataset.append(i)
    return label, dataset, testLabel


if __name__ == '__main__':
    #导入 train.csv 和 test.csv
    train = open(r'train.csv','r')
    test = open(r'test.csv','r')
    train_csv = list(csv.reader(train))
    test_csv = list(csv.reader(test))
    #处理train.csv和test.csv的数据
    label, dataset, testLabel = main(train_csv)
    testData = []
    test_csv.pop(0)
    for i in test_csv:
        i = list(map(float,i)) 
        testData.append(i)
    #创建树
    tree = createBranch(dataset)
    #利用testData中的数据计算预测正确的个数
    correct = 0
    for i in testData:
        result = classify(i,tree)
        if result == True:
            correct += 1
    #计算正确率
    accuracy = correct/len(testData)
    print('Accuracy:', accuracy)