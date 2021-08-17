from Bucketing import bucketing
# knn
# # 留一法对小数据集是合适的，但大多数情况下我们会选择十折交叉验证。
# 衡量分类器准确率的方式是使用以下公式：正确分类的记录数÷记录总数。
# 需要一个更为详细的评价结果，这时就会用到一个称为混淆矩阵的可视化表格。
# 表格的行表示测试用例实际所属的类别，列则表示分类器的判断结果。
# 混淆矩阵可以帮助我们快速识别出分类器到底在哪些类别上发生了混淆
# 混淆矩阵的对角线（绿色字体）表示分类正确的人数，因此求得的准确率是：正确人数除以总人数
# 十折交叉验证也可以评估分类的准确率
#
######################################################################################################
# Knn分类步骤：
# 1.数据集分桶，将数据集分为10份，
# 2.将桶中数据读取到指定的数据集合中，并进行数据初始化
# 3.计算数据集中每个特征值与其中位数的差值以及标准差
# 4.对数据集中的每一个特征值进行标准化
# 5.最后进行分类操作
#   5.1 通过计算向量之间的曼哈顿距离得出相似度结果
#   5.2 通过相似度结果计算每个向量相似度最大的N个向量
#   5.3 确定最近的N个向量所属的类别，得票数最高的即为该向量所属的类别
# 6.使用指定的数据测试集进行测试
# 7.进行十折交叉进行训练和测试
# 8.通过混淆矩阵验证算法的有效性
########################################################################################################

import random
import heapq

class knnClassifier:
    def __init__(self, bucketPrefix, testBucketNumber, dataFormat,k):
        """
        该分类器程序将从bucketPrefix指定的一系列文件中读取数据，
        并留出testBucketNumber指定的桶来做测试集，其余的做训练集。
        dataFormat用来表示数据的格式，如：
        "class    num    num    num    num    num    comment"
        """
        # 分桶
        # bucketing.buckets('./data/mpgTrainingSet.txt','mpgSet','\t',1)

        self.k = k
        # 特征值与中位数的差值
        self.medianAndDeviation = []
        self.format = dataFormat.strip().split()
        # print(len(self.format))
        self.data =[]

        for i in range(1,11):
            if i != testBucketNumber:
                filename = "%s-%02i" % (bucketPrefix,i)
                f = open(filename)
                lines = f.readlines()
                f.close()

                for line in lines[1:]:
                    fields = line.strip().split('\t')
                    ignore = []
                    vector = []
                    for j in range(len(fields)):
                        if self.format[j] == "num":
                            vector.append(float(fields[j]))
                        elif self.format[j] == "comment":
                            ignore.append(fields[j])
                        elif self.format[j] == "class":
                            classification = fields[j]
                    self.data.append((classification,vector,ignore))

        self.rawData = list(self.data)
        # 获取特征向量的长度
        self.vlen = len(self.data[0][1])
        # 标准化数据
        for i in range(self.vlen):
            self.normalizeColumn(i)

    # 计算中位数
    def getMedian(self, alist):
        """return median of alist"""
        if alist == []:
            return []
        # 当数据量很大时，可以使用选择排序或者其他排序算法
        blist = sorted(alist)
        length = len(alist)
        if length % 2 == 1:
            # length of list is odd so return middle element
            return blist[int(((length + 1) / 2) -  1)]
        else:
            # length of list is even so compute midpoint
            v1 = blist[int(length / 2)]
            v2 =blist[(int(length / 2) - 1)]
            return (v1 + v2) / 2.0

    # 计算标准差
    def getAbsoluteStandardDeviation(self, alist, median):
        """given alist and median return absolute standard deviation"""
        sum = 0
        for item in alist:
            sum += abs(item - median)
        return sum / len(alist)

    # 标准化集合的特定列，特征值减去中位数除以绝对标准差
    def normalizeColumn(self, columnNumber):
       """given a column number, normalize that column in self.data"""
       # first extract values to list
       col = [v[1][columnNumber] for v in self.data]
       median = self.getMedian(col)
       asd = self.getAbsoluteStandardDeviation(col, median)
       #print("Median: %f   ASD = %f" % (median, asd))
       self.medianAndDeviation.append((median, asd))
       for v in self.data:
           v[1][columnNumber] = (v[1][columnNumber] - median) / asd

    # 标准化向量
    def normalizeVector(self, v):
        """We have stored the median and asd for each column.
        We now use them to normalize vector v"""
        vector = list(v)
        for i in range(len(vector)):
            (median, asd) = self.medianAndDeviation[i]
            vector[i] = (vector[i] - median) / asd
        return vector

    # 计算向量之间距离
    def manhattan(self, vector1, vector2):
        """Computes the Manhattan distance."""
        return sum(map(lambda v1, v2: abs(v1 - v2), vector1, vector2))

    # 未排除自身，相似度
    def nearestNeighbor(self, itemVector):
        """return nearest neighbor to itemVector"""
        result =min([ (self.manhattan(itemVector, item[1]), item)
                     for item in self.data])
        # print(result)
        return result

    def knn(self,itemVector):
        """使用kNN算法判断itemVector所属类别"""
        # 使用heapq.nsmallest来获得k个近邻
        neighbors = heapq.nsmallest(self.k,[(self.manhattan(itemVector,item[1]),item) for item in self.data])

        # print(neighbors)
        results = {}
        for neighbor in neighbors:
            theCalss = neighbor[1][0]
            results.setdefault(theCalss,0)
            results[theCalss] += 1

        resultList = sorted([(i[1],i[0]) for i in results.items()], reverse=True)

        # 获取得票最高的分类
        maxVotes = resultList[0][0]
        possibleAnswers = [i[1] for i in resultList if i[0] == maxVotes]
        answers = random.choice(possibleAnswers)
        return answers

    # 分类
    def classify(self, itemVector):
        """Return class we think item Vector is in"""
        # return(self.nearestNeighbor(self.normalizeVector(itemVector))[1][0])
        return (self.knn(self.normalizeVector(itemVector)))

    # 使用指定测试集进行测试
    def tesBuckets(self,bucketPrefix,testBucketNumber):
        """读取bucketPrefix-bucketNumber所指定的文件作为测试集"""
        filename = "%s-%02i" % (bucketPrefix,testBucketNumber)
        f = open(filename)
        lines = f.readlines()
        totals ={}
        f.close()

        for line in lines[1:]:
            data = line.strip().split('\t')
            vector = []
            classColumn = -1
            for i in range(len(self.format)):
                if self.format[i] == 'num':
                    vector.append(float(data[i]))
                elif self.format[i] == 'class':
                    classColumn = i
            theRealClass = data[classColumn]
            classifiedAs = self.classify(vector)
            totals.setdefault(theRealClass,{})
            totals[theRealClass].setdefault(classifiedAs,0)
            totals[theRealClass][classifiedAs] += 1
        return totals


# 十折交叉验证
def tenFolds(bucketPrefix,dataFormat,k):
    result ={}
    for i in range(1,11):
        c = knnClassifier(bucketPrefix,i, dataFormat,k)
        t= c.tesBuckets(bucketPrefix,i)
        for (key,value) in t.items():
            result.setdefault(key,{})
            for (ckey,cvalue) in value.items():
                result[key].setdefault(ckey,0)
                result[key][ckey] += cvalue
    print(result)
    # 输出结果
    categories = list(result.keys())
    categories.sort()
    print(   "Classified as: ")
    header =    "        "
    subheader = "      +"
    for category in categories:
        header += category + "   "
        subheader += "----+"
    print (header)
    print (subheader)
    total = 0.0
    correct = 0.0
    for category in categories:
        row = category + "    |"
        for c2 in categories:
            if c2 in result[category]:
                count = result[category][c2]
            else:
                count = 0
            row += " %2i |" % count
            total += count
            if c2 == category:
                correct += count
        print(row)
    print(subheader)
    print("\n%5.3f percent correct" %((correct * 100) / total))
    print("total of %i instances" % total)




print("SMALL DATA SET")
tenFolds("./data/pimaSmall/pimaSmall",
        "num    num num num num num num num class", 1)
print("\n\nLARGE DATA SET")

tenFolds("./data/pima/pima",
        "num    num num num num num num num class", 1)
# tenFolds("./data/pima/pima", "num    num    num    num    num    num    num    num    class",3)
# tenFolds("./test/mpgSet", "class    num    num    num    num    num    comment",3)
# k = knnClassifier('mpgSet',2,'class    num    num    num    num    num    comment')

# print(k.tesBuckets('mpgSet',2))

