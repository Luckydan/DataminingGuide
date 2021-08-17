# 分类器是指通过物品特征来判断它应该属于哪个组或类别的程序！
# 分类器程序会基于一组已经做过分类的物品进行学习，从而判断新物品的所属类别。
# “正规化”表示将值的范围缩小到0和1之间；（计算方法：特征值减去最小值然后除以最大值和最小值之间的差值）
# “标准化”则是将特征值转换为均值为0的一组数，其中每个数表示偏离均值的程度（即标准偏差或绝对偏差）。
# 本例代码中使用的修正的标准分就是属于后者。
# 留一法对小数据集是合适的，但大多数情况下我们会选择十折交叉验证。
# 衡量分类器准确率的方式是使用以下公式：正确分类的记录数÷记录总数。
# 需要一个更为详细的评价结果，这时就会用到一个称为混淆矩阵的可视化表格。
# 表格的行表示测试用例实际所属的类别，列则表示分类器的判断结果。
# 混淆矩阵可以帮助我们快速识别出分类器到底在哪些类别上发生了混淆
class Classifier:

    def __init__(self, filename):
        # 特征值与中位数差值集合
        self.medianAndDeviation = []
        # reading the data in from the file
        f = open(filename)
        lines = f.readlines()
        f.close()
        self.format = lines[0].strip().split('\t')
        # print(len(self.format))
        self.data = []
        for line in lines[1:]:
            fields = line.strip().split('\t')
            # print(len(fields))
            ignore = []
            vector = []
            for i in range(len(fields)):
                # print(fields)
                if self.format[i] == 'num':
                    vector.append(float(fields[i]))
                elif self.format[i] == 'comment':
                    ignore.append(fields[i])
                elif self.format[i] == 'class':
                    classification = fields[i]

            self.data.append((classification, vector, ignore))
        self.rawData = list(self.data)
        # get length of instance vector
        self.vlen = len(self.data[0][1])
        # now normalize the data
        for i in range(self.vlen):
            self.normalizeColumn(i)

    ##################################################
    ###
    ###  CODE TO COMPUTE THE MODIFIED STANDARD SCORE

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

    ###
    ### END NORMALIZATION
    ##################################################
    # map(fuction,iterable)
    # v1=[1,2,3,4]
    # v2=[2,3,4,5]
    # print([v for v in map(lambda x,y:(x*y,x+y),v1,v2)])
    def manhattan(self, vector1, vector2):
        """Computes the Manhattan distance."""
        return sum(map(lambda v1, v2: abs(v1 - v2), vector1, vector2))

    # 未排除自身
    def nearestNeighbor(self, itemVector):
        """return nearest neighbor to itemVector"""
        result =min([ (self.manhattan(itemVector, item[1]), item)
                     for item in self.data])
        print(result)
        return result

    def classify(self, itemVector):
        """Return class we think item Vector is in"""
        return(self.nearestNeighbor(self.normalizeVector(itemVector))[1][0])


def unitTest():
    classifier = Classifier('./data/athletesTrainingSet.txt')
    br = ('Basketball', [72, 162], ['Brittainey Raven'])
    nl = ('Gymnastics', [61, 76], ['Viktoria Komova'])
    cl = ("Basketball", [74, 190], ['Crystal Langhorne'])
    # first check normalize function
    brNorm = classifier.normalizeVector(br[1])
    nlNorm = classifier.normalizeVector(nl[1])
    clNorm = classifier.normalizeVector(cl[1])
    assert(brNorm == classifier.data[1][1])
    assert(nlNorm == classifier.data[-1][1])
    print('normalizeVector fn OK')
    # check distance
    assert (round(classifier.manhattan(clNorm, classifier.data[1][1]), 5) == 1.16823)
    assert(classifier.manhattan(brNorm, classifier.data[1][1]) == 0)
    assert(classifier.manhattan(nlNorm, classifier.data[-1][1]) == 0)
    print('Manhattan distance fn OK')
    # Brittainey Raven's nearest neighbor should be herself
    result = classifier.nearestNeighbor(brNorm)
    assert(result[1][2]== br[2])
    # Nastia Liukin's nearest neighbor should be herself
    result = classifier.nearestNeighbor(nlNorm)
    assert(result[1][2]== nl[2])
    # Crystal Langhorne's nearest neighbor is Jennifer Lacy"
    assert(classifier.nearestNeighbor(clNorm)[1][2][0] == "Jennifer Lacy")
    print("Nearest Neighbor fn OK")
    # Check if classify correctly identifies sports
    assert(classifier.classify(br[1]) == 'Basketball')
    assert(classifier.classify(cl[1]) == 'Basketball')
    assert(classifier.classify(nl[1]) == 'Gymnastics')
    print('Classify fn OK')

def test(training_filename, test_filename):
    """Test the classifier on a test set of data"""
    classifier = Classifier(training_filename)
    f = open(test_filename)
    lines = f.readlines()
    f.close()
    numCorrect = 0.0
    for line in lines:
        data = line.strip().split('\t')
        vector = []
        classInColumn = -1
        # print(len(data))
        for i in range(len(classifier.format)):
              if classifier.format[i] == 'num':
                  vector.append(float(data[i]))
              elif classifier.format[i] == 'class':
                  classInColumn = i
        theClass= classifier.classify(vector)
        prefix = '-'
        if theClass == data[classInColumn]:
            # it is correct
            numCorrect += 1
            prefix = '+'
        print("%s  %12s  %s" % (prefix, theClass, line))
    print("%4.2f%% correct" % (numCorrect * 100/ len(lines)))


# unitTest()
# classifier = Classifier('./data/athletesTrainingSet.txt')
# print(classifier.classify([74,190]))
# print(classifier.data)
# test('./data/athletesTrainingSet.txt','./data/athletesTestSet.txt')
# test('./data/irisTrainingSet.data', './data/irisTestSet.data')


# print([1,2,3][-1])
