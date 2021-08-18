# nativeBayesClassifier
import math

# ############################################################################################################
# 朴素贝叶斯方法的优缺点
# 优点：
# 1.实现简单（只需计数即可）
# 2.需要的训练集较少
# 3.运算效率高
# 缺点：
# 1.无法学习特征之间的相互影响。比如我喜欢奶酪，也喜欢米饭，但是不喜欢两者一起吃。
# 总结：使用朴素贝叶斯计算得到的概率其实是真实概率的一种估计，而真实概率是对全量数据做统计得到的；显然，对全量数据做统计是不现实的。
# 大部分情况下，这种估计都是接近于真实概率的。但当真实概率非常小时，这种抽样统计的做法就会有问题了。
# 在朴素贝叶斯中，概率为0的影响是很大的，甚至会不顾其他概率的大小。此外，抽样统计的另一个问题是会低估真实概率。
# 通过加入分类型特征值个数m和其先验概率p来解决该问题，详细内容请看笔记
#
# KNN 算法的优缺点
# 优点：
# 1.实现也比较简单
# 2.不需要按特定形式准备数据
# 缺点：
# 1.需要大量内存保存训练集数据
# 总结：当训练集不大时，kNN算法是一个不错的选择。这个算法的用途很广，包括推荐系统、蛋白质分析、图片分类等。
#
# #############################################################################################################
class nativeBayesClassifier:
    def __init__(self,bucketPrefix,testBucketNumber,dataFormat):
        """
        bucketPrefix:表示数据集分桶后的前缀名称
        testBucketNumber:表示测试集的桶编号
        dataFormat:表示数据集的格式样式：attr    attr    attr    attr    class

        self.conditional数据格式：
        {'i500': {1: {'appearance': 0.3333333333333333, 'health': 0.4444444444444444,
              'both': 0.2222222222222222},
          2: {'active': 0.4444444444444444, 'sedentary': 0.2222222222222222,
              'moderate': 0.3333333333333333},
          3: {'aggressive': 0.6666666666666666, 'moderate': 0.3333333333333333},
          4: {'yes': 0.6666666666666666, 'no': 0.3333333333333333}},
         'i100': {1: {'both': 0.5, 'health': 0.16666666666666666,
                      'appearance': 0.3333333333333333},
                  2: {'active': 0.3333333333333333, 'sedentary': 0.5,
                      'moderate': 0.16666666666666666},
                  3: {'aggressive': 0.16666666666666666, 'moderate': 0.8333333333333334},
                  4: {'yes': 0.3333333333333333, 'no': 0.6666666666666666}}}

        self.counts 数据格式：
        {'i500': {1: {'appearance': 3, 'health': 4, 'both': 2},
          2: {'active': 4, 'sedentary': 2, 'moderate': 3},
          3: {'aggressive': 6, 'moderate': 3},
          4: {'yes': 6, 'no': 3}},
         'i100': {1: {'both': 3, 'health': 1, 'appearance': 2},
                  2: {'active': 2, 'sedentary': 3, 'moderate': 1},
                  3: {'aggressive': 1, 'moderate': 5},
                  4: {'yes': 2, 'no': 4}}}

        self.classes 数据格式：
        {'i500': 9, 'i100': 6}

        self.totals 数据格式：
        totals = {'1': {1: 8, 2: 378, 3: 182, 4: 102, 5: 1141,
                6: 98.2, 7: 2.036, 8: 141},
          '0': {1: 3, 2: 323, 3: 242, 4: 96, 5: 214,
                6: 98.1, 7: 2.006, 8: 76}}

        self.numericValue 数据格式：
        totals = {'1': {1: 8, 2: 378, 3: 182, 4: 102, 5: 1141,
                6: 98.2, 7: 2.036, 8: 141},
          '0': {1: 3, 2: 323, 3: 242, 4: 96, 5: 214,
                6: 98.1, 7: 2.006, 8: 76}}
        """
        total = 0
        classes = {}
        # 对分类型数据进行计数
        counts = {}

        # 对数值型数据进行就和
        # 我们会用下面两个变量来计算每个分类各个特征的平均值和样本标准差
        totals = {}
        numericValues = {}

        # 从文件中读取数据
        self.format = dataFormat.strip().split('\t')
        # 先验概率
        self.prior = {}
        # 条件概率
        self.conditiational = {}

        # 遍历十个桶
        for i in range(1,11):
            # 跳过测试桶
            if i != testBucketNumber:
                filename = "%s-%02i" % (bucketPrefix,i)
                f = open(filename)
                lines = f.readlines()
                f.close()
                for line in lines:
                    fields = line.strip().split('\t')
                    ignore = []
                    vector = []
                    nums = []
                    for j in range(len(fields)):
                        if self.format[j] == "num":
                            nums.append(float(fields[j]))
                        if self.format[j] == "attr":
                            vector.append(fields[j])
                        if self.format[j] == "comment":
                            ignore.append(fields[j])
                        if self.format[j] == "class":
                            category = fields[j]

                    # 处理该条数据
                    total += 1
                    classes.setdefault(category,0)
                    counts.setdefault(category,{})

                    # 数值型数据处理初始化
                    totals.setdefault(category,{})
                    numericValues.setdefault(category,{})

                    classes[category] += 1

                    # 处理分类型的各个属性
                    col = 0
                    for columnValue in vector:
                        col += 1
                        counts[category].setdefault(col,{})
                        counts[category][col].setdefault(columnValue,0)
                        counts[category][col][columnValue] += 1

                    # 处理数值型的各个特征值
                    col = 0
                    for columnValue in nums:
                        col +=1
                        totals[category].setdefault(col,0)
                        totals[category][col] += columnValue

                        numericValues[category].setdefault(col,[])
                        numericValues[category][col].append(columnValue)

        # 记数结束，开始计算概率

        # 计算先验概率P(h)
        for (category,count) in classes.items():
            self.prior[category] = count / total

        # 计算条件概率
        for (category,columns) in counts.items():
            self.conditiational.setdefault(category,{})
            for(col,valueCounts) in columns.items():
                self.conditiational[category].setdefault(col,{})
                for (attr,columnValue) in valueCounts.items():
                    self.conditiational[category][col][attr] = (columnValue / classes[category])

        self.tmp = counts

        # 计算数值型特征值的平均值和标准差
        self.mean = {}
        self.ssd = {}

        # 计算平均值
        for (category,columns) in totals.items():
            self.mean.setdefault(category,{})
            for (col,cTotal) in columns.items():
                self.mean[category][col] = cTotal / classes[category]

        # 计算标准差
        for (category,columns) in numericValues.items():
            self.ssd.setdefault(category,{})
            for (col,values) in columns.items():
                SumOfSquareDifferences = 0
                theMean = self.mean[category][col]
                for value in values:
                    SumOfSquareDifferences += (value - theMean) **2
                columns[col] = 0
                self.ssd[category][col] = math.sqrt(SumOfSquareDifferences / (classes[category] -1))



    # 进行分类
    def classify(self,itemVector,numVector):
        """返回vector所属的类别"""
        results = []

        for (category,prior) in self.prior.items():
            # print((category,prior))
            prob = prior
            # 分类型特征值进行分类计算
            col = 1
            for attrValue in itemVector:
                # print(self.conditiational[category][col])
                if not attrValue in self.conditiational[category][col]:
                    # 属性不存在，返回规律值为0
                    prob = 0
                else:
                    # 所有特征值的条件概率乘积
                    prob = prob * self.conditiational[category][col][attrValue]
                col += 1

            # 数值型特征值进行分类计算
            col = 1
            for x in numVector:
                mean = self.mean[category][col]
                ssd = self.mean[category][1]
                ePart = math.pow(math.e,-(x-mean)**2/(2 * ssd**2))
                prob = prob * ((1.0 /(math.sqrt(2 * math.pi) * ssd)) * ePart)
            results.append((prob,category))

        print(results)
        return max(results)[1]

# 测试
# c = nativeBayesClassifier('./data/iHealth/i', 10, 'attr\tattr\tattr\tattr\tclass')
# test=c.classify(['health','moderate', 'moderate', 'yes'])
# print(test)

c = nativeBayesClassifier('./data/pimaSmall/pimaSmall', 1, 'num\tnum\tnum\tnum\tnum\tnum\tnum\tnum\tclass')
test=c.classify([],[4,110,76,20,100,28.4,0.118,27])
print(test)
