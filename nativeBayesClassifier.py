# nativeBayesClassifier

class nativeBayesClassifier:
    def __init__(self,bucketPrefix,testBucketNumber,dataFormat):
        """
        bucketPrefix:表示数据集分桶后的前缀名称
        testBucketNumber:表示测试集的桶编号
        dataFormat:表示数据集的格式样式：attr    attr    attr    attr    class
        """
        total = 0
        classes = {}
        counts = {}

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
                    for j in range(len(fields)):
                        if self.format[j] == "num":
                            vector.append(float(fields[j]))
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
                    classes[category] += 1

                    # 处理各个属性
                    col = 0
                    for columnValue in vector:
                        col += 1
                        counts[category].setdefault(col,{})
                        counts[category][col].setdefault(columnValue,0)
                        counts[category][col][columnValue] += 1

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

    # 进行分类
    def classify(self,vector):
        """返回vector所属的类别"""
        results = []

        for (category,prior) in self.prior.items():
            # print((category,prior))
            prob = prior
            col = 1
            for attrValue in vector:
                # print(self.conditiational[category][col])
                if not attrValue in self.conditiational[category][col]:
                    # 属性不存在，返回规律值为0
                    prob = 0
                else:
                    # 所有特征值的条件概率乘积
                    prob = prob * self.conditiational[category][col][attrValue]
                col += 1
            results.append((prob,category))
        print(results)
        return max(results)[1]

# 测试
c = nativeBayesClassifier('./data/iHealth/i', 10, 'attr\tattr\tattr\tattr\tclass')
test=c.classify(['health','moderate', 'moderate', 'yes'])
print(test)

