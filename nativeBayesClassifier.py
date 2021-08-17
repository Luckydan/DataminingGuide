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
        slef.prior = {}
        # 条件概率
        self.conditiational = {}

        # 遍历十个桶
        for i in range(1,11):
            # 跳过测试桶
            if i != testBucketNumber:
                filename = "%s-%2i" % (bucketPrefix,i)
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
        for category,count) in classes.items():
            slef.prior{category} = count / total

        # 计算条件概率
        for (category,columns) in counts.items():
            slef.conditiational.setdefault(category,{})
            for(col,valueCounts) in columns.items():
                self.conditiational[category].setdefault(col,{})
                for (attr,columnValue) in valueCounts.items():
                    self.conditiational[category][col][attr] = (columnValue / classes[category])

        self.tmp = counts
