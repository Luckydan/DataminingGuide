import random

############################################################################################
# 十折交叉验证，便用于更大的数据集。
# 十折交叉验证要求数据集等分成10份
############################################################################################
class bucketing:
    # 将数据进行分桶
    def buckets(filename,bucketName,separator,classColumn):
        """filename是源文件名
        bucketName是十个目标文件的前缀名
        separator是分隔符，如制表符、逗号等
        classColumn是表示数据所属分类的那一列的序号"""

        # 将数据分为10份
        numberOfBuckets = 10
        data = {}

        # 获取分类，然后按类放置
        with open(filename) as f:
            lines =f.readlines()

        for line in lines:
            if separator != '\t':
                line = line.replace(separator,'\t')
            # 获取分类
            category = line.split()[classColumn]
            data.setdefault(category,[])
            data[category].append(line)

        buckets = []
        # 初始化同
        for i in range(numberOfBuckets):
            buckets.append([])

        # 将各个类别的数据均匀的放置到各个桶中
        for k in data.keys():
            # 打乱顺序
            random.shuffle(data[k])
            bNum = 0

            # 分桶
            for item in data[k]:
                buckets[bNum].append(item)
                bNum = (bNum + 1) % numberOfBuckets

        # 将桶中的数据写到文件中
        for bNum in range(numberOfBuckets):
            f = open(".//%s-%02i" % (bucketName,bNum+1),'w')
            for item in buckets[bNum]:
                f.write(item)
            f.close()
