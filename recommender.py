# recommender.py
import codecs
import numpy as np
from math import sqrt
import pymysql as mysql


# 测试数据
users = {"Angelica": {"Blues Traveler": 3.5, "Broken Bells": 2.0, "Norah Jones": 4.5, "Phoenix": 5.0, "Slightly Stoopid": 1.5, "The Strokes": 2.5, "Vampire Weekend": 2.0},
         "Bill":{"Blues Traveler": 2.0, "Broken Bells": 3.5, "Deadmau5": 4.0, "Phoenix": 2.0, "Slightly Stoopid": 3.5, "Vampire Weekend": 3.0},
         "Chan": {"Blues Traveler": 5.0, "Broken Bells": 1.0, "Deadmau5": 1.0, "Norah Jones": 3.0, "Phoenix": 5, "Slightly Stoopid": 1.0},
         "Dan": {"Blues Traveler": 3.0, "Broken Bells": 4.0, "Deadmau5": 4.5, "Phoenix": 3.0, "Slightly Stoopid": 4.5, "The Strokes": 4.0, "Vampire Weekend": 2.0},
         "Hailey": {"Broken Bells": 4.0, "Deadmau5": 1.0, "Norah Jones": 4.0, "The Strokes": 4.0, "Vampire Weekend": 1.0},
         "Jordyn":  {"Broken Bells": 4.5, "Deadmau5": 4.0, "Norah Jones": 5.0, "Phoenix": 5.0, "Slightly Stoopid": 4.5, "The Strokes": 4.0, "Vampire Weekend": 4.0},
         "Sam": {"Blues Traveler": 5.0, "Broken Bells": 2.0, "Norah Jones": 3.0, "Phoenix": 5.0, "Slightly Stoopid": 4.0, "The Strokes": 5.0},
         "Veronica": {"Blues Traveler": 3.0, "Norah Jones": 5.0, "Phoenix": 4.0, "Slightly Stoopid": 2.5, "The Strokes": 3.0}
        }

# 测试数据
users2 = {"Amy": {"Taylor Swift": 4, "PSY": 3, "Whitney Houston": 4},
          "Ben": {"Taylor Swift": 5, "PSY": 2},
          "Clara": {"PSY": 3.5, "Whitney Houston": 4},
          "Daisy": {"Taylor Swift": 5, "Whitney Houston": 3}}

# 完整的推荐，通过调用数据库中的数据，完成相应的物品推荐工作
class recommender:
    def __init__(self,data,k=4,metric='consine',n=5):
        """ 初始化推荐模块
        data   训练数据
        k      K邻近算法中的值
        metric 使用何种距离计算方式
        n      推荐结果的数量
        """
        self.k = k
        self.n = n
        self.username2id = {}
        self.userid2name = {}
        self.productid2name = {}
        # 将距离计算方式保存下来
        self.metric = metric
        if self.metric == 'pearson':
            self.fn = self.pearson
        else:
            self.fn = self.cosine

        # 如果data的数据类型为字典类型，则保存下来
        if type(data).__name__ == 'dict':
            self.data = data
        # 表示同时评价过物品item1,item1的用户数
        self.frequencies = {}
        # 表示物品item 与item1之间的评分差值
        self.deviations = {}

    #  获取数据库连接
    def getConnection(self):
        connection = mysql.connect(host="localhost",user="luckydan",password="123",database="book-crossing",cursorclass=mysql.cursors.DictCursor)
        return conection


    def convertProductID2name(self,id):
        """ 通过Id 获取产品名称 """
        if id in self.productid2name:
            return self.productid2name[id]
        else:
            return id

    # 返回该用户评分最高的物品
    def userRatings(self,id,n):
        print("Rating for " + self.username2id[id])
        ratings = self.data[id]
        print(len(ratings))
        ratings = list(ratings.items())
        ratings = [(self.convertProductID2name(k),v) for (k,v) in ratings]
        ratings = ratings[:n]
        for rating in ratings:
            print("%s \t %i" % (rating[0],rating[1]))

    # 加载BX数据集，connection为数据库连接
    def loadBookDB(self):
        connection = mysql.connect(host="localhost",user="luckydan",password="123",database="book-crossing",cursorclass=mysql.cursors.DictCursor)
        self.data = {}
        i = 0
        with connection:
            with connection.cursor() as cursor:
                # 将书籍评分数据放入self.data
                sql1 = "select * from `bx-book-ratings`"
                cursor.execute(sql1)
                result =cursor.fetchall()
                for item in result:
                    i += 1
                    user = item["User-ID"]
                    book = item["ISBN"]
                    rating = int(item["Book-Rating"])
                    if user in self.data:
                        currentRatings = self.data[user]
                    else:
                        currentRatings = {}
                    currentRatings[book] = rating
                    self.data[user] = currentRatings
                # print(self.data)

                 # 将书籍信息存入self.productid2name
                # 包括isbn号、书名、作者等
                sql2 = "select * from `bx-books`;"
                cursor.execute(sql2)
                result =cursor.fetchall()
                for item in result:
                    i +=1
                    isbn =item["ISBN"]
                    title = item["Book-Title"]
                    author = item["Book-Author"]
                    title = title + " by " + author
                    self.productid2name[isbn] = title
                # print(self.productid2name)

                # 将用户信息存入self.userid2name和self.username2id
                sql3 = "select * from `bx-users`;"
                cursor.execute(sql3)
                result =cursor.fetchall()
                for item in result:
                    i +=1
                    userid =item["User-ID"]
                    location = item["Location"]
                    if len(item) > 3:
                        age = item["age"]
                    else:
                        age = 'NULL'
                    if age != 'NULL':
                        value = location + '  (age: ' + age + ')'
                    else:
                        value = location
                    self.userid2name[userid] = value
                    self.username2id[location] = userid
        # print(i)

    # 加载MovieLens数据集数据集
    def loadMovieLens(self,path="E:\\temp\\dataset\\ml-latest-small\\"):
        self.data = {}
        # 加载电影名称相关数据
        movie = codecs.open(path+"movies.csv",'r','ascii','ignore')
        i = 0
        for line in movie:
            i +=1
            if i ==1:continue
            fields = line.split(",")
            movieId = fields[0].strip('"').strip(" ")
            title = fields[1].strip('"').strip(' ')
            # genres = fields[2].strip(" ")
            self.productid2name[movieId] = title
        movie.close()

        # print(self.productid2name)

        # 打开文件,加载电影评价数据
        f = codecs.open(path+"ratings.csv",'r','utf-8')
        i = 0
        for line in f:
            i +=1
            if i==1: continue
            fields = line.split(",")
            userId = fields[0].strip(" ")
            movieId = fields[1].strip(" ")
            rating = float(fields[2].strip(" "))
            if rating > 5:
                print("EXCEEDING ", rating)
            if userId in self.data:
                currentRatings = self.data[userId]
            else:
                currentRatings = {}
            currentRatings[self.convertProductID2name(movieId)] = rating
            self.data[userId] = currentRatings
        f.close()
        # print(self.data)


        # print(self.productid2name)

    # 皮尔逊相关系数
    def pearson(self,rating1,rating2):
        distance = 0
        sum_x = 0
        sum_y =0
        sum_xy =0
        sum_x2 =0
        sum_y2=0
        n = 0
        # print(rating1)
        # print(rating2)
        for key in rating1:
            if key in rating2:
                n +=1
                x = rating1[key]
                y = rating2[key]
                sum_x += x
                sum_y += y
                sum_xy += x * y
                sum_x2 += pow(x,2)
                sum_y2 += pow(y,2)
        # print(n)
        # 分母非0判断
        denominator = sqrt(sum_x2 - pow(sum_x,2)/n) * sqrt(sum_y2 - pow(sum_y,2)/n)
        if denominator == 0:
            return 0
        distance = (sum_xy - (sum_x * sum_y)/n) / denominator
        return distance

    # 余弦相似度，对于稀疏数据集比较适用
    def cosine(self,rating1,rating2):
        distance = 0
        sum_xy= 0
        sum_x2 = 0
        sum_y2 = 0
        for key in rating1:
            if key in rating2:
                x = rating1[key]
                y = rating2[key]
                sum_xy += x * y
                sum_x2 = pow(x,2)
                sum_y2 = pow(y,2)
        if sum_x2 ==0 or sum_y2 ==0 or sum_xy ==0:
            return 0
        else:
            distance = sum_xy / sqrt(sum_x2) * sqrt(sum_y2)
        return distance

    # 修正的余弦相似度
    # def amendConsine(self,rating1,rating2)：


    # 获取近邻用户
    def computeNearestNeighbor(self,username):
        distances = []
        for instance in self.data:
            if instance != username:
                print(username)
                print(self.data[username])
                distance = self.fn(self.data[username],self.data[instance])
                distances.append((instance,distance))
        # 按距离排序，距离近的排在前面
        distances.sort(key=lambda artistTuple: artistTuple[1],
                       reverse=True)
        return distances

    # 基于用户的推荐
    def recommend(self,user):
        # 返回推荐列表
        recommendations = {}
        # 1.获取近邻用户
        nearest = self.computeNearestNeighbor(user)

        # 2.获取用户评价的物品
        userRatings = self.data[user]

        # 3.计算总距离
        totalDistance = 0.0
        for i in range(self.k):
            totalDistance += nearest[i][1]

        if totalDistance == 0:
            totalDistance = 1
        # 4.汇总K近邻用户的评分
        for i in range(self.k):
            # 4.1 计算饼图的每个分片
            weight = nearest[i][1] / totalDistance

            # 获取用户名称
            name = nearest[i][0]

            # 获取用户评分
            neighborRatings = self.data[name]

            # 获得没有评价过的商品
            for artist in neighborRatings:
                if artist not in userRatings:
                    if artist not in recommendations:
                        recommendations[artist] = neighborRatings[artist] * weight
                    else:
                        recommendations[artist] = recommendations[artist] * neighborRatings[artist] * weight

            # 5.开始推荐
            recommendations = list(recommendations.items())

            recommendations = [(self.convertProductID2name(k), v)
                          for (k, v) in recommendations]
            # 排序并返回
            recommendations.sort(key=lambda artistTuple: artistTuple[1],
                                reverse = True)
            # 返回前n个结果
            return recommendations

    # 基于物品的推荐 ---- Slope One 算法的差值计算
    def computeDeviations(self):
        # 获取每位用户的评分数据
        for ratings in self.data.values():
            # 对每个用户的评分项进行处理
            for (item,rating) in ratings.items():
                self.frequencies.setdefault(item,{})
                self.deviations.setdefault(item,{})
                # 再次遍历用户的每个评分项
                for (item1,rating1) in ratings.items():
                    if item != item1:
                    # 将评分的差异保存到变量中
                        self.frequencies[item].setdefault(item1,0)
                        self.deviations[item].setdefault(item1,0.0)
                        self.frequencies[item][item1] += 1
                        self.deviations[item][item1] += rating - rating1

        # 计算物品之间的差值
        for (item,ratings) in self.deviations.items():
            for item2 in ratings:
                ratings[item2] /= self.frequencies[item][item2]


    # 基于物品的推荐 ---- 加权的Slope One算法：推荐逻辑的实现
    def slopeOneRecommendations(self, userRatings):
        recommendations = {}
        frequencies = {}

        # 遍历目标用户的评分项（物品，评分）
        for (userItem,userRating) in userRatings.items():
            # 遍历物品之间的评分差值集合
            for (diffItem,diffRating) in self.deviations.items():
                if diffItem not in userRatings and userItem in self.deviations[diffItem]:
                    freq = self.frequencies[diffItem][userItem]
                    recommendations.setdefault(diffItem,0.0)
                    frequencies.setdefault(diffItem,0)

                    # 分子
                    recommendations[diffItem] += (diffRating[userItem] + userRating) * freq

                    # 分母
                    frequencies[diffItem] += freq

        # 推荐物品以及对应的预测评分值
        recommendations = [(k,v/frequencies[k]) for (k,v) in recommendations.items()]

        # 排序并返回
        recommendations.sort(key= lambda artistTuple:artistTuple[1],reverse=True)
        return recommendations


# connection = mysql.connect(host="localhost",user="luckydan",password="123",database="book-crossing",cursorclass=mysql.cursors.DictCursor)
# users ={}
r = recommender(users)
# # r.loadBookDB()
# r.computeDeviations()
# # print(r.deviations)
# print(r.slopeOneRecommendations(users['Bill']))

r.loadMovieLens()
r.computeDeviations()
# print(r.deviations)
print(r.slopeOneRecommendations(r.data['283']))
