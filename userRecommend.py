# userRecommend.py
from math import sqrt

# 如果数据存在“分数膨胀”问题，就使用皮尔逊相关系数。
# 如果数据比较“密集”，变量之间基本都存在公有值，且这些距离数据是非常重要的，那就使用欧几里得或曼哈顿距离。
# 如果数据是稀疏的，则使用余弦相似度。\
# 协同过滤：利用他人的喜好来进行推荐，也就是说，是大家一起产生的推荐。

users = {"Angelica": {"Blues Traveler": 3.5, "Broken Bells": 2.0, "Norah Jones": 4.5, "Phoenix": 5.0, "Slightly Stoopid": 1.5, "The Strokes": 2.5, "Vampire Weekend": 2.0},
         "Bill":{"Blues Traveler": 2.0, "Broken Bells": 3.5, "Deadmau5": 4.0, "Phoenix": 2.0, "Slightly Stoopid": 3.5, "Vampire Weekend": 3.0},
         "Chan": {"Blues Traveler": 5.0, "Broken Bells": 1.0, "Deadmau5": 1.0, "Norah Jones": 3.0, "Phoenix": 5, "Slightly Stoopid": 1.0},
         "Dan": {"Blues Traveler": 3.0, "Broken Bells": 4.0, "Deadmau5": 4.5, "Phoenix": 3.0, "Slightly Stoopid": 4.5, "The Strokes": 4.0, "Vampire Weekend": 2.0},
         "Hailey": {"Broken Bells": 4.0, "Deadmau5": 1.0, "Norah Jones": 4.0, "The Strokes": 4.0, "Vampire Weekend": 1.0},
         "Jordyn":  {"Broken Bells": 4.5, "Deadmau5": 4.0, "Norah Jones": 5.0, "Phoenix": 5.0, "Slightly Stoopid": 4.5, "The Strokes": 4.0, "Vampire Weekend": 4.0},
         "Sam": {"Blues Traveler": 5.0, "Broken Bells": 2.0, "Norah Jones": 3.0, "Phoenix": 5.0, "Slightly Stoopid": 4.0, "The Strokes": 5.0},
         "Veronica": {"Blues Traveler": 3.0, "Norah Jones": 5.0, "Phoenix": 4.0, "Slightly Stoopid": 2.5, "The Strokes": 3.0}
        }

# 计算曼哈顿距离
def manhattan(rating1,rating2):
	distance = 0
	for key in rating1:
		if key in rating2:
			distance += abs(rating1[key] - rating2[key])

	return distance

# 闵可夫斯基距离
def minkowski(rating1,rating2,r=2):
	distance = 0
	for key in rating1:
		if key in rating2:
			distance += pow(abs(rating1[key]-rating2[key]),r)
	# print(pow(distance,1/r))
	return pow(distance,1/r)

# 皮尔逊相关系数
# 皮尔逊相关系数可以有效解决分数膨胀问题，即评分趋高，范围窄，分布集中(过度悲观和乐观两种情况)。
def pearson(rating1,rating2):
	distance = 0
	sum_x = 0
	sum_y =0
	sum_xy =0
	sum_x2 =0
	sum_y2=0
	n = 0
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
	# 分母非0判断
	denominator = sqrt(sum_x2 - pow(sum_x,2)/n) * sqrt(sum_y2 - pow(sum_y,2)/n)
	if denominator == 0:
		return 0
	distance = (sum_xy - (sum_x * sum_y)/n) / denominator
	return distance

# 余弦相似度在文本挖掘中应用得较多，在协同过滤中也会使用到。
def Cosine(rating1,rating2):
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
    if sum_x2 ==0 || sum_y2 ==0 || sum_xy ==0:
        return 0
    else:
        distance = sum_xy / sqrt(sum_x2) * sqrt(sum_y2)
    return distance

# 计算指定用户与所有其他用户之间的相似度
def computeNearestNeighbor(username,users,distanceFunction=manhattan):
	distances = []
	for user in users:
		if username != user:
			distance = distanceFunction(users[username],users[user])
			distances.append((user,distance))
	distances.sort(key=getSecond,reverse=True)
	# print(distances)
	return distances

#  获取元组中的第二个元素
def getSecond(oneTuple):
	return oneTuple[1]

#  对指定用户进行推荐
def recommend(username,users):
	nearest = computeNearestNeighbor(username,users,distanceFunction=minkowski)[-1][0]
	recommendations = []
	neighborRatings = users[nearest]
	for artist  in neighborRatings:
		if artist  not in users[username]:
			# 推荐的乐队名称和对应的最相似用户对该乐队的评分
			recommendations.append((artist,neighborRatings[artist]))
	return sorted(recommendations,key=lambda artistTuple:artistTuple[1],reverse=True)
# 问题：1
# 当用户为Angelica时，没有任何推荐，获取Angelica的相似用户列表可以发现，
# 最相似的用户为Veronica，因为Veronica没有评价的乐队，Angelica也没有进行评价
# 类似问题暂时未处理


if __name__ == '__main__':
	result = recommend("Hailey",users)
	print(pearson(users['Angelica'], users['Bill']))
	print(result)
