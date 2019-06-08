import json

with open(r'E:\Project\Python\test\master.json', 'r') as fr:
    data = json.load(fr)

ThemeRiverData = dict()
Countries = []
Sui_nums = [[], []]
Years = [[], []]
years = list(range(1985, 2017))
SEX = ['MALE', 'FEMALE']

# 得到所有国家名称的序列
for item in data:
    if item['country'] not in Countries:
        Countries.append(item['country'])

for country in Countries:
    Sui_nums[0].append([])
    Sui_nums[1].append([])
    Years[0].append([])
    Years[1].append([])

for item in data:
    temp1 = Countries.index(item['country'])  # 国家名所对应的索引
    temp2 = item["suicides_no"]  # 当前年份死亡人数
    temp3 = item["year"]  # 当前年份
    temp4 = item["sex"]     # 性别
    if temp3 not in Years[0][temp1]:
        Sui_nums[0][temp1].append(temp2)
        Years[0][temp1].append(temp3)
    elif temp3 not in Years[1][temp1]:
        Sui_nums[1][temp1].append(temp2)
        Years[1][temp1].append(temp3)
    else:
        if temp4 == "male":
            ind = Years[0][temp1].index(temp3)
            Sui_nums[0][temp1][ind] += temp2
        elif temp4 == "female":
            ind = Years[1][temp1].index(temp3)
            Sui_nums[1][temp1][ind] += temp2

for i in range(len(Countries)):
    count = []
    for sex in range(2):
        for y in years:
            temp = []
            s = str(y)+'/01/01'
            temp.append(s)
            if y not in Years[sex][i]:
                temp.append(None)
            else:
                temp.append(Sui_nums[sex][i][Years[sex][i].index(y)])
            temp.append(SEX[sex])
            count.append(temp)
    ThemeRiverData[Countries[i]] = count
# print(ThemeRiverData)

# 生成列表文件
with open(r'.\ThemeRiverData.json', 'w') as fw:
    json.dump(ThemeRiverData, fw, sort_keys=False, indent=4)
