import json

with open(r'E:\Project\Python\test\master.json', 'r') as fr:
    data = json.load(fr)

MultiLineData = []
Countries = []
Sui_nums = []
Years = []
years = list(range(1985, 2017))

# 得到所有国家名称的序列
for item in data:
    if item['country'] not in Countries:
        Countries.append(item['country'])

for country in Countries:
    Sui_nums.append([])
    Years.append([])

for item in data:
    temp1 = Countries.index(item['country'])  # 国家名所对应的索引
    temp2 = item["suicides_no"]  # 当前年份死亡人数
    temp3 = item["year"]  # 当前年份
    if temp3 not in Years[temp1]:
        Sui_nums[temp1].append(temp2)
        Years[temp1].append(temp3)
    else:
        ind = Years[temp1].index(temp3)
        Sui_nums[temp1][ind] += temp2

for i in range(len(Countries)):
    MultiLineData.append([])
    MultiLineData[i].append(Countries[i])
    for y in years:
        if y not in Years[i]:
            MultiLineData[i].append(None)
        else:
            MultiLineData[i].append(Sui_nums[i][Years[i].index(y)])

# print(MultiLineData)
# 生成列表文件
with open(r'.\MultiLineData.json', 'w') as fw:
    json.dump(MultiLineData, fw, sort_keys=False, indent=4)
