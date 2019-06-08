import json

with open(r'E:\Project\Python\test\master.json', 'r') as fr:
    data = json.load(fr)

HeatMapData = []
Countries = []
Rates = []
Sui_nums = []
GDP = []
Years = []

# 得到所有国家名称的序列
for item in data:
    if item['country'] not in Countries:
        Countries.append(item['country'])


# 得到所有国家所有年份死亡率的序列并处理得到平均值
temp_rate_num = []
temp_gdp = []

# 为每个国家建立一个空列表存储死亡率和年份
for country in Countries:
    temp_rate_num.append([0, 0, 0])
    temp_gdp.append([])
    Years.append([])

# 统计各国家死亡率、死亡人数及GDP
for item in data:
    temp1 = Countries.index(item['country'])  # 国家名所对应的索引
    temp2 = item["suicides/100k pop"]  # 当前年份死亡率
    temp3 = item["suicides_no"]     # 当前年份死亡人数
    temp4 = item["gdp_for_year($)"]     # 当前年份GDP
    temp5 = item["year"]    # 当前年份
    temp_rate_num[temp1][1] += temp2    # 统计所有年份死亡率总和
    temp_rate_num[temp1][2] += temp3    # 统计所有年份死亡人数总和
    if temp5 not in Years[temp1]:       # 若当前年份未统计
        Years[temp1].append(temp5)      # 将当前年份加入已统计年份序列
        temp_rate_num[temp1][0] += 1    # 统计数据中共包含几个年份
        temp_gdp[temp1].append(temp4)   # 统计每个国家当前年份GDP
# print(temp_rate_num)

# 计算平均死亡率及死亡人数总和
for item in temp_rate_num:
    Rates.append(item[1]/item[0])   # 平均死亡率
    Sui_nums.append(item[2])    # 死亡人数总和
# 取GDP中位数
for lst in temp_gdp:
    half = len(lst)//2
    mean = (lst[half] + lst[-half])/2
    GDP.append(mean)

# print(Countries)
# print(Rates)
# print(Sui_nums)
# print(GDP)

# 生成列表文件
for i in range(len(Countries)):
    HeatMapData.append([Countries[i], Rates[i], Sui_nums[i], GDP[i]])
# print(HeatMapData)
with open(r'.\HeatMapData.json', 'w') as fw:
    json.dump(HeatMapData, fw, sort_keys=False, indent=4)

