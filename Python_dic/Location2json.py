import pandas as pd
import json

# 读取master.csv中的国家名称
df = pd.read_csv('.\\master.csv', usecols=['country']).values.tolist()
for i in range(len(df)):
    df[i] = df[i][0]
df = list(set(df))
df.sort()

# 读取CountryLocation.csv中的国家经纬度数据
f = open('CountryLocation.csv', 'r')
raw = []
lst = f.readlines()
for i in range(len(lst)):
    raw.append(lst[i].replace('\n', '').split('\t'))

# 生成字典目录
location = []
for i in df:
    temp_dic = dict()
    temp_dic['country'] = i
    for items in raw:
        for values in items:
            if values == i:
                temp_dic['longitude'] = float(items[2])
                temp_dic['latitude'] = float(items[3])
    location.append(temp_dic)

#     生成json文件
fw = open(".\\CountryLocation.json", "w")  # 打开json文件
json.dump(location, fw, sort_keys=False, indent=4)
f.close()
fw.close()
