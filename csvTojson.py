from functools import reduce
import json

path_csv = ".\\master.csv"
path_json = ".\\master.json"
fo = open(path_csv, "r")  # 打开csv文件
raw = []


def str2float(s):
    lst = s.split('.')
    return reduce(lambda x, y: y+x*10, map(int, lst[0]))+reduce(lambda x, y: y+x*10, map(int, lst[1]))/10**len(lst[1])


for line in fo.readlines():
    line = line.replace('\n', '')
    raw.append(line.split(','))
fo.close()
for i in range(1, len(raw)):
    for j in range(10, 10+len(raw[i][10:-2:])):
        raw[i][j] = raw[i][j-1]+raw[i][j]
    for j in raw[i][9:-3]:
        del raw[i][9]
    raw[i][9] = raw[i][9].replace('"', '')
columns = raw[0]
columns[0] = 'country'
del raw[0]
for i in range(len(raw)):
    for j in [1, 4, 5, 9, 10]:
        if raw[i][j] == '':
            raw[i][j] = 0
        else:
            raw[i][j] = int(raw[i][j])
    for k in [6, 8]:
        if raw[i][k] == '' or raw[i][k] == '0':
            raw[i][k] = 0
        elif '.' in raw[i][k]:
            raw[i][k] = str2float(raw[i][k])
        else:
            raw[i][k] = float(int(raw[i][k]))
fw = open(".\\master.json", "w")  # 打开json文件
for i in range(len(raw)):  # 遍历文件的每一行内容，除了列名
    raw[i] = dict(zip(columns, raw[i]))  # columns为列名，所以为key,raw[i]为value,
    # zip()是一个内置函数，将两个长度相同的列表组合成一个关系对
json.dump(raw, fw, sort_keys=False, indent=4)
