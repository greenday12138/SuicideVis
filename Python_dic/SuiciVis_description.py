#author:yw



import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#源数据路径
data_path=r'C:\Users\lenovo\Data_visualization\SuiciVis\data\master.csv'
#结果路径
file_path=r'C:\Users\lenovo\Data_visualization\SuiciVis\data\result.csv'

df=pd.read_csv(data_path)

#数据查看
#df['year'].value_counts()

'''
自定义函数分区
'''
#平均年龄函数
def mean_age(l):
    age=[]
    for i in l:
        if i=='75+ years':
            age.append(75)
        else:
            part1=i.partition('-')
            part2=part1[2].partition('years')
            age.append((int(part1[0])+int(part2[0]))/2)
    return age

#两列表比较返回匹配列表
def returnMatches(a,b):
    return list(set(a).intersection(set(b)))

#两列表比较返回匹配元素索引列表
def returnMatchesIndex(a,b):
    return [i for i,x in enumerate(a) if x==b]

#特殊tuple函数
def SpecialTuple(a,b):
    a,b=list(a),list(b)
    temp=[]
    if len(a)==len(b):
        for i in range(len(a)):
            t=tuple([a[i],b[i]])
            temp.append(t)
    return temp




'''
对于单个国家来说，需要描述的数据：
第i年死亡女性平均年龄
第i年死亡男性平均年龄
第i年死亡人群平均年龄
第i年死亡女性年龄众数
第i年死亡男性年龄众数
第i年死亡人群年龄众数

所有年份死亡女性平均年龄
所有年份死亡男性平均年龄
所有年份死亡人群平均年龄
所有年份死亡女性年龄众数
所有年份死亡男性年龄众数
所有年份死亡人群年龄众数
'''

'''
变量及函数命名规则

结果变量：
    以字典格式存储，键为标签名，值为对应数值
    #键：标签名。例，在dict对象country_year_sex_mean,country_year_sex_mode中一标签为('Uzbekistan2014','male')
    #值：标签对应的mean值/mode值

变量名：
    1.均为小写
    2.名称即表明键值。如某一键值为('Albania1987','female')的平均值dict对象，其名称为country_year_sex_mean
    
函数名：
    1.单词首字母大写。如CountryYearSexMode()
    2.一般与变量名一一对应。如计算country_year_sex_mode的函数即名为CountryYearSexMode()
'''



df['mean_age']=mean_age(df['age'])
#分组
df1=df.groupby(['country-year','sex'])#以('country-year','sex')作为索引
df2=df.groupby('country-year')
df3=df.groupby(['country','sex'])
df4=df.groupby('country')



'''所有国家第i年男性/女性死亡平均年龄'''
#变量名country_year_sex_mean表示标签对国家、年份、性别做了区分
country_year_sex_mean=dict()
temp1=list(df1['mean_age'])
temp2=list(df1['suicides_no'])

def MeanAge( mean_age, no,mean):
    for i in range(len(mean_age)):
        if np.sum(list(no[i][1])):
            mean[mean_age[i][0]]=np.average(list(mean_age[i][1]),axis=0,weights=list(no[i][1]))
        else:
            mean[mean_age[i][0]]=None
    return

MeanAge(temp1,temp2,country_year_sex_mean)

'''所有国家第i年死亡人群平均年龄'''
country_year_mean=dict()
temp1=list(df2['mean_age'])
temp2=list(df2['suicides_no'])

MeanAge(temp1,temp2,country_year_mean)

'''所有国家所有年份死亡男性/女性平均年龄'''

country_sex_mean=dict()
temp1=list(df3['mean_age'])
temp2=list(df3['suicides_no'])
MeanAge(temp1,temp2,country_sex_mean)

'''所有国家所有年份死亡人群平均年龄'''

country_mean=dict()
temp1=list(df4['mean_age'])
temp2=list(df4['suicides_no'])
MeanAge(temp1,temp2,country_mean)

'''所有国家第i年男性/女性死亡平均年龄众数'''
country_year_sex_mode={}
r=df1['suicides/100k pop'].agg([('max','max')])

def CountryYearSexMode(r,mode):
    for i in range(len( r['max'])):
        index3,index4=np.arange(len(list(df['suicides/100k pop']))),np.arange(len(SpecialTuple(df['country-year'],df['sex'])))
        temp3,temp4=index3[list(df['suicides/100k pop'])==r['max'][i]],returnMatchesIndex(SpecialTuple(df['country-year'],df['sex']),r['max'].index[i])
        temp5=returnMatches(temp3,temp4)
        mode[r['max'].index[i]]=df['mean_age'][temp5[0]]
    return

CountryYearSexMode(r,country_year_sex_mode)

'''所有国家第i年死亡人群平均年龄众数'''
country_year_mode=dict()
r=df2['suicides/100k pop'].agg([('max','max')])

def CountryYearMode(r,mode):
    for i in range(len(r['max'])):
        index3,index4=np.arange(len(list(df['suicides/100k pop']))),np.arange(len(tuple(df['country-year'])))
        temp3,temp4=index3[list(df['suicides/100k pop'])==r['max'][i]],returnMatchesIndex(tuple(df['country-year']),r['max'].index[i])
        temp5=returnMatches(temp3,temp4)
        mode[r['max'].index[i]]=df['mean_age'][temp5[0]]
    return

CountryYearMode(r,country_year_mode)

'''所有国家所有年份死亡男性/女性众数'''
country_sex_mode=dict()
r=df3['suicides/100k pop'].agg([('max','max')])

def CountrySexMode(r,mode):
    for i in range(len(r['max'])):
        index3,index4=np.arange(len(list(df['suicides/100k pop']))),np.arange(len(SpecialTuple(df['country'],df['sex'])))
        temp3,temp4=index3[list(df['suicides/100k pop'])==r['max'][i]],returnMatchesIndex(SpecialTuple(df['country'],df['sex']),r['max'].index[i])
        temp5=returnMatches(temp3,temp4)
        mode[r['max'].index[i]]=df['mean_age'][temp5[0]]
    return

CountrySexMode(r,country_sex_mode)

'''所有国家所有年份死亡人群众数'''

country_mode=dict()
r=df4['suicides/100k pop'].agg([('max','max')])

def CountryMode(r,mode):
    for i in range(len(r['max'])):
        index3,index4=np.arange(len(list(df['suicides/100k pop']))),np.arange(len(tuple(df['country'])))
        temp3,temp4=index3[list(df['suicides/100k pop'])==r['max'][i]],returnMatchesIndex(tuple(df['country']),r['max'].index[i])
        temp5=returnMatches(temp3,temp4)
        mode[r['max'].index[i]]=df['mean_age'][temp5[0]]
    return

CountryMode(r,country_mode)
