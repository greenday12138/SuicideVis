#author:yw
'''
任务定义：
    描述：想象一下，当你观察这份数据时，你最想分析哪些问题？

    1.由数据中('suicides/100k pop','HDI for year','gdp_per_capita')三项，对比两个或多个国家或地区在同一年份某一年龄段的自杀率
'''

import numpy as np
import pandas as pd
import gc
'''
import os
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import warnings
'''
import tensorly as tl
from tensorly.decomposition import parafac

data_path=r'C:\Users\lenovo\Data_visualization\SuiciVis\data\master.csv'

dp=pd.read_csv(data_path)
dp.dataframeName='The World Suicides Data.csv'
#dp.head()
'''
#数据清洗

#dp.describe()
#dp.info()
'''数据中'HDI for year'列存在缺失'''
#计算每列缺失数据所占百分比
#dp.isnull().sum()*100/dp.isnull().count()

#去掉缺失数据项
#由于HID一列缺失率高达69.94%，因此我认为不适合简单删除处理
'''
country_list=dp.country.unique()
fill_list=['HDI for year']
for country in country_list:
    dp.loc[dp['country']==country,fill_list]=dp.loc[dp['country']==country,fill_list].interpolate()

dp.dropna(inplace=True)

#dp.isnull().sum()
'''

'''如何处理二值属性'''
'''如何区分离群点'''
'''如何分别对离群点和正常值进行分析'''

dp.groupby(['country','age']).suicides_no.sum().nlargest(10).plot(kind='barh')
dp.groupby(['country','age'])['suicides/100k pop'].sum().nlargest(10).plot(kind='barh')

ax=sns.catplot(x='sex',y='suicides_no',col='age',data=dp,estimator=np.median,height=4,aspect=.7,kind='bar')
bx=sns.catplot(x='sex',y='suicides/100k pop',col='age',data=dp,estimator=np.median,height=4,aspect=.7,kind='bar')

dp.groupby(by=['age','sex'])['suicides_no'].sum().unstack().reset_index().melt(id_vars='age')
dp.groupby(by=['age','sex'])['suicides_no','suicides/100k pop'].sum().unstack().reset_index().melt(id_vars='age')

sns.catplot('age','suicides_no',hue='sex',col='year',data=dp,kind='bar',col_wrap=3,estimator=np.sum)

sns.set(style='darkgrid')
g=sns.FacetGrid(dp,row='sex',col='age',margin_titles=True)
g.map(plt.scatter,'suicides_no','population',edgecolor='w')

sns.set(style='darkgrid')
g=sns.FacetGrid(dp,row='year',col='age',hue='sex',margin_titles=True)
g.map(plt.scatter,'suicides_no','population',edgecolor='w').add_legend()
'''

#构建张量

'''自定义函数区'''
def List2Dict(list):
    '''
    :param list:
    :return:
    '''
    temp_dict=dict()
    count=0
    for i in list:
        temp_dict[i]=count
        count+=1
    return temp_dict

def ShapeofIndex(index):
    '''
    :param index:list
    :return:
    '''
    shape=[]
    for i in index:
        shape.append(len(i))
    return tuple(shape)

def IndexinTensor(new_index,temp_index):
    index=[]
    count=0
    for i in temp_index:
        index.append(new_index[count][i])
        count+=1
    return tuple(index)

def DataFrame2Ndarray(df,index,drop_columns_list=None):
    '''
    :param df:type:DataFrame
    :param index: type:list
    :param drop_columns_list: type:list
    :return:
    '''
    df=df.drop(drop_columns_list,axis=1)
    new_index=[]
    for i in index:
        new_index.append(list(set(df[i].values)))
    count=0
    df=df.set_index(index)
    shape=ShapeofIndex(new_index)
    for i in new_index:
        j=0
        new_index[count]=List2Dict(i)
        #new_index列表每一个元素都是一个维度，如country维，year维
        count+=1
    del count
    tensor=np.zeros(shape=shape)
    #tensor=tensor.astype(list)
    for i in range(df.values.shape[0]):

        temp=df.values[i,:]
        temp_index=df.index[i]
        tensor[IndexinTensor(new_index,temp_index)]=temp
    return tensor

#kernel function
DataFrame2Ndarray(dp,['country','year','age','sex'],['suicides_no', 'country-year', 'generation'])#此句报"Memory Error"


