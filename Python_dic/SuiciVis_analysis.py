#author:yw
'''
任务定义：
    描述：想象一下，当你观察这份数据时，你最想分析哪些问题？
    1.由数据中('suicides/100k pop','HDI for year','gdp_per_capita')三项，对比两个或多个国家或地区在同一年份某一年龄段的自杀率
'''
import numpy as np
import pandas as pd
import gc
import psutil
import sklearn
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn import cluster
from numpy import matrix
'''
import os
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import warnings
'''

data_path=r'C:\Users\lenovo\Data_visualization\SuiciVis\data\master.csv'

dp=pd.read_csv(data_path)
dp.dataframeName='The World Suicides Data.csv'
#dp.head()
'''
#数据清洗

#dp.describe()
#dp.info()
#数据中'HDI for year'列存在缺失
#计算每列缺失数据所占百分比
#dp.isnull().sum()*100/dp.isnull().count()

#去掉缺失数据项
#由于HID一列缺失率高达69.94%，因此我认为不适合简单删除处理

country_list=dp.country.unique()
fill_list=['HDI for year']
for country in country_list:
    dp.loc[dp['country']==country,fill_list]=dp.loc[dp['country']==country,fill_list].interpolate()

dp.dropna(inplace=True)

#dp.isnull().sum()


#如何处理二值属性
#如何区分离群点
#如何分别对离群点和正常值进行分析

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

def DataFrame2Ndarray(df,index,mode,return_index=False):
    '''
    :param df:type:DataFrame
    :param index: type:list
    :param mode:string
    :return:
    '''
    new_index=[]
    for i in index:
        new_index.append(list(set(df[i].values)))
    df=df.set_index(index)
    shape=ShapeofIndex(new_index)
    for i in range(len(new_index)):
        j=0
        new_index[i]=List2Dict(new_index[i])
        #new_index列表每一个元素都是一个维度，如country维，year维
    tensor=np.zeros(shape=shape)
    for i in range(df.values.shape[0]):
        temp=df[mode].values[i]
        temp_index=df[mode].index[i]
        tensor[IndexinTensor(new_index,temp_index)]=temp
        '''
        gc.collect()
        mem=psutil.virtual_memory()
        print('Memory available in GB'+str(mem.available/(1024**3)))
        '''
        #del temp
        #del temp_index
    if return_index==True:
        return tensor,new_index
    else:
        return tensor

index_list=['country','year','age','sex']
#mode_list=['suicides_no','population','suicides/100k pop','HDI for year',' gdp_for_year ($)','gdp_per_capita ($)']
#倒数第二个mode项有问题
mode_list=['suicides_no','population','suicides/100k pop','HDI for year','gdp_per_capita ($)']


#number of components
rank=1
'''
mode0_tensor,tensor_index=DataFrame2Ndarray(dp,index_list,mode_list[0],return_index=True)
mode0_factors_errors=tl.decomposition.parafac(mode0_tensor,rank=rank,return_errors=True)
mode0_reconstructed_tensor=tl.kruskal_to_tensor(mode0_factors_errors[0])

mode2_tensor=DataFrame2Ndarray(dp,index_list,mode_list[2])
mode2_factors_errors=tl.decomposition.parafac(mode2_tensor,rank=rank,return_errors=True)
mode2_reconstructed_tensor=tl.kruskal_to_tensor(mode2_factors_errors[0])
#mode2_deviation_percentage=(mode2_tensor-mode2_reconstructed_tensor)/mode2_tensor
#running time warning
'''

#聚类算法

#len(mode2_factors_errors[0][0])

class mode_feature:
    def __init__(self,mode=None,tensor=None):
        self.mode=mode
        self.tensor=tensor
        if tensor.all==None:
            self.factors=None
            self.reconstructed_tensor=None
        else:
            self.factors,self.errors=tl.decomposition.parafac(self.tensor,rank=1,return_errors=True)
            self.reconstructed_tensor=tl.kruskal_to_tensor(self.factors)

    def create_mode_features(self,mode_list,tensor_list=None,func=DataFrame2Ndarray,df=None,index_list=None):
        mode_features=[]
        if tensor_list!=None:
            for i in range(len(mode_list)):
                mode_features.append(mode_feature(mode_list[i],tensor_list[i]))
            return mode_features
        else:
            for i in range(len(mode_list)):
                mode_features.append(mode_feature(mode=i,tensor=DataFrame2Ndarray(df,index_list,mode=mode_list[i])))
            return mode_features

    #def factors_cluster(self):

'''
mode0_feature=mode_feature(0,DataFrame2Ndarray(dp,index_list,mode_list[0]))
mode1_feature=mode_feature(1,DataFrame2Ndarray(dp,index_list,mode_list[1]))
mode2_feature=mode_feature(2,DataFrame2Ndarray(dp,index_list,mode_list[2]))
mode3_feature=mode_feature(3,DataFrame2Ndarray(dp,index_list,mode_list[3]))
'''
#mode_features=mode_feature(mode=None,tensor=None)
#mode_features=mode_features.create_mode_features(mode_list=mode_list,func=DataFrame2Ndarray,df=dp,index_list=index_list)
mode0_feature=mode_feature(0,DataFrame2Ndarray(dp,index_list,mode_list[0]))
[temp,tensor_index]=DataFrame2Ndarray(dp,index_list,mode=mode_list[0],return_index=True)
del temp

#维度'country'上的特征矩阵

mode0_tensor=DataFrame2Ndarray(dp,index_list,mode_list[0])
mode0_feature_factors=tl.decomposition.parafac(mode0_tensor,rank=rank)
mode0_feature_country=np.matmul(mode0_feature_factors[0],np.matrix.transpose(tl.tenalg.khatri_rao([mode0_feature_factors[3],mode0_feature_factors[2],mode0_feature_factors[1]])))

#K-means聚类
n_clusters=3
kmeans=sklearn.cluster.KMeans(n_clusters=n_clusters,random_state=0).fit(mode0_feature_country)
'''
mode0_country_partition_labels=dict()
for i in range(len(kmeans.labels_)):
    mode0_country_partition_labels[tensor_index[0].get() == i] = i
'''


