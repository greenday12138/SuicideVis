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
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
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
#https://scikit-learn.org/stable/modules/clustering.html#optics

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
mode0_feature_factors,mode0_feature_erors=tl.decomposition.parafac(mode0_tensor,rank=rank,return_errors=True)
mode0_feature_reconstructed_tensor=tl.kruskal_to_tensor(mode0_feature_factors)
#该句注意除数不能为0，即要运行该句必须进行数据清洗
#mode0_feature_deviation=(mode0_tensor-mode0_feature_reconstructed_tensor)/mode0_tensor
mode0_feature_country=np.matmul(mode0_feature_factors[0],np.matrix.transpose(tl.tenalg.khatri_rao([mode0_feature_factors[3],mode0_feature_factors[2],mode0_feature_factors[1]])))

#查看降维后数据与原数据的差异
#以下语句均需自行在python console进行运行
#mode0_feature_factors[3]
#tensor_index[3]
#sex维度上男性自杀人数远远高于女性，符合常识，因此认为降维得到的factors比较准确地概括了数据本身的特征

mode0_feature_deviation=mode0_tensor-mode0_feature_reconstructed_tensor
mode0_feature_country_deviation_quartile_25=np.percentile(mode0_feature_deviation,q=25,axis=(1,2,3),interpolation='linear')
mode0_feature_country_deviation_quartile_50=np.percentile(mode0_feature_deviation,q=50,axis=(1,2,3),interpolation='linear')
mode0_feature_country_deviation_quartile_75=np.percentile(mode0_feature_deviation,q=75,axis=(1,2,3),interpolation='linear')

#K-means clustering
n_clusters=10
kmeans=sklearn.cluster.KMeans(n_clusters=n_clusters,random_state=0).fit(mode0_feature_country)

'''
mode0_country_partition_labels=dict()
for i in range(len(kmeans.labels_)):
    mode0_country_partition_labels[tensor_index[0].get() == i] = i
'''


#Hierarchical clustering

#different linkage strategies
#https://scikit-learn.org/stable/auto_examples/cluster/plot_digits_linkage.html#sphx-glr-auto-examples-cluster-plot-digits-linkage-py
clustering=[]
mode0_feature_factors_red=sklearn.manifold.SpectralEmbedding(n_components=2).fit_transform(mode0_feature_country)
for linkage in ('ward','average','complete','single'):
    clustering.append(cluster.AgglomerativeClustering(linkage=linkage,n_clusters=10))
    clustering[len(clustering)-1].fit(mode0_feature_factors_red)
#此句须在python console自行运行
#clustering[0].labels_


#adding connectivity constraints
#https://scikit-learn.org/stable/auto_examples/cluster/plot_ward_structured_vs_unstructured.html#sphx-glr-auto-examples-cluster-plot-ward-structured-vs-unstructured-py

ward=[]
ward.append(AgglomerativeClustering(n_clusters=10,linkage='ward').fit(mode0_feature_country))
#此句须在python console自行运行
#ward[0].labels_
from sklearn.neighbors import kneighbors_graph
connectivity=kneighbors_graph(mode0_feature_country,n_neighbors=10,include_self=False)
ward.append(AgglomerativeClustering(n_clusters=10,connectivity=connectivity,linkage='ward').fit(mode0_feature_country))
#ward[1].labels_

#DBSCAN
#https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
#Key Parameters:eps, min_samples,
from sklearn.cluster import DBSCAN

db=DBSCAN(eps=25,min_samples=3).fit(mode0_feature_country)
#此句须在python console自行运行
#db.labels_
#sum(db.labels_==-1)#outliers的数量
#np.max(db.labels_)+1#聚类算法生成的簇的数量
#labels_中不同正数代表不同类别，-1表示为outliers
'''参数选择'''

#OPTICS
#https://scikit-learn.org/stable/auto_examples/cluster/plot_optics.html#sphx-glr-auto-examples-cluster-plot-optics-py
#Key Parameters:min_samples,
from sklearn.cluster import OPTICS,cluster_optics_dbscan

clust=OPTICS(min_samples=5,xi=.05,min_cluster_size=0.05)
clust.fit(mode0_feature_country)
reachability=clust.reachability_[clust.ordering_]
labels=clust.labels_[clust.ordering_]
#labels中不同正数代表不同类别，-1表示为outliers
