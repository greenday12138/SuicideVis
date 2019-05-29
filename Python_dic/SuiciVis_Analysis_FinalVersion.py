#author:yw
'''
Figure:Fig.1(TPFlow: Progressive Partition and Multidimensional Pattern Extraction for Large-Scale Spatio-Temporal Data Analysis)

1.确定所要分析的元素
    与Fig.1不同的是，Fig.1所要分析的只有sale这一元素，而我们可以在element_list=['suicides_no','population','suicides/100k pop','HDI for year','gdp_per_capita ($)']中选取任一元素进行分析
2.a区域--overview clustering tree
    Mode:Country
    Element:suicides_no
    1）颜色
    Figure区域：a1,a2
        由element0_feature_country_deviation_mean每一项的大小(该数据项未经normalization)决定
        由论文中
        When analysts hover over a node, a menu (a4 in Fig. 1 a?) will pop up displaying different options. The options include the dimension to perform partition on, the number of clusters
        to create and the clustering algorithm to use
        知，若用户选取mode为year或其他，则颜色应当由对应element0_feature_year_deviation_mean每一项的大小决定
    2)Data Partition
    Figure区域：a1,a2
        a) 聚类算法(基于原数据降维后的特征矩阵element0_feature_country)
            例：
                kmeans.labels_
                    kmeans.labels.shape:(101,)
                    每一行的数值即为对应簇编号
            各算法介绍参见https://scikit-learn.org/stable/modules/clustering.html#optics
            不同算法需要用户提供的各种参数及返回值如下：
                Kmeans:
                    Parameters:
                        cluster:string,所选用聚类算法名称，如kmeans算法对应的cluster即为"kmeans"
                        n_clusters:integer,产生聚类的个数
                    Return:
                        {cluster:(n_clusters,kmeans_labels)}
                        cluster:string,'kmeans'

                        n_clusters:integer,产生聚类个数
                        kmeans_labels:list,每个值对应该国家所属簇

                Hierarchical clustering:
                    Parameters:
                        cluster:string,'hierarchical'
                        n_clusters:integer,产生聚类的个数
                        linkage: ('ward','average','complete','single')中之一或其中几个
                        connectivity:bool
                    Return:
                        {cluster:(n_clusters,hierarchical_labels)}；若linkage为多个，则返回值为元素为{cluster:(n_clusters,hierarchical_labels)}的tuple（元组）
                        cluster:string,'hierarchical'
                        n_clusters:integer,产生聚类个数
                        hierarchical_labels:list,每个值对应该国家所属簇

                DBSCAN:
                    Parameters:
                        cluster:string,'DBSCAN'
                        eps:正数。Maximun radius of th neighborhood.Any sample that is not a core sample, and is at least eps in distance from any core sample, is considered an outlier by the algorithm.
                        min_samples:integer
                    Return:
                        {cluster:(n_clusters,n_outliers,db_labels)}
                        cluster:string,'DBSCAN'
                        n_clusters:integer,产生聚类的个数
                        n_outliers:integer,outliers的个数
                        db_labels:list,每个正整数值对应该国家所属簇；-1表示该国家被归入了outliers

                OPTICS
                    Parameters:
                        cluster:string,'OPTICS'
                        min_samples:integer
                        xi:
                        min_cluster_size:
                    Return:
                        {cluster:(n_clusters,n_outliers,op_labels)}
                        cluster:string,'OPTICS'
                        n_clusters:integer,产生聚类的个数
                        n_outliers:integer,outliers的个数
                        op_labels:list,每个正整数值对应该国家所属簇；-1表示该国家被归入了outliers

        b) 用户自行划分
            操作一：在country维度上用户自行划分几个类
                Figure: Fig.3(TPFlow: Progressive Partition and Multidimensional Pattern Extraction for Large-Scale Spatio-Temporal Data Analysis)
                相当于将element0_tensor分为几个sub-tensor
    3)
    Figure区域：a3
    由于该区域是由用户自行划分出来的，后端可根据用户操作划分sub-tensor
    参见论文Fig. 6d区域的解释

3.b,c区域
    Mode:Country
    Element:suicides_no

    bar_line:
        Parameters:
            selected_modes:tuple,例：('China','Iceland','German')
            axis:tuple,例：(1985,2006)
        Return:
            bar_charts:list,其元素为{selected_mode:value_list}
            selected_mode:string,例，'China'
            value_list:
4.d区域
    Mode:Country
    Element: suicides_no

    bubble:
        Parameters:

            selected_modes:tuple,例：('China','Iceland','German')
        Return:


'''

import numpy as np
import pandas as pd
import sklearn
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn import cluster
from numpy import matrix
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS,cluster_optics_dbscan

'''数据读取'''

#原数据存储路径
data_path=r'C:\Users\lenovo\Data_visualization\SuiciVis\data\master.csv'

#读取原数据
dp=pd.read_csv(data_path)
dp.dataframeName='The World Suicides Data.csv'

'''数据清洗'''

#dp.describe()
#dp.info()
#数据中'HDI for year'列存在缺失
#计算每列缺失数据所占百分比
#dp.isnull().sum()*100/dp.isnull().count()



'''构建张量'''

#张量的四个维度
index_list=['country','year','age','sex']
#张量元素的种类
element_list=['suicides_no','population','suicides/100k pop','HDI for year','gdp_per_capita ($)']
#element_list=['suicides_no','population','suicides/100k pop','HDI for year',' gdp_for_year ($)','gdp_per_capita ($)']
#倒数第二个项即' gdp_for_year ($)'读取有问题

rank=1

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

def DataFrame2Ndarray(df,index,element,return_index=False):
    '''
    :param df:type:DataFrame
    :param index: type:list
    :param element:string
    :return:
    '''
    new_index=[]
    for i in index:
        new_index.append(list(set(df[i].values)))
        new_index[len(new_index)-1].sort()
    df=df.set_index(index)
    shape=ShapeofIndex(new_index)
    for i in range(len(new_index)):
        j=0
        new_index[i]=List2Dict(new_index[i])
        #new_index列表每一个元素都是一个维度，如country维，year维
    tensor=np.zeros(shape=shape)
    for i in range(df.values.shape[0]):
        temp=df[element].values[i]
        temp_index=df[element].index[i]
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

#[temp,tensor_index]=DataFrame2Ndarray(dp,index_list,element=element_list[0],return_index=True)
#del temp

class tensor():
    #类属性
    index_list=['country','year','age','sex']
    rank=1
    #mode_matrix=None
    def __init__(self,df,element):
        self.element=element
        #self.index_list=index_list
        self.tensor,self.tensor_index=DataFrame2Ndarray(df,index_list,element,True)
        self.feature_factors=tl.decomposition.parafac(self.tensor,rank=rank)
        self.reconstructed_tensor=tl.kruskal_to_tensor(self.feature_factors)
        self.deviation=self.tensor-self.reconstructed_tensor

    def partition(self,list,select=True):
        pass

    def choose_mode(self,mode):
        #获得用户所选mode在index_list中的索引
        index=index_list.index(mode)
        #temp=self.feature_factors
        temp=self.feature_factors.copy()
        temp.pop(index)
        temp=np.matmul(self.feature_factors[index],np.matrix.transpose(tl.tenalg.khatri_rao(temp)))
        return temp#返回值待定

    '''聚类算法'''
    def kmeans(self,n_clusters,mode):
        kmeans=sklearn.cluster.KMeans(n_clusters=n_clusters,random_state=0).fit(self.choose_mode(mode))
        return kmeans.labels_#返回值待定

    def hierarchical(self,mode,n_clusters,linkage,connectivity=False,n_neighbors=False):
        temp=self.choose_mode(mode)
        if connectivity==True:
            #https://scikit-learn.org/stable/auto_examples/cluster/plot_ward_structured_vs_unstructured.html#sphx-glr-auto-examples-cluster-plot-ward-structured-vs-unstructured-py
            connectivity=kneighbors_graph(temp,n_neighbors=n_neighbors,include_self=False)
            clustering=AgglomerativeClustering(n_clusters=10,connectivity=connectivity,linkage=linkage).fit(temp)
        else:
            #https://scikit-learn.org/stable/auto_examples/cluster/plot_digits_linkage.html#sphx-glr-auto-examples-cluster-plot-digits-linkage-py
            factors_red=sklearn.manifold.SpectralEmbedding(n_components=2).fit_transform(temp)
            clustering=cluster.AgglomerativeClustering(linkage=linkage,n_clusters=n_clusters)
            clustering.fit(factors_red)
        return clustering.labels_

    def DBSCAN(self):
        pass
    def OPTICS(self):
        pass
    def Clustering(self,cluster,mode):
        pass

temp=tensor(dp,element_list[0])
mode='country'
#len(temp.choose_mode(mode))
#element0_feature_country1=tensor.choose_mode(temp,mode=mode)

n_clusters=10
#temp.kmeans(n_clusters=n_clusters,mode=mode)
temp.hierarchical(mode=mode,n_clusters=n_clusters,linkage='ward',connectivity=True,n_neighbors=10)
class a:
    a=1

    def change(cls):
        cls.a=2

b=a()
b.a
b.change()
b.a
c=a()
c.a

