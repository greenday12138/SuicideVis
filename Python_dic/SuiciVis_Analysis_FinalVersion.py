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
            selected_modes:tuple,例：('China','Iceland','Germany')
            axis:tuple,,表示坐标轴表示的区间，例：(1985,2006)即表示展示1985至2006年的数据
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS,cluster_optics_dbscan
#import matplotlib.pyplot as plt

'''数据读取'''

#原数据存储路径
data_path=r'C:\Users\lenovo\Data_visualization\SuiciVis\data\master.csv'

#读取原数据
dp=pd.read_csv(data_path)
dp.dataframeName='The World Suicides Data.csv'

'''数据清洗'''

#dp.describe()
#dp.info()

#计算每列缺失数据所占百分比
#dp.isnull().sum()*100/dp.isnull().count()
#数据中'HDI for year'列缺失率达69%

'''构建张量'''

#张量的四个维度
index_list=['country','year','age','sex']
#张量元素的种类
element_list=['suicides_no','population','suicides/100k pop','gdp_per_capita ($)']
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
    if return_index==True:
        return tensor,new_index
    else:
        return tensor

def return_tensor_inex(key,search_list,element):
    #key,search_list,index_list,tensor_list
    index=index_list.index(key)
    return_list=[]
    temp,tensor_index=DataFrame2Ndarray(dp,index_list,element,True)
    del temp
    for i in search_list:
        return_list.append(tensor_index[index][i])
    return return_list
#return_tensor_inex('country',['Germany','Albania'],'suicides_no')


def partition(tensor,dict,element,select=True):
    '''
    :param dict:
    :param select:
    :return:
    '''
    '''
    {'country':['China','Germany'],'year':[1985,2001,2003]}
    '''
    #temp_tensor=np.copy(self.tensor)
    if select==False:
        flag=0
        for i in range(len(index_list)):
            if type(dict.get(index_list[i]))!= type(None):
                flag+=1
                select_index_list=return_tensor_inex(index_list[i],dict[index_list[i]],element)
                if flag==1:
                    temp_tensor=np.delete(tensor,select_index_list,axis=i)
                    continue
                temp_tensor=np.delete(temp_tensor,select_index_list,axis=i)

    else:
        flag=0
        for i in range(len(index_list)):
            if type(dict.get(index_list[i]))!= type(None):
                flag+=1
                select_index_list=return_tensor_inex(index_list[i],dict[index_list[i]],element)
                temp_index_list=list(range(tensor.shape[i]))
                for j in select_index_list:
                    temp_index_list.remove(j)
                if flag==1:
                    temp_tensor=np.delete(tensor,temp_index_list,axis=i)
                    continue
                temp_tensor=np.delete(temp_tensor,temp_index_list,axis=i)
    return temp_tensor

class tensor():
    #类属性
    index_list=['country','year','age','sex']
    rank=1
    #mode_matrix=None
    def __init__(self,df,element):
        self.element=element
        self.tensor,self.tensor_index=DataFrame2Ndarray(df,index_list,element,True)
        self.feature_factors=tl.decomposition.parafac(self.tensor,rank=rank)
        self.reconstructed_tensor=tl.kruskal_to_tensor(self.feature_factors)
        self.deviation=self.tensor-self.reconstructed_tensor

    def partition(self,dict,select=True):
        '''
        :param dict:
        :param select:
        :return:
        '''
        '''
        {'country':['China','Germany'],'year':[1985,2001,2003]}
        '''
        #temp_tensor=np.copy(self.tensor)
        if select==False:
            flag=0
            for i in range(len(index_list)):
                if type(dict.get(index_list[i]))!= type(None):
                    flag+=1
                    select_index_list=return_tensor_inex(index_list[i],dict[index_list[i]],self.element)
                    if flag==1:
                        temp_tensor=np.delete(self.tensor,select_index_list,axis=i)
                        continue
                    temp_tensor=np.delete(temp_tensor,select_index_list,axis=i)

        else:
            flag=0
            for i in range(len(index_list)):
                if type(dict.get(index_list[i]))!= type(None):
                    flag+=1
                    select_index_list=return_tensor_inex(index_list[i],dict[index_list[i]],self.element)
                    temp_index_list=list(range(self.tensor.shape[i]))
                    for j in select_index_list:
                        temp_index_list.remove(j)
                    if flag==1:
                        temp_tensor=np.delete(self.tensor,temp_index_list,axis=i)
                        continue
                    temp_tensor=np.delete(temp_tensor,temp_index_list,axis=i)
        return temp_tensor

    def choose_mode(self,mode):
        #获得用户所选mode在index_list中的索引
        index=index_list.index(mode)
        #temp=self.feature_factors
        temp=self.feature_factors.copy()
        temp.pop(index)
        temp=np.matmul(self.feature_factors[index],np.matrix.transpose(tl.tenalg.khatri_rao(temp)))
        return temp#返回值待定

    '''聚类算法'''
    #各个算法的说明见：
    #https://scikit-learn.org/stable/modules/clustering.html
    def kmeans(self,n_clusters,mode):
        kmeans=sklearn.cluster.KMeans(n_clusters=n_clusters,random_state=0).fit(self.choose_mode(mode))
        return kmeans.labels_,n_clusters#返回值待定

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
        return clustering.labels_,n_clusters

    def DBSCAN(self,eps,min_samples):
        #https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
        '''
        #sum(db.labels_==-1)#outliers的数量
        #np.max(db.labels_)+1#聚类算法生成的簇的数量
        #labels_中不同正数代表不同类别，-1表示为outliers
        '''
        temp=self.choose_mode(mode)
        db=DBSCAN(eps=eps,min_samples=min_samples).fit(temp)
        return db.labels_,np.max(db.labels_)+1,sum(db.labels_==-1),sum(db.labels_==-1)/len(db.labels_)

    def OPTICS(self,min_samples,xi,min_cluster_size):
        #https://scikit-learn.org/stable/auto_examples/cluster/plot_optics.html#sphx-glr-auto-examples-cluster-plot-optics-py
        '''
        :param min_samples:
        :param xi:
        :param min_cluster_size:
        :return:
        '''
        temp=self.choose_mode(mode)
        clustering=OPTICS(min_samples=min_samples,xi=xi,min_cluster_size=min_cluster_size)
        clustering.fit(temp)
        reachability=clustering.reachability_[clustering.ordering_]
        labels=clustering.labels_[clustering.ordering_]
        return labels,np.max(labels)+1,sum(labels==-1),sum(labels==-1)/len(labels),reachability

    def bar_line(self,selected_modes,axis,show_modes=False):
        '''
            selected_modes:dict,例：{'country':('China','Iceland','Germany')}
            axis:dict,,表示坐标轴表示的区间，例：{'year':(1998,1999,2000,2001}即表示展示1998至2001年的数据
            show_modes:dict,例：{'sex':('male','female')}
        Return:
            bar_charts:list,其元素为{selected_mode:value_list}
            selected_mode:string,例，'China'
            value_list:
        '''
        bar_charts=dict()
        temp_tensor=partition(self.tensor,selected_modes,self.element,True)
        temp_tensor=partition(temp_tensor,axis,self.element,True)
        temp_index=list(range(len(index_list)))
        #del_index=[]
        #del_index.append(index_list.index(selected_modes.keys))
        #del_index.append(index_list.index(axis.keys))
        modes_keys,axis_keys=list(selected_modes.keys()),list(axis.keys())
        temp_index.remove(index_list.index(modes_keys[0]))
        temp_index.remove(index_list.index(axis_keys[0]))
        #sum=np.sum(temp_tensor,axis=)
        if type(show_modes)==type(False):
            temp_list=list(selected_modes.values())
            count=0
            for i in list(selected_modes.values()): #np.sum(temp_tensor,axis=temp_index)
                bar_charts[i]=
        else:
            pass
    def bubble(self):
        pass
temp=tensor(dp,element_list[0])
mode='country'
n_clusters=10

#len(temp.choose_mode(mode))
#temp.kmeans(n_clusters=n_clusters,mode=mode)
#temp.hierarchical(mode=mode,n_clusters=n_clusters,linkage='ward',connectivity=True,n_neighbors=10)
#temp.DBSCAN(eps=25,min_samples=3)
#temp.OPTICS(5,.05,0.05)

#temp.partition({'country':['Albania','Germany'],'year':[1985,2001,2003]},False).shape
#temp.partition({'country':['Albania','Germany'],'year':[1985,2001,2003]},True).shape

temp.bar_line({'country':['Albania','Germany']},{'year':[1985,2001,2003]})
'''
temp={'country':'Germany'}
temp.values()
temp.
'''
