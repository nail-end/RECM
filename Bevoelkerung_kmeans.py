#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as math

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage

#from sklearn.cluster import AgglomerativeClustering

  


# fuer Ztraffo
from scipy import stats

import geopandas as gpd
from geopandas.plotting import plot_polygon_collection

# um Dateien zu scannen
import os#, fnmatch, shutil



# In[2]:


import inspect

#to read sourcecode of functions
#print(inspect.getsource(__function__))


# In[4]:


# Define functions

def detect_csv():
    '''
    All .csv files starting with 'NUTS3' are detected and names and number  of files are returned
    
    !!! Discriptive Columns header should begin with "!" to seperate data from declaration
    
    '''
    
    path_input = os.getcwd() + '/Data_input' 

    files_input = []
    #files_input = ['space_keeper']
    #files_names = []
    for root, dirs, files in os.walk(path_input):
        for file in files:
            if file.endswith('.csv'):
                if file.startswith('NUTS3'):
                    files_input.append(file)
                    #files_names.append(file.split('.')[0])
                
# Transfertabelle eig nicht noetig
                #elif file.startswith('Transfer'):
                #    files_input[0] = file
    
    number_files = len(files_input)
    
    #return(files_input, files_names, number_files)
    return(files_input, number_files)



def read_input(files_input):
    '''
    Read in .csv Files (utf-8 encoding). Create Dictionarry of DataFrames & chooose NUTS3 column as Index
    '''
    dict_df_input = {}
    for gen in files_input:
    
        # import Data
        dict_df_input[gen] = pd.read_csv('./Data_input/' + gen ,index_col=0, sep=';', encoding='utf-8')
        # set index
        dict_df_input[gen].set_index('!NUTS3', drop=True, inplace=True)
        
    return(dict_df_input)


def ztransform_df(df_input):
    '''
    Z-Transform input DataFram: only columns not beginning with "!", columnwise
    using: stats.zscore : arithemtic mean:
    (Z - mean( column(Z) ) ) / std( column(Z) )
    '''
    
    df_output = df_input.copy()
    
    mask_data = ~df_input.columns.str.startswith('!')
    df_output.loc[:,mask_data] = stats.zscore(df_input.loc[:,mask_data], axis=0)
    
    return(df_output)


# In[5]:


#files_input, files_names, number_files = detect_csv()
files_input, number_files = detect_csv()


# In[6]:


dict_df = read_input(files_input)


# In[7]:


df_test = dict_df[list(dict_df)[0]]


# In[8]:


df_test.head()


# In[9]:


######### ENDE Daten einlesen


# In[10]:


# Z-Transformation arithmetic Mean to list

list_df_transform = []

for gen_int, gen_str in enumerate(dict_df):
    list_df_transform.append(ztransform_df(dict_df[gen_str]))


# In[11]:


# Z-Transformation arithmetic Mean to dict

dict_df_transform = {}

for gen in dict_df:
    dict_df_transform[gen] = ztransform_df(dict_df[gen])


# In[11]:


# Z-Transformation (Milligan & Cooper, 1988)





# In[12]:


####### ENDE Z-Transformation


# In[13]:


## Factor Analysis


# In[14]:


deco.FactorAnalysis


# In[14]:


##### ENDE Faktorenanalyse


# In[ ]:


dict_df_transform


# In[13]:


### Data Migration


# In[ ]:


######skipp here


# In[ ]:


#### Get rid of "!" Columns in dict

dict_df_transform_rid = {}

for gen in dict_df_transform:
    dict_df_transform_rid[gen] = dict_df_transform[gen].loc[:, ~dict_df_transform[gen].columns.str.startswith('!')]


# In[ ]:


#### Get rid of "!" Columns in list
list_df_transform_rid = []

for gen,_ in enumerate(list_df_transform):
    list_df_transform_rid.append(list_df_transform[gen].loc[:, ~list_df_transform[gen].columns.str.startswith('!')] )


# In[12]:


### Convert Data to one DataFrame after got rid of "!"

df_alldata = pd.DataFrame()
for gen,_ in enumerate(list_df_transform_rid):
    df_alldata = pd.concat([df_alldata, list_df_transform_rid[gen]], axis=1)


# In[13]:


####### start here again


# In[14]:


## get indexof all Frames from transfer table
index_all = df_test.index # nur uebergangsweise
index_all


# In[15]:


## check if index of all Frames are the same


# In[16]:


# Convert Data to one DataFrame. All columns without "!" in one step

df_alldata = pd.DataFrame(index = index_all)

for gen,_ in enumerate(list_df_transform):
    df_alldata = pd.concat( [ df_alldata , list_df_transform[gen].loc \
        [:, ~list_df_transform[gen].columns.str.startswith('!')] ], \
                            sort=False, axis=1)
    


# In[20]:


##### ENDE Data Migration


# In[ ]:





# In[21]:


##### ENDE Faktorenanalyse


# In[20]:


### K-means


# In[35]:


### ER SAM
# Importiere die Input.xlsx in das Dataframe df_input
#df_input = pd.read_excel('Data_input/Input_sam.xlsx', sheet_name='Tabelle6', dtype = str)


# In[18]:


X = df_alldata.values


# fig, ax1 = plt.subplots(figsize=(40,20))
# #fig.figure(figsize=(40,20))
# 
# ax1.scatter(range(401), f1, s=20 ,c='r')
# ax1.plot(range(401), f1, c='r')
# 
# ax2 = ax1.twinx()
# 
# ax2.scatter(range(401), f2, s=20, c='g')
# ax2.plot(range(401), f2, c='g')
# 
# plt.show()

# # Punktediagramm 
# plt.scatter(X[:,7],X[:,8], s=5)
# plt.xlabel('Bevoelkerung *1000')
# plt.ylabel('Flaeche')
# plt.show()

# In[19]:


####### K-Means mit variabler Clusteranzahl
distanceX = []

for k in range(1,11):
    kmeanTest = KMeans(n_clusters=k) # Einstellungen ueberpruefen
    #fit the model
    kmeanTest.fit(X)
    distanceX.append(sum(np.min(cdist(X, kmeanTest.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    # distance = sum ( min je Spalte(cdist)) / Anzahl Landkreise


# In[46]:


kmeanTest


# In[48]:


cdist(X, kmeanTest.cluster_centers_, 'euclidean').shape


# In[50]:


help(KMeans)


# In[20]:


# Elbow Method
plt.figure(figsize=(10,10))
#plt.subplot(1,2)
plt.plot(distanceX, 'bx-', color='r',)

plt.xlabel('Number of Clusters')
plt.ylabel('mean Distance to Zentroid')
plt.legend(['str(name_projekt)'])
plt.title('Elbow Method Graph ')
#plt.ylim((0,max(distanceX)+0.5))
#plt.text(1, 4, s, bbox=dict(facecolor='red', alpha=0.5),ha='right', wrap=True)
plt.show()


# In[66]:


# lineare Regressio
#slope, intercept, r_value, p_value, std_err = stats.linregress(distanceX,range(len(distanceX)))


# In[21]:


# nicht wirklich Sinnvoll
# Steigung berechnen
slope = [x - y for x, y in zip(distanceX[:-1], distanceX[1:])]
# clusteranzahl vor der geringsten Steiung
print(slope.index(min(slope))-1)
# geringste Aenderung der Steigung
grad_slope = [x - y for x, y in zip(slope[:-1], slope[1:])]
print(grad_slope.index(min(grad_slope))-1)


# In[22]:


#K-means Berechnung nach Auswahl der optimalen Cluster
kmeans_4 = KMeans(n_clusters=4).fit(X)
# Centroide in centers speichern
centers_4=kmeans_4.cluster_centers_
# Ordnet die Eingangsdaten den Clustern zu
y_kmeans_4 = kmeans_4.predict(X)


df_alldata['Cluster'] = kmeans_4.labels_


# In[23]:


df_alldata.head()


# plt.figure(figsize=(40,20))
# #plt.plot(df_input.Distance_Feature, df_input.Speeding_Feature,'.', alpha=0.5)
# plt.grid(True)
# #plt.xlim(0,250)
# #plt.ylim(-20,120)
# plt.xlabel('Bevoelkerung *k')
# plt.ylabel('Flaeche')
# plt.title('Kmeans mit 4 Clustern')
# 
# 
# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans_4, s=10, cmap='plasma')
# plt.scatter(centers_4[:, 0],centers_4[:, 1], c='red', s=200, alpha=0.5);
# 
#     
# #plt.tight_layout()
# plt.show()

# In[41]:


# Agglomerative Clustering # nicht so gut


# In[147]:


cluster_ward= AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')  
cluster =  cluster_ward.fit_predict(X) # only Clusterlabels


# In[144]:


# more infos
test_ward  = cluster_ward.fit(X)
test_ward.labels_ 


# In[148]:


# besseres Agglomerative mit Scipy
#from scipy.cluster.hierarchy import dendrogram, linkage


# In[154]:


Z = linkage(X, 'ward')#, 'euclidean')


# In[152]:


fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[42]:


# Typregion identifizieren
# Mittelwerte innerhalb der Cluster ermitteln
centers1 = np.array(kmeans_4.cluster_centers_)


# In[44]:


centers1.shape


# In[46]:


df_alldata.shape


# In[47]:


for index, row in df_alldata.iterrows():
    print(index)
    print(row)
    print('====')


# In[50]:


for index, row in df_alldata.iterrows():
    
    if row.Cluster==0:
        fehler = np.sum([(row.Bev-centers1[0,0])**2, (row.Flaeche-centers1[0,1])**2])
        df1.set_value(index,'Fehler',fehler)
    if row.Cluster==1:
        fehler = np.sum([(row.Bev-centers1[1,0])**2, (row.Flaeche-centers1[1,1])**2])
        df1.set_value(index,'Fehler',fehler) 
    if row.Cluster==2:
        fehler = np.sum([(row.Bev-centers1[2,0])**2, (row.Flaeche-centers1[2,1])**2])
        df1.set_value(index,'Fehler',fehler) 
    if row.Cluster==3:
        fehler = np.sum([(row.Bev-centers1[3,0])**2, (row.Flaeche-centers1[3,1])**2])
        df1.set_value(index,'Fehler',fehler) 


# In[59]:


# Identifiziere die Regionen mit dem kleinsten Fehler je Cluster
# Für jedes Cluster:
for i in range(4):
    cluster = i
    mask = (df1['Cluster']==cluster)
    min_row = np.argmin(df1.loc[mask]['Fehler'].values)
    print('Cluster '+str(cluster)+':')
    print(df1.loc[mask].values[min_row])
    


# ## Visualisierung

# In[25]:


len(df_alldata.Cluster.unique())


# list_df_karte = []
# for gen in df_alldata.Cluster.unique():
#     mask = (df_alldata.Cluster== gen)
#     list_df_karte.append(df_alldata.loc[mask])

# mask.append(df_alldata.Cluster==0)
# df_karte0 = df_alldata.loc[mask]
# 
# mask = (df_alldata.Cluster==1)
# df_karte1 = df_alldata.loc[mask]
# 
# mask = (df_alldata.Cluster==2)
# df_karte2 = df_alldata.loc[mask]
# 
# mask = (df_alldata.Cluster==3)
# df_karte3 = df_alldata.loc[mask]

# In[26]:


# detect shape-file
path_map = os.getcwd() 

files_map = []
for root, dirs, files in os.walk(path_map):
    for file in files:
        if file.endswith('NUTS3.shp'):
            files_map.append(root)   
            files_map.append(file)
            #if file.startswith('NUTS3'):
                #files_map.append(file)
path2= files_map[0] + '/'+ files_map[1]


# In[27]:


path2


# In[31]:


df_map1 = gpd.read_file(path2)
df_map_basic = gpd.read_file(path2)


# In[32]:


# Aenderungen der Nutscodes
df_map1.at[200,'NUTS_CODE'] = 'DE91C'# Goettingen
df_map1.at[204,'NUTS_CODE'] = 'DE91C'# Osterode Harz
df_map1.at[300,'NUTS_CODE'] = 'DEB1C'# Cochem-Zell
df_map1.at[303,'NUTS_CODE'] = 'DEB1D'# Rhein-Hunsrueck-Kreis


# In[33]:


list_df_karte = []
for gen in df_alldata.Cluster.unique():
    mask = (df_alldata.Cluster== gen)
    list_df_karte.append(pd.merge(df_map1, df_alldata[['Cluster']].loc[mask], right_index=True, left_on='NUTS_CODE'))
    


# mische1_df = map1.merge(df_karte0, on ='NUTS_CODE')
# mische2_df = map1.merge(df_karte1, on = 'NUTS_CODE')
# mische3_df = map1.merge(df_karte2, on = 'NUTS_CODE')
# mische4_df = map1.merge(df_karte3, on = 'NUTS_CODE')

# In[34]:


cmaps=['seismic', 'viridis', 'spring', 'Wistia']


# In[37]:


f, ax = plt.subplots(1,figsize=(12, 12))
ax = df_map_basic.plot(ax=ax, facecolor = 'brg', alpha = 0.6)
for gen in range(len(list_df_karte)):
    ax = list_df_karte[gen].plot(ax=ax, column = 'Cluster', cmap=cmaps[gen])
    '''
ax = mische1_df.plot(ax=ax, column = 'Cluster', cmap = 'seismic')
ax = mische2_df.plot(ax=ax, column = 'Cluster', cmap = 'viridis')
ax = mische3_df.plot(ax=ax, column = 'Cluster', cmap = 'spring')
ax = mische4_df.plot(ax=ax, column = 'Cluster', cmap = 'Wistia')
'''

ax.set_axis_off()
f.suptitle('Clusterzugehörigkeit der NUTS-3 Kreise in Deutschland')
lims = plt.axis('equal')


# In[ ]:





# In[ ]:




