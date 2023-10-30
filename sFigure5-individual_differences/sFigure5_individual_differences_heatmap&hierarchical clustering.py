# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 14:35:44 2023

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from math import pi
from scipy.cluster.hierarchy import dendrogram, linkage



InputData_path_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\sFigure5-individual_differences'


skip_file_list = [1,3,28,29,110,122] 

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = int(file_name.split('-')[1])
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(USN,date)
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)


Movement_Label_path = get_path(InputData_path_dir,'revised_Movement_Labels.csv')
Feature_space_path = get_path(InputData_path_dir,'Feature_Space.csv')
#Speed_distance_path =  



animal_info_csv = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Table_S1_animal_information.csv'             ## 动物信息表位置
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]

Forenoon_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Forenoon') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-off')]


movement_order = ['running','trotting','walking','right_turning','left_turning','stepping',
                  'jumping','climbing','rearing','hunching','rising','sniffing',
                  'grooming','pausing']

cluster_order = ['locomotion','exploration','maintenance','nap']

mv_sort_order2 = {}
for i in range(len(cluster_order)):
    mv_sort_order2.setdefault(cluster_order[i],i)


movement_color_dict = {'running':'#DD2C00',
                       'trotting':'#EC407A',
                       'walking':'#FF5722',
                       'left_turning':'#FFAB91',
                       'right_turning':'#FFCDD2',
                       'stepping':'#BCAAA4',
                       'sniffing':'#26A69A',
                       'climbing':'#43A047',
                       'rearing':'#66BB6A',
                       'hunching':'#0288D1',
                       'rising':'#9CCC65',
                       'jumping':'#FFB74D',
                       'grooming':'#AB47BC',
                       'pausing':'#90A4AE',} 


def extract_segment2(df):
    mv_dict = {}
    for mv in movement_order:
        mv_dict.setdefault(mv,0)
    count_df = pd.DataFrame(mv_dict,index=[0])
    
    label_count = df.value_counts('revised_movement_label')
    for count_mv in label_count.index:
        count_df[count_mv]  = label_count[count_mv]
    count_df['travel_distance'] =  df['locomotion_speed_smooth'].sum() * (1/30)
    count_df['back_z'] =  df['back_z'].mean()
    return(count_df)

feature_vector_list = []
video_id = []


for i in animal_info.index:
    video_index = animal_info.loc[i,'video_index']
    gender = animal_info.loc[i,'gender']
    ExperimentTime = animal_info.loc[i,'ExperimentTime']
    LightingCondition = animal_info.loc[i,'LightingCondition']
    mouse_id =  animal_info.loc[i,'mouse_id']
    video_id.append('video{}_mouse:{}'.format(video_index,mouse_id))
    
    Mov_file_data = pd.read_csv(Movement_Label_path[video_index])
    MoV_60_count = extract_segment2(Mov_file_data)
    
    FeA_file_data = pd.read_csv(Feature_space_path[i])
    MoV_60_count['event_number'] = len(FeA_file_data)
    MoV_60_count['gender'] = gender
    MoV_60_count['group'] = ExperimentTime + '_' + LightingCondition
    feature_vector_list.append(MoV_60_count)
            
df = pd.concat(feature_vector_list)
df.reset_index(drop=True,inplace=True)

df["groud_id"] = pd.factorize(df["group"])[0].astype(int)


feat_cols = df.columns[:-3]
X = df[feat_cols].values
y = df['groud_id']

print(X.shape, y.shape)

scaled_X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_X)

Z = linkage(pca_result, 'ward',)

plt.figure(figsize=(10, 5),dpi=300)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z, leaf_rotation=90., leaf_font_size=8,labels=df['group'].values)
plt.savefig('{}\individual_dendrogram.png'.format(output_dir), dpi=300)
plt.show()




my_palette = dict(zip(df.group.unique(), ["#F5B25E","#398FCB","#926736",'#003960']))
row_colors = df.group.map(my_palette)



col_colors = []
col_colors.extend(['#E84240']*6)
col_colors.extend(['#127991']*5)
col_colors.extend(['#894991']*1)
col_colors.extend(['#D3D4D4']*1)

#fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(20,10),dpi=300)
fig = sns.clustermap(df.iloc[:,:-3], metric="euclidean", method="ward", cmap="YlGnBu",figsize=(15,25),dendrogram_ratio=0.2,
               square=True,linewidths=2,tree_kws=dict(linewidths=2,colors='black',label=False),xticklabels=False,
               standard_scale=1, row_colors=row_colors,col_colors=None,col_cluster=False,yticklabels=video_id)

fig.cax.set_visible(False)
#fig.ax_row_dendrogram.set_visible(False)
fig.ax_col_dendrogram.set_visible(False)
plt.savefig('{}\individual_differences.png'.format(output_dir), dpi=300)
plt.show()