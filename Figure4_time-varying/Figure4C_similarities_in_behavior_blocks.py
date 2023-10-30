# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:18:25 2023

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


InputData_path_dir =  r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure4_time-varying\others'


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



animal_info_csv = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Table_S1_animal_information.csv'
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]

Morning_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-off')]


movement_order = ['running','trotting','walking','right_turning','left_turning','stepping',
                  'jumping','climbing','rearing','hunching','rising','sniffing',
                  'grooming','pausing']


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


def movement_count2(df,min_i):    
    label_count = df.value_counts('revised_movement_label')  
    df = pd.DataFrame(data=0,index=movement_order,columns=[min_i])
    for i in label_count.index:
        df.loc[i,min_i] =  label_count[i]
    return(df.T)


dataset = Night_lightOn_info
dataset_name = 'Night_lightOn_info'



MoV_matrix_list = []
step = 5
start = 0
end = 0
for min_i in range(step,61,step):
    end = min_i*30*60
    for key in dataset['video_index']:                   # nightLightOff_info    dayLightOn_info
        if key in skip_file_list:
            pass
        else:
            Mov_data = pd.read_csv(Movement_Label_path[key])
            Mov_data = Mov_data.iloc[start:end,:]
            MoV_matrix = movement_count2(Mov_data,min_i)
            #distance_data = pd.read_csv(Distance_path_day[key],usecols=['back_x','back_y','smooth_speed_distance'])
            #distance_data = distance_data.iloc[start:end,:]
            #distance_data['r'] = np.sqrt(np.square(distance_data['back_x']-200)+np.square(distance_data['back_y']-200))
            #MoV_matrix['normalized_distance'] = distance_data['r'].mean()
            #MoV_matrix['travel_distance'] = distance_data['smooth_speed_distance'].mean()
            MoV_matrix['time_window'] = min_i
            MoV_matrix_list.append(MoV_matrix)
    start = end
    
df_all = pd.concat(MoV_matrix_list)
df_all.reset_index(drop=True,inplace=True)


df_compare = pd.DataFrame()

num = 0
for i in df_all.index:
    time_window1 = df_all.loc[i,'time_window']
    arr1 =  np.array(df_all.iloc[i,:-1])
    for j in df_all.index:
        if i == j:
            pass
        else:
            time_window2 = df_all.loc[j,'time_window']
            arr2 =  np.array(df_all.iloc[j,:-1])
            
            similarity = pdist(np.vstack([arr1,arr2]),metric='correlation')

            df_compare.loc[num,'time_window1'] = time_window1
            df_compare.loc[num,'time_window2'] = time_window2
            df_compare.loc[num,'similarity'] = 1 - similarity[0]
            num += 1
    


df_heatmap = pd.DataFrame()
for i in range(step,61,step):
    for j in range(step,61,step):
        temp_df = df_compare[(df_compare['time_window1']==i) & (df_compare['time_window2']==j)]
        average_sim = temp_df['similarity'].mean()
        df_heatmap.loc[i,j] = average_sim

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300)
sns.heatmap(df_heatmap,square=True,vmin=0.5,vmax=1,cbar=False,xticklabels=[],yticklabels=[],cmap='Spectral_r',linecolor='black',linewidths=1)   #Spectral_r
plt.savefig('{}/{}_60min_temporal_similarities.png'.format(output_dir,dataset_name),transparent=True,dpi=300)


