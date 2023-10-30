# -*- coding: utf-8 -*-
"""
Created on Thu May  4 17:22:55 2023

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
import scipy.stats
from scipy.interpolate import make_interp_spline
import matplotlib.patheffects as pe


new_generate_data_day = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure7_Movement_sequence_prediction\predicted_movement_sequences\new_sequence-MOTP_MODP'
new_generate_data_night = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure7_Movement_sequence_prediction\predicted_movement_sequences\new_sequence-NFTP_NFDP'

InputData_path_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure7_Movement_sequence_prediction\predicted_movement_sequences'


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

def get_path2(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = file_name.split('-')[1]
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(USN,date)
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)


Feature_space_path = get_path(InputData_path_dir,'Feature_Space.csv')
Feature_space_path_predict_day = get_path2(new_generate_data_day,'Feature_Space.csv')
Feature_space_path_predict_night = get_path2(new_generate_data_night,'Feature_Space.csv')


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
     

def calculate_entroy(df):
    
    count_df = df['revised_movement_label'].value_counts()
    mv_entroy = pd.DataFrame()
    for mv in movement_order:        
        if mv in count_df.index:
            mv_count = count_df[mv]

        else:
            mv_count = 0
        mv_entroy.loc[mv,'count'] = mv_count
    
    #print(mv_entroy)
    entroy = scipy.stats.entropy(mv_entroy,base=2)
    #print(entroy)
    return(entroy)

dataset1 = Morning_lightOn_info
dataset1_name = 'Morning_lightOn'

dataset2 = Night_lightOff_info
dataset2_name = 'Night_lightOff'

dataset3 = Feature_space_path_predict_day
dataset3_name = 'Predicted_Morning_lightOn'

dataset4 = Feature_space_path_predict_night
dataset4_name = 'Predicted_Night_lightOff'


df_trans_speed = pd.DataFrame()

start = 0
end = 0
num = 0
for i in range(5,61,5):
    end = i *30*60
    for index in dataset1.index:
        video_index = dataset1.loc[index,'video_index']
        ExperimentTime = dataset1.loc[index,'ExperimentTime']
        LightingCondition = dataset1.loc[index,'LightingCondition']
        FeA_data = pd.read_csv(Feature_space_path[video_index])
        temp_df = FeA_data[(FeA_data['segBoundary_start']>start) & (FeA_data['segBoundary_end']<end)]
        entroy = calculate_entroy(temp_df)
        df_trans_speed.loc[num,'video_index'] = index
        df_trans_speed.loc[num,'ExperimentTime'] = dataset1_name
        df_trans_speed.loc[num,'time_tag'] = i
        df_trans_speed.loc[num,'trans_speed'] = len(temp_df) /5 # (len(temp_df) /1) * entroy
        df_trans_speed.loc[num,'entroy'] = entroy
        num += 1
    for index in dataset2.index:
        video_index = dataset2.loc[index,'video_index']
        ExperimentTime = dataset2.loc[index,'ExperimentTime']
        LightingCondition = dataset2.loc[index,'LightingCondition']
        FeA_data = pd.read_csv(Feature_space_path[video_index])
        temp_df = FeA_data[(FeA_data['segBoundary_start']>start) & (FeA_data['segBoundary_end']<end)]
        entroy = calculate_entroy(temp_df)
        df_trans_speed.loc[num,'video_index'] = index
        df_trans_speed.loc[num,'ExperimentTime'] = dataset2_name
        df_trans_speed.loc[num,'time_tag'] = i
        df_trans_speed.loc[num,'trans_speed'] = len(temp_df) /5 # (len(temp_df) /1) * entroy
        df_trans_speed.loc[num,'entroy'] = entroy
        num += 1
    for key in dataset3.keys():
        FeA_data = pd.read_csv(Feature_space_path_predict_day[key])
        temp_df = FeA_data[(FeA_data['segBoundary_start']>start) & (FeA_data['segBoundary_end']<end)]
        entroy = calculate_entroy(temp_df)
        df_trans_speed.loc[num,'video_index'] = index
        df_trans_speed.loc[num,'ExperimentTime'] = dataset3_name
        df_trans_speed.loc[num,'time_tag'] = i
        df_trans_speed.loc[num,'trans_speed'] = len(temp_df) /5 # (len(temp_df) /1) * entroy
        df_trans_speed.loc[num,'entroy'] = entroy
        num += 1
    for key in dataset4.keys():
        FeA_data = pd.read_csv(Feature_space_path_predict_night[key])
        temp_df = FeA_data[(FeA_data['segBoundary_start']>start) & (FeA_data['segBoundary_end']<end)]
        entroy = calculate_entroy(temp_df)
        df_trans_speed.loc[num,'video_index'] = index
        df_trans_speed.loc[num,'ExperimentTime'] = dataset4_name
        df_trans_speed.loc[num,'time_tag'] = i
        df_trans_speed.loc[num,'trans_speed'] = len(temp_df) /5 # (len(temp_df) /1) * entroy
        df_trans_speed.loc[num,'entroy'] = entroy
        num += 1
    start = end        

average_df = pd.DataFrame()
num = 0   
for i in range(5,61,5):
    for k in [dataset1_name,dataset2_name,dataset3_name,dataset4_name]:
        temp_df = df_trans_speed[(df_trans_speed['time_tag']==i) &(df_trans_speed['ExperimentTime']==k) ]
        average_trans_speed = np.mean(temp_df['trans_speed'])
        average_entroy = np.mean(temp_df['entroy'])
        average_df.loc[num,'time_tag'] = i
        average_df.loc[num,'ExperimentTime'] = k
        average_df.loc[num,'average_trans_speed'] = average_trans_speed
        average_df.loc[num,'average_entroy'] = average_entroy
        num += 1

        
color_list = {dataset1_name:'#F5B25E', dataset2_name:'#003960',dataset3_name:'#E6E243', dataset4_name:'#71BFCB'}
    
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10,5),dpi=300)
for k in  [dataset1_name,dataset2_name,dataset3_name,dataset4_name]:
     ExperimentTime_df = average_df[average_df['ExperimentTime']==k]
     x = ExperimentTime_df['time_tag']
     y = ExperimentTime_df['average_trans_speed']
     x_smooth = np.linspace(x.min(), x.max(), 300)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
     y_smooth = make_interp_spline(x, y)(x_smooth)
     color = color_list[k]
     ax[0].plot(x_smooth, y_smooth,color = color,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground=color,alpha=0.6), pe.Normal()])

ax[0].set_xlabel('Time(min)')
ax[0].set_ylabel('Average movement trans number')
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)    

for k in  [dataset1_name,dataset2_name,dataset3_name,dataset4_name]:
     ExperimentTime_df = average_df[average_df['ExperimentTime']==k]
     x = ExperimentTime_df['time_tag']
     y = ExperimentTime_df['average_entroy']
     x_smooth = np.linspace(x.min(), x.max(), 300)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
     y_smooth = make_interp_spline(x, y)(x_smooth)
     color = color_list[k]
     ax[1].plot(x_smooth, y_smooth,color = color,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground=color,alpha=0.6), pe.Normal()])

ax[1].set_xlabel('Time(min)')
ax[1].set_ylabel('Average_entroy rate')
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)            

plt.savefig('{}/movement_transition_speed.png'.format(output_dir),dpi=300)
        
        
        
        
        
        
        
        
        