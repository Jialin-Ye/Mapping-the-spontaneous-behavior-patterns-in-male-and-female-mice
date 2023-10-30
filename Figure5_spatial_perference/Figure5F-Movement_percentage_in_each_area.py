# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 22:06:37 2023

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



InputData_path_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label" 
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure5_spatial_perference'


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

cluster_order = ['locomotion','exploration','maintenance','nap']

mv_sort_order2 = {}
for i in range(len(cluster_order)):
    mv_sort_order2.setdefault(cluster_order[i],i)


movement_color_dict = {'running':'#FF3030',
                       'trotting':'#F06292',
                       'walking':'#EB6148',
                       'left_turning':'#F6BBC6',
                       'right_turning':'#F29B78',
                       'stepping':'#E4CF7B',                       
                       'jumping':'#ECAD4F',
                       'climbing':'#2E7939',                       
                       'rearing':'#88AF26',
                       'hunching':'#7AB69F',
                       'rising':'#80DEEA',
                       'sniffing':'#2C93CB',                       
                       'grooming':'#A13E97',
                       'pausing':'#D3D4D4',}



dataset = Morning_lightOn_info
count_variable = 'movement_cluster_label'     ### 'movement_label'  OriginalDigital_label, movement_label,revised_movement_label, movement_cluster_label

time_window1 = 0
time_window2 = 60


def movement_label_count(df):
    data = df.copy()
    label_number = data[count_variable].value_counts()

    df_output = pd.DataFrame()
    num = 0
    for mv in movement_order:
        df_output.loc[num,count_variable] = mv
        if not mv in label_number.index:
            df_output.loc[num,'label_frequency'] = 0
        else:
            df_output.loc[num,'label_frequency'] = label_number[mv] / 30
        num += 1
    return(df_output)

def movement_cluster_label_count(df):
    data = df.copy()
    label_number = data[count_variable].value_counts()

    df_output = pd.DataFrame()
    num = 0
    for mv in cluster_order:
        df_output.loc[num,count_variable] = mv
        if not mv in label_number.index:
            df_output.loc[num,'label_frequency'] = 0
        else:
            df_output.loc[num,'label_frequency'] = label_number[mv] / 30
        num += 1
    return(df_output)


def region_MVcount(df):
    MV_cluster_loc_list = []
    for loc in ['center','perimeter','corner']:
        location_df = df[df['location']==loc]
        MV_cluster_loc = movement_cluster_label_count(location_df)
        MV_cluster_loc['location'] = loc
        MV_cluster_loc_list.append(MV_cluster_loc)
    MV_cluster_loc_df = pd.concat(MV_cluster_loc_list)
    return(MV_cluster_loc_df)
    


LocMV_df_list = []
for video_index in Morning_lightOn_info['video_index']:
    Mov_data = pd.read_csv(Movement_Label_path[video_index])
    Mov_data = Mov_data.iloc[time_window1*30*60:time_window2*30*60,:]
    LocMV_df = region_MVcount(Mov_data)
    LocMV_df['ExpermentTime'] = 'Morning'
    LocMV_df['mouse_id'] = video_index
    LocMV_df['order'] = LocMV_df[count_variable].map(mv_sort_order2)
    LocMV_df_list.append(LocMV_df)

for video_index in Night_lightOff_info['video_index']:
    Mov_data = pd.read_csv(Movement_Label_path[video_index])
    Mov_data = Mov_data.iloc[time_window1*30*60:time_window2*30*60,:]
    LocMV_df = region_MVcount(Mov_data) 
    LocMV_df['ExpermentTime'] = 'Night'
    LocMV_df['mouse_id'] = video_index
    LocMV_df['order'] = LocMV_df[count_variable].map(mv_sort_order2)
    LocMV_df_list.append(LocMV_df)

df_out = pd.concat(LocMV_df_list)

def removeNAN_align(df):
    M = df.to_numpy()
    #get True or False depending on the null status of each entry
    condition = ~np.isnan(M)

    #for each array, get entries that are not null
    step1 = [np.compress(ent,arr) for ent,arr in zip(condition,M)]
    step1

    #concatenate each dataframe 
    step2 = pd.concat([pd.DataFrame(ent).T for ent in step1],ignore_index=True)
    step2.index = df.index
    return(step2)

df_out1 = df_out.pivot_table(values='label_frequency', index=['ExpermentTime',count_variable,'location','order'], columns='mouse_id',)
df_out1.to_csv(r'{}\Full_info_{}_LocMVcluster_{}-{}min.csv'.format(output_dir,count_variable,time_window1, time_window2))
df_out2 = removeNAN_align(df_out1)
df_out2 = df_out2.sort_index(level='order',)
df_out2.to_csv(r'{}\Simplify_{}_LocMVcluster_{}-{}.csv'.format(output_dir,count_variable,time_window1, time_window2))














































