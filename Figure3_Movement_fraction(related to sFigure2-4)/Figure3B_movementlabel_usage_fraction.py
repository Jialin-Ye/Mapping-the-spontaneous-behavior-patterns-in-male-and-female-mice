#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 16:25:05 2022

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



InputData_path_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label"                                         
output_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure3_Movement_fraction\movement_fraction" 

if not os.path.exists(output_dir):                                                                       
    os.mkdir(output_dir)

skip_file_list = [1,3,28,29,110,122]                                                                      

count_variable = 'movement_cluster_label'   ### 'movement_label'  OriginalDigital_label, movement_label,revised_movement_label, movement_cluster_label

time_window1 = 0
time_window2 = 60                                             


movement_order = ['running','trotting','walking','right_turning','left_turning','stepping',
                  'jumping','climbing','rearing','hunching','rising','sniffing',
                  'grooming','pausing']                                                                    

cluster_order = ['locomotion','exploration','maintenance','nap']                                 
     

animal_info_csv = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Table_S1_animal_information.csv'            
animal_info = pd.read_csv(animal_info_csv)                                                                         
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]                                           

variables = ['gender','ExperimentTime','LightingCondition']   #'estrous_cycle'   add if need to analysis estrous_cycle movement                                                                                 




def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            video_index = int(file_name.split('-')[1])
            file_path_dict.setdefault(video_index,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir,'revised_Movement_Labels.csv')



def original_label_count(df):
    data = df.copy()
    label_number = data[count_variable].value_counts()

    df_output = pd.DataFrame()
    num = 0
    for mv in range(1,41):
        df_output.loc[num,count_variable] = mv
        if not mv in label_number.index:
            df_output.loc[num,'label_frequency'] = 0
        else:
            df_output.loc[num,'label_frequency'] = (label_number[mv] / label_number.values.sum())*100
        df_output.loc[num,'accumulative_time'] = df_output.loc[num,'label_frequency']*(time_window2-time_window1)
        num += 1
    return(df_output)    

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
            df_output.loc[num,'label_frequency'] = (label_number[mv] / label_number.values.sum())*100
        df_output.loc[num,'accumulative_time'] = df_output.loc[num,'label_frequency']*(time_window2-time_window1)
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
            df_output.loc[num,'label_frequency'] = (label_number[mv] / label_number.values.sum()) *100
        df_output.loc[num,'accumulative_time'] = df_output.loc[num,'label_frequency']*(time_window2-time_window1)
        num += 1
    return(df_output)


mv_sort_order = {}
for i in range(len(movement_order)):
    mv_sort_order.setdefault(movement_order[i],i)

mv_sort_order2 = {}
for i in range(len(cluster_order)):
    mv_sort_order2.setdefault(cluster_order[i],i)



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


repeat_list = []
for ID in animal_info['video_index'].unique():
    if len(animal_info[animal_info['video_index']==ID]) >=2:
        repeat_list.append(ID)
print('repeated video index : {}'.format(repeat_list))             ## chekck repeat file name

movement_frequency_each_mice = []

test_df_list = []

Morning_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-off')]

for i in animal_info.index:
    video_index = animal_info.loc[i,'video_index']
    
    if video_index in skip_file_list:
        pass
    else:
        mouse_id = animal_info.loc[i,'mouse_id']                                                               
        if count_variable == 'OriginalDigital_label':
            Movement_Label_file = pd.read_csv(Movement_Label_path[video_index])
            Movement_Label_file = Movement_Label_file.iloc[time_window1*30*60:time_window2*30*60,:]
            df = original_label_count(Movement_Label_file)
      
            for v in variables:
                df[v] = animal_info.loc[i,v]
    
            df['mouse_id'] = mouse_id
            df['order'] = df['OriginalDigital_label']
            movement_frequency_each_mice.append(df)
            
        elif count_variable == 'revised_movement_label':
            Movement_Label_file = pd.read_csv(Movement_Label_path[video_index])
            #Movement_Label_file = add_movement_label(Movement_Label_file)
            Movement_Label_file = Movement_Label_file.iloc[time_window1*30*60:time_window2*30*60,:]
            
            df = movement_label_count(Movement_Label_file)
            
            
            for v in variables:
                df[v] = animal_info.loc[i,v]
            df['mouse_id'] = mouse_id
            df['order'] = df['revised_movement_label'].map(mv_sort_order)
            
            movement_frequency_each_mice.append(df)
            
            
        elif count_variable == 'movement_cluster_label':
            Movement_Label_file = pd.read_csv(Movement_Label_path[video_index])
            #Movement_Label_file = add_movement_label(Movement_Label_file)
            #Movement_Label_file = add_cluster(Movement_Label_file)
            Movement_Label_file = Movement_Label_file.iloc[time_window1*30*60:time_window2*30*60,:]
            df = movement_cluster_label_count(Movement_Label_file)
            for v in variables:
                df[v] = animal_info.loc[i,v]
            df['mouse_id'] = mouse_id
            df['order'] = df['movement_cluster_label'].map(mv_sort_order2)
            movement_frequency_each_mice.append(df)
    

variables.insert(0, count_variable)
variables.insert(-1, 'order')


df_out = pd.concat(movement_frequency_each_mice)

df_out1 = df_out.pivot_table(values='label_frequency', index=variables, columns='mouse_id',)
df_out1.to_csv(r'{}\Full_info_{}_count_{}-{}min.csv'.format(output_dir,count_variable,time_window1, time_window2))
df_out2 = removeNAN_align(df_out1)
df_out2 = df_out2.sort_index(level='order',)
df_out2.to_csv(r'{}\Simplify_{}_count_{}-{}_min.csv'.format(output_dir,count_variable,time_window1, time_window2))


