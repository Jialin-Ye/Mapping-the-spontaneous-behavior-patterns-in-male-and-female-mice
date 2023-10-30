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
output_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure4_time-varying\Temporal_change_in_movement_cluster" 

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
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(video_index,date)
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
            df_output.loc[num,'label_frequency'] = label_number[mv] / label_number.values.sum()
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
            df_output.loc[num,'label_frequency'] = label_number[mv] / label_number.values.sum()
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
            df_output.loc[num,'label_frequency'] = label_number[mv] / label_number.values.sum()
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
print('repeated video index : {}'.format(repeat_list))

movement_frequency_each_mice = []

test_df_list = []

Morning_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-off')]


movement_frequency_each_mice = []

dataset1 = Night_lightOff_info
datasetname1 = 'Night_lightOff'


time_window1 = 0
time_window2 = 0                                           #### set time_window for calculation

for min_i in range(15,61,15):
    time_window2 = min_i
    for i in dataset1.index:
        video_index = dataset1.loc[i,'video_index']
        Movement_Label_file = pd.read_csv(Movement_Label_path[video_index])
        #Movement_Label_file = add_movement_label(Movement_Label_file)
        #Movement_Label_file = add_cluster(Movement_Label_file)
        Movement_Label_file = Movement_Label_file.iloc[time_window1*30*60:time_window2*30*60,:]
        df_count = movement_cluster_label_count(Movement_Label_file)
        df_count['time'] = min_i
        movement_frequency_each_mice.append(df_count)
    time_window1 = time_window2

df_out = pd.concat(movement_frequency_each_mice)


df_statistic = pd.DataFrame()

for min_i in range(15,61,15):
    for j in cluster_order:
        value = df_out.loc[(df_out['movement_cluster_label']==j)&(df_out['time']==min_i),'label_frequency'].values.mean()
        df_statistic.loc[j,min_i] = value


df_statistic.to_csv('{}/{}_temporal_change_in_MovementCluster.csv'.format(output_dir,datasetname1))




