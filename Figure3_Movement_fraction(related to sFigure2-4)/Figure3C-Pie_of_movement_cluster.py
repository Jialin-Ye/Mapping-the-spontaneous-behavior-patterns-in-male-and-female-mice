# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:14:38 2023

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
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure3_Movement_fraction\03_night-lightOn&night-lightOff_related_to_sFigure3'



def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            video_index = int(file_name.split('-')[1])
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(video_index,date)
            file_path_dict.setdefault(video_index,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir,'Movement_Labels.csv')       



skip_file_list = [1,3,28,29,110,122] 

animal_info_csv =r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Table_S1_animal_information.csv'             
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]


Morning_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-off')]


movement_order = ['running','trotting','walking','right_turning','left_turning','stepping',
                  'jumping','climbing','rearing','hunching','rising','sniffing',
                  'grooming','pausing']


cluster_color_dict={'locomotion':'#DC2543',                     
                     'exploration':'#009688',
                     'maintenance':'#A13E97',
                     'nap':'#D3D4D4'}

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


movement_cluster_label_order = {'locomotion':1,'exploration':2,'maintenance':3,'nap':4,}



dataset1 = Night_lightOn_info
dataset2 = Night_lightOff_info

dataset1_name = 'Night_lightOn'
dataset2_name = 'Night_lightOff'


dataset1_list = []
for index in dataset1.index:
    video_index = dataset1.loc[index,'video_index']
    gender = dataset1.loc[index,'gender']
    annoMV_data = pd.read_csv(Movement_Label_path[video_index],usecols=['movement_cluster_label'])
    annoMV_data['gender'] = gender
    dataset1_list.append(annoMV_data)
    
df_dataset1_all = pd.concat(dataset1_list,axis=0)


dataset2_list = []
for index in dataset2.index:
    video_index = dataset2.loc[index,'video_index']
    gender = dataset2.loc[index,'gender']
    annoMV_data = pd.read_csv(Movement_Label_path[video_index],usecols=['movement_cluster_label'])
    annoMV_data['gender'] = gender
    dataset2_list.append(annoMV_data)
    
df_dataset2_all = pd.concat(dataset2_list,axis=0)


dataset_list = []

### movement
for sex in ['male','female']:
#for loc in ['wall']:
    temp_df_dataset1 = df_dataset1_all[df_dataset1_all['gender']==sex]
    dataset1_count_df = temp_df_dataset1.value_counts('movement_cluster_label').to_frame(name='count')
    dataset1_count_df['percentage'] = dataset1_count_df['count'] / dataset1_count_df['count'].sum()
    dataset1_count_df['color'] = dataset1_count_df.index.map(cluster_color_dict)
    dataset1_count_df['plot_order'] = dataset1_count_df.index.map(movement_cluster_label_order)
    dataset1_count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)
    dataset1_count_df['gender'] = sex
    dataset1_count_df['ExperimentTime'] = dataset1_name.split('_')[0]
    dataset1_count_df['LightingCondition'] = dataset1_name.split('_')[1]
    dataset_list.append(dataset1_count_df)
    
    temp_df_dataset2 = df_dataset2_all[df_dataset2_all['gender']==sex]
    dataset2_count_df = temp_df_dataset2.value_counts('movement_cluster_label').to_frame(name='count')
    dataset2_count_df['percentage'] = dataset2_count_df['count'] / dataset2_count_df['count'].sum()
    dataset2_count_df['color'] = dataset2_count_df.index.map(cluster_color_dict)
    dataset2_count_df['plot_order'] = dataset2_count_df.index.map(movement_cluster_label_order)
    dataset2_count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)
    dataset2_count_df['gender'] = sex
    dataset2_count_df['ExperimentTime'] = dataset2_name.split('_')[0]
    dataset2_count_df['LightingCondition'] = dataset2_name.split('_')[1]
    day_radius = night_radius = 1
    dataset_list.append(dataset2_count_df)
# =============================================================================
#     if len(temp_df_day) >= len(temp_df_night):
#         night_radius = 1
#         day_radius = len(temp_df_day)/len(temp_df_night)
#     else:
#         day_radius = 1
#         night_radius = len(temp_df_night)/len(temp_df_day)
#     
# =============================================================================
    #plt.figure(figsize=(10, 10),dpi=300)
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=600)
    ax.pie(dataset1_count_df['percentage']/2,counterclock=True,normalize=False,startangle = 90,
            #explode = [0.08,0.08,0.08,0.08],
            colors = dataset1_count_df['color'],
            radius = day_radius,
            #autopct= '%1.1f%%',pctdistance=1.2,
            wedgeprops = {'linewidth':3, 'edgecolor': "black"},)
            #wedgeprops = {'linewidth':3, 'edgecolor': "#F5B25E"},)
            #hatch=['.', '.', '.', '.'])
    plt.pie(dataset2_count_df['percentage']/2,counterclock=False,normalize=False,startangle = 90,
            #explode = [0.08,0.08,0.08,0.08],
            colors = dataset2_count_df['color'],
            radius = night_radius,
            #autopct= '%1.1f%%',pctdistance=1.2,
            wedgeprops = {'linewidth':3, 'edgecolor': "black"})
            #wedgeprops = {'linewidth':3, 'edgecolor': "#003960"},)
            #hatch=['.', '.', '.', '.'])
    plt.pie([1], colors = ['#ffffff'], radius = 0.5)
    plt.savefig('{}/{}-{}_{}.png'.format(output_dir,dataset1_name,dataset2_name,sex),transparent=True,dpi=600)



df_summary = pd.concat(dataset_list,axis=0)
#df_summary.to_csv('{}/{}-{}.csv'.format(output_dir,dataset1_name,dataset2_name))























'''



location_dir_day = r'D:\Personal_File\yejiailn\3D_spontaneous_data\DayAMLightOn_NightLightOn\result\add_location_16'
location_dir_night = r'D:\Personal_File\yejiailn\3D_spontaneous_data\DayPMLightOn_NightLightOff\result\add_location_16'

location_path_day = get_path(location_dir_day,'addlocation.csv')
location_path_night = get_path(location_dir_night,'addlocation.csv')

Mov_dir_day = r'D:\Personal_File\yejiailn\3D_spontaneous_data\DayAMLightOn_NightLightOn\result\anno_MV_csv-3sThreshold'
Mov_dir_night = r'D:\Personal_File\yejiailn\3D_spontaneous_data\DayPMLightOn_NightLightOff\result\anno_MV_csv-3sThreshold'

Mov_path_day = get_path(Mov_dir_day,'Movement_Labels.csv')
Mov_path_night = get_path(Mov_dir_night,'Movement_Labels.csv')


movement_list = ['running','walking','right_turning','left_turning','stepping',
                 'jumping','climb_up','rearing','hunching','rising','sniffing',
                 'grooming','pause']

location_list = ['center','wall','corner']

location_area = {'center':25*25,
                 'wall':4*25*12.5,
                 'corner':4*12.5*12.5}

movement_color_dict = {'running':'#F44336','walking':'#FF5722','left_turning':'#FFAB91','right_turning':'#FFCDD2','stepping':'#BCAAA4',
                     'sniffing':'#26A69A','climb_up':'#43A047','rearing':'#66BB6A','hunching':'#81C784','rising':'#9CCC65','jumping':'#FFB74D',
                     'grooming':'#AB47BC','pause':'#90A4AE',}

cluster_color_dict = {'locomotion':'#F94040',
                       'exploration':'#077E97',
                       'maintenance':'#914C99',
                       'nap':'#D4D4D4',
                       }

movement_cluster_label_order = {'locomotion':1,
                       'exploration':2,
                       'maintenance':3,
                       'nap':4,
                       }

movementlaber_order = {'running':1,
                       'walking':2,
                       'left_turning':3,
                       'right_turning':4,
                       'stepping':5,
                       'jumping':6,
                       'climb_up':7,
                       'rearing':8,
                       'hunching':9,
                       'rising':10,
                       'sniffing':11,
                       'grooming':12,
                       'pause':13,

                       }

def add_location_16(df):

    center_zone = [6,7,10,11]
    df.loc[df['location'].isin(center_zone),'location_word'] = 'center'
    wall_zone = [2,3,5,9,8,12,14,15]
    df.loc[df['location'].isin(wall_zone),'location_word'] = 'wall'
    corner_zone = [1,4,13,16]
    df.loc[df['location'].isin(corner_zone),'location_word'] = 'corner'    
    return(df)

def add_movement_cluster_label(df):
    df_copy = df.copy()

    big_movement_cluster_label_dict4 = {'locomotion':['running','walking','left_turning','right_turning','stepping'],
                         'exploration':['climb_up','rearing','hunching','rising','sniffing','jumping'],
                         'maintenance':['grooming'],
                         'nap':['pause']
                         }

    
    df_copy.loc[df_copy['new_label'].isin(big_movement_cluster_label_dict4['locomotion']),'movement_cluster_label'] = 'locomotion'
    df_copy.loc[df_copy['new_label'].isin(big_movement_cluster_label_dict4['exploration']),'movement_cluster_label'] = 'exploration'
    df_copy.loc[df_copy['new_label'].isin(big_movement_cluster_label_dict4['maintenance']),'movement_cluster_label'] = 'maintenance' 
    df_copy.loc[df_copy['new_label'].isin(big_movement_cluster_label_dict4['nap']),'movement_cluster_label'] = 'nap'
    return(df_copy)

day_list = []
for index in dayLightOn_info.index:
    file_name = dayLightOn_info.loc[index,'video_index']
    gender = dayLightOn_info.loc[index,'gender']
    
    annoMV_data = pd.read_csv(Mov_path_day[file_name],usecols=['new_label'])
    annoMV_data = add_movement_cluster_label(annoMV_data)
    annoMV_data['gender'] = gender
    
    loc_data = pd.read_csv(location_path_day[file_name],usecols=['location'])
    loc_data = add_location_16(loc_data)
    df_combine = pd.concat([annoMV_data,loc_data],axis=1)
    day_list.append(df_combine)
    
df_day_all = pd.concat(day_list,axis=0)


night_list = []
for index in nightLightOff_info.index:
    file_name = nightLightOff_info.loc[index,'video_index']
    gender = nightLightOff_info.loc[index,'gender']
    
    annoMV_data = pd.read_csv(Mov_path_night[file_name],usecols=['new_label'])
    annoMV_data = add_movement_cluster_label(annoMV_data)
    annoMV_data['gender'] = gender
    
    loc_data = pd.read_csv(location_path_night[file_name],usecols=['location'])
    loc_data = add_location_16(loc_data)
    df_combine = pd.concat([annoMV_data,loc_data],axis=1)
    night_list.append(df_combine)
    
df_night_all = pd.concat(night_list,axis=0)




# =============================================================================
# ## movement_cluster_label 
# for loc in ['center','wall','corner']:
# #for loc in ['wall']:
#     temp_df_day = df_day_all[df_day_all['location_word']==loc]
#     day_count_df = temp_df_day.value_counts('movement_cluster_label').to_frame(name='count')
#     day_count_df['percentage'] = day_count_df['count'] / day_count_df['count'].sum()
#     day_count_df['color'] = day_count_df.index.map(cluster_color_dict)
#     day_count_df['plot_order'] = day_count_df.index.map(movement_cluster_label_order)
#     day_count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)
#     
#     temp_df_night = df_night_all[df_night_all['location_word']==loc]
#     night_count_df = temp_df_night.value_counts('movement_cluster_label').to_frame(name='count')
#     night_count_df['percentage'] = night_count_df['count'] / night_count_df['count'].sum()
#     night_count_df['color'] = night_count_df.index.map(cluster_color_dict)
#     night_count_df['plot_order'] = night_count_df.index.map(movement_cluster_label_order)
#     night_count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)
#     
#     if len(temp_df_day) >= len(temp_df_night):
#         night_radius = 1
#         day_radius = len(temp_df_day)/len(temp_df_night)
#     else:
#         day_radius = 1
#         night_radius = len(temp_df_night)/len(temp_df_day)
#     
#     #plt.figure(figsize=(10, 10),dpi=300)
#     fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300)
#     ax.pie(day_count_df['percentage']/2.01,counterclock=True,normalize=False,startangle = 90,
#             explode = [0.08,0.08,0.08,0.08],
#             colors = day_count_df['color'],
#             radius = day_radius,
#             wedgeprops = {'linewidth':5, 'edgecolor': "#F5B25E"},)
#             #hatch=['.', '.', '.', '.'])
#     plt.pie(night_count_df['percentage']/2.01,counterclock=False,normalize=False,startangle = 90,
#             explode = [0.08,0.08,0.08,0.08],
#             colors = night_count_df['color'],
#             radius = night_radius,
#             wedgeprops = {'linewidth':5, 'edgecolor': "#003960"},)
#             #hatch=['.', '.', '.', '.'])
#     plt.pie([1], colors = ['#ffffff'], radius = 0.5)
#     #plt.title(loc)
# =============================================================================



### movement
for sex in ['male','female']:
#for loc in ['wall']:
    temp_df_day = df_day_all[df_day_all['gender']==sex]
    day_count_df = temp_df_day.value_counts('movement_cluster_label').to_frame(name='count')
    day_count_df['percentage'] = day_count_df['count'] / day_count_df['count'].sum()
    day_count_df['color'] = day_count_df.index.map(cluster_color_dict)
    day_count_df['plot_order'] = day_count_df.index.map(movement_cluster_label_order)
    day_count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)
    
    temp_df_night = df_night_all[df_night_all['gender']==sex]
    night_count_df = temp_df_night.value_counts('movement_cluster_label').to_frame(name='count')
    night_count_df['percentage'] = night_count_df['count'] / night_count_df['count'].sum()
    night_count_df['color'] = night_count_df.index.map(cluster_color_dict)
    night_count_df['plot_order'] = night_count_df.index.map(movement_cluster_label_order)
    night_count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)
    
    day_radius = night_radius = 1
    
# =============================================================================
#     if len(temp_df_day) >= len(temp_df_night):
#         night_radius = 1
#         day_radius = len(temp_df_day)/len(temp_df_night)
#     else:
#         day_radius = 1
#         night_radius = len(temp_df_night)/len(temp_df_day)
#     
# =============================================================================
    #plt.figure(figsize=(10, 10),dpi=300)
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300)
    ax.pie(day_count_df['percentage']/2,counterclock=True,normalize=False,startangle = 90,
            #explode = [0.08,0.08,0.08,0.08],
            colors = day_count_df['color'],
            radius = day_radius,
            #autopct= '%1.1f%%',pctdistance=1.2,
            wedgeprops = {'linewidth':3, 'edgecolor': "black"},)
            #wedgeprops = {'linewidth':3, 'edgecolor': "#F5B25E"},)
            #hatch=['.', '.', '.', '.'])
    plt.pie(night_count_df['percentage']/2,counterclock=False,normalize=False,startangle = 90,
            #explode = [0.08,0.08,0.08,0.08],
            colors = night_count_df['color'],
            radius = night_radius,
            #autopct= '%1.1f%%',pctdistance=1.2,
            wedgeprops = {'linewidth':3, 'edgecolor': "black"})
            #wedgeprops = {'linewidth':3, 'edgecolor': "#003960"},)
            #hatch=['.', '.', '.', '.'])
    plt.pie([1], colors = ['#ffffff'], radius = 0.5)
    plt.savefig(r'D:\Personal_File\yejiailn\课题\文章\图\第二版-figure_code\figure3\movement_cluster_label_pie_{}_开光灯.png'.format(sex),transparent=True,dpi=300)


'''




