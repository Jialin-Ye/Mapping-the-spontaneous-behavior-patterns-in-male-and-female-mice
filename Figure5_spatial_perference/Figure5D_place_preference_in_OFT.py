# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:19:29 2023

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
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure5_spatial_perference\place_preference'


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


animal_info_csv = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Table_S1_animal_information.csv'
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]

Morning_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-off')]


dataset1 = Morning_lightOn_info
dataset1_name = 'Morning_lightOn'
dataset1_color = '#F5B25E'                      ## morning '#F5B25E' afternoon 936736   night-light-off '#003960'    night-lightOn '#3498DB'


dataset2 = Night_lightOff_info
dataset2_name = 'Night_lightOff'
dataset2_color = '#003960'

location_list = ['center','wall','corner']

location_area = {'center':(18*2)*(18*2),
                 'wall':4*(25-18)*(18*2),
                 'corner':4*(25-18)*(25-18)}                                                ### cm

location_df = pd.DataFrame()

num = 0
for index in dataset1.index:
    video_index = animal_info.loc[index,'video_index']
    ExperimentTime = animal_info.loc[index,'ExperimentTime']
    LightingCondition = animal_info.loc[index,'LightingCondition']
    
    location_df.loc[num,'video_index'] = video_index
    location_df.loc[num,'experiment_condition'] = ExperimentTime+'_'+LightingCondition
    
    Mov_loc_data = pd.read_csv(Movement_Label_path[video_index])
    
    center = Mov_loc_data[Mov_loc_data['location']=='center']
    wall =  Mov_loc_data[Mov_loc_data['location']=='wall']
    corner =  Mov_loc_data[Mov_loc_data['location']=='corner']
    
    center_time = round((len(center)/len(Mov_loc_data))*100,2)
    center_time_area = round(((len(center)/(30*60))/location_area['center'])*100,2)          ### min/ cm2
    
    location_df.loc[num,'center_time_p'] = center_time
    location_df.loc[num,'center_time_area'] = center_time_area
    
    wall_time = round((len(wall)/len(Mov_loc_data))*100,2)
    wall_time_area = round(((len(wall)/(30*60))/location_area['wall'])*100,2)
    
    location_df.loc[num,'wall_time_p'] = wall_time
    location_df.loc[num,'wall_time_area'] = wall_time_area
    
    corner_time = round((len(corner)/len(Mov_loc_data))*100,2)
    corner_time_area = round(((len(corner)/(30*60))/location_area['corner'])*100,2)
    
    location_df.loc[num,'corner_time_p'] = corner_time
    location_df.loc[num,'corner_time_area'] = corner_time_area
    num += 1

for index in dataset2.index:
    video_index = animal_info.loc[index,'video_index']
    ExperimentTime = animal_info.loc[index,'ExperimentTime']
    LightingCondition = animal_info.loc[index,'LightingCondition']
    
    location_df.loc[num,'video_index'] = video_index
    location_df.loc[num,'experiment_condition'] = ExperimentTime+'_'+LightingCondition
    
    Mov_loc_data = pd.read_csv(Movement_Label_path[video_index])
    
    center = Mov_loc_data[Mov_loc_data['location']=='center']
    wall =  Mov_loc_data[Mov_loc_data['location']=='wall']
    corner =  Mov_loc_data[Mov_loc_data['location']=='corner']
    
    center_time = round((len(center)/len(Mov_loc_data))*100,2)
    center_time_area = round(((len(center)/(30*60))/location_area['center'])*100,2)          ### min/ cm2
    
    location_df.loc[num,'center_time_p'] = center_time
    location_df.loc[num,'center_time_area'] = center_time_area
    
    wall_time = round((len(wall)/len(Mov_loc_data))*100,2)
    wall_time_area = round(((len(wall)/(30*60))/location_area['wall'])*100,2)
    
    location_df.loc[num,'wall_time_p'] = wall_time
    location_df.loc[num,'wall_time_area'] = wall_time_area
    
    corner_time = round((len(corner)/len(Mov_loc_data))*100,2)
    corner_time_area = round(((len(corner)/(30*60))/location_area['corner'])*100,2)
    
    location_df.loc[num,'corner_time_p'] = corner_time
    location_df.loc[num,'corner_time_area'] = corner_time_area
    num += 1


info_dict = {'time_percentage':[],'time_percentage_perArea':[],'value1':[],'value2':[],'experiment_condition':[]}
for i in location_df.index:
    experiment_condition = location_df.loc[i,'experiment_condition']
    center_time = location_df.loc[i,'center_time_p']
    center_time_area = location_df.loc[i,'center_time_area']
    info_dict['time_percentage'].append('center')
    info_dict['time_percentage_perArea'].append('center')
    info_dict['value1'].append(center_time)
    info_dict['value2'].append(center_time_area)
    info_dict['experiment_condition'].append(experiment_condition)
    
    wall_time = location_df.loc[i,'wall_time_p']
    wall_time_area = location_df.loc[i,'wall_time_area']
    info_dict['time_percentage'].append('wall')
    info_dict['time_percentage_perArea'].append('wall')
    info_dict['value1'].append(wall_time)
    info_dict['value2'].append(wall_time_area)
    info_dict['experiment_condition'].append(experiment_condition)
    
    corner_time = location_df.loc[i,'corner_time_p']
    corner_time_area = location_df.loc[i,'corner_time_area']
    info_dict['time_percentage'].append('corner')
    info_dict['time_percentage_perArea'].append('corner')
    info_dict['value1'].append(corner_time)
    info_dict['value2'].append(corner_time_area)
    info_dict['experiment_condition'].append(experiment_condition)

df_plot = pd.DataFrame(info_dict)
df_plot.to_csv(r'{}\{}_{}place_preference.csv'.format(output_dir,dataset1_name,dataset2_name))


fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10,5),dpi=300)

ax[0].boxplot(location_df[['center_time_p','wall_time_p','corner_time_p']],labels=['center_time_p','wall_time_p','corner_time_p'])
ax[0].set_title('Total stay time')

ax[1].boxplot(location_df[['center_time_area','wall_time_area','corner_time_area']],labels=['center_time','wall_time','corner_time'])
ax[1].set_title('Area normalized')

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(5,4),dpi=300)
sns.stripplot(data=df_plot, x='time_percentage',y='value1',hue='experiment_condition',jitter=True,dodge=True,edgecolor='black',linewidth=1,ax=ax,palette=[dataset1_color,dataset2_color],alpha=0.9)
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 3},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            saturation=1,
            dodge = True,
            width = 0.8,
            x="time_percentage",
            y="value1",
            hue='experiment_condition',
            data=df_plot,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax)
ax.legend_.remove()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_ylabel('Time in differnece areas (%)')
ax.set_xlabel('')
ax.set_title('Time in differnece areas (%)')
ax.set_yticks(range(0,101,20))
plt.savefig('{}\{}_{}_time_percentage(%).png'.format(output_dir,dataset1_name,dataset2_name),dpi=300)


fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(5,4),dpi=300)
sns.stripplot(data=df_plot, x='time_percentage_perArea',y='value2',hue='experiment_condition',jitter=True,dodge=True,edgecolor='black',linewidth=1,ax=ax,palette=[dataset1_color,dataset2_color],alpha=0.9)
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 3},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            saturation=1,
            dodge = True,
            width = 0.8,
            x="time_percentage_perArea",
            y="value2",
            hue='experiment_condition',
            data=df_plot,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax)
ax.legend_.remove()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_ylabel('Time in per unit areas (min/cm2)')
ax.set_xlabel('')
ax.set_title('Time in per unit areas (min/cm2)')
ax.set_yticks(range(0,31,10))
plt.savefig('{}\{}_{}_NormalizedArea_Time(min_cm2).png'.format(output_dir,dataset1_name,dataset2_name),dpi=300)