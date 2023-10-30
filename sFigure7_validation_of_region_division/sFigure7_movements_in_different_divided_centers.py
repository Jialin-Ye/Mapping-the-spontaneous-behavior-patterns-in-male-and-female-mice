# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 09:47:40 2023

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
import matplotlib.patches as patches

### 统计位置比例和动作

InputData_path_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\sFigure7_validation_of_region_division\validation_of_region_division'

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

candidated_center_area = (18*2) * (18*2) - 25*25
traditional_center = 25* 25
wall_area = 4 * (25 - 18) * (18*2)
corner_area = 4 *(25 - 18) * (25 - 18)    #18

area_dict = {'traditional_center':traditional_center,
             'candidated_center':candidated_center_area,
             'wall':wall_area,
             'corner':corner_area}

def add_location18_validation(df):  #70, 430    
    df_copy = df.copy()

    df_copy['location'] = 'candidated_center'
    
    df_copy.loc[(df_copy['back_x']>= 125)&(df_copy['back_x']<=375)&(df_copy['back_y']>= 125)&(df_copy['back_y']<=375),'location'] = 'traditional_center'
    
    df_copy.loc[(df_copy['back_x']< 70)&(df_copy['back_y']<= 430)&(df_copy['back_y']>= 70),'location'] = 'wall'
    df_copy.loc[(df_copy['back_x']> 430)&(df_copy['back_y']<= 430)&(df_copy['back_y']>= 70),'location'] = 'wall'
    df_copy.loc[(df_copy['back_y']< 70)&(df_copy['back_x']<= 430)&(df_copy['back_x']>= 70),'location'] = 'wall'
    df_copy.loc[(df_copy['back_y']> 430)&(df_copy['back_x']<= 430)&(df_copy['back_x']>= 70),'location'] = 'wall'
    
    df_copy.loc[(df_copy['back_x']>430) & (df_copy['back_y']> 430),'location'] = 'corner'
    df_copy.loc[(df_copy['back_x']< 70) & (df_copy['back_y']< 70),'location'] = 'corner'
    df_copy.loc[(df_copy['back_x']< 70) & (df_copy['back_y']> 430),'location'] = 'corner'
    df_copy.loc[(df_copy['back_x']> 430) & (df_copy['back_y']< 70),'location'] = 'corner'
    
    return(df_copy)


def plot(df,file_index):
    #lut = dict(zip(df['location'].unique(), sns.color_palette("husl", 4)))
    
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10),constrained_layout=True,dpi=300)
    x0 = 250
    y0 = 250
    ax.scatter(df['back_x'],df['back_y'],s=1,c=df['color'])
    ax.scatter(x0,y0,s=100,c='red')
    ax.add_patch(
    patches.Rectangle(
        (0, 0),   # (x,y)
        500,          # width
        500,          # height
        alpha=1,ec="black", fc=None,fill=False,lw=2)
        )
    ax.add_patch(
    patches.Rectangle(
        (125, 125),   # (x,y)
        250,          # width
        250,          # height
        alpha=1,ec="black", fc=None,fill=False,lw=1)
        )
    fig.suptitle('rec-{}-add_location_18test'.format(file_index),fontsize=25)
    plt.show()
    
dataset1 = Morning_lightOn_info
dataset2 = Night_lightOff_info



start = 0
end = 60 *60 *30

    
df_location = pd.DataFrame()
info_dict = {'video_index':[],'gender':[],'ExperimentTime':[],'LightingCondition':[],'location':[],'movement_label':[],'count':[],'mv_per_unit_area':[]} 
num = 0
num2 = 0
for index in animal_info.index:
    video_index = animal_info.loc[index,'video_index']
    ExperimentTime = animal_info.loc[index,'ExperimentTime']
    LightingCondition = animal_info.loc[index,'LightingCondition']
    gender = animal_info.loc[index,'gender']
    MoV_file = pd.read_csv(Movement_Label_path[video_index])
    MoV_file = add_location18_validation(MoV_file)
    #lut = dict(zip(['traditional_center','candidated_center','wall','corner','unknown'], ['orange','green','red','blue','grey']))
    #MoV_file['color'] = MoV_file['location'].map(lut)
    #plot(MoV_file,index)
    
    MoV_file = MoV_file.iloc[start:end,:]
    Tcenter_zone_time = len(MoV_file[MoV_file['location']=='traditional_center']) / 30                       ### unit: s
    center_zone_time = len(MoV_file[MoV_file['location']=='candidated_center']) / 30                         ### unit: s
    wall_time = len(MoV_file[MoV_file['location']=='wall']) / 30
    corner_time = len(MoV_file[MoV_file['location']=='corner']) / 30
    
    df_location.loc[num,'ExperimentTime'] = ExperimentTime
    df_location.loc[num,'LightingCondition'] = LightingCondition 
    df_location.loc[num,'gender'] = gender 

    df_location.loc[num,'Tcenter_zone_time(s)'] = Tcenter_zone_time
    df_location.loc[num,'center_zone_time(s)'] = center_zone_time 
    df_location.loc[num,'wall_time(s)'] = wall_time
    df_location.loc[num,'corner_time(s)'] = corner_time
    df_location.loc[num,'Tcenter_zone_time_area(s/cm2)'] = Tcenter_zone_time / traditional_center
    df_location.loc[num,'center_zone_time_area(s/cm2)'] = center_zone_time /candidated_center_area          #unit: s/cm2
    df_location.loc[num,'wall_time_area(s/cm2)'] =  wall_time/wall_area
    df_location.loc[num,'corner_time_area(s/cm2)'] = corner_time/corner_area
    
    
    for loc in ['traditional_center','candidated_center','wall','corner']:
        temp_df = MoV_file[MoV_file['location']==loc]
        count_df = temp_df['revised_movement_label'].value_counts() / len(temp_df)

        for mv in movement_order:
            if mv in count_df.index:
                mv_count = count_df[mv]
            else:
                mv_count = 0
            
            info_dict['video_index'].append(video_index)
            info_dict['gender'].append(gender)
            info_dict['ExperimentTime'].append(ExperimentTime)
            info_dict['LightingCondition'].append(LightingCondition)
            info_dict['location'].append(loc)
            info_dict['movement_label'].append(mv)
            info_dict['count'].append(mv_count)
            info_dict['mv_per_unit_area'].append(mv_count/area_dict[loc])
    
    num +=1

df_location.to_csv(r'{}\location4partValiation{}_{}_min.csv'.format(output_dir,start,end))
df_location_mvCount = pd.DataFrame(info_dict)
df_location_mvCount.to_csv(r'{}\location4partValiation_movement_{}_{}_min.csv'.format(output_dir,start,end))  



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    