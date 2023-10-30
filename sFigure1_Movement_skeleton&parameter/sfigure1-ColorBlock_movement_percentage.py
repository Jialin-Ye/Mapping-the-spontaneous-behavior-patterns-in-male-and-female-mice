# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 18:26:05 2023

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



skip_file_list = [1,3,28,29,110,122] 
animal_info_csv = r'H:\spontaneous_behavior_data\Table_S1_animal_information.csv'             ## 动物信息表位置
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]

Morning_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-off')]



anno_Mov_csv_dir = r'H:\spontaneous_behavior_data\BehaviorAtlas_outputs_data\revised_movement_label'
output_dir = r'H:\spontaneous_behavior_data\Figure&panel_code\sFigure1_Movement_skeleton&parameter'

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            USN = int(file_name.split('-')[1])
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(USN,date)
            file_path_dict.setdefault(USN,file_dir+'/'+file_name)
    return(file_path_dict)



annoMV_path_dict = get_path(anno_Mov_csv_dir,'revised_Movement_Labels.csv')





data_list = []

for file_name in animal_info['video_index']:
    annoMV_data = pd.read_csv(annoMV_path_dict[file_name])
    data_list.append(annoMV_data)


df = pd.concat(data_list)



# =============================================================================
# movement_color_dict = {'running':'#DD2C00',
#                        'trotting':'#EC407A',
#                        'walking':'#FF5722',
#                        'left_turning':'#FFAB91',
#                        'right_turning':'#FFCDD2',
#                        'stepping':'#BCAAA4',
#                        'sniffing':'#26A69A',
#                        'climbing':'#43A047',
#                        'rearing':'#66BB6A',
#                        'hunching':'#0288D1',
#                        'rising':'#9CCC65',
#                        'jumping':'#FFB74D',
#                        'grooming':'#AB47BC',
#                        'pausing':'#90A4AE',} 
# =============================================================================


# =============================================================================
# movement_color_dict = {'running':'#E60012',
#                        'trotting':'#DC4173',
#                        'walking':'#EA572A',
#                        'left_turning':'#F8CACE',
#                        'right_turning':'#F4A68E',
#                        'stepping':'#BBA8A3',
#                        
#                        'sniffing':'#1080C3',
#                        'climbing':'#9A9926',
#                        'rearing':'#97C464',
#                        'hunching':'#4CA74F',
#                        'rising':'#299F93',
#                        'jumping':'#F6B24F',
#                        'grooming':'#914B99',
#                        'pausing':'#8DA1AB',} 
# =============================================================================
category_color_dict={'locomotion':'#DC2543',                     
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



movement_order = ['running','trotting','walking','right_turning','left_turning','stepping',
           'jumping','climbing','rearing','hunching','rising','sniffing',
           'grooming','pausing']

category_order = ['locomotion','exploration','maintenance','nap']

category_df = pd.DataFrame()
num = 0
for category in category_order:
    num += 1
    category_df.loc[num,'label'] = category
    category_df.loc[num,'percentage'] = len(df[df['movement_cluster_label']==category]) / len(df)
    category_df.loc[num,'color'] = category_color_dict[category]
category_df = category_df.iloc[::-1]


movement_df = pd.DataFrame()
num = 0
for mv in movement_order:
    num += 1
    movement_df.loc[num,'label'] = mv
    movement_df.loc[num,'percentage'] = len(df[df['revised_movement_label']==mv]) / len(df)
    movement_df.loc[num,'color'] = movement_color_dict[mv]
movement_df = movement_df.iloc[::-1]

fig, ax = plt.subplots(nrows=1,ncols=1,figsize = (10,8),dpi=600)

ax.scatter(0,0.8,c='white')
ax.scatter(1,0,c='white')

# =============================================================================
# height = 0
# for i in category_df.index:
#     value = category_df.loc[i,'percentage']
#     color = category_df.loc[i,'color']
# 
#     ax.add_patch(
#     patches.Rectangle(
#         (0, height),   # (x,y)
#         0.05,          # width
#         value,          # height
#         #alpha=0.3,ec="#FF7F50", fc="#FFBF00",lw=5)
#         alpha=1,ec="white", fc=color,lw=1)
#         )
#     height += value
#     height += 0.05
# =============================================================================

big_skeleton = ['right_turning','left_turning','jumping','climbing','rearing','grooming']
middle_skeleton = ['hunching','rising']

height = 0
skeleton = 0
num = 0
for i in movement_df.index:
    if num == 0:
        c_value = category_df.loc[4,'percentage']
        c_color = category_df.loc[4,'color']
    
        ax.add_patch(
        patches.Rectangle(
            (0, height),   # (x,y)
            0.03,          # width
            c_value,          # height
            #alpha=0.3,ec="#FF7F50", fc="#FFBF00",lw=5)
            alpha=1,ec="black", fc=c_color,lw=1)
            )
    if num == 1:
        c_value = category_df.loc[3,'percentage']
        c_color = category_df.loc[3,'color']
        ax.add_patch(
        patches.Rectangle(
            (0, height),   # (x,y)
            0.03,          # width
            c_value,          # height
            #alpha=0.3,ec="#FF7F50", fc="#FFBF00",lw=5)
            alpha=1,ec="black", fc=c_color,lw=1)
            )
    if num == 2:
        c_value = category_df.loc[2,'percentage']
        c_color = category_df.loc[2,'color']
        ax.add_patch(
        patches.Rectangle(
            (0, height),   # (x,y)
            0.03,          # width
            c_value,          # height
            #alpha=0.3,ec="#FF7F50", fc="#FFBF00",lw=5)
            alpha=1,ec="black", fc=c_color,lw=1)
            )
    if num == 8:
        c_value = category_df.loc[1,'percentage']
        c_color = category_df.loc[1,'color']
        ax.add_patch(
        patches.Rectangle(
            (0, height),   # (x,y)
            0.03,          # width
            c_value,          # height
            #alpha=0.3,ec="#FF7F50", fc="#FFBF00",lw=5)
            alpha=1,ec="black", fc=c_color,lw=1)
            )
    movement = movement_df.loc[i,'label']  
    value = movement_df.loc[i,'percentage']
    color = movement_df.loc[i,'color']
    ax.add_patch(
    patches.Rectangle(
        (0.18, height),   # (x,y)
        0.05,          # width
        value,          # height
        #alpha=0.3,ec="#FF7F50", fc="#FFBF00",lw=5)
        alpha=1,ec="black", fc=color,lw=1)
        )
    
    height += value
    height += 0.0335
    
    if movement in big_skeleton:
        height_value = 0.11
    elif movement in big_skeleton:
        height_value = 0.07
    else:
        height_value = 0.07
    
    ax.add_patch(
    patches.Rectangle(
        (0.3, skeleton),   # (x,y)
        0.4,          # width
        height_value,          # height
        #alpha=0.3,ec="#FF7F50", fc="#FFBF00",lw=5)
        alpha=0.1,ec="black", fc=color,lw=1)
        )
    skeleton += height_value
    skeleton += 0.017
    
    num += 1

ax.axis('off')
plt.savefig(r'{}\behavior_skeleton.png'.format(output_dir),dpi=600)