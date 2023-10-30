# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:01:00 2023

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


output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure2_skeleton_kinematics\movement_parametter'
InputData_path_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label"
InputData_path_dir2 = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\01_BehaviorAtlas_collated_data"

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            video_index = int(file_name.split('-')[1])
            file_path_dict.setdefault(video_index,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir,'revised_Movement_Labels.csv')
Skeleton_path = get_path(InputData_path_dir2,'normalized_skeleton_XYZ.csv')
#Skeleton_path = get_path(InputData_path_dir2,'Cali_Data3d.csv')

skip_file_list = [1,3,28,29,110,122] 

animal_info_csv = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Table_S1_animal_information.csv'             ## 动物信息表位置
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]

Light_on_info = animal_info[animal_info['LightingCondition']=='Light-on']

Morning_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-off')]


movement_order = ['running','trotting','walking','right_turning','left_turning','stepping',
                  'jumping','climbing','rearing','hunching','rising','sniffing',
                  'grooming','pausing']


OriginalDigital_label_list = range(40)

locomotion_list = ['running','trotting','walking','left_turning','right_turning','stepping']
standUP_list = ['sniffing','rising','hunching','rearing','climbing']

turning_list = ['left_turning','right_turning']

#pausing_list = ['pausing']
pausing_list = [39,20,21]

grooming_list = [37,40,15]






def add_BodyPart_distance(df):
    count_part = ['neck','back','left_front_limb','right_front_limb','left_hind_limb','right_hind_limb','root_tail']
    count_done =[]
    new_col_num = 0
    for k in count_part:
        count_done.append(k)
        for l in count_part:
            if l in count_done:
                pass
            else:
                new_col_num += 1
                part1_name = k
                part2_name = l
                     
                part1_x = part1_name + '_x'
                part1_y = part1_name + '_y'
                part1_z = part1_name + '_z'
                        
                part2_x = part2_name + '_x'
                part2_y = part2_name + '_y'
                part2_z = part2_name + '_z'
                     

                df['{}-{}_distance'.format(part1_name,part2_name)] = np.sqrt(np.square(df[part1_x]-df[part2_x])+np.square(df[part1_y]-df[part2_y])+np.square(df[part1_z]-df[part2_z]))
                
    df['sum_body_distance'] = df.iloc[:,-new_col_num:-1].sum(axis=1)
    df['body_stretch_rate'] = (df['sum_body_distance']/df['sum_body_distance'].mean()) * 100
    #df = df[df['body_stretch_rate']<200]
    return(df)


def add_BodyPart_distance2(df):
    df['nose-back_distance'] = np.sqrt(np.square(df['nose_x']-df['back_x'])+np.square(df['nose_y']-df['back_y'])+np.square(df['nose_z']-df['back_z']))
    df['neck-back_distance'] = np.sqrt(np.square(df['neck_x']-df['neck_x'])+np.square(df['neck_y']-df['back_y'])+np.square(df['neck_z']-df['back_z']))
    df['left_front_limb-back_distance'] = np.sqrt(np.square(df['left_front_limb_x']-df['back_x'])+np.square(df['left_front_limb_y']-df['back_y'])+np.square(df['left_front_limb_z']-df['back_z']))
    df['right_front_limb-back_distance'] = np.sqrt(np.square(df['right_front_limb_x']-df['back_x'])+np.square(df['right_front_limb_y']-df['back_y'])+np.square(df['right_front_limb_z']-df['back_z']))
    df['left_hind_limb-back_distance'] = np.sqrt(np.square(df['left_hind_limb_x']-df['back_x'])+np.square(df['left_hind_limb_y']-df['back_y'])+np.square(df['left_hind_limb_z']-df['back_z']))
    df['right_hind_limb-back_distance'] = np.sqrt(np.square(df['right_hind_limb_x']-df['back_x'])+np.square(df['right_hind_limb_y']-df['back_y'])+np.square(df['right_hind_limb_z']-df['back_z']))
    df['root_tail-back_distance'] = np.sqrt(np.square(df['root_tail_x']-df['back_x'])+np.square(df['root_tail_y']-df['back_y'])+np.square(df['root_tail_z']-df['back_z']))
    
    df['sum_body_distance'] = df[['nose-back_distance','neck-back_distance','left_front_limb-back_distance','right_front_limb-back_distance','left_hind_limb-back_distance','right_hind_limb-back_distance','root_tail-back_distance']].sum(axis=1)
    df['body_stretch_rate'] = ((df['sum_body_distance']-df['sum_body_distance'].mean())/df['sum_body_distance'].mean()) * 100
    #df = df[df['body_stretch_rate']<200]
    return(df)

ske_data_list = []
for video_index in list(Morning_lightOn_info['video_index']):
#for index in Morning_lightOn_info.index:
    
    ske_data = pd.read_csv(Skeleton_path[video_index])
    ske_data = add_BodyPart_distance(ske_data)
    Mov_data = pd.read_csv(Movement_Label_path[video_index],usecols=['OriginalDigital_label','revised_movement_label','locomotion_speed_smooth'])
    conbime_data = pd.concat([Mov_data,ske_data],axis=1)
    
    #select_data = conbime_data[conbime_data['revised_movement_label'].isin(pausing_list)]
    #select_data = conbime_data[conbime_data['OriginalDigital_label'].isin(grooming_list)]
    ske_data_list.append(conbime_data)


all_df = pd.concat(ske_data_list,axis=0)
all_df.reset_index(drop=True,inplace=True)


all_df = all_df[(all_df['back_z']>10)&(all_df['back_z']<300)]
all_df = all_df[(all_df['body_stretch_rate']>0)&(all_df['body_stretch_rate']<200)]
# =============================================================================
# df_select_list = []
# for MV in all_df['OriginalDigital_label'].unique():
#     df_singleLocomotion = all_df[all_df['OriginalDigital_label']==MV]
#     if len(df_singleLocomotion) < 10000:
#         df_frame_select = df_singleLocomotion
#     else:
#         df_frame_select = df_singleLocomotion.sample(n=10000, random_state=1) # weights='locomotion_speed_smooth',
#     df_select_list.append(df_frame_select)
# =============================================================================


df_select_list = []
for MV in all_df['revised_movement_label'].unique():
    df_singleLocomotion = all_df[all_df['revised_movement_label']==MV]
    if len(df_singleLocomotion) < 10000:
        df_frame_select = df_singleLocomotion
    else:
        df_frame_select = df_singleLocomotion.sample(n=10000, random_state=22) # weights='locomotion_speed_smooth',
    df_select_list.append(df_frame_select)


df_select = pd.concat(df_select_list)
df_select.reset_index(drop=True,inplace=True)



singleMov_back_height_list = []
for i in movement_order:
     df_singleMov = df_select[df_select['revised_movement_label'] == i]
     df = pd.DataFrame(data=df_singleMov['back_z'].values,index=range(len(df_singleMov)))
     df.columns = [i]
     singleMov_back_height_list.append(df)

df_output = pd.concat(singleMov_back_height_list,axis=1)
df_output.to_csv(r'{}\back_height.csv'.format(output_dir))   



singleMov_stretch_rate_list = []
for i in movement_order:
     df_singleMov = df_select[df_select['revised_movement_label'] == i]
     df = pd.DataFrame(data=df_singleMov['body_stretch_rate'].values,index=range(len(df_singleMov)))
     df.columns = [i]
     singleMov_stretch_rate_list.append(df)

df_output = pd.concat(singleMov_stretch_rate_list,axis=1)
  
df_output.to_csv(r'{}\body_sketch_rate.csv'.format(output_dir))  
