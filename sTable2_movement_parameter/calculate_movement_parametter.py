# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:19:49 2023

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
import math



InputData_path_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label"
InputData_path_dir2 = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\01_BehaviorAtlas_collated_data"
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\sTable2_movement_parameter'


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
Skeleton_path = get_path(InputData_path_dir2,'normalized_skeleton_XYZ.csv')


skip_file_list = [1,3,28,29,110,122] 

animal_info_csv = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Table_S1_animal_information.csv'
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


dateset = Morning_lightOn_info



def cal_ang(point_1, point_2, point_3):
    a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
    b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
    c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
    A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
    B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    C=math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))
    return (B)
 
def Srotate(angle,valuex,valuey,pointx,pointy):  ### 顺时针
    valuex = np.array(valuex)  
    valuey = np.array(valuey)  
    sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx  
    sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy  
    return((sRotatex,sRotatey))
def Nrotate(angle,valuex,valuey,pointx,pointy):  
    valuex = np.array(valuex)  
    valuey = np.array(valuey)  
    nRotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
    nRotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
    return((nRotatex,nRotatey))


def plot(nose_point_new,neck_point_new,back_point_new,root_tail_point_new):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8),constrained_layout=True,dpi=300)
    ax.axvline(0,-5,5)
    ax.axhline(0,-5,5)
    ax.scatter(0,0,s=20,c='black')
    ax.scatter(nose_point_new[0],nose_point_new[1],s=10,c='orange')
    ax.scatter(neck_point_new[0],neck_point_new[1],s=10,c='red')
    ax.scatter(root_tail_point_new[0],root_tail_point_new[1],s=10,c='green')
    plt.xlim(-100,100)
    plt.ylim(-100,100)
    plt.show()

def point_align(nose_point,neck_point,back_point,root_tail_point):
    back_point_x = back_point[0]
    back_point_y = back_point[1]
    back_point_x_revise = back_point_x - back_point_x
    back_point_y_revise = back_point_y - back_point_y
    nose_point_x_revise = nose_point[0] - back_point_x
    nose_point_y_revise = nose_point[1] - back_point_y
    neck_point_x_revise = neck_point[0] - back_point_x
    neck_point_y_revise = neck_point[1] - back_point_y
    root_tail_point_x_revise = root_tail_point[0] - back_point_x
    root_tail_point_y_revise = root_tail_point[1] - back_point_y
    back_point_new = (back_point_x_revise,back_point_y_revise)

    if root_tail_point_x_revise == 0:
        nose_point_new = (nose_point_x_revise,nose_point_y_revise)
        neck_point_new = (neck_point_x_revise,neck_point_y_revise)
        root_tail_point_new = (0,root_tail_point_y_revise)
    else:
        clockwise_revise_angle = cal_ang((root_tail_point_x_revise,root_tail_point_y_revise), (0,0), (0,-1))

        if root_tail_point_x_revise>0:
            nose_point_new = Srotate(math.radians(clockwise_revise_angle),nose_point_x_revise,nose_point_y_revise,0,0)
            neck_point_new = Srotate(math.radians(clockwise_revise_angle),neck_point_x_revise,neck_point_y_revise,0,0)
            root_tail_point_new = Srotate(math.radians(clockwise_revise_angle),root_tail_point_x_revise,root_tail_point_y_revise,0,0)
        else:
            nose_point_new = Nrotate(math.radians(clockwise_revise_angle),nose_point_x_revise,nose_point_y_revise,0,0)
            neck_point_new = Nrotate(math.radians(clockwise_revise_angle),neck_point_x_revise,neck_point_y_revise,0,0)
            root_tail_point_new = Nrotate(math.radians(clockwise_revise_angle),root_tail_point_x_revise,root_tail_point_y_revise,0,0)
    #plot(nose_point_new,neck_point_new,back_point_new,root_tail_point_new)
    return(nose_point_new,neck_point_new,(0,0),root_tail_point_new)


def calculate_average_angle(df):
    
    df_singleLocomotion = df.copy()
    df_singleLocomotion.reset_index(drop=True,inplace=True)
    df_singleLocomotion.loc[len(df_singleLocomotion.index)] = df_singleLocomotion.mean(axis=0)
    
    nose_average = [round(df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_x'],3),round(df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_y'],3)]
    neck_average = [round(df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_x'],3),round(df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_y'],3)]
    back_average = [round(df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_x'],3),round(df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_y'],3)]
    root_tail_average = [round(df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_x'],3),round(df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_y'],3)]

    nose_point_new,neck_point_new,back_point_new,root_tail_point_new = point_align(nose_average,neck_average,back_average,root_tail_average)

    angle1 = cal_ang(nose_point_new, neck_point_new, back_point_new)
    angle2 = cal_ang(neck_point_new, back_point_new,root_tail_point_new)
    

    ## 计算角度
    if nose_point_new[0]<0:
        angle1 = -angle1
    
    if neck_point_new[0]<0:
        angle2 = -angle2
    #angle2 = cal_ang(neck_average, back_average,root_tail_average)
    #angle3 = cal_ang(back_average,root_tail_average,mid_tail_average)

    return(angle1,angle2)


def calculate_average_angle3D(df):
    
    df_singleLocomotion = df.copy()
    df_singleLocomotion.reset_index(drop=True,inplace=True)
    df_singleLocomotion.loc[len(df_singleLocomotion.index)] = df_singleLocomotion.mean(axis=0)
    
    nose_average = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_z']]
    neck_average = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_z']]
    back_average = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_z']]
    root_tail_average = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_z']]
    
    
    ## 计算角度
    angle1 =  cal_ang(nose_average, neck_average, back_average)
    angle2 = cal_ang(neck_average, back_average,root_tail_average)
    #angle3 = cal_ang(back_average,root_tail_average,mid_tail_average)
    
    return(angle1,angle2)






ske_data_list = []
for video_index in list(Morning_lightOn_info['video_index']):
#for index in Morning_lightOn_info.index:
    
    ske_data = pd.read_csv(Skeleton_path[video_index],index_col=0)
    Mov_data = pd.read_csv(Movement_Label_path[video_index],usecols=['OriginalDigital_label','revised_movement_label','locomotion_speed_smooth'])
    conbime_data = pd.concat([Mov_data,ske_data],axis=1)
    
    #select_data = conbime_data[conbime_data['revised_movement_label'].isin(pausing_list)]
    #select_data = conbime_data[conbime_data['OriginalDigital_label'].isin(grooming_list)]
    ske_data_list.append(conbime_data)


all_df = pd.concat(ske_data_list,axis=0)
all_df.reset_index(drop=True,inplace=True)


average_dict = {'revised_movement_label':[],
               'smooth_speed':[],'smooth_speed_std':[],

               'nose_z':[],'nose_z_std':[],
               'back_z':[],'back_z_std':[],
               'angle1':[],
               'angle2':[],
               'angle3':[],
               'angle4':[],}

for mv in all_df['revised_movement_label'].unique():
    temp_df = all_df[all_df['revised_movement_label']==mv]
    #temp_df.reset_index(drop=True,inplace=True)
    average_dict['revised_movement_label'].append(mv)
 
    average_dict['smooth_speed'].append(temp_df['locomotion_speed_smooth'].mean(axis=0))
    average_dict['smooth_speed_std'].append(temp_df['locomotion_speed_smooth'].std(axis=0))
    
    average_dict['nose_z'].append(temp_df['nose_z'].mean(axis=0))
    average_dict['nose_z_std'].append(temp_df['nose_z'].std(axis=0))
    
    average_dict['back_z'].append(temp_df['back_z'].mean(axis=0))
    average_dict['back_z_std'].append(temp_df['back_z'].std(axis=0))
    
    according_Ske = all_df.loc[temp_df.index,['nose_x','nose_y','neck_x','neck_y','back_x','back_y','root_tail_x','root_tail_y']]
    according_Ske.reset_index(drop=True,inplace=True)
    a,b = calculate_average_angle(according_Ske)
    average_dict['angle1'].append(a)
    average_dict['angle2'].append(b)

    according_Ske3D = all_df.loc[temp_df.index,['nose_x','nose_y','nose_z','neck_x','neck_y','neck_z','back_x','back_y','back_z','root_tail_x','root_tail_y','root_tail_z']]
    according_Ske3D.reset_index(drop=True,inplace=True)
    c,d = calculate_average_angle3D(according_Ske3D)
    average_dict['angle3'].append(c)
    average_dict['angle4'].append(d)
    
    #body_shrinkage = bodyShrinkage(temp_df)
    #average_dict['body_shrinkage'].append(body_shrinkage)
    
    
summary_df = pd.DataFrame(average_dict)
summary_df.to_csv('{}\movement_para.csv'.format(output_dir),index=None)