# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 09:54:44 2023

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
import random

output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure2_skeleton_kinematics\movement_parametter'
InputData_path_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label"
SkeData_path_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\01_BehaviorAtlas_collated_data'
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

normal_ske_path = get_path(SkeData_path_dir,'normalized_skeleton_XYZ.csv')

animal_info_csv = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Table_S1_animal_information.csv'             ## 动物信息表位置
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]

Morning_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-off')]


movement_order = ['running','trotting','walking','right_turning','left_turning','stepping',
                  'jumping','climbing','rearing','hunching','rising','sniffing',
                  'grooming','pausing']


movement_color_dict = {'running':'#DD2C00',
                       'trotting':'#EC407A',
                       'walking':'#FF5722',
                       'left_turning':'#FFAB91',
                       'right_turning':'#FFCDD2',
                       'stepping':'#BCAAA4',
                       'sniffing':'#26A69A',
                       'climbing':'#43A047',
                       'rearing':'#66BB6A',
                       'hunching':'#0288D1',
                       'rising':'#9CCC65',
                       'jumping':'#FFB74D',
                       'grooming':'#AB47BC',
                       'pausing':'#90A4AE',}



def cal_ang(point_1, point_2, point_3):
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
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


# =============================================================================
# def calculate_average_angle(df):
# 
#     angle1_list = []
#     angle2_list = []
#     for i in df.index:
#         nose_point = (df.loc[i,'nose_x'],df.loc[i,'nose_y'])
#         neck_point = (df.loc[i,'neck_x'],df.loc[i,'neck_y'])
#         back_point = (df.loc[i,'back_x'],df.loc[i,'back_y'])
#         root_tail_point = (df.loc[i,'root_tail_x'],df.loc[i,'root_tail_y'])
#         nose_point_new,neck_point_new,back_point_new,root_tail_point_new = point_align(nose_point,neck_point,back_point,root_tail_point)
# 
#         angle1 = cal_ang(nose_point_new,back_point_new,root_tail_point_new)
#         angle2 = cal_ang(neck_point_new,back_point_new,root_tail_point_new)
#         if nose_point_new[0] < 0:
#             angle1 = -angle1
#         elif nose_point_new[0] >=0:
#             angle1 = angle1
#         if neck_point_new[0] < 0:
#             angle2 = -angle2
#         elif neck_point_new[0] >=0:
#             angle2 = angle2
#         #angle1 = clockwise_angle(nose_point,back_point,root_tail_point)
#         #angle2 = clockwise_angle(neck_point,back_point,root_tail_point)
#         angle1_list.append(angle1)
#         angle2_list.append(angle2)
#         arr1 = np.array(angle1_list)
#         arr2 = np.array(angle2_list)
#     return(np.mean(arr1),np.mean(arr2))
# =============================================================================


def bodyShrinkage(df):
   
    df_singleMov = df.copy()
    angle_info_dict = {'nose-back_distance':[],'neck-back_distance':[],'root_tail-back_distance':[],'left_front_limb-back_distance':[],'right_front_limb-back_distance':[],'left_hind_limb-back_distance':[],'right_hind_limb-back_distance':[]}
    #for i in df_select['origin_label'].unique():
    #    df_singleMov = df_select[df_select['origin_label'] == i]
    #    df_singleMov.reset_index(drop=True,inplace=True)
    #    movement_label = df_singleMov['new_label'][0]
       
    df_singleMov['nose-back_distance'] = np.sqrt(np.square(df_singleMov['nose_x']-df_singleMov['back_x'])+np.square(df_singleMov['nose_y']-df_singleMov['back_y'])+np.square(df_singleMov['nose_z']-df_singleMov['back_z']))
    df_singleMov['neck-back_distance'] = np.sqrt(np.square(df_singleMov['neck_x']-df_singleMov['back_x'])+np.square(df_singleMov['neck_y']-df_singleMov['back_y'])+np.square(df_singleMov['neck_z']-df_singleMov['back_z']))
    df_singleMov['root_tail-back_distance'] = np.sqrt(np.square(df_singleMov['root_tail_x']-df_singleMov['back_x'])+np.square(df_singleMov['root_tail_y']-df_singleMov['back_y'])+np.square(df_singleMov['root_tail_z']-df_singleMov['back_z']))
    df_singleMov['left_front_limb-back_distance'] = np.sqrt(np.square(df_singleMov['left_front_limb_x']-df_singleMov['back_x'])+np.square(df_singleMov['left_front_limb_y']-df_singleMov['back_y'])+np.square(df_singleMov['left_front_limb_z']-df_singleMov['back_z']))
    df_singleMov['right_front_limb-back_distance'] = np.sqrt(np.square(df_singleMov['right_front_limb_x']-df_singleMov['back_x'])+np.square(df_singleMov['right_front_limb_y']-df_singleMov['back_y'])+np.square(df_singleMov['right_front_limb_z']-df_singleMov['back_z']))
    df_singleMov['left_hind_limb-back_distance'] = np.sqrt(np.square(df_singleMov['left_hind_limb_x']-df_singleMov['back_x'])+np.square(df_singleMov['left_hind_limb_y']-df_singleMov['back_y'])+np.square(df_singleMov['left_hind_limb_z']-df_singleMov['back_z']))
    df_singleMov['right_hind_limb-back_distance'] = np.sqrt(np.square(df_singleMov['right_hind_limb_x']-df_singleMov['back_x'])+np.square(df_singleMov['right_hind_limb_y']-df_singleMov['back_y'])+np.square(df_singleMov['right_hind_limb_z']-df_singleMov['back_z']))
    
    df_singleMov['sum'] = df_singleMov.sum(axis=1)
    
    return(df_singleMov['sum'].mean())



dataset = Morning_lightOn_info


df_list = []
for file_name in dataset['video_index']:

    Mov_data = pd.read_csv(Movement_Label_path[file_name])
    #Ske_data = pd.read_csv(Ske_path[file_name],usecols=['nose_z','neck_z','back_z','nose_x','nose_y','neck_x','neck_y','back_x','back_y','root_tail_x','root_tail_y',
    #                                                    'root_tail_z',
    #                                                    'left_front_limb_x','left_front_limb_y','left_front_limb_z',
    #                                                    'right_front_limb_x','right_front_limb_y','right_front_limb_z',
    #                                                    'left_hind_limb_x','left_hind_limb_y','left_hind_limb_z',
    #                                                    ])
    Ske_data = pd.read_csv(normal_ske_path[file_name],usecols=['nose_z','neck_z','back_z','nose_x','nose_y','neck_x','neck_y','back_x','back_y','root_tail_x','root_tail_y',
                                                        'root_tail_z',
                                                        'left_front_limb_x','left_front_limb_y','left_front_limb_z',
                                                        'right_front_limb_x','right_front_limb_y','right_front_limb_z',
                                                        'right_hind_limb_x','right_hind_limb_y','right_hind_limb_z',
                                                        'left_hind_limb_x','left_hind_limb_y','left_hind_limb_z',
                                                        ])

    df = pd.concat([Mov_data,Ske_data],axis=1)
    df_list.append(df)

all_df = pd.concat(df_list)
all_df.reset_index(drop=True,inplace=True)
#all_df = all_df.sample(frac=1, replace=False, random_state=1)




average_dict = {'revised_movement_label':[],
               #'paraSpeed':[],'paraSpeed_std':[],
               'smooth_speed':[],'smooth_speed_std':[],

               'nose_z':[],'nose_z_std':[],
               'back_z':[],'back_z_std':[],}
               #'angle1':[],
               #'angle2':[],
               #'angle3':[],
               #'angle4':[],
               #'body_shrinkage':[],}


for mv in movement_order:
    temp_df = all_df[all_df['revised_movement_label']==mv]
    #temp_df.reset_index(drop=True,inplace=True)
    average_dict['revised_movement_label'].append(mv)
    #average_dict['paraSpeed'].append(temp_df['speed_back'].mean(axis=0))
    #average_dict['paraSpeed_std'].append(temp_df['speed_back'].std(axis=0))
 
    average_dict['smooth_speed'].append(temp_df['locomotion_speed_smooth'].mean(axis=0))
    average_dict['smooth_speed_std'].append(temp_df['locomotion_speed_smooth'].std(axis=0))
    
    average_dict['nose_z'].append(temp_df['nose_z'].mean(axis=0))
    average_dict['nose_z_std'].append(temp_df['nose_z'].std(axis=0))
    average_dict['back_z'].append(temp_df['back_z'].mean(axis=0))
    average_dict['back_z_std'].append(temp_df['back_z'].std(axis=0))
    
# =============================================================================
#     
#     
#     according_Ske = all_df.loc[temp_df.index,['nose_x','nose_y','neck_x','neck_y','back_x','back_y','root_tail_x','root_tail_y']]
#     according_Ske.reset_index(drop=True,inplace=True)
#     a,b = calculate_average_angle(according_Ske)
#     average_dict['angle1'].append(a)
#     average_dict['angle2'].append(b)
# 
#     according_Ske3D = all_df.loc[temp_df.index,['nose_x','nose_y','nose_z','neck_x','neck_y','neck_z','back_x','back_y','back_z','root_tail_x','root_tail_y','root_tail_z']]
#     according_Ske3D.reset_index(drop=True,inplace=True)
#     c,d = calculate_average_angle3D(according_Ske3D)
#     average_dict['angle3'].append(c)
#     average_dict['angle4'].append(d)
#     
#     body_shrinkage = bodyShrinkage(temp_df)
#     average_dict['body_shrinkage'].append(body_shrinkage)
#     
# =============================================================================
    
summary_df = pd.DataFrame(average_dict)
summary_df.to_csv('{}/movement_label_para.csv'.format(output_dir))