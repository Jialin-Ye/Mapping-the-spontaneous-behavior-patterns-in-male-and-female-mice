#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 17:26:41 2022

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""


import os 
import pandas as pd 
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from itertools import chain
import mpl_scatter_density
import math
import scipy.optimize as optimize
import time

output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure2_skeleton_kinematics\stride'
normalized_ske_file_dir   = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\01_BehaviorAtlas_collated_data'
revised_data_path_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label'


def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if file_name.endswith(content):
            USN = int(file_name.split('-')[1])
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(USN,date)
            file_path_dict.setdefault(USN,file_dir+'/'+file_name)
    return(file_path_dict)


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

Skeleton_path = get_path(normalized_ske_file_dir,'normalized_skeleton_XYZ.csv')
Movement_Label_path = get_path(revised_data_path_dir,'revised_Movement_Labels.csv')
Feature_space_path = get_path(revised_data_path_dir,'Feature_Space.csv')

animal_info_csv =r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Table_S1_animal_information.csv'             ## 动物信息表位置
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]


Morning_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-off')]


movement_order = ['running','trotting','walking','stepping', 'right_turning','left_turning',
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


locomotion_list = ['running','trotting','walking','stepping'] #'left_turning','right_turning',


plot_dataset = Morning_lightOn_info

def judge_ascending(alist,blist):       ## 判断递增
    new_list1 = []               #all_index
    new_list2 = []               #all_index
    peak_true = []               #peak_index
    valley_true = []              #value_index
    for i in range(1,len(alist)):
        v1 = alist[i-1]
        v2 = alist[i]
        v3 = blist[i-1]
        v4 = blist[i]
        if v1 < v2:
            new_list1.append(v1)
            if v3 > 0:
                peak_true.append(v1)
                new_list2.append(v3)
            else:
                valley_true.append(v1)
                new_list2.append(-v3)
        else:
            break
    return(peak_true,valley_true,new_list1,new_list2)


def find_sequence_peaks(arr,height,distance,prominence):
    indices= find_peaks(arr,height=height, threshold=None, distance=distance, prominence=prominence, width=None, wlen=None, rel_height=None, plateau_size=None)
    time_index = indices[0]
    value = indices[1]['peak_heights']
    return(time_index,value)
    

def calculate_GapAndValue(distance_name,data,mv):
    if mv == 'walking':
        height1=height2 = 45
        distance = 3
        prominence = 2
        length = 10
    if mv == 'running':
        height1=height2 = 50
        distance = 1
        prominence = 1
        length = 4
    if mv == 'trotting':
        height1=height2 = 50
        distance = 1
        prominence = 1
        length = 4
    if mv == 'stepping':
        height1=height2 = 45
        distance = 5
        prominence = 0.5
        length = 4
    if mv == 'right_turning':
        height1 = 45
        height2 = 50
        distance = 5
        prominence = 0.5
        length = 4
    if mv == 'left_turning':
        height1 = 40
        height2 = 50
        distance = 1
        prominence = 0.5
        length = 4
    num = 0
    step_info_dict = {'revised_movement_label':[],'step_type':[],'bouts':[],'valley_time_gap':[],'vally_value':[],'peak_time_gap':[],'peak_value':[]}
    #peak1_indices= find_peaks(data,height=height1, threshold=None, distance=distance, prominence=prominence, width=None, wlen=None, rel_height=None, plateau_size=None)
    #valley1_indices = find_peaks(-data,height=-height2, threshold=None, distance=distance, prominence=prominence, width=None, wlen=None, rel_height=None, plateau_size=None)
    #peak1_time_index = peak1_indices[0]
    #peak1_value = peak1_indices[1]['peak_heights']
    #valley1_time_index = valley1_indices[0]
    #valley1_value = -valley1_indices[1]['peak_heights']
    
    peak1_time_index,peak1_value = find_sequence_peaks(data,height=height1,distance=distance,prominence=prominence)
    valley1_time_index,valley1_value = find_sequence_peaks(-data,height=-height2,distance=distance,prominence=prominence)
    
    if len(peak1_time_index) >0 and len(valley1_time_index) >0:
        #new_time_sequence,new_value = judge_ascending2(peak1_time_index,valley1_time_index,peak1_value,valley1_value)
        if peak1_time_index[0] < valley1_time_index[0]:
            time_sequence = list(chain.from_iterable(zip(peak1_time_index, valley1_time_index)))
            value = list(chain.from_iterable(zip(peak1_value, valley1_value)))
        else:
            time_sequence = list(chain.from_iterable(zip(valley1_time_index, peak1_time_index)))
            value = list(chain.from_iterable(zip(valley1_value, peak1_value)))
        
        #new_time_sequence,new_value = judge_ascending(time_sequence,value)
        #peak_true_index_list,valley_true_index_list= judge_ascending(time_sequence,value)
        peak_true_index_list,valley_true_index_list,new_time_sequence,new_value = judge_ascending(time_sequence,value)

        if len(peak_true_index_list+valley_true_index_list)>length:
            peak_gap_average = np.mean(np.diff(peak_true_index_list))
            peak_value_average = np.mean(data[peak_true_index_list])
            valley_gap_average = np.mean(np.diff(valley_true_index_list))
            valley_value_average = np.mean(data[valley_true_index_list])
            step_info_dict['peak_time_gap'].append(peak_gap_average)
            step_info_dict['peak_value'].append(peak_value_average)
            step_info_dict['valley_time_gap'].append(valley_gap_average)
            step_info_dict['vally_value'].append(valley_value_average)
            step_info_dict['revised_movement_label'].append(mv)
            step_info_dict['bouts'].append(num)
            step_info_dict['step_type'].append(distance_name)
            
            
            #fig = plt.figure(figsize=(10,10),constrained_layout=True,dpi=300)
            #ax = fig.add_subplot(1, 1, 1)
            #ax.plot(data)
            #ax.plot(new_time_sequence,new_value,c='orange')
            #ax.plot(peak1_indices[0], data[peak1_indices[0]], 'o',c='red')
            #ax.plot(valley1_indices[0], data[valley1_indices[0]], 'o',c='green')
            #ax.plot(valley_true_index_list, data[valley_true_index_list]-2, '^',c='green')
    df = pd.DataFrame(step_info_dict)
    return(df)

def averageXYsquence(peak_value,peak_gap,valley_value,valley_gap,num):
    
    x_list = [0,]
    y_list = [valley_value,]
    
    x = 0
    y = valley_value
    for num in range(1,num):
        if num%2 != 0:
            x += peak_gap
            y = peak_value
            x_list.append(x)
            y_list.append(y)
        else:
            x += valley_gap
            y = valley_value
            x_list.append(x)
            y_list.append(y)
    return(x_list,y_list)

def sinY(arr,peak_value,valley_value,peak_gap,valley_gap):
    A = (peak_value-valley_value)/2            #幅值
    B = (2*np.pi)/(peak_gap+valley_gap)        #周期
    C = 4.75                                   #相位
    D = (peak_value + valley_value)/2          #上移
    arrY = []
    for x in arr:
        y = A*math.sin(B*x + C) + D
        arrY.append(y)
    return(arrY)

def plot_curve(all_df):

    
    fig = plt.figure(figsize=(15,10),dpi=1200)
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    for i in all_df['revised_movement_label'].unique():

        color = movement_color_dict[i]
        peak_time_gap_value = all_df.loc[all_df['revised_movement_label']==i,'peak_time_gap'].round(3)
        peak_value_value = all_df.loc[all_df['revised_movement_label']==i,'peak_value'].round(3)
        valley_time_gap_value = all_df.loc[all_df['revised_movement_label']==i,'valley_time_gap'].round(3)
        valley_alue_value = all_df.loc[all_df['revised_movement_label']==i,'vally_value'].round(3)
        
        peak_gap = np.mean(peak_time_gap_value)
        peak_value = np.mean(peak_value_value)
        valley_gap = np.mean(valley_time_gap_value)
        valley_value = np.mean(valley_alue_value)
        
        if i == 'running':
            num = 7
        elif i == 'trotting':
            num = 7
        else:
            num = 5
            
        step_strike_x, step_strike_y = averageXYsquence(peak_value,peak_gap,valley_value,valley_gap,num)
        
        if i == 'running':
            linestype = '-'
        elif i == 'trotting':
            linestype = '--'
        elif i == 'walking':
            linestype = '-'
        elif i == 'stepping':
            linestype = '--'
            
        t = np.arange(0.0, round(step_strike_x[-1],2),0.1)
        s = sinY(t,peak_value,valley_value,peak_gap,valley_gap)
        ax.plot(t, s,c=color,lw=5,linestyle= linestype)
        #ax.plot(step_strike_x,step_strike_y,c=color,lw=5)
    
    ax.set_yticks(np.arange(42,59,4))
    ax.set_yticklabels(range(42,59,4),family='arial',color='black', weight='bold', size = 30)
    ax.set_xticks(np.arange(0,41,10))
    ax.set_xticklabels(range(0,41,10),family='arial',color='black', weight='bold', size = 30)
    ax.spines['bottom'].set_linewidth(2)###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2)####设置左边坐标轴的粗细
    ax.spines['top'].set_visible(False) #去掉上边框
    ax.spines['right'].set_visible(False) #去掉右边框
    plt.tick_params(pad=10)
    plt.savefig('{}/strike_info.png'.format(output_dir),dpi=1200)
    
    #ax.scatter_density(peak_time_gap_value.to_list(),peak_value_value.to_list(),c=color,alpha=0.5)
    #ax.scatter_density(valley_time_gap_value, valley_alue_value, 'o')



t1 = time.time()
df_list = []
#for key in list(FeA_path.keys())[0:5]:
for key in plot_dataset['video_index']:
    FeA_file = pd.read_csv(Feature_space_path[key])
    FeA_file = FeA_file[FeA_file['revised_movement_label'].isin(locomotion_list)]
    Skeleton_file = pd.read_csv(Skeleton_path[key],index_col=0)
    for mv in ['running','trotting','walking','stepping']:#FeA_file['movement_label'].unique(): #,'left_turning','right_turning'
        singleLocomotionFeA = FeA_file[FeA_file['revised_movement_label']==mv]
        num = 0
        for i in singleLocomotionFeA.index:
            time_sequence = [0]
            value = [0]
            segBoundary_start = singleLocomotionFeA.loc[i,'segBoundary_start']
            segBoundary_end = singleLocomotionFeA.loc[i,'segBoundary_end']
            frame_length = singleLocomotionFeA.loc[i,'frame_length']
            distance1 = np.array(np.sqrt((Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_front_claw_x']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_hind_claw_x'])**2+(Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_front_claw_y']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_hind_claw_y'])**2))
            distance2 = np.array(np.sqrt((Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_front_claw_x']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_hind_claw_x'])**2+(Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_front_claw_y']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_hind_claw_y'])**2))
            distance3 = np.array(np.sqrt((Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_front_claw_x']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_hind_claw_x'])**2+(Skeleton_file.loc[segBoundary_start:segBoundary_end,'left_front_claw_y']-Skeleton_file.loc[segBoundary_start:segBoundary_end,'right_hind_claw_y'])**2))
            
            #fig = plt.figure(figsize=(10,10),constrained_layout=True,dpi=300)
            #ax = fig.add_subplot(1, 1, 1)
            #ax.plot(distance1)

            df_distance1 = calculate_GapAndValue('distance1',distance1,mv)
            
            if df_distance1.shape[0] > 0:
                num += 1
                df_distance1['bouts'] = num
                df_list.append(df_distance1)

all_df = pd.concat(df_list,axis=0)
all_df.reset_index(drop=True,inplace=True)
all_df.to_csv('{}/strike_info.csv'.format(output_dir))

plot_curve(all_df)
t2 = time.time()
print('Time comsue:{:.2f} min \n'.format((t2-t1)/60))
# 拟合sin曲线


                
