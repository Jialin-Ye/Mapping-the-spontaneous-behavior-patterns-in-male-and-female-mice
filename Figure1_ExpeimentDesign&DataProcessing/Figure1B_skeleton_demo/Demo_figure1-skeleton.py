#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 09:22:20 2022

@author: yejohnny
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from scipy.stats import gaussian_kde
import mpl_scatter_density
import os 

output_dir = r'D:\Personal_File\yejiailn\课题\文章\图\第一版\figure_demo'

new_ske_file = r'D:\Personal_File\yejiailn\3D_spontaneous_data\DayAMLightOn_NightLightOn\result\normalized_skeleton_z_all/rec-1-G1_normalized_skeleton_locomotion.csv'
mov_file = r'D:\Personal_File\yejiailn\3D_spontaneous_data\DayAMLightOn_NightLightOn\result\anno_MV_csv-3sThreshold/rec-1-G1-anno_Movement_Labels.csv'





locomotion_list = ['running','walking','left_turning','right_turning','stepping']

def label(x, color, label):
    ax = plt.gca()
    ax.text(1, .2, label, color='black', fontsize=13,
            ha="left", va="center", transform=ax.transAxes)
def cauculate_speed_para(df):
    df_dict = {'movement_label':[],'average_speed':[],'min_speed':[],'max_speed':[],'speed_std':[]}
    for i in locomotion_list:
        temp_df = df[df['new_label']==i]
        average_speed = temp_df['new_speed_back'].mean()
        min_speed = temp_df['new_speed_back'].min()
        max_speed = temp_df['new_speed_back'].max()
        speed_std = temp_df['new_speed_back'].std()
        df_dict['movement_label'].append(i)
        df_dict['average_speed'].append(average_speed)
        df_dict['min_speed'].append(min_speed)
        df_dict['max_speed'].append(max_speed)
        df_dict['speed_std'].append(speed_std)
    df_out = pd.DataFrame(df_dict)
    return(df_out)

## 速度分布
def speed_distribution(df):
    df_temp = all_df[['new_label','new_speed_back']]
    df_locomotion = df_temp[df_temp['new_label'].isin(locomotion_list)]
    movement_color_dict={'running':'#F44336','walking':'#FF5722','left_turning':'#FFAB91','right_turning':'#FFCDD2','stepping':'#BCAAA4',
                     'sniffing':'#26A69A','climb_up':'#43A047','rearing':'#66BB6A','hunching':'#81C784','rising':'#9CCC65','jumping':'#FFB74D',
                     'grooming':'#AB47BC','pause':'#90A4AE',}
    speed_para = cauculate_speed_para(df_locomotion)
    print(speed_para)
    
    #g = sns.kdeplot(data=df_sort,x="average_speed", fill=True, alpha=0.7)
    
    palette = movement_color_dict
    #palette = Mov_color_dict
    
    g = sns.FacetGrid(df_locomotion, palette=palette, row="new_label", hue="new_label",row_order=locomotion_list, aspect=5, height=1.5)
    g.map_dataframe(sns.kdeplot, x="new_speed_back", fill=True, alpha=0.7,common_norm=True,cut=1,gridsize=1000)
    g.map_dataframe(sns.kdeplot, x="new_speed_back", color='black',common_norm=True,cut=1,gridsize=1000)
    
    g.map(label, "new_label")
    g.fig.subplots_adjust(hspace=0.1)
    g.set(xlim=(0, 500))
    g.set_titles("")
    g.set(yticks=[], xlabel="new_speed_back")
    g.despine(left=True)
    
    plt.suptitle('Speed of all_locomotion_fragement', y=1)



ske_data = pd.read_csv(new_ske_file,index_col=0)
Mov_data = pd.read_csv(mov_file,usecols=['new_label'])
conbime_data = pd.concat([Mov_data,ske_data],axis=1)

#select_data = conbime_data[conbime_data['new_label'].isin(select_list)]
select_data = conbime_data.iloc[1188:1327+51,:]


#speed_distribution(df_select)
## 步频


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

def plot_TopView_skeleton(df_select):
    df_singleLocomotion = df_select
    #df_singleLocomotion = df_select[df_select['origin_label'] == i]
    df_singleLocomotion.reset_index(drop=True,inplace=True)
    df_singleLocomotion.loc[len(df_singleLocomotion.index)] = df_singleLocomotion.mean(axis=0)
   
    nose = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_y']]
    left_ear = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_y']]
    right_ear = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_y']]
    neck = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_y']]
    left_front_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_y']]
    right_front_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_y']]
    left_hind_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_y']]
    right_hind_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_y']]
    left_front_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_y']]
    right_front_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_y']]
    left_hind_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_y']]
    right_hind_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_y']]
    back = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_y']]
    root_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_y']]
    mid_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_y']]
    tip_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_y']]
    
    fig = plt.figure(figsize=(10,10),constrained_layout=True,dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    #ax.scatter(nose[0],nose[1],s=60,c='green',alpha = 0.3)
    ax.scatter(nose[0],nose[1],s=120,c='#9C27B0',marker='o',ec = 'black')
    ax.scatter(left_ear[0],left_ear[1],s=60,c='green',alpha = 0.3)
    ax.scatter(right_ear[0],right_ear[1],s=60,c='green',alpha = 0.3)
    
    ax.plot([nose[0],left_ear[0]],[nose[1],left_ear[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([nose[0],right_ear[0]],[nose[1],right_ear[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([left_ear[0],right_ear[0]],[left_ear[1],right_ear[1]],c='black',alpha = 0.3,lw=5)
    
    ax.scatter(neck[0],neck[1],s=60,c='#607D5B',alpha = 0.3)
    ax.scatter(back[0],back[1],s=60,c='#607D5B',alpha = 0.3)
    ax.scatter(root_tail[0],root_tail[1],s=60,c='#607D5B',alpha = 0.3)
    ax.scatter(mid_tail[0],mid_tail[1],s=60,c='#607D5B',alpha = 0.3)
    #ax.scatter(tip_tail[0],tip_tail[1],s=40,c='orange')
    
    ax.scatter(left_front_limb[0],left_front_limb[1],s=60,c='blue',alpha = 0.3)
    ax.scatter(right_front_limb[0],right_front_limb[1],s=60,c='blue',alpha = 0.3)
    ax.scatter(left_hind_limb[0],left_hind_limb[1],s=60,c='blue',alpha = 0.3)
    ax.scatter(right_hind_limb[0],right_hind_limb[1],s=60,c='blue',alpha = 0.3)
    
    ax.scatter(left_front_claw[0],left_front_claw[1],s=250,c='#FF1744',marker='^',ec = 'black')
    ax.scatter(right_front_claw[0],right_front_claw[1],s=250,c='#F57C00',marker='^',ec = 'black')
    ax.scatter(left_hind_claw[0],left_hind_claw[1],s=250,c='#00E676',marker='^',ec = 'black')
    ax.scatter(right_hind_claw[0],right_hind_claw[1],s=250,c='#00E5FF',marker='^',ec = 'black')
    
    ax.plot([left_front_limb[0],left_hind_limb[0]],[left_front_limb[1],left_hind_limb[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([left_hind_limb[0],right_hind_limb[0]],[left_hind_limb[1],right_hind_limb[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([right_hind_limb[0],right_front_limb[0]],[right_hind_limb[1],right_front_limb[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([right_front_limb[0],left_front_limb[0]],[right_front_limb[1],left_front_limb[1]],c='black',alpha = 0.3,lw=5)
    
    ax.plot([left_ear[0],neck[0]],[left_ear[1],neck[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([right_ear[0],neck[0]],[right_ear[1],neck[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([left_front_limb[0],neck[0]],[left_front_limb[1],neck[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([right_front_limb[0],neck[0]],[right_front_limb[1],neck[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([left_hind_limb[0],root_tail[0]],[left_hind_limb[1],root_tail[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([right_hind_limb[0],root_tail[0]],[right_hind_limb[1],root_tail[1]],c='black',alpha = 0.3,lw=5)
    
    ax.plot([root_tail[0],mid_tail[0]],[root_tail[1],mid_tail[1]],c='black',alpha = 0.3,lw=5)
    #ax.plot([mid_tail[0],tip_tail[0]],[mid_tail[1],tip_tail[1]],c='black')
    

    ax.plot([left_front_limb[0],back[0]],[left_front_limb[1],back[1]],c='black',alpha = 0.3,lw=3)
    ax.plot([right_front_limb[0],back[0]],[right_front_limb[1],back[1]],c='black',alpha = 0.3,lw=3)
    ax.plot([left_hind_limb[0],back[0]],[left_hind_limb[1],back[1]],c='black',alpha = 0.3,lw=3)
    ax.plot([right_hind_limb[0],back[0]],[right_hind_limb[1],back[1]],c='black',alpha = 0.3,lw=3)
    
    
    
    ax.scatter_density(df_singleLocomotion.loc[:,'left_front_claw_x'], df_singleLocomotion.loc[:,'left_front_claw_y'], color = '#FF1744',alpha=1)
    ax.scatter_density(df_singleLocomotion.loc[:,'right_front_claw_x'], df_singleLocomotion.loc[:,'right_front_claw_y'], color = '#F57C00',alpha=1)
    ax.scatter_density(df_singleLocomotion.loc[:,'left_hind_claw_x'], df_singleLocomotion.loc[:,'left_hind_claw_y'], color ='#00E676',alpha=1)
    ax.scatter_density(df_singleLocomotion.loc[:,'right_hind_claw_x'], df_singleLocomotion.loc[:,'right_hind_claw_y'], color ='#00E5FF',alpha=1)
    ax.scatter_density(df_singleLocomotion.loc[:,'nose_x'], df_singleLocomotion.loc[:,'nose_y'], color ='#9C27B0',alpha=1)
    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)
        
        #plt.savefig('{}/{}_skeleton_topview.png'.format(output_dir,i),dpi=300)
#plot_TopView_skeleton(select_data)

def plot_3DsideView_skeleton(df_select,i):
#### 3D###
    body_color_dict = {'nose':'#1E2C59',
                       'left_ear':'#192887',
                       'right_ear':'#1B3A95',
                       'neck':'#204FA1',
                       'left_front_limb':'#1974BA',
                       'right_front_limb':'#1EA2BA',
                       'left_hind_limb':'#42B799',
                       'right_hind_limb':'#5CB979',
                       'left_front_claw':'#7BBF57',
                       'right_front_claw':'#9EC036',
                       'left_hind_claw':'#BEAF1F',
                       'right_hind_claw':'#C08719',
                       'back':'#BF5D1C',
                       'root_tail':'#BE3320',
                       'mid_tail':'#9B1F24',
                       'tip_tail':'#6A1517',
                                    
        }

    df_singleLocomotion = df_select
    df_singleLocomotion.reset_index(drop=True,inplace=True)
    df_singleLocomotion.loc[len(df_singleLocomotion.index)] = df_singleLocomotion.mean(axis=0)
    
    
    nose = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_z']]
    left_ear = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_z']]
    right_ear = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_z']]
    neck = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_z']]
    left_front_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_z']]
    right_front_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_z']]
    left_hind_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_z']]
    right_hind_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_z']]
    left_front_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_z']]
    right_front_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_z']]
    left_hind_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_z']]
    right_hind_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_z']]
    back = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_z']]
    root_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_z']]
    mid_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_z']]
    tip_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_z']]

    fig = plt.figure(figsize=(8,8),constrained_layout=True,dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(nose[0],nose[1],nose[2],s=60,c=body_color_dict['nose'],alpha = 0.5)
    ax.scatter(left_ear[0],left_ear[1],left_ear[2],s=60,c=body_color_dict['left_ear'],alpha = 0.5)
    ax.scatter(right_ear[0],right_ear[1],right_ear[2],s=60,c=body_color_dict['right_ear'],alpha = 0.5)
    ax.plot([nose[0],left_ear[0]],[nose[1],left_ear[1]],[nose[2],left_ear[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([nose[0],right_ear[0]],[nose[1],right_ear[1]],[nose[2],right_ear[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([left_ear[0],right_ear[0]],[left_ear[1],right_ear[1]],[left_ear[2],right_ear[2]],c='black',alpha = 0.3,lw=5)
    
    ax.scatter(neck[0],neck[1],neck[2],s=60,c=body_color_dict['neck'],alpha = 0.5)
    ax.scatter(back[0],back[1],back[2],s=60,c=body_color_dict['back'],alpha = 0.5)
    ax.scatter(root_tail[0],root_tail[1],root_tail[2],s=60,c=body_color_dict['root_tail'],alpha = 0.5)
    ax.scatter(mid_tail[0],mid_tail[1],mid_tail[2],s=60,c=body_color_dict['mid_tail'],alpha = 0.5)
    #ax.scatter(tip_tail[0],tip_tail[1],s=40,c='orange')
    
    ax.scatter(left_front_limb[0],left_front_limb[1],left_front_limb[2],s=60,c='black',alpha = 0.5)
    ax.scatter(right_front_limb[0],right_front_limb[1],right_front_limb[2],s=60,c='black',alpha = 0.5)
    ax.scatter(left_hind_limb[0],left_hind_limb[1],left_hind_limb[2],s=60,c='black',alpha = 0.5)
    ax.scatter(right_hind_limb[0],right_hind_limb[1],right_hind_limb[2],s=60,c='black',alpha = 0.5)
    
    ax.scatter(left_front_claw[0],left_front_claw[1],left_front_claw[2],s=60,c=body_color_dict['left_front_claw'],marker='o',ec = 'black',alpha = 0.5)
    ax.scatter(right_front_claw[0],right_front_claw[1],right_front_claw[2],s=60,c=body_color_dict['right_front_claw'],marker='o',ec = 'black',alpha = 0.5)
    ax.scatter(left_hind_claw[0],left_hind_claw[1],left_hind_claw[2],s=60,c=body_color_dict['left_hind_claw'],marker='o',ec = 'black',alpha = 0.5)
    ax.scatter(right_hind_claw[0],right_hind_claw[1],right_hind_claw[2],s=60,c=body_color_dict['right_hind_claw'],marker='o',ec = 'black',alpha = 0.5)
    
    ax.plot([left_front_limb[0],left_hind_limb[0]],[left_front_limb[1],left_hind_limb[1]],[left_front_limb[2],left_hind_limb[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([left_hind_limb[0],right_hind_limb[0]],[left_hind_limb[1],right_hind_limb[1]],[left_hind_limb[2],right_hind_limb[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([right_hind_limb[0],right_front_limb[0]],[right_hind_limb[1],right_front_limb[1]],[right_hind_limb[2],right_front_limb[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([right_front_limb[0],left_front_limb[0]],[right_front_limb[1],left_front_limb[1]],[right_front_limb[2],left_front_limb[2]],c='black',alpha = 0.5,lw=5)
    
    ax.plot([left_ear[0],neck[0]],[left_ear[1],neck[1]],[left_ear[2],neck[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([right_ear[0],neck[0]],[right_ear[1],neck[1]],[right_ear[2],neck[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([left_front_limb[0],neck[0]],[left_front_limb[1],neck[1]],[left_front_limb[2],neck[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([right_front_limb[0],neck[0]],[right_front_limb[1],neck[1]],[right_front_limb[2],neck[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([left_hind_limb[0],root_tail[0]],[left_hind_limb[1],root_tail[1]],[left_hind_limb[2],root_tail[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([right_hind_limb[0],root_tail[0]],[right_hind_limb[1],root_tail[1]],[right_hind_limb[2],root_tail[2]],c='black',alpha = 0.5,lw=5)
    
    ax.plot([root_tail[0],mid_tail[0]],[root_tail[1],mid_tail[1]],[root_tail[2],mid_tail[2]],c='black',alpha = 0.5,lw=5)
    #ax.plot([mid_tail[0],tip_tail[0]],[mid_tail[1],tip_tail[1]],c='black')
    

    ax.plot([left_front_limb[0],back[0]],[left_front_limb[1],back[1]],[left_front_limb[2],back[2]],c='black',alpha = 0.5,lw=3)
    ax.plot([right_front_limb[0],back[0]],[right_front_limb[1],back[1]],[right_front_limb[2],back[2]],c='black',alpha = 0.5,lw=3)
    ax.plot([left_hind_limb[0],back[0]],[left_hind_limb[1],back[1]],[left_hind_limb[2],back[2]],c='black',alpha = 0.5,lw=3)
    ax.plot([right_hind_limb[0],back[0]],[right_hind_limb[1],back[1]],[right_hind_limb[2],back[2]],c='black',alpha = 0.5,lw=3)
    
    ax.plot([left_front_limb[0],left_front_claw[0]],[left_front_limb[1],left_front_claw[1]],[left_front_limb[2],left_front_claw[2]],c='black',alpha = 0.5,lw=3)
    ax.plot([right_front_limb[0],right_front_claw[0]],[right_front_limb[1],right_front_claw[1]],[right_front_limb[2],right_front_claw[2]],c='black',alpha = 0.5,lw=3)
    ax.plot([left_hind_limb[0],left_hind_claw[0]],[left_hind_limb[1],left_hind_claw[1]],[left_hind_limb[2],left_hind_claw[2]],c='black',alpha = 0.5,lw=3)
    ax.plot([right_hind_limb[0],right_hind_claw[0]],[right_hind_limb[1],right_hind_claw[1]],[right_hind_limb[2],right_hind_claw[2]],c='black',alpha = 0.5,lw=3)
    
    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)
    ax.set_zlim(-50,150)

    ax.view_init(20, 160)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(False)
    plt.axis('off')
    #ax.scatter(df_singleLocomotion.loc[:,'left_front_claw_x'], df_singleLocomotion.loc[:,'left_front_claw_y'], df_singleLocomotion.loc[:,'left_front_claw_z'],color = '#FF1744',alpha=0.2,s=0.5)
    #ax.scatter(df_singleLocomotion.loc[:,'right_front_claw_x'], df_singleLocomotion.loc[:,'right_front_claw_y'],df_singleLocomotion.loc[:,'right_front_claw_z'], color = '#F57C00',alpha=0.2,s=0.5)
    #ax.scatter(df_singleLocomotion.loc[:,'left_hind_claw_x'], df_singleLocomotion.loc[:,'left_hind_claw_y'], df_singleLocomotion.loc[:,'left_hind_claw_z'],color ='#00E676',alpha=0.2,s=0.5)
    #ax.scatter(df_singleLocomotion.loc[:,'right_hind_claw_x'], df_singleLocomotion.loc[:,'right_hind_claw_y'], df_singleLocomotion.loc[:,'right_hind_claw_z'],color ='#00E5FF',alpha=0.2,s=0.5)
    plt.savefig('{}/{}.png'.format(output_dir,i),dpi=300,transparent=True)
start =0
end = 0
for i in range(0,len(select_data),10):
    end = i
    temp_df = select_data.iloc[start:end,:]
    plot_3DsideView_skeleton(temp_df,i)
    start = end

def plot_2DsideView_skeleton(df_select):    
    #### sideview
    df_singleLocomotion = df_select
    df_singleLocomotion.reset_index(drop=True,inplace=True)
    df_singleLocomotion.loc[len(df_singleLocomotion.index)] = df_singleLocomotion.mean(axis=0)
    
    nose = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_z']]
    left_ear = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_z']]
    right_ear = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_z']]
    neck = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_z']]
    left_front_limb = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_z']]
    right_front_limb = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_z']]
    left_hind_limb = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_z']]
    right_hind_limb = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_z']]
    left_front_claw = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_z']]
    right_front_claw = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_z']]
    left_hind_claw = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_z']]
    right_hind_claw = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_z']]
    back = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_z']]
    root_tail = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_z']]
    mid_tail = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_z']]#     tip_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_z']]

    fig = plt.figure(figsize=(9,9),constrained_layout=True,dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    #ax.scatter(nose[0],nose[1],s=60,c='green',alpha = 0.3)
    ax.scatter(nose[0],nose[1],s=120,c='#9C27B0',marker='o',ec = 'black')
    ax.scatter(left_ear[0],left_ear[1],s=60,c='green',alpha = 0.3)
    ax.scatter(right_ear[0],right_ear[1],s=60,c='green',alpha = 0.3)
    
    ax.plot([nose[0],left_ear[0]],[nose[1],left_ear[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([nose[0],right_ear[0]],[nose[1],right_ear[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([left_ear[0],right_ear[0]],[left_ear[1],right_ear[1]],c='black',alpha = 0.3,lw=5)
    
    ax.scatter(neck[0],neck[1],s=60,c='#607D5B',alpha = 0.3)
    ax.scatter(back[0],back[1],s=60,c='#607D5B',alpha = 0.3)
    ax.scatter(root_tail[0],root_tail[1],s=60,c='#607D5B',alpha = 0.3)
    ax.scatter(mid_tail[0],mid_tail[1],s=60,c='#607D5B',alpha = 0.3)
    #ax.scatter(tip_tail[0],tip_tail[1],s=40,c='orange')
    
    ax.scatter(left_front_limb[0],left_front_limb[1],s=60,c='blue',alpha = 0.3)
    ax.scatter(right_front_limb[0],right_front_limb[1],s=60,c='blue',alpha = 0.3)
    ax.scatter(left_hind_limb[0],left_hind_limb[1],s=60,c='blue',alpha = 0.3)
    ax.scatter(right_hind_limb[0],right_hind_limb[1],s=60,c='blue',alpha = 0.3)
    
    ax.scatter(left_front_claw[0],left_front_claw[1],s=250,c='#FF1744',marker='^',ec = 'black')
    ax.scatter(right_front_claw[0],right_front_claw[1],s=250,c='#F57C00',marker='^',ec = 'black')
    ax.scatter(left_hind_claw[0],left_hind_claw[1],s=250,c='#00E676',marker='^',ec = 'black')
    ax.scatter(right_hind_claw[0],right_hind_claw[1],s=250,c='#00E5FF',marker='^',ec = 'black')
    
    ax.plot([left_front_limb[0],left_hind_limb[0]],[left_front_limb[1],left_hind_limb[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([left_hind_limb[0],right_hind_limb[0]],[left_hind_limb[1],right_hind_limb[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([right_hind_limb[0],right_front_limb[0]],[right_hind_limb[1],right_front_limb[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([right_front_limb[0],left_front_limb[0]],[right_front_limb[1],left_front_limb[1]],c='black',alpha = 0.3,lw=5)
    
    ax.plot([left_ear[0],neck[0]],[left_ear[1],neck[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([right_ear[0],neck[0]],[right_ear[1],neck[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([left_front_limb[0],neck[0]],[left_front_limb[1],neck[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([right_front_limb[0],neck[0]],[right_front_limb[1],neck[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([left_hind_limb[0],root_tail[0]],[left_hind_limb[1],root_tail[1]],c='black',alpha = 0.3,lw=5)
    ax.plot([right_hind_limb[0],root_tail[0]],[right_hind_limb[1],root_tail[1]],c='black',alpha = 0.3,lw=5)
    
    ax.plot([root_tail[0],mid_tail[0]],[root_tail[1],mid_tail[1]],c='black',alpha = 0.3,lw=5)
    #ax.plot([mid_tail[0],tip_tail[0]],[mid_tail[1],tip_tail[1]],c='black')
    
    
    ax.plot([left_front_limb[0],back[0]],[left_front_limb[1],back[1]],c='black',alpha = 0.3,lw=3)
    ax.plot([right_front_limb[0],back[0]],[right_front_limb[1],back[1]],c='black',alpha = 0.3,lw=3)
    ax.plot([left_hind_limb[0],back[0]],[left_hind_limb[1],back[1]],c='black',alpha = 0.3,lw=3)
    ax.plot([right_hind_limb[0],back[0]],[right_hind_limb[1],back[1]],c='black',alpha = 0.3,lw=3)
    
    ax.plot([left_front_limb[0],left_front_claw[0]],[left_front_limb[1],left_front_claw[1]],c='black',alpha = 0.3,lw=3)
    ax.plot([right_front_limb[0],right_front_claw[0]],[right_front_limb[1],right_front_claw[1]],c='black',alpha = 0.3,lw=3)
    ax.plot([left_hind_limb[0],left_hind_claw[0]],[left_hind_limb[1],left_hind_claw[1]],c='black',alpha = 0.3,lw=3)
    ax.plot([right_hind_limb[0],right_hind_claw[0]],[right_hind_limb[1],right_hind_claw[1]],c='black',alpha = 0.3,lw=3)
    
    
    ax.plot([neck[0],nose[0]],[neck[1],nose[1]],c='#2E7D32',alpha = 0.5,lw=5)
    ax.plot([neck[0],back[0]],[neck[1],back[1]],c='#2E7D32',alpha = 0.5,lw=5)
    ax.plot([root_tail[0],back[0]],[root_tail[1],back[1]],c='#2E7D32',alpha = 0.5,lw=5)
    
    ax.scatter_density(-df_singleLocomotion.loc[:,'left_front_claw_y'], df_singleLocomotion.loc[:,'left_front_claw_z'], color = '#FF1744',alpha=1)
    ax.scatter_density(-df_singleLocomotion.loc[:,'right_front_claw_y'], df_singleLocomotion.loc[:,'right_front_claw_z'], color = '#F57C00',alpha=1)
    ax.scatter_density(-df_singleLocomotion.loc[:,'left_hind_claw_y'], df_singleLocomotion.loc[:,'left_hind_claw_z'], color ='#00E676',alpha=1)
    ax.scatter_density(-df_singleLocomotion.loc[:,'right_hind_claw_y'], df_singleLocomotion.loc[:,'right_hind_claw_z'], color ='#00E5FF',alpha=1)
    ax.scatter_density(-df_singleLocomotion.loc[:,'nose_y'], df_singleLocomotion.loc[:,'nose_z'], color ='#9C27B0',alpha=1)
    

    
    ax.set_xlim(-100,100)
    ax.set_ylim(-50,150)
    #plt.savefig('{}/{}_skeleton_sideview.png'.format(output_dir,i),dpi=600)
#plot_2DsideView_skeleton(select_data)



def plot_angle_sideview(df_select):         ### for stand up
    angle_info_dict = {'movement_label':[],'nose-neck-back':[],'neck-back-tail':[],'back-tail1-tail2':[]}
    for i in df_select['new_label'].unique():
        df_singleLocomotion = df_select[df_select['new_label'] == i]
        df_singleLocomotion.reset_index(drop=True,inplace=True)
        df_singleLocomotion.loc[len(df_singleLocomotion.index)] = df_singleLocomotion.mean(axis=0)
        
        nose_average = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_z']]
        neck_average = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_z']]
        back_average = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_z']]
        root_tail_average = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_z']]
        mid_tail_average = [-df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_z']]
        
        ## 计算角度
        angle1 = cal_ang(nose_average, neck_average, back_average)
        angle2 = cal_ang(neck_average, back_average,root_tail_average)
        angle3 = cal_ang(back_average,root_tail_average,mid_tail_average)
        angle_info_dict['movement_label'].append(i)
        angle_info_dict['nose-neck-back'].append(angle1)
        angle_info_dict['neck-back-tail'].append(angle2)
        angle_info_dict['back-tail1-tail2'].append(angle3)
        
        fig = plt.figure(figsize=(9,9),constrained_layout=True,dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        for j in df_singleLocomotion.index[0:-2]:
            nose = [-df_singleLocomotion.loc[j,'nose_y'],df_singleLocomotion.loc[j,'nose_z']]
            neck = [-df_singleLocomotion.loc[j,'neck_y'],df_singleLocomotion.loc[j,'neck_z']]
            back = [-df_singleLocomotion.loc[j,'back_y'],df_singleLocomotion.loc[j,'back_z']]
            root_tail = [-df_singleLocomotion.loc[j,'root_tail_y'],df_singleLocomotion.loc[j,'root_tail_z']]
            mid_tail = [-df_singleLocomotion.loc[j,'mid_tail_y'],df_singleLocomotion.loc[j,'mid_tail_z']]
            

            ax.scatter(nose[0],nose[1],s=40,c='#FF1744',alpha = 0.1)
            ax.scatter(neck[0],neck[1],s=40,c='#F57C00',alpha = 0.1)
            ax.scatter(back[0],back[1],s=40,c='#00E676',alpha = 0.1)
            ax.scatter(root_tail[0],root_tail[1],s=40,c='#00E5FF',alpha = 0.1)
            ax.scatter(mid_tail[0],mid_tail[1],s=40,c='#607D5B',alpha = 0.1)
            
            ax.plot([nose[0],neck[0]],[nose[1],neck[1]],c='#90A4AE',alpha = 0.1,lw=1)
            ax.plot([neck[0],back[0]],[neck[1],back[1]],c='#90A4AE',alpha = 0.1,lw=1)
            ax.plot([back[0],root_tail[0]],[back[1],root_tail[1]],c='#90A4AE',alpha = 0.1,lw=1)
            ax.plot([root_tail[0],mid_tail[0]],[root_tail[1],mid_tail[1]],c='#90A4AE',alpha = 0.1,lw=1)
        
        ax.scatter(nose_average[0],nose_average[1],s=260,c='#FF1744',ec='black',alpha = 1)
        ax.scatter(neck_average[0],neck_average[1],s=260,c='#F57C00',ec='black',alpha = 1)
        ax.scatter(back_average[0],back_average[1],s=260,c='#00E676',ec='black',alpha = 1)
        ax.scatter(root_tail_average[0],root_tail_average[1],s=260,ec='black',c='#00E5FF',alpha = 1)
        ax.scatter(mid_tail_average[0],mid_tail_average[1],s=260,c='#607D5B',ec='black',alpha = 1)
        
        ax.plot([nose_average[0],neck_average[0]],[nose_average[1],neck_average[1]],c='black',alpha = 1,lw=5)
        ax.plot([neck_average[0],back_average[0]],[neck_average[1],back_average[1]],c='black',alpha = 1,lw=5)
        ax.plot([back_average[0],root_tail_average[0]],[back_average[1],root_tail_average[1]],c='black',alpha = 1,lw=5)
        ax.plot([root_tail_average[0],mid_tail_average[0]],[root_tail_average[1],mid_tail_average[1]],c='black',alpha = 1,lw=5)
        plt.title(i,fontsize = 25)
        
        ax.set_xlim(-100,100)
        ax.set_ylim(-50,150)
        
        df = pd.DataFrame(angle_info_dict)
        print(df)
        #plt.savefig('{}/{}_angle_sideview.png'.format(output_dir,i),dpi=600)
#plot_angle_sideview(df_select)      

##### 计算身体弯曲程度
def plot_angle(df_select):
    angle_info_dict = {'movement_label':[],'nose-neck-back':[],'neck-back-tail':[],'back-tail1-tail2':[]}
    for i in df_select['new_label'].unique():
        df_singleLocomotion = df_select[df_select['new_label'] == i]
        df_singleLocomotion.reset_index(drop=True,inplace=True)
        df_singleLocomotion.loc[len(df_singleLocomotion.index)] = df_singleLocomotion.mean(axis=0)
        
        nose_average = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_y']]
        neck_average = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_y']]
        back_average = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_y']]
        root_tail_average = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_y']]
        mid_tail_average = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_y']]
        
        ## 计算角度
        angle1 = cal_ang(nose_average, neck_average, back_average)
        angle2 = cal_ang(neck_average, back_average,root_tail_average)
        angle3 = cal_ang(back_average,root_tail_average,mid_tail_average)
        angle_info_dict['movement_label'].append(i)
        angle_info_dict['nose-neck-back'].append(angle1)
        angle_info_dict['neck-back-tail'].append(angle2)
        angle_info_dict['back-tail1-tail2'].append(angle3)
        
        fig = plt.figure(figsize=(9,9),constrained_layout=True,dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    
        for j in df_singleLocomotion.index[0:-2]:
            nose = [df_singleLocomotion.loc[j,'nose_x'],df_singleLocomotion.loc[j,'nose_y']]
            neck = [df_singleLocomotion.loc[j,'neck_x'],df_singleLocomotion.loc[j,'neck_y']]
            back = [df_singleLocomotion.loc[j,'back_x'],df_singleLocomotion.loc[j,'back_y']]
            root_tail = [df_singleLocomotion.loc[j,'root_tail_x'],df_singleLocomotion.loc[j,'root_tail_y']]
            mid_tail = [df_singleLocomotion.loc[j,'mid_tail_x'],df_singleLocomotion.loc[j,'mid_tail_y']]
            
            if (i == 'left_turning') & (nose[0] < 20) & (nose[1]< 75):
                ax.scatter(nose[0],nose[1],s=60,c='#FF1744',alpha = 0.1)
                ax.scatter(neck[0],neck[1],s=60,c='#F57C00',alpha = 0.1)
                ax.scatter(back[0],back[1],s=60,c='#00E676',alpha = 0.1)
                ax.scatter(root_tail[0],root_tail[1],s=60,c='#00E5FF',alpha = 0.1)
                ax.scatter(mid_tail[0],mid_tail[1],s=60,c='#607D5B',alpha = 0.1)
                
                ax.plot([nose[0],neck[0]],[nose[1],neck[1]],c='#90A4AE',alpha = 0.1,lw=1)
                ax.plot([neck[0],back[0]],[neck[1],back[1]],c='#90A4AE',alpha = 0.1,lw=1)
                ax.plot([back[0],root_tail[0]],[back[1],root_tail[1]],c='#90A4AE',alpha = 0.1,lw=1)
                ax.plot([root_tail[0],mid_tail[0]],[root_tail[1],mid_tail[1]],c='#90A4AE',alpha = 0.1,lw=1)
            elif (i == 'walking') & (nose[0] > -30) & (nose[0]< 30) & (nose[1]< 75):
                ax.scatter(nose[0],nose[1],s=60,c='#FF1744',alpha = 0.1)
                ax.scatter(neck[0],neck[1],s=60,c='#F57C00',alpha = 0.1)
                ax.scatter(back[0],back[1],s=60,c='#00E676',alpha = 0.1)
                ax.scatter(root_tail[0],root_tail[1],s=60,c='#00E5FF',alpha = 0.1)
                ax.scatter(mid_tail[0],mid_tail[1],s=60,c='#607D5B',alpha = 0.1)
                
                ax.plot([nose[0],neck[0]],[nose[1],neck[1]],c='#90A4AE',alpha = 0.1,lw=1)
                ax.plot([neck[0],back[0]],[neck[1],back[1]],c='#90A4AE',alpha = 0.1,lw=1)
                ax.plot([back[0],root_tail[0]],[back[1],root_tail[1]],c='#90A4AE',alpha = 0.1,lw=1)
                ax.plot([root_tail[0],mid_tail[0]],[root_tail[1],mid_tail[1]],c='#90A4AE',alpha = 0.1,lw=1)
            elif (i == 'right_turning') & (nose[0] > -20) & (nose[1]< 75):
                ax.scatter(nose[0],nose[1],s=60,c='#FF1744',alpha = 0.1)
                ax.scatter(neck[0],neck[1],s=60,c='#F57C00',alpha = 0.1)
                ax.scatter(back[0],back[1],s=60,c='#00E676',alpha = 0.1)
                ax.scatter(root_tail[0],root_tail[1],s=60,c='#00E5FF',alpha = 0.1)
                ax.scatter(mid_tail[0],mid_tail[1],s=60,c='#607D5B',alpha = 0.1)
                
                ax.plot([nose[0],neck[0]],[nose[1],neck[1]],c='#90A4AE',alpha = 0.1,lw=1)
                ax.plot([neck[0],back[0]],[neck[1],back[1]],c='#90A4AE',alpha = 0.1,lw=1)
                ax.plot([back[0],root_tail[0]],[back[1],root_tail[1]],c='#90A4AE',alpha = 0.1,lw=1)
                ax.plot([root_tail[0],mid_tail[0]],[root_tail[1],mid_tail[1]],c='#90A4AE',alpha = 0.1,lw=1)
            elif (i == 'stepping') &  (nose[0] > -30) & (nose[0]< 30) & (nose[1]< 75):
                ax.scatter(nose[0],nose[1],s=60,c='#FF1744',alpha = 0.1)
                ax.scatter(neck[0],neck[1],s=60,c='#F57C00',alpha = 0.1)
                ax.scatter(back[0],back[1],s=60,c='#00E676',alpha = 0.1)
                ax.scatter(root_tail[0],root_tail[1],s=60,c='#00E5FF',alpha = 0.1)
                ax.scatter(mid_tail[0],mid_tail[1],s=60,c='#607D5B',alpha = 0.1)
                
                ax.plot([nose[0],neck[0]],[nose[1],neck[1]],c='#90A4AE',alpha = 0.1,lw=1)
                ax.plot([neck[0],back[0]],[neck[1],back[1]],c='#90A4AE',alpha = 0.1,lw=1)
                ax.plot([back[0],root_tail[0]],[back[1],root_tail[1]],c='#90A4AE',alpha = 0.1,lw=1)
                ax.plot([root_tail[0],mid_tail[0]],[root_tail[1],mid_tail[1]],c='#90A4AE',alpha = 0.1,lw=1)
            elif (i == 'running') &  (nose[0] > -30) & (nose[0]< 30) & (nose[1]< 75):
                ax.scatter(nose[0],nose[1],s=60,c='#FF1744',alpha = 0.1)
                ax.scatter(neck[0],neck[1],s=60,c='#F57C00',alpha = 0.1)
                ax.scatter(back[0],back[1],s=60,c='#00E676',alpha = 0.1)
                ax.scatter(root_tail[0],root_tail[1],s=60,c='#00E5FF',alpha = 0.1)
                ax.scatter(mid_tail[0],mid_tail[1],s=60,c='#607D5B',alpha = 0.1)
                
                ax.plot([nose[0],neck[0]],[nose[1],neck[1]],c='#90A4AE',alpha = 0.1,lw=1)
                ax.plot([neck[0],back[0]],[neck[1],back[1]],c='#90A4AE',alpha = 0.1,lw=1)
                ax.plot([back[0],root_tail[0]],[back[1],root_tail[1]],c='#90A4AE',alpha = 0.1,lw=1)
                ax.plot([root_tail[0],mid_tail[0]],[root_tail[1],mid_tail[1]],c='#90A4AE',alpha = 0.1,lw=1)
        ax.scatter(nose_average[0],nose_average[1],s=260,c='#FF1744',ec='black',alpha = 1)
        ax.scatter(neck_average[0],neck_average[1],s=260,c='#F57C00',ec='black',alpha = 1)
        ax.scatter(back_average[0],back_average[1],s=260,c='#00E676',ec='black',alpha = 1)
        ax.scatter(root_tail_average[0],root_tail_average[1],s=260,ec='black',c='#00E5FF',alpha = 1)
        ax.scatter(mid_tail_average[0],mid_tail_average[1],s=260,c='#607D5B',ec='black',alpha = 1)
        
        ax.plot([nose_average[0],neck_average[0]],[nose_average[1],neck_average[1]],c='black',alpha = 1,lw=5)
        ax.plot([neck_average[0],back_average[0]],[neck_average[1],back_average[1]],c='black',alpha = 1,lw=5)
        ax.plot([back_average[0],root_tail_average[0]],[back_average[1],root_tail_average[1]],c='black',alpha = 1,lw=5)
        ax.plot([root_tail_average[0],mid_tail_average[0]],[root_tail_average[1],mid_tail_average[1]],c='black',alpha = 1,lw=5)
        plt.title(i,fontsize = 25)
        ax.set_xlim(-120,120)
        ax.set_ylim(-120,120)
        #plt.savefig('{}/{}_skeleton_angle.png'.format(output_dir,i),dpi=600)
    df = pd.DataFrame(angle_info_dict)
    print(df)

#plot_angle(df_select)


def bodyShrinkage(df_select):
   angle_info_dict = {'origin_label':[],'movement_label':[],'nose-back_distance':[],'neck-back_distance':[],'root_tail-back_distance':[],'left_front_limb-back_distance':[],'right_front_limb-back_distance':[],'left_hind_limb-back_distance':[],'right_hind_limb-back_distance':[]}
   for i in df_select['origin_label'].unique():
       df_singleMov = df_select[df_select['origin_label'] == i]
       df_singleMov.reset_index(drop=True,inplace=True)
       movement_label = df_singleMov['new_label'][0]
       
       df_singleMov['nose-back_distance'] = np.sqrt(np.square(df_singleMov['nose_x']-df_singleMov['back_x'])+np.square(df_singleMov['nose_y']-df_singleMov['back_y'])+np.square(df_singleMov['nose_z']-df_singleMov['back_z']))
       df_singleMov['neck-back_distance'] = np.sqrt(np.square(df_singleMov['neck_x']-df_singleMov['back_x'])+np.square(df_singleMov['neck_y']-df_singleMov['back_y'])+np.square(df_singleMov['neck_z']-df_singleMov['back_z']))
       df_singleMov['root_tail-back_distance'] = np.sqrt(np.square(df_singleMov['root_tail_x']-df_singleMov['back_x'])+np.square(df_singleMov['root_tail_y']-df_singleMov['back_y'])+np.square(df_singleMov['root_tail_z']-df_singleMov['back_z']))
       df_singleMov['left_front_limb-back_distance'] = np.sqrt(np.square(df_singleMov['left_front_limb_x']-df_singleMov['back_x'])+np.square(df_singleMov['left_front_limb_y']-df_singleMov['back_y'])+np.square(df_singleMov['left_front_limb_z']-df_singleMov['back_z']))
       df_singleMov['right_front_limb-back_distance'] = np.sqrt(np.square(df_singleMov['right_front_limb_x']-df_singleMov['back_x'])+np.square(df_singleMov['right_front_limb_y']-df_singleMov['back_y'])+np.square(df_singleMov['right_front_limb_z']-df_singleMov['back_z']))
       df_singleMov['left_hind_limb-back_distance'] = np.sqrt(np.square(df_singleMov['left_hind_limb_x']-df_singleMov['back_x'])+np.square(df_singleMov['left_hind_limb_y']-df_singleMov['back_y'])+np.square(df_singleMov['left_hind_limb_z']-df_singleMov['back_z']))
       df_singleMov['right_hind_limb-back_distance'] = np.sqrt(np.square(df_singleMov['right_hind_limb_x']-df_singleMov['back_x'])+np.square(df_singleMov['right_hind_limb_y']-df_singleMov['back_y'])+np.square(df_singleMov['right_hind_limb_z']-df_singleMov['back_z']))
       
       angle_info_dict['origin_label'].append(i)
       angle_info_dict['movement_label'].append(movement_label)
       angle_info_dict['nose-back_distance'].append(df_singleMov['nose-back_distance'].mean())
       angle_info_dict['neck-back_distance'].append(df_singleMov['neck-back_distance'].mean())
       angle_info_dict['root_tail-back_distance'].append(df_singleMov['root_tail-back_distance'].mean())
       angle_info_dict['left_front_limb-back_distance'].append(df_singleMov['left_front_limb-back_distance'].mean())
       angle_info_dict['right_front_limb-back_distance'].append(df_singleMov['right_front_limb-back_distance'].mean())
       angle_info_dict['left_hind_limb-back_distance'].append(df_singleMov['left_hind_limb-back_distance'].mean())
       angle_info_dict['right_hind_limb-back_distance'].append(df_singleMov['right_hind_limb-back_distance'].mean())
   
   df = pd.DataFrame(angle_info_dict)
   df['sum'] = df.iloc[:,2:].mean(axis=1)
   return(df)
    
           
#a = bodyShrinkage(df_select)


'''

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

def align_normailied_coordinates(df):
    df_new = df.copy()
    for i in df_new.index:
        nose = [df_new.loc[i,'nose_x'],df_new.loc[i,'nose_y']]
        left_ear = [df_new.loc[i,'left_ear_x'],df_new.loc[i,'left_ear_y']]
        right_ear = [df_new.loc[i,'right_ear_x'],df_new.loc[i,'right_ear_y']]
        neck = [df_new.loc[i,'neck_x'],df_new.loc[i,'neck_y']]
        left_front_limb = [df_new.loc[i,'left_front_limb_x'],df_new.loc[i,'left_front_limb_y']]
        right_front_limb = [df_new.loc[i,'right_front_limb_x'],df_new.loc[i,'right_front_limb_y']]
        left_hind_limb = [df_new.loc[i,'left_hind_limb_x'],df_new.loc[i,'left_hind_limb_y']]
        right_hind_limb = [df_new.loc[i,'right_hind_limb_x'],df_new.loc[i,'right_hind_limb_y']]
        left_front_claw = [df_new.loc[i,'left_front_claw_x'],df_new.loc[i,'left_front_claw_y']]
        right_front_claw = [df_new.loc[i,'right_front_claw_x'],df_new.loc[i,'right_front_claw_y']]
        left_hind_claw = [df_new.loc[i,'left_hind_claw_x'],df_new.loc[i,'left_hind_claw_y']]
        right_hind_claw = [df_new.loc[i,'right_hind_claw_x'],df_new.loc[i,'right_hind_claw_y']]
        back = [df_new.loc[i,'back_x'],df_new.loc[i,'back_y']]
        root_tail = [df_new.loc[i,'root_tail_x'],df_new.loc[i,'root_tail_y']]
        mid_tail = [df_new.loc[i,'mid_tail_x'],df_new.loc[i,'mid_tail_y']]
        tip_tail = [df_new.loc[i,'tip_tail_x'],df_new.loc[i,'tip_tail_y']]
        
        clockwise_revise_angle = cal_ang((root_tail[0],root_tail[1]), (0,0), (0,-1))
        if root_tail[0]>=0:
            nose = Srotate(math.radians(clockwise_revise_angle),nose[0],nose[1],0,0)
            left_ear = Srotate(math.radians(clockwise_revise_angle),left_ear[0],left_ear[1],0,0)
            right_ear = Srotate(math.radians(clockwise_revise_angle),right_ear[0],right_ear[1],0,0)
            neck = Srotate(math.radians(clockwise_revise_angle),neck[0],neck[1],0,0)
            left_front_limb = Srotate(math.radians(clockwise_revise_angle),left_front_limb[0],left_front_limb[1],0,0)
            right_front_limb = Srotate(math.radians(clockwise_revise_angle),right_front_limb[0],right_front_limb[1],0,0)
            left_hind_limb = Srotate(math.radians(clockwise_revise_angle),left_hind_limb[0],left_hind_limb[1],0,0)
            right_hind_limb = Srotate(math.radians(clockwise_revise_angle),right_hind_limb[0],right_hind_limb[1],0,0)
            left_front_claw = Srotate(math.radians(clockwise_revise_angle),left_front_claw[0],left_front_claw[1],0,0)
            right_front_claw = Srotate(math.radians(clockwise_revise_angle),right_front_claw[0],right_front_claw[1],0,0)
            left_hind_claw = Srotate(math.radians(clockwise_revise_angle),left_hind_claw[0],left_hind_claw[1],0,0)
            right_hind_claw = Srotate(math.radians(clockwise_revise_angle),right_hind_claw[0],right_hind_claw[1],0,0)
            back = Srotate(math.radians(clockwise_revise_angle),back[0],back[1],0,0)
            root_tail = Srotate(math.radians(clockwise_revise_angle),root_tail[0],root_tail[1],0,0)
            mid_tail = Srotate(math.radians(clockwise_revise_angle),mid_tail[0],mid_tail[1],0,0)
            tip_tail = Srotate(math.radians(clockwise_revise_angle),tip_tail[0],tip_tail[1],0,0)
        else:
            nose = Nrotate(math.radians(clockwise_revise_angle),nose[0],nose[1],0,0)
            left_ear = Nrotate(math.radians(clockwise_revise_angle),left_ear[0],left_ear[1],0,0)
            right_ear = Nrotate(math.radians(clockwise_revise_angle),right_ear[0],right_ear[1],0,0)
            neck = Nrotate(math.radians(clockwise_revise_angle),neck[0],neck[1],0,0)
            left_front_limb = Nrotate(math.radians(clockwise_revise_angle),left_front_limb[0],left_front_limb[1],0,0)
            right_front_limb = Nrotate(math.radians(clockwise_revise_angle),right_front_limb[0],right_front_limb[1],0,0)
            left_hind_limb = Nrotate(math.radians(clockwise_revise_angle),left_hind_limb[0],left_hind_limb[1],0,0)
            right_hind_limb = Nrotate(math.radians(clockwise_revise_angle),right_hind_limb[0],right_hind_limb[1],0,0)
            left_front_claw = Nrotate(math.radians(clockwise_revise_angle),left_front_claw[0],left_front_claw[1],0,0)
            right_front_claw = Nrotate(math.radians(clockwise_revise_angle),right_front_claw[0],right_front_claw[1],0,0)
            left_hind_claw = Nrotate(math.radians(clockwise_revise_angle),left_hind_claw[0],left_hind_claw[1],0,0)
            right_hind_claw = Nrotate(math.radians(clockwise_revise_angle),right_hind_claw[0],right_hind_claw[1],0,0)
            back = Nrotate(math.radians(clockwise_revise_angle),back[0],back[1],0,0)
            root_tail = Nrotate(math.radians(clockwise_revise_angle),root_tail[0],root_tail[1],0,0)
            mid_tail = Nrotate(math.radians(clockwise_revise_angle),mid_tail[0],mid_tail[1],0,0)
            tip_tail = Nrotate(math.radians(clockwise_revise_angle),tip_tail[0],tip_tail[1],0,0)

        back_tail_long = abs(root_tail[1])
        back_tail_theoretical_long = 5
        nose = [(nose[0]-back[0])*(back_tail_theoretical_long/back_tail_long),(nose[1]-back[1])*(back_tail_theoretical_long/back_tail_long)]
        left_ear = [(left_ear[0]-back[0])*(back_tail_theoretical_long/back_tail_long),(left_ear[1]-back[1])*(back_tail_theoretical_long/back_tail_long)]
        right_ear = [(right_ear[0]-back[0])*(back_tail_theoretical_long/back_tail_long),(right_ear[1]-back[1])*(back_tail_theoretical_long/back_tail_long)]
        neck = [(neck[0]-back[0])*(back_tail_theoretical_long/back_tail_long),(neck[1]-back[1])*(back_tail_theoretical_long/back_tail_long)]
        left_front_limb = [(left_front_limb[0]-back[0])*(back_tail_theoretical_long/back_tail_long),(left_front_limb[1]-back[1])*(back_tail_theoretical_long/back_tail_long)]
        right_front_limb = [(right_front_limb[0]-back[0])*(back_tail_theoretical_long/back_tail_long),(right_front_limb[1]-back[1])*(back_tail_theoretical_long/back_tail_long)]
        left_hind_limb = [(left_hind_limb[0]-back[0])*(back_tail_theoretical_long/back_tail_long),(left_hind_limb[1]-back[1])*(back_tail_theoretical_long/back_tail_long)]
        right_hind_limb = [(right_hind_limb[0]-back[0])*(back_tail_theoretical_long/back_tail_long),(right_hind_limb[1]-back[1])*(back_tail_theoretical_long/back_tail_long)]
        left_front_claw = [(left_front_claw[0]-back[0])*(back_tail_theoretical_long/back_tail_long),(left_front_claw[1]-back[1])*(back_tail_theoretical_long/back_tail_long)]
        right_front_claw = [(right_front_claw[0]-back[0])*(back_tail_theoretical_long/back_tail_long),(right_front_claw[1]-back[1])*(back_tail_theoretical_long/back_tail_long)]
        left_hind_claw = [(left_hind_claw[0]-back[0])*(back_tail_theoretical_long/back_tail_long),(left_hind_claw[1]-back[1])*(back_tail_theoretical_long/back_tail_long)]
        right_hind_claw = [(right_hind_claw[0]-back[0])*(back_tail_theoretical_long/back_tail_long),(right_hind_claw[1]-back[1])*(back_tail_theoretical_long/back_tail_long)]
        root_tail = [(root_tail[0]-back[0])*(back_tail_theoretical_long/back_tail_long),(root_tail[1]-back[1])*(back_tail_theoretical_long/back_tail_long)]
        mid_tail = [(mid_tail[0]-back[0])*(back_tail_theoretical_long/back_tail_long),(mid_tail[1]-back[1])*(back_tail_theoretical_long/back_tail_long)]
        tip_tail = [(nose[0]-back[0])*(back_tail_theoretical_long/back_tail_long),(tip_tail[1]-back[1])*(back_tail_theoretical_long/back_tail_long)]
        
        df_new.loc[i,'nose_x'] = nose[0]
        df_new.loc[i,'nose_y'] = nose[1]
        df_new.loc[i,'left_ear_x'] = left_ear[0]
        df_new.loc[i,'left_ear_y'] = left_ear[1]
        df_new.loc[i,'right_ear_x'] = right_ear[0]
        df_new.loc[i,'right_ear_y'] = right_ear[1]
        df_new.loc[i,'neck_x'] = neck[0]
        df_new.loc[i,'neck_y'] = neck[1]
        df_new.loc[i,'left_front_limb_x'] = left_front_limb[0]
        df_new.loc[i,'left_front_limb_y'] = left_front_limb[1]
        df_new.loc[i,'right_front_limb_x'] = right_front_limb[0]
        df_new.loc[i,'right_front_limb_y'] = right_front_limb[1]
        df_new.loc[i,'left_hind_limb_x'] = left_hind_limb[0]
        df_new.loc[i,'left_hind_limb_y'] = left_hind_limb[1]
        df_new.loc[i,'right_hind_limb_x'] = right_hind_limb[0]
        df_new.loc[i,'right_hind_limb_y'] = right_hind_limb[1]
        df_new.loc[i,'left_front_claw_x'] = left_front_claw[0]
        df_new.loc[i,'left_front_claw_y'] = left_front_claw[1]
        df_new.loc[i,'right_front_claw_x'] = right_front_claw[0]
        df_new.loc[i,'right_front_claw_y'] = right_front_claw[1]
        df_new.loc[i,'left_hind_claw_x'] = left_hind_claw[0]
        df_new.loc[i,'left_hind_claw_y'] = left_hind_claw[1]
        df_new.loc[i,'right_hind_claw_x'] = right_hind_claw[0]
        df_new.loc[i,'right_hind_claw_y'] = right_hind_claw[1]
        df_new.loc[i,'back_x'] = back[0]
        df_new.loc[i,'back_y'] = back[1]
        df_new.loc[i,'root_tail_x'] = root_tail[0]
        df_new.loc[i,'root_tail_y'] = root_tail[1]
        df_new.loc[i,'mid_tail_x'] = mid_tail[0]
        df_new.loc[i,'mid_tail_y'] = mid_tail[1]
        df_new.loc[i,'tip_tail_x'] = tip_tail[0]
        df_new.loc[i,'tip_tail_y'] = tip_tail[1]
    return(df_new)













for i in df_locomotion['new_label'].unique():
    df_singleLocomotion = df_locomotion[df_locomotion['new_label'] == i]
    df_singleLocomotion.reset_index(drop=True,inplace=True)
    df_singleLocomotion.loc[len(df_singleLocomotion.index)] = df_singleLocomotion.mean(axis=0)
    
    nose = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_y']]
    left_ear = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_y']]
    right_ear = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_y']]
    neck = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_y']]
    left_front_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_y']]
    right_front_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_y']]
    left_hind_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_y']]
    right_hind_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_y']]
    left_front_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_y']]
    right_front_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_y']]
    left_hind_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_y']]
    right_hind_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_y']]
    back = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_y']]
    root_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_y']]
    mid_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_y']]
    tip_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_y']]

    #back and neck
    
    clockwise_revise_angle = cal_ang((root_tail[0],root_tail[1]), (0,0), (0,-1))
    if root_tail[0]>=0:
        nose = Srotate(math.radians(clockwise_revise_angle),nose[0],nose[1],0,0)
        left_ear = Srotate(math.radians(clockwise_revise_angle),left_ear[0],left_ear[1],0,0)
        right_ear = Srotate(math.radians(clockwise_revise_angle),right_ear[0],right_ear[1],0,0)
        neck = Srotate(math.radians(clockwise_revise_angle),neck[0],neck[1],0,0)
        left_front_limb = Srotate(math.radians(clockwise_revise_angle),left_front_limb[0],left_front_limb[1],0,0)
        right_front_limb = Srotate(math.radians(clockwise_revise_angle),right_front_limb[0],right_front_limb[1],0,0)
        left_hind_limb = Srotate(math.radians(clockwise_revise_angle),left_hind_limb[0],left_hind_limb[1],0,0)
        right_hind_limb = Srotate(math.radians(clockwise_revise_angle),right_hind_limb[0],right_hind_limb[1],0,0)
        left_front_claw = Srotate(math.radians(clockwise_revise_angle),left_front_claw[0],left_front_claw[1],0,0)
        right_front_claw = Srotate(math.radians(clockwise_revise_angle),right_front_claw[0],right_front_claw[1],0,0)
        left_hind_claw = Srotate(math.radians(clockwise_revise_angle),left_hind_claw[0],left_hind_claw[1],0,0)
        right_hind_claw = Srotate(math.radians(clockwise_revise_angle),right_hind_claw[0],right_hind_claw[1],0,0)
        back = Srotate(math.radians(clockwise_revise_angle),back[0],back[1],0,0)
        root_tail = Srotate(math.radians(clockwise_revise_angle),root_tail[0],root_tail[1],0,0)
        mid_tail = Srotate(math.radians(clockwise_revise_angle),mid_tail[0],mid_tail[1],0,0)
        tip_tail = Srotate(math.radians(clockwise_revise_angle),tip_tail[0],tip_tail[1],0,0)
    else:
        nose = Nrotate(math.radians(clockwise_revise_angle),nose[0],nose[1],0,0)
        left_ear = Nrotate(math.radians(clockwise_revise_angle),left_ear[0],left_ear[1],0,0)
        right_ear = Nrotate(math.radians(clockwise_revise_angle),right_ear[0],right_ear[1],0,0)
        neck = Nrotate(math.radians(clockwise_revise_angle),neck[0],neck[1],0,0)
        left_front_limb = Nrotate(math.radians(clockwise_revise_angle),left_front_limb[0],left_front_limb[1],0,0)
        right_front_limb = Nrotate(math.radians(clockwise_revise_angle),right_front_limb[0],right_front_limb[1],0,0)
        left_hind_limb = Nrotate(math.radians(clockwise_revise_angle),left_hind_limb[0],left_hind_limb[1],0,0)
        right_hind_limb = Nrotate(math.radians(clockwise_revise_angle),right_hind_limb[0],right_hind_limb[1],0,0)
        left_front_claw = Nrotate(math.radians(clockwise_revise_angle),left_front_claw[0],left_front_claw[1],0,0)
        right_front_claw = Nrotate(math.radians(clockwise_revise_angle),right_front_claw[0],right_front_claw[1],0,0)
        left_hind_claw = Nrotate(math.radians(clockwise_revise_angle),left_hind_claw[0],left_hind_claw[1],0,0)
        right_hind_claw = Nrotate(math.radians(clockwise_revise_angle),right_hind_claw[0],right_hind_claw[1],0,0)
        back = Nrotate(math.radians(clockwise_revise_angle),back[0],back[1],0,0)
        root_tail = Nrotate(math.radians(clockwise_revise_angle),root_tail[0],root_tail[1],0,0)
        mid_tail = Nrotate(math.radians(clockwise_revise_angle),mid_tail[0],mid_tail[1],0,0)
        tip_tail = Nrotate(math.radians(clockwise_revise_angle),tip_tail[0],tip_tail[1],0,0)
    
    back_tail_long = abs(root_tail[1])
    nose = [(nose[0]-back[0])*(2/back_tail_long),(nose[1]-back[1])*(2/back_tail_long)]
    left_ear = [(left_ear[0]-back[0])*(2/back_tail_long),(left_ear[1]-back[1])*(2/back_tail_long)]
    right_ear = [(right_ear[0]-back[0])*(2/back_tail_long),(right_ear[1]-back[1])*(2/back_tail_long)]
    neck = [(neck[0]-back[0])*(2/back_tail_long),(neck[1]-back[1])*(2/back_tail_long)]
    left_front_limb = [(left_front_limb[0]-back[0])*(2/back_tail_long),(left_front_limb[1]-back[1])*(2/back_tail_long)]
    right_front_limb = [(right_front_limb[0]-back[0])*(2/back_tail_long),(right_front_limb[1]-back[1])*(2/back_tail_long)]
    left_hind_limb = [(left_hind_limb[0]-back[0])*(2/back_tail_long),(left_hind_limb[1]-back[1])*(2/back_tail_long)]
    right_hind_limb = [(right_hind_limb[0]-back[0])*(2/back_tail_long),(right_hind_limb[1]-back[1])*(2/back_tail_long)]
    left_front_claw = [(left_front_claw[0]-back[0])*(2/back_tail_long),(left_front_claw[1]-back[1])*(2/back_tail_long)]
    right_front_claw = [(right_front_claw[0]-back[0])*(2/back_tail_long),(right_front_claw[1]-back[1])*(2/back_tail_long)]
    left_hind_claw = [(left_hind_claw[0]-back[0])*(2/back_tail_long),(left_hind_claw[1]-back[1])*(2/back_tail_long)]
    right_hind_claw = [(right_hind_claw[0]-back[0])*(2/back_tail_long),(right_hind_claw[1]-back[1])*(2/back_tail_long)]
    root_tail = [(root_tail[0]-back[0])*(2/back_tail_long),(root_tail[1]-back[1])*(2/back_tail_long)]
    mid_tail = [(mid_tail[0]-back[0])*(2/back_tail_long),(mid_tail[1]-back[1])*(2/back_tail_long)]
    tip_tail = [(nose[0]-back[0])*(2/back_tail_long),(tip_tail[1]-back[1])*(2/back_tail_long)]
    
    
    
    fig,ax = fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5),constrained_layout=True)
    ax.scatter(nose[0],nose[1],s=40,c='green',alpha = 0.2)
    ax.scatter(left_ear[0],left_ear[1],s=40,c='green',alpha = 0.2)
    ax.scatter(right_ear[0],right_ear[1],s=40,c='green',alpha = 0.2)
    
    ax.plot([nose[0],left_ear[0]],[nose[1],left_ear[1]],c='black',alpha = 0.2)
    ax.plot([nose[0],right_ear[0]],[nose[1],right_ear[1]],c='black',alpha = 0.2)
    ax.plot([left_ear[0],right_ear[0]],[left_ear[1],right_ear[1]],c='black',alpha = 0.2)
    
    ax.scatter(neck[0],neck[1],s=40,c='#607D8B',alpha = 0.2)
    ax.scatter(back[0],back[1],s=40,c='#607D8B',alpha = 0.2)
    ax.scatter(root_tail[0],root_tail[1],s=40,c='#607D8B',alpha = 0.2)
    ax.scatter(mid_tail[0],mid_tail[1],s=40,c='#607D8B',alpha = 0.2)
    #ax.scatter(tip_tail[0],tip_tail[1],s=40,c='orange')
    
    ax.scatter(left_front_limb[0],left_front_limb[1],s=40,c='blue',alpha = 0.2)
    ax.scatter(right_front_limb[0],right_front_limb[1],s=40,c='blue',alpha = 0.2)
    ax.scatter(left_hind_limb[0],left_hind_limb[1],s=40,c='blue',alpha = 0.2)
    ax.scatter(right_hind_limb[0],right_hind_limb[1],s=40,c='blue',alpha = 0.2)
    
    ax.scatter(left_front_claw[0],left_front_claw[1],s=200,c='#FF7043',marker='^',ec = 'black')
    ax.scatter(right_front_claw[0],right_front_claw[1],s=200,c='#FF7043',marker='^',ec = 'black')
    ax.scatter(left_hind_claw[0],left_hind_claw[1],s=200,c='#FF7043',marker='^',ec = 'black')
    ax.scatter(right_hind_claw[0],right_hind_claw[1],s=200,c='#FF7043',marker='^',ec = 'black')
    
    ax.plot([left_front_limb[0],left_hind_limb[0]],[left_front_limb[1],left_hind_limb[1]],c='black',alpha = 0.2)
    ax.plot([left_hind_limb[0],right_hind_limb[0]],[left_hind_limb[1],right_hind_limb[1]],c='black',alpha = 0.2)
    ax.plot([right_hind_limb[0],right_front_limb[0]],[right_hind_limb[1],right_front_limb[1]],c='black',alpha = 0.2)
    ax.plot([right_front_limb[0],left_front_limb[0]],[right_front_limb[1],left_front_limb[1]],c='black',alpha = 0.2)
    
    ax.plot([left_ear[0],neck[0]],[left_ear[1],neck[1]],c='black',alpha = 0.2)
    ax.plot([right_ear[0],neck[0]],[right_ear[1],neck[1]],c='black',alpha = 0.2)
    ax.plot([left_front_limb[0],neck[0]],[left_front_limb[1],neck[1]],c='black',alpha = 0.2)
    ax.plot([right_front_limb[0],neck[0]],[right_front_limb[1],neck[1]],c='black',alpha = 0.2)
    ax.plot([left_hind_limb[0],root_tail[0]],[left_hind_limb[1],root_tail[1]],c='black',alpha = 0.2)
    ax.plot([right_hind_limb[0],root_tail[0]],[right_hind_limb[1],root_tail[1]],c='black',alpha = 0.2)
    
    ax.plot([root_tail[0],mid_tail[0]],[root_tail[1],mid_tail[1]],c='black',alpha = 0.2)
    #ax.plot([mid_tail[0],tip_tail[0]],[mid_tail[1],tip_tail[1]],c='black')
    

    ax.plot([left_front_limb[0],back[0]],[left_front_limb[1],back[1]],c='black',alpha = 0.2)
    ax.plot([right_front_limb[0],back[0]],[right_front_limb[1],back[1]],c='black',alpha = 0.2)
    ax.plot([left_hind_limb[0],back[0]],[left_hind_limb[1],back[1]],c='black',alpha = 0.2)
    ax.plot([right_hind_limb[0],back[0]],[right_hind_limb[1],back[1]],c='black',alpha = 0.2)
    
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    plt.title(i,fontsize = 25)
## 转弯角度计算

## 步宽

## 高度分布

## 弯腰程度

## 身体收缩程度


'''